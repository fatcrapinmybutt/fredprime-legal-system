import os
import shutil
import argparse
import logging
import platform
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm

# Mapping of file extensions to categories
CATEGORIES = {
    "Documents": [
        ".pdf",
        ".doc",
        ".docx",
        ".txt",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".csv",
        ".odt",
        ".ods",
        ".odp",
    ],
    "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".svg"],
    "Music": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma"],
    "Videos": [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv"],
    "Archives": [".zip", ".rar", ".7z", ".tar", ".gz", ".bz2"],
    "Code": [
        ".py",
        ".js",
        ".html",
        ".css",
        ".java",
        ".c",
        ".cpp",
        ".cs",
        ".rb",
        ".php",
    ],
}

DEFAULT_CATEGORY = "Other"
ORGANIZED_FOLDER = "Organized"

# Required drives for litigation system
REQUIRED_DRIVES = ["Q", "D", "Z"]
FORBIDDEN_DRIVE = "C"


def get_drive_letter(path: Path) -> str | None:
    """Extract the drive letter from a path on Windows.
    
    Returns the uppercase drive letter without colon, or None if not a Windows drive path.
    """
    if platform.system() != "Windows":
        return None
    
    # Resolve to handle symbolic links and junctions
    try:
        resolved_path = path.resolve()
    except (OSError, RuntimeError):
        # If resolution fails, use the original path
        resolved_path = path
    
    path_str = str(resolved_path)
    if len(path_str) >= 2 and path_str[1] == ":":
        return path_str[0].upper()
    return None


def validate_drive_path(path: Path, required_drives: list[str] = None) -> tuple[bool, str]:
    """Validate that a path meets drive requirements for the litigation system.
    
    Args:
        path: The path to validate
        required_drives: List of required drive letters (defaults to REQUIRED_DRIVES)
        
    Returns:
        tuple: (is_valid, error_message)
               is_valid is True if validation passes, False otherwise
               error_message describes the validation failure, or empty string on success
    """
    if required_drives is None:
        required_drives = REQUIRED_DRIVES
    
    # Check if path exists
    if not path.exists():
        return False, f"Path does not exist: {path}"
    
    # For non-Windows systems, skip drive validation
    if platform.system() != "Windows":
        return True, ""
    
    # Get the drive letter (this also resolves symlinks/junctions)
    drive_letter = get_drive_letter(path)
    
    if drive_letter is None:
        return False, f"Could not determine drive letter for path: {path}"
    
    # Check if it's the forbidden C: drive
    if drive_letter == FORBIDDEN_DRIVE:
        return False, f"C: drive is forbidden for litigation data. Path resolves to: {path.resolve()}"
    
    # Check if it's one of the required drives
    if drive_letter not in required_drives:
        return False, f"Drive {drive_letter}: is not one of the required drives: {', '.join(required_drives)}"
    
    return True, ""


def check_required_drives_exist(required_drives: list[str] = None) -> tuple[bool, list[str]]:
    """Check if all required drives are present on the system.
    
    Args:
        required_drives: List of required drive letters (defaults to REQUIRED_DRIVES)
        
    Returns:
        tuple: (all_present, missing_drives)
               all_present is True if all drives exist, False otherwise
               missing_drives is a list of drive letters that are missing
    """
    if required_drives is None:
        required_drives = REQUIRED_DRIVES
    
    # Skip check on non-Windows systems
    if platform.system() != "Windows":
        return True, []
    
    missing = []
    for drive in required_drives:
        drive_path = Path(f"{drive}:/")
        if not drive_path.exists():
            missing.append(drive)
    
    return len(missing) == 0, missing


def get_category(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    for category, extensions in CATEGORIES.items():
        if ext in extensions:
            return category
    return DEFAULT_CATEGORY


def safe_move(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        base = dest.stem
        suffix = dest.suffix
        counter = 1
        while True:
            new_name = f"{base}_{counter}{suffix}"
            new_dest = dest.with_name(new_name)
            if not new_dest.exists():
                dest = new_dest
                break
            counter += 1
    shutil.move(str(src), str(dest))


def move_file(base_output: Path, file_path: Path) -> None:
    try:
        category = get_category(file_path)
        dest_dir = base_output / category
        destination = dest_dir / file_path.name
        safe_move(file_path, destination)
        logging.info("Moved %s -> %s", file_path, destination)
    except Exception as e:
        logging.error("Failed to move %s: %s", file_path, e)


def remove_empty_dirs(base_path: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(base_path, topdown=False):
        p = Path(dirpath)
        if not list(p.glob("*")):
            try:
                p.rmdir()
                logging.info("Removed empty directory %s", p)
            except Exception as e:
                logging.error("Failed to remove %s: %s", p, e)


def organize_drive(target_path: Path, output_path: Path | None = None) -> None:
    base_output = output_path.resolve() if output_path else target_path / ORGANIZED_FOLDER
    base_output.mkdir(exist_ok=True)

    files_to_move = []
    for root, dirs, files in os.walk(target_path):
        # Skip the output directory itself
        if ORGANIZED_FOLDER in Path(root).parts:
            continue
        for file in files:
            files_to_move.append(Path(root) / file)

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        for f in tqdm(files_to_move, desc="Organizing", unit="file"):
            executor.submit(move_file, base_output, f)

    remove_empty_dirs(target_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Organize files on a drive")
    parser.add_argument("path", nargs="?", default="F:/", help="Path to organize (default F:/)")
    parser.add_argument("--log", default="organize_drive.log", help="Log file path")
    parser.add_argument("--output", default=None, help="Optional output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        filename=args.log,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    target_path = Path(args.path)
    
    # Validate the path
    is_valid, error_msg = validate_drive_path(target_path)
    if not is_valid:
        print(f"Error: {error_msg}")
        return 1
    
    # Check if required drives exist
    all_present, missing = check_required_drives_exist()
    if not all_present:
        print(f"Warning: Missing required drives: {', '.join(missing)}")
        # Note: We continue anyway as the target drive is valid
    
    output_path = Path(args.output).resolve() if args.output else None
    organize_drive(target_path, output_path)
    print("Organization complete.")
    return 0


if __name__ == "__main__":
    main()
