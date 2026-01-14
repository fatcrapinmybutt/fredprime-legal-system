import os
import shutil
import argparse
import logging
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
    if not target_path.exists():
        print(f"Path {target_path} does not exist.")
        return
    output_path = Path(args.output).resolve() if args.output else None
    organize_drive(target_path, output_path)
    print("Organization complete.")


if __name__ == "__main__":
    main()
