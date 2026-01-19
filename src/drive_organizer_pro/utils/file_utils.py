"""
Safe file operation utilities for DriveOrganizerPro.

© 2026 MBP LLC. All rights reserved.
"""

import shutil
from pathlib import Path
from typing import Optional
from ..utils.logger import logger


def safe_move(src: Path, dest: Path, overwrite: bool = False) -> bool:
    """
    Safely move a file from source to destination.

    Args:
        src: Source file path
        dest: Destination file path
        overwrite: Whether to overwrite existing files

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create destination directory
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Handle existing file
        if dest.exists() and not overwrite:
            dest = resolve_name_collision(dest)

        # Move file
        shutil.move(str(src), str(dest))
        logger.debug(f"Moved: {src} -> {dest}")
        return True

    except Exception as e:
        logger.error(f"Failed to move {src} to {dest}: {e}")
        return False


def safe_copy(src: Path, dest: Path, overwrite: bool = False) -> bool:
    """
    Safely copy a file from source to destination.

    Args:
        src: Source file path
        dest: Destination file path
        overwrite: Whether to overwrite existing files

    Returns:
        True if successful, False otherwise
    """
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists() and not overwrite:
            dest = resolve_name_collision(dest)

        shutil.copy2(str(src), str(dest))
        logger.debug(f"Copied: {src} -> {dest}")
        return True

    except Exception as e:
        logger.error(f"Failed to copy {src} to {dest}: {e}")
        return False


def resolve_name_collision(path: Path, max_attempts: int = 9999) -> Path:
    """
    Resolve file name collisions by appending a counter.

    Args:
        path: Original file path
        max_attempts: Maximum number of attempts

    Returns:
        New path with unique name
    """
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent

    for i in range(1, max_attempts + 1):
        new_path = parent / f"{stem} ({i}){suffix}"
        if not new_path.exists():
            return new_path

    # Fallback with timestamp
    import time
    timestamp = int(time.time())
    return parent / f"{stem}__{timestamp}{suffix}"


def remove_empty_dirs(root: Path, preserve_root: bool = True) -> int:
    """
    Recursively remove empty directories.

    Args:
        root: Root directory to start from
        preserve_root: Whether to keep the root directory

    Returns:
        Number of directories removed
    """
    count = 0

    for dirpath in sorted(root.rglob('*'), reverse=True):
        if not dirpath.is_dir():
            continue

        try:
            if not any(dirpath.iterdir()):
                if preserve_root and dirpath == root:
                    continue
                dirpath.rmdir()
                count += 1
                logger.debug(f"Removed empty directory: {dirpath}")
        except Exception as e:
            logger.error(f"Failed to remove directory {dirpath}: {e}")

    return count


def get_file_size(path: Path) -> Optional[int]:
    """
    Get file size in bytes.

    Args:
        path: File path

    Returns:
        File size in bytes or None if error
    """
    try:
        return path.stat().st_size
    except Exception as e:
        logger.error(f"Failed to get size for {path}: {e}")
        return None


def is_file_accessible(path: Path) -> bool:
    """
    Check if file is accessible for reading.

    Args:
        path: File path

    Returns:
        True if accessible, False otherwise
    """
    try:
        with open(path, 'rb'):
            pass
        return True
    except Exception:
        return False
