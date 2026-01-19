"""
Path manipulation utilities for cross-platform support.

© 2026 MBP LLC. All rights reserved.
"""

import os
from pathlib import Path, PureWindowsPath, PurePosixPath
from typing import Optional


def sanitize_filename(filename: str, replacement: str = '_') -> str:
    """
    Sanitize filename by removing invalid characters.

    Args:
        filename: Original filename
        replacement: Character to replace invalid chars with

    Returns:
        Sanitized filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, replacement)
    return filename.strip()


def get_relative_path(path: Path, base: Path) -> Optional[Path]:
    """
    Get relative path from base to path.

    Args:
        path: Target path
        base: Base path

    Returns:
        Relative path or None if not relative
    """
    try:
        return path.relative_to(base)
    except ValueError:
        return None


def normalize_path(path: Path) -> Path:
    """
    Normalize path to absolute resolved path.

    Args:
        path: Input path

    Returns:
        Normalized absolute path
    """
    return path.resolve()


def is_path_safe(path: Path, base: Path) -> bool:
    """
    Check if path is safe (within base directory).

    Args:
        path: Path to check
        base: Base directory

    Returns:
        True if safe, False otherwise
    """
    try:
        normalized = path.resolve()
        normalized_base = base.resolve()
        return normalized.is_relative_to(normalized_base)
    except Exception:
        return False


def ensure_windows_long_path(path: Path) -> str:
    """
    Ensure Windows long path support (\\\\?\\ prefix).

    Args:
        path: Input path

    Returns:
        Path string with long path prefix if needed
    """
    path_str = str(path.resolve())

    if os.name == 'nt' and not path_str.startswith('\\\\?\\'):
        if len(path_str) > 260:
            return f"\\\\?\\{path_str}"

    return path_str


def split_path_components(path: Path) -> list[str]:
    """
    Split path into individual components.

    Args:
        path: Input path

    Returns:
        List of path components
    """
    return list(path.parts)
