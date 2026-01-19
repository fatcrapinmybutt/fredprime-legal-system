"""
Hashing utilities for file deduplication.

© 2026 MBP LLC. All rights reserved.
"""

import hashlib
from pathlib import Path
from typing import Optional, Callable
from ..utils.logger import logger


def calculate_md5(file_path: Path, chunk_size: int = 8192,
                  progress_callback: Optional[Callable[[int, int], None]] = None) -> Optional[str]:
    """
    Calculate MD5 hash of a file.

    Args:
        file_path: Path to file
        chunk_size: Size of chunks to read
        progress_callback: Optional callback(bytes_read, total_bytes)

    Returns:
        MD5 hash string or None if error
    """
    try:
        md5_hash = hashlib.md5()
        total_size = file_path.stat().st_size
        bytes_read = 0

        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                md5_hash.update(chunk)
                bytes_read += len(chunk)
                if progress_callback:
                    progress_callback(bytes_read, total_size)

        return md5_hash.hexdigest()

    except Exception as e:
        logger.error(f"Failed to calculate MD5 for {file_path}: {e}")
        return None


def calculate_sha256(file_path: Path, chunk_size: int = 8192,
                     progress_callback: Optional[Callable[[int, int], None]] = None) -> Optional[str]:
    """
    Calculate SHA256 hash of a file.

    Args:
        file_path: Path to file
        chunk_size: Size of chunks to read
        progress_callback: Optional callback(bytes_read, total_bytes)

    Returns:
        SHA256 hash string or None if error
    """
    try:
        sha256_hash = hashlib.sha256()
        total_size = file_path.stat().st_size
        bytes_read = 0

        with open(file_path, 'rb') as f:
            while chunk := f.read(chunk_size):
                sha256_hash.update(chunk)
                bytes_read += len(chunk)
                if progress_callback:
                    progress_callback(bytes_read, total_size)

        return sha256_hash.hexdigest()

    except Exception as e:
        logger.error(f"Failed to calculate SHA256 for {file_path}: {e}")
        return None


def calculate_file_hash(file_path: Path, algorithm: str = 'md5',
                       chunk_size: int = 8192) -> Optional[str]:
    """
    Calculate hash of a file using specified algorithm.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5' or 'sha256')
        chunk_size: Size of chunks to read

    Returns:
        Hash string or None if error
    """
    if algorithm.lower() == 'md5':
        return calculate_md5(file_path, chunk_size)
    elif algorithm.lower() == 'sha256':
        return calculate_sha256(file_path, chunk_size)
    else:
        logger.error(f"Unsupported hash algorithm: {algorithm}")
        return None
