"""
Duplicate file detection and handling.

© 2026 MBP LLC. All rights reserved.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from ..utils.hash_utils import calculate_md5, calculate_sha256
from ..utils.logger import logger


class DuplicateHandler:
    """Handles duplicate file detection and management."""

    def __init__(self, index_file: Optional[Path] = None, algorithm: str = 'md5'):
        """
        Initialize duplicate handler.

        Args:
            index_file: Path to persistent dedupe index file
            algorithm: Hash algorithm to use ('md5' or 'sha256')
        """
        self.algorithm = algorithm
        self.index_file = index_file
        self.hash_index: Dict[str, Path] = {}
        self.duplicates: List[tuple[Path, Path]] = []

        if index_file and index_file.exists():
            self.load_index()

    def calculate_hash(self, file_path: Path) -> Optional[str]:
        """
        Calculate hash for a file.

        Args:
            file_path: Path to file

        Returns:
            Hash string or None if error
        """
        if self.algorithm == 'md5':
            return calculate_md5(file_path)
        elif self.algorithm == 'sha256':
            return calculate_sha256(file_path)
        else:
            logger.error(f"Unsupported algorithm: {self.algorithm}")
            return None

    def check_duplicate(self, file_path: Path) -> Optional[Path]:
        """
        Check if file is a duplicate.

        Args:
            file_path: Path to file

        Returns:
            Path to original file if duplicate, None otherwise
        """
        file_hash = self.calculate_hash(file_path)

        if not file_hash:
            return None

        if file_hash in self.hash_index:
            original = self.hash_index[file_hash]
            if original.exists() and original != file_path:
                logger.info(f"Duplicate detected: {file_path} -> {original}")
                return original

        # Not a duplicate, add to index
        self.hash_index[file_hash] = file_path
        return None

    def scan_directory(self, directory: Path, recursive: bool = True) -> List[tuple[Path, Path]]:
        """
        Scan directory for duplicates.

        Args:
            directory: Directory to scan
            recursive: Whether to scan recursively

        Returns:
            List of (duplicate, original) tuples
        """
        self.duplicates.clear()
        pattern = '**/*' if recursive else '*'

        files = [f for f in directory.glob(pattern) if f.is_file()]
        total = len(files)

        logger.info(f"Scanning {total} files for duplicates...")

        for i, file_path in enumerate(files, 1):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{total} files scanned")

            original = self.check_duplicate(file_path)
            if original:
                self.duplicates.append((file_path, original))

        logger.info(f"Found {len(self.duplicates)} duplicates")
        return self.duplicates

    def quarantine_duplicates(self, quarantine_dir: Path, dry_run: bool = False) -> int:
        """
        Move duplicate files to quarantine directory.

        Args:
            quarantine_dir: Directory to move duplicates to
            dry_run: If True, don't actually move files

        Returns:
            Number of files quarantined
        """
        if not self.duplicates:
            logger.warning("No duplicates to quarantine")
            return 0

        quarantine_dir.mkdir(parents=True, exist_ok=True)
        count = 0

        for duplicate, original in self.duplicates:
            try:
                if dry_run:
                    logger.info(f"[DRY RUN] Would quarantine: {duplicate}")
                else:
                    dest = quarantine_dir / duplicate.name
                    # Resolve name collision
                    if dest.exists():
                        stem = dest.stem
                        suffix = dest.suffix
                        counter = 1
                        while dest.exists():
                            dest = quarantine_dir / f"{stem}_{counter}{suffix}"
                            counter += 1

                    duplicate.rename(dest)
                    logger.info(f"Quarantined: {duplicate} -> {dest}")
                count += 1

            except Exception as e:
                logger.error(f"Failed to quarantine {duplicate}: {e}")

        return count

    def load_index(self) -> bool:
        """
        Load persistent dedupe index.

        Returns:
            True if successful, False otherwise
        """
        if not self.index_file or not self.index_file.exists():
            return False

        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Convert string paths back to Path objects
            self.hash_index = {h: Path(p) for h, p in data.items()}
            logger.info(f"Loaded {len(self.hash_index)} entries from index")
            return True

        except Exception as e:
            logger.error(f"Failed to load index from {self.index_file}: {e}")
            return False

    def save_index(self) -> bool:
        """
        Save persistent dedupe index.

        Returns:
            True if successful, False otherwise
        """
        if not self.index_file:
            return False

        try:
            self.index_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert Path objects to strings for JSON
            data = {h: str(p) for h, p in self.hash_index.items()}

            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.hash_index)} entries to index")
            return True

        except Exception as e:
            logger.error(f"Failed to save index to {self.index_file}: {e}")
            return False

    def clear_index(self) -> None:
        """Clear the hash index."""
        self.hash_index.clear()
        logger.info("Cleared hash index")

    def get_statistics(self) -> Dict:
        """
        Get statistics about duplicates.

        Returns:
            Dictionary with statistics
        """
        return {
            'total_indexed': len(self.hash_index),
            'duplicates_found': len(self.duplicates),
            'algorithm': self.algorithm
        }
