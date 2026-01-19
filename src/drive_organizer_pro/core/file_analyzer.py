"""
File analyzer for bucket classification.

© 2026 MBP LLC. All rights reserved.
"""

from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
from ..utils.logger import logger


class FileAnalyzer:
    """Analyzes files for classification and organization."""

    def __init__(self, config_manager):
        """
        Initialize file analyzer.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager

    def get_bucket_for_file(self, file_path: Path) -> str:
        """
        Determine bucket for a file based on extension.

        Args:
            file_path: Path to file

        Returns:
            Bucket name
        """
        extension = file_path.suffix.lower()
        return self.config_manager.get_bucket_for_extension(extension)

    def get_file_metadata(self, file_path: Path) -> Optional[Dict]:
        """
        Extract file metadata.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with file metadata or None if error
        """
        try:
            stat = file_path.stat()
            return {
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'accessed': datetime.fromtimestamp(stat.st_atime),
                'extension': file_path.suffix.lower(),
                'name': file_path.name,
                'stem': file_path.stem
            }
        except Exception as e:
            logger.error(f"Failed to get metadata for {file_path}: {e}")
            return None

    def should_skip_file(self, file_path: Path) -> bool:
        """
        Check if file should be skipped.

        Args:
            file_path: Path to file

        Returns:
            True if should skip, False otherwise
        """
        # Skip hidden files
        if file_path.name.startswith('.'):
            return True

        # Skip system files
        system_files = ['desktop.ini', 'thumbs.db', '.ds_store']
        if file_path.name.lower() in system_files:
            return True

        return False

    def get_file_category_info(self, file_path: Path) -> Dict:
        """
        Get comprehensive category information for a file.

        Args:
            file_path: Path to file

        Returns:
            Dictionary with category info
        """
        metadata = self.get_file_metadata(file_path)
        bucket = self.get_bucket_for_file(file_path)

        return {
            'bucket': bucket,
            'metadata': metadata,
            'should_skip': self.should_skip_file(file_path)
        }
