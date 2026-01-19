"""
Configuration manager for DriveOrganizerPro.

© 2026 MBP LLC. All rights reserved.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from ..utils.logger import logger


class ConfigManager:
    """Manages configuration loading and saving."""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            config_dir = Path(__file__).parent
        self.config_dir = Path(config_dir)
        self.buckets: Dict[str, List[str]] = {}
        self.sub_buckets: Dict[str, List[str]] = {}
        self.user_settings: Dict = {}

    def load_buckets(self, config_file: Optional[Path] = None) -> Dict[str, List[str]]:
        """
        Load bucket configuration.

        Args:
            config_file: Path to bucket config file

        Returns:
            Dictionary mapping bucket names to extensions
        """
        if config_file is None:
            config_file = self.config_dir / "default_buckets.json"

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.buckets = json.load(f)
            logger.info(f"Loaded {len(self.buckets)} buckets from {config_file}")
            return self.buckets
        except Exception as e:
            logger.error(f"Failed to load buckets from {config_file}: {e}")
            return self._get_default_buckets()

    def load_sub_buckets(self, config_file: Optional[Path] = None) -> Dict:
        """
        Load sub-bucket configuration.

        Args:
            config_file: Path to sub-bucket config file

        Returns:
            Dictionary with sub-bucket definitions and keyword mappings
        """
        if config_file is None:
            config_file = self.config_dir / "sub_buckets.json"

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self.sub_buckets = json.load(f)
            logger.info(f"Loaded sub-bucket config from {config_file}")
            return self.sub_buckets
        except Exception as e:
            logger.error(f"Failed to load sub-buckets from {config_file}: {e}")
            return {"sub_buckets": [], "keyword_mappings": {}}

    def save_buckets(self, config_file: Path) -> bool:
        """
        Save bucket configuration.

        Args:
            config_file: Path to save config file

        Returns:
            True if successful, False otherwise
        """
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.buckets, f, indent=2)
            logger.info(f"Saved bucket config to {config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save buckets to {config_file}: {e}")
            return False

    def get_bucket_for_extension(self, extension: str) -> str:
        """
        Get bucket name for a file extension.

        Args:
            extension: File extension (with or without dot)

        Returns:
            Bucket name
        """
        if not extension.startswith('.'):
            extension = f'.{extension}'

        extension = extension.lower()

        for bucket, extensions in self.buckets.items():
            if extension in extensions:
                return bucket

        return "Miscellaneous"

    def _get_default_buckets(self) -> Dict[str, List[str]]:
        """Get default bucket configuration."""
        return {
            "Documents": [".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt"],
            "Spreadsheets": [".xls", ".xlsx", ".csv", ".ods"],
            "Images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg", ".webp"],
            "Videos": [".mp4", ".avi", ".mkv", ".mov", ".wmv"],
            "Audio": [".mp3", ".wav", ".flac", ".aac", ".ogg"],
            "Archives": [".zip", ".rar", ".7z", ".tar", ".gz"],
            "Code": [".py", ".js", ".html", ".css", ".cpp", ".java"],
            "Miscellaneous": []
        }

    def add_bucket(self, name: str, extensions: List[str]) -> bool:
        """
        Add a new bucket.

        Args:
            name: Bucket name
            extensions: List of file extensions

        Returns:
            True if successful, False otherwise
        """
        try:
            self.buckets[name] = extensions
            logger.info(f"Added bucket: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add bucket {name}: {e}")
            return False

    def remove_bucket(self, name: str) -> bool:
        """
        Remove a bucket.

        Args:
            name: Bucket name

        Returns:
            True if successful, False otherwise
        """
        try:
            if name in self.buckets:
                del self.buckets[name]
                logger.info(f"Removed bucket: {name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove bucket {name}: {e}")
            return False
