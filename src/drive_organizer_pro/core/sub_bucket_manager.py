"""
Sub-bucket manager for keyword-based organization.

© 2026 MBP LLC. All rights reserved.
"""

import re
from pathlib import Path
from typing import Optional, List
from ..utils.logger import logger


class SubBucketManager:
    """Manages sub-bucket organization based on keywords."""

    def __init__(self, sub_bucket_config: dict):
        """
        Initialize sub-bucket manager.

        Args:
            sub_bucket_config: Sub-bucket configuration dictionary
        """
        self.sub_buckets = sub_bucket_config.get('sub_buckets', [])
        self.keyword_mappings = sub_bucket_config.get('keyword_mappings', {})

    def detect_sub_bucket(self, file_path: Path) -> Optional[str]:
        """
        Detect appropriate sub-bucket for a file based on keywords.

        Args:
            file_path: Path to file

        Returns:
            Sub-bucket name or None if no match
        """
        filename_lower = file_path.name.lower()
        stem_lower = file_path.stem.lower()

        # Check each sub-bucket's keywords
        for sub_bucket, keywords in self.keyword_mappings.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()
                # Check if keyword appears in filename
                if keyword_lower in filename_lower or keyword_lower in stem_lower:
                    logger.debug(f"File {file_path.name} matched sub-bucket {sub_bucket} (keyword: {keyword})")
                    return sub_bucket

        return None

    def get_sub_bucket_path(self, bucket_path: Path, file_path: Path) -> Path:
        """
        Get the full path including sub-bucket if applicable.

        Args:
            bucket_path: Main bucket path
            file_path: File to organize

        Returns:
            Path with sub-bucket if applicable, otherwise bucket path
        """
        sub_bucket = self.detect_sub_bucket(file_path)

        if sub_bucket:
            return bucket_path / sub_bucket
        else:
            return bucket_path

    def create_sub_buckets(self, bucket_path: Path) -> bool:
        """
        Create all sub-bucket directories within a bucket.

        Args:
            bucket_path: Main bucket path

        Returns:
            True if successful, False otherwise
        """
        try:
            for sub_bucket in self.sub_buckets:
                sub_path = bucket_path / sub_bucket
                sub_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created sub-bucket: {sub_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create sub-buckets in {bucket_path}: {e}")
            return False

    def list_sub_buckets(self) -> List[str]:
        """
        Get list of all sub-bucket names.

        Returns:
            List of sub-bucket names
        """
        return self.sub_buckets.copy()

    def get_keywords_for_sub_bucket(self, sub_bucket: str) -> List[str]:
        """
        Get keywords associated with a sub-bucket.

        Args:
            sub_bucket: Sub-bucket name

        Returns:
            List of keywords
        """
        return self.keyword_mappings.get(sub_bucket, [])
