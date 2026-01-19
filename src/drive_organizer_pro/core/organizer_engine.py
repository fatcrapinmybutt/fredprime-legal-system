"""
Main organizer engine for DriveOrganizerPro.

© 2026 MBP LLC. All rights reserved.
"""

from pathlib import Path
from typing import List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from .file_analyzer import FileAnalyzer
from .sub_bucket_manager import SubBucketManager
from .duplicate_handler import DuplicateHandler
from .backup_manager import BackupManager
from ..config.config_manager import ConfigManager
from ..utils.file_utils import safe_move, remove_empty_dirs
from ..utils.logger import logger


class OrganizerEngine:
    """Main drive organization engine."""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize organizer engine.

        Args:
            config_dir: Configuration directory path
        """
        self.config_manager = ConfigManager(config_dir)
        self.config_manager.load_buckets()
        sub_bucket_config = self.config_manager.load_sub_buckets()

        self.file_analyzer = FileAnalyzer(self.config_manager)
        self.sub_bucket_manager = SubBucketManager(sub_bucket_config)
        self.duplicate_handler = DuplicateHandler()
        self.backup_manager = BackupManager()

        self.stats = {
            'files_processed': 0,
            'files_moved': 0,
            'files_skipped': 0,
            'duplicates_found': 0,
            'errors': 0,
            'buckets_used': {}
        }

    def organize_drive(
        self,
        source_path: Path,
        output_path: Optional[Path] = None,
        drives: Optional[List[str]] = None,
        dry_run: bool = False,
        remove_empty: bool = True,
        handle_duplicates: bool = True,
        create_sub_buckets: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        max_workers: Optional[int] = None
    ) -> Dict:
        """
        Organize files on a drive or directory.

        Args:
            source_path: Path to organize
            output_path: Output directory (defaults to source_path/Organized)
            drives: List of drive letters to organize (e.g. ['E:', 'F:'])
            dry_run: If True, don't actually move files
            remove_empty: Whether to remove empty directories
            handle_duplicates: Whether to detect and handle duplicates
            create_sub_buckets: Whether to create sub-buckets
            progress_callback: Optional callback(current, total, status)
            max_workers: Number of worker threads

        Returns:
            Dictionary with organization statistics
        """
        # Reset stats
        self.stats = {
            'files_processed': 0,
            'files_moved': 0,
            'files_skipped': 0,
            'duplicates_found': 0,
            'errors': 0,
            'buckets_used': {}
        }

        # Start backup session
        if not dry_run:
            session_name = self.backup_manager.start_session()
            logger.info(f"Started session: {session_name}")

        # Determine output path
        if output_path is None:
            output_path = source_path / "Organized"

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Organizing: {source_path} -> {output_path}")

        # Create bucket directories
        self._create_buckets(output_path, create_sub_buckets)

        # Discover all files
        logger.info("Discovering files...")
        files_to_process = self._discover_files(source_path, output_path)

        total_files = len(files_to_process)
        logger.info(f"Found {total_files} files to process")

        if progress_callback:
            progress_callback(0, total_files, "Starting organization...")

        # Process files
        if max_workers is None:
            max_workers = min(os.cpu_count() or 4, 8)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for file_path in files_to_process:
                future = executor.submit(
                    self._process_file,
                    file_path,
                    output_path,
                    dry_run,
                    handle_duplicates
                )
                futures.append(future)

            # Process results as they complete
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    self._update_stats(result)

                    if progress_callback:
                        status = f"Processing: {result.get('file_name', 'unknown')}"
                        progress_callback(i, total_files, status)

                except Exception as e:
                    logger.error(f"Error processing file: {e}")
                    self.stats['errors'] += 1

        # Remove empty directories
        if remove_empty and not dry_run:
            logger.info("Removing empty directories...")
            removed_count = remove_empty_dirs(source_path, preserve_root=True)
            logger.info(f"Removed {removed_count} empty directories")

        # Save backup session
        if not dry_run:
            self.backup_manager.save_session()

        # Log final stats
        self._log_final_stats()

        return self.stats

    def _create_buckets(self, output_path: Path, create_sub_buckets: bool) -> None:
        """Create bucket directories."""
        for bucket_name in self.config_manager.buckets.keys():
            bucket_path = output_path / bucket_name
            bucket_path.mkdir(parents=True, exist_ok=True)

            if create_sub_buckets:
                self.sub_bucket_manager.create_sub_buckets(bucket_path)

        # Create duplicates bucket
        duplicates_path = output_path / "_Duplicates"
        duplicates_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Created {len(self.config_manager.buckets)} bucket directories")

    def _discover_files(self, source_path: Path, output_path: Path) -> List[Path]:
        """Discover all files to process."""
        files = []

        # Skip the output directory itself
        output_path = output_path.resolve()

        for root, dirs, filenames in os.walk(source_path):
            root_path = Path(root)

            # Skip output directory
            if root_path.resolve() == output_path or output_path in root_path.resolve().parents:
                continue

            for filename in filenames:
                file_path = root_path / filename

                # Skip if should be skipped
                if not self.file_analyzer.should_skip_file(file_path):
                    files.append(file_path)

        return files

    def _process_file(
        self,
        file_path: Path,
        output_path: Path,
        dry_run: bool,
        handle_duplicates: bool
    ) -> Dict:
        """Process a single file."""
        result = {
            'file_name': file_path.name,
            'processed': False,
            'moved': False,
            'skipped': False,
            'duplicate': False,
            'error': None
        }

        try:
            # Check for duplicates
            if handle_duplicates:
                original = self.duplicate_handler.check_duplicate(file_path)
                if original:
                    result['duplicate'] = True
                    result['processed'] = True

                    # Move to duplicates folder
                    if not dry_run:
                        duplicates_path = output_path / "_Duplicates"
                        safe_move(file_path, duplicates_path / file_path.name, overwrite=False)

                    return result

            # Determine bucket
            bucket = self.file_analyzer.get_bucket_for_file(file_path)
            bucket_path = output_path / bucket

            # Determine sub-bucket
            final_path = self.sub_bucket_manager.get_sub_bucket_path(bucket_path, file_path)
            sub_bucket = final_path.name if final_path != bucket_path else ""

            # Move file
            if not dry_run:
                destination = final_path / file_path.name
                if safe_move(file_path, destination, overwrite=False):
                    result['moved'] = True

                    # Log to backup manager
                    self.backup_manager.log_move(
                        file_path,
                        destination,
                        bucket=bucket,
                        sub_bucket=sub_bucket
                    )
            else:
                result['moved'] = True  # Simulated

            result['processed'] = True
            result['bucket'] = bucket
            result['sub_bucket'] = sub_bucket

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error processing {file_path}: {e}")

        return result

    def _update_stats(self, result: Dict) -> None:
        """Update statistics from a file processing result."""
        self.stats['files_processed'] += 1

        if result.get('moved'):
            self.stats['files_moved'] += 1

            # Track bucket usage
            bucket = result.get('bucket', 'Unknown')
            self.stats['buckets_used'][bucket] = self.stats['buckets_used'].get(bucket, 0) + 1

        if result.get('skipped'):
            self.stats['files_skipped'] += 1

        if result.get('duplicate'):
            self.stats['duplicates_found'] += 1

        if result.get('error'):
            self.stats['errors'] += 1

    def _log_final_stats(self) -> None:
        """Log final statistics."""
        logger.info("=" * 60)
        logger.info("ORGANIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Files processed: {self.stats['files_processed']}")
        logger.info(f"Files moved: {self.stats['files_moved']}")
        logger.info(f"Files skipped: {self.stats['files_skipped']}")
        logger.info(f"Duplicates found: {self.stats['duplicates_found']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info("")
        logger.info("Bucket distribution:")
        for bucket, count in sorted(self.stats['buckets_used'].items()):
            logger.info(f"  {bucket}: {count} files")
        logger.info("=" * 60)

    def revert_last_organization(self, dry_run: bool = False) -> int:
        """
        Revert the last organization operation.

        Args:
            dry_run: If True, don't actually move files

        Returns:
            Number of files reverted
        """
        sessions = self.backup_manager.list_sessions()

        if not sessions:
            logger.error("No sessions found to revert")
            return 0

        latest_session = sessions[0]['session_id']
        logger.info(f"Reverting latest session: {latest_session}")

        return self.backup_manager.revert_session(latest_session, dry_run=dry_run)
