"""
Backup and revert system for DriveOrganizerPro.

© 2026 MBP LLC. All rights reserved.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from ..utils.logger import logger


class BackupManager:
    """Manages backup and revert operations."""

    def __init__(self, backup_dir: Optional[Path] = None):
        """
        Initialize backup manager.

        Args:
            backup_dir: Directory to store backup logs
        """
        if backup_dir is None:
            backup_dir = Path.home() / '.driveorganizerpro' / 'backups'

        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.current_session: Optional[str] = None
        self.move_log: List[Dict] = []

    def start_session(self, session_name: Optional[str] = None) -> str:
        """
        Start a new backup session.

        Args:
            session_name: Optional custom session name

        Returns:
            Session ID
        """
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"session_{timestamp}"

        self.current_session = session_name
        self.move_log = []

        logger.info(f"Started backup session: {session_name}")
        return session_name

    def log_move(self, source: Path, destination: Path, bucket: str = "",
                 sub_bucket: str = "", renamed: bool = False) -> None:
        """
        Log a file move operation.

        Args:
            source: Original file path
            destination: New file path
            bucket: Bucket name
            sub_bucket: Sub-bucket name (if any)
            renamed: Whether file was renamed
        """
        entry = {
            'source': str(source.absolute()),
            'destination': str(destination.absolute()),
            'bucket': bucket,
            'sub_bucket': sub_bucket,
            'renamed': renamed,
            'timestamp': datetime.now().isoformat()
        }

        self.move_log.append(entry)
        logger.debug(f"Logged move: {source} -> {destination}")

    def save_session(self) -> bool:
        """
        Save current session to disk.

        Returns:
            True if successful, False otherwise
        """
        if not self.current_session:
            logger.error("No active session to save")
            return False

        try:
            session_file = self.backup_dir / f"{self.current_session}.json"

            session_data = {
                'session_id': self.current_session,
                'created': datetime.now().isoformat(),
                'total_moves': len(self.move_log),
                'moves': self.move_log
            }

            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2)

            logger.info(f"Saved session to {session_file} ({len(self.move_log)} moves)")
            return True

        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False

    def load_session(self, session_name: str) -> bool:
        """
        Load a session from disk.

        Args:
            session_name: Session name or file path

        Returns:
            True if successful, False otherwise
        """
        try:
            session_file = self.backup_dir / f"{session_name}.json"

            if not session_file.exists():
                logger.error(f"Session file not found: {session_file}")
                return False

            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            self.current_session = session_data['session_id']
            self.move_log = session_data['moves']

            logger.info(f"Loaded session: {session_name} ({len(self.move_log)} moves)")
            return True

        except Exception as e:
            logger.error(f"Failed to load session {session_name}: {e}")
            return False

    def revert_session(self, session_name: Optional[str] = None, dry_run: bool = False) -> int:
        """
        Revert all moves from a session.

        Args:
            session_name: Session to revert (uses current if None)
            dry_run: If True, don't actually move files

        Returns:
            Number of files reverted
        """
        if session_name:
            if not self.load_session(session_name):
                return 0
        elif not self.current_session:
            logger.error("No session to revert")
            return 0

        count = 0
        errors = 0

        logger.info(f"Reverting session: {self.current_session} ({len(self.move_log)} moves)")

        # Revert in reverse order
        for entry in reversed(self.move_log):
            source = Path(entry['source'])
            destination = Path(entry['destination'])

            if dry_run:
                logger.info(f"[DRY RUN] Would revert: {destination} -> {source}")
                count += 1
                continue

            try:
                if not destination.exists():
                    logger.warning(f"Destination not found, skipping: {destination}")
                    continue

                # Create source directory if needed
                source.parent.mkdir(parents=True, exist_ok=True)

                # Move file back
                destination.rename(source)
                logger.info(f"Reverted: {destination} -> {source}")
                count += 1

            except Exception as e:
                logger.error(f"Failed to revert {destination}: {e}")
                errors += 1

        logger.info(f"Reverted {count} files ({errors} errors)")
        return count

    def list_sessions(self) -> List[Dict]:
        """
        List all available backup sessions.

        Returns:
            List of session info dictionaries
        """
        sessions = []

        for session_file in self.backup_dir.glob("*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                sessions.append({
                    'session_id': data['session_id'],
                    'created': data['created'],
                    'total_moves': data['total_moves'],
                    'file': str(session_file)
                })

            except Exception as e:
                logger.error(f"Failed to read session file {session_file}: {e}")

        return sorted(sessions, key=lambda x: x['created'], reverse=True)

    def delete_session(self, session_name: str) -> bool:
        """
        Delete a backup session.

        Args:
            session_name: Session to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            session_file = self.backup_dir / f"{session_name}.json"

            if not session_file.exists():
                logger.error(f"Session not found: {session_name}")
                return False

            session_file.unlink()
            logger.info(f"Deleted session: {session_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete session {session_name}: {e}")
            return False

    def get_session_summary(self) -> Dict:
        """
        Get summary of current session.

        Returns:
            Dictionary with session statistics
        """
        if not self.current_session:
            return {'active': False}

        buckets = {}
        for entry in self.move_log:
            bucket = entry['bucket']
            buckets[bucket] = buckets.get(bucket, 0) + 1

        return {
            'active': True,
            'session_id': self.current_session,
            'total_moves': len(self.move_log),
            'buckets': buckets
        }
