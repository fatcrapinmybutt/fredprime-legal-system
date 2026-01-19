"""
Basic test suite for DriveOrganizerPro.

© 2026 MBP LLC. All rights reserved.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from drive_organizer_pro.config.config_manager import ConfigManager
from drive_organizer_pro.core.file_analyzer import FileAnalyzer
from drive_organizer_pro.core.sub_bucket_manager import SubBucketManager
from drive_organizer_pro.utils.file_utils import safe_move, resolve_name_collision


class TestConfigManager:
    """Test configuration management."""

    def test_load_default_buckets(self):
        """Test loading default bucket configuration."""
        config = ConfigManager()
        buckets = config.load_buckets()

        assert isinstance(buckets, dict)
        assert len(buckets) > 0
        assert "Documents" in buckets
        assert "Images" in buckets

    def test_get_bucket_for_extension(self):
        """Test bucket lookup by extension."""
        config = ConfigManager()
        config.load_buckets()

        assert config.get_bucket_for_extension(".pdf") == "Documents"
        assert config.get_bucket_for_extension(".jpg") == "Images"
        assert config.get_bucket_for_extension(".mp4") == "Videos"
        assert config.get_bucket_for_extension(".unknown") == "Miscellaneous"


class TestFileAnalyzer:
    """Test file analysis."""

    def test_get_bucket_for_file(self, tmp_path):
        """Test bucket determination for files."""
        config = ConfigManager()
        config.load_buckets()
        analyzer = FileAnalyzer(config)

        # Create test files
        pdf_file = tmp_path / "test.pdf"
        pdf_file.touch()

        jpg_file = tmp_path / "image.jpg"
        jpg_file.touch()

        assert analyzer.get_bucket_for_file(pdf_file) == "Documents"
        assert analyzer.get_bucket_for_file(jpg_file) == "Images"

    def test_should_skip_file(self, tmp_path):
        """Test file skip logic."""
        config = ConfigManager()
        config.load_buckets()
        analyzer = FileAnalyzer(config)

        # Hidden file
        hidden = tmp_path / ".hidden"
        hidden.touch()
        assert analyzer.should_skip_file(hidden) is True

        # System file
        system = tmp_path / "desktop.ini"
        system.touch()
        assert analyzer.should_skip_file(system) is True

        # Normal file
        normal = tmp_path / "document.pdf"
        normal.touch()
        assert analyzer.should_skip_file(normal) is False


class TestSubBucketManager:
    """Test sub-bucket management."""

    def test_detect_sub_bucket(self, tmp_path):
        """Test sub-bucket detection."""
        sub_bucket_config = {
            "sub_buckets": ["Test1", "Test2"],
            "keyword_mappings": {
                "Test1": ["test1", "alpha"],
                "Test2": ["test2", "beta"]
            }
        }

        manager = SubBucketManager(sub_bucket_config)

        # Test keyword detection
        test1_file = tmp_path / "test1_document.pdf"
        test1_file.touch()
        assert manager.detect_sub_bucket(test1_file) == "Test1"

        test2_file = tmp_path / "beta_file.txt"
        test2_file.touch()
        assert manager.detect_sub_bucket(test2_file) == "Test2"

        # No keyword match
        normal_file = tmp_path / "normal.pdf"
        normal_file.touch()
        assert manager.detect_sub_bucket(normal_file) is None


class TestFileUtils:
    """Test file utility functions."""

    def test_resolve_name_collision(self, tmp_path):
        """Test name collision resolution."""
        # Create existing file
        existing = tmp_path / "test.txt"
        existing.touch()

        # Resolve collision
        new_path = resolve_name_collision(existing)

        assert new_path != existing
        assert new_path.parent == existing.parent
        assert "test" in new_path.stem
        assert new_path.suffix == ".txt"

    def test_safe_move(self, tmp_path):
        """Test safe file move operation."""
        # Create source file
        source = tmp_path / "source.txt"
        source.write_text("test content")

        # Create destination directory
        dest_dir = tmp_path / "destination"
        dest_dir.mkdir()
        dest = dest_dir / "source.txt"

        # Move file
        result = safe_move(source, dest)

        assert result is True
        assert not source.exists()
        assert dest.exists()
        assert dest.read_text() == "test content"


class TestIntegration:
    """Integration tests."""

    def test_basic_organization(self, tmp_path):
        """Test basic file organization."""
        from drive_organizer_pro.core.organizer_engine import OrganizerEngine

        # Create test files
        source_dir = tmp_path / "source"
        source_dir.mkdir()

        (source_dir / "document.pdf").touch()
        (source_dir / "image.jpg").touch()
        (source_dir / "video.mp4").touch()

        # Organize
        engine = OrganizerEngine()
        stats = engine.organize_drive(
            source_path=source_dir,
            dry_run=True,  # Safe mode
            remove_empty=False,
            handle_duplicates=False,
            create_sub_buckets=False
        )

        # Verify stats
        assert stats['files_processed'] == 3
        assert stats['errors'] == 0


def test_package_imports():
    """Test that all major modules can be imported."""
    # Core modules
    from drive_organizer_pro.core import organizer_engine
    from drive_organizer_pro.core import file_analyzer
    from drive_organizer_pro.core import duplicate_handler
    from drive_organizer_pro.core import backup_manager
    from drive_organizer_pro.core import sub_bucket_manager

    # Config
    from drive_organizer_pro.config import config_manager

    # Utils
    from drive_organizer_pro.utils import logger
    from drive_organizer_pro.utils import file_utils
    from drive_organizer_pro.utils import hash_utils
    from drive_organizer_pro.utils import path_utils

    # GUI
    from drive_organizer_pro.gui import themes
    from drive_organizer_pro.gui import components

    assert True  # All imports successful


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
