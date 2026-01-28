"""
Comprehensive test suite for organize_drive module, focusing on drive validation logic.

Tests cover:
- Missing required drives scenarios
- Non-existent drive paths
- C: drive rejection (direct and via symbolic links/junctions)
- Successful validation with valid Q/D/Z drives
"""

import os
import platform
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add parent directory to path to import organize_drive
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import organize_drive


class TestDriveLetterExtraction:
    """Test drive letter extraction from paths."""

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_get_drive_letter_valid_path(self, tmp_path):
        """Test extracting drive letter from a valid path."""
        # Create a mock Windows path
        with patch("platform.system", return_value="Windows"):
            with patch.object(Path, "resolve", return_value=Path("Q:/test/path")):
                path = Path("Q:/test/path")
                result = organize_drive.get_drive_letter(path)
                assert result == "Q"

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_get_drive_letter_normalized(self, tmp_path):
        """Test that drive letter is uppercase normalized."""
        with patch("platform.system", return_value="Windows"):
            with patch.object(Path, "resolve", return_value=Path("d:/test/path")):
                path = Path("d:/test/path")
                result = organize_drive.get_drive_letter(path)
                assert result == "D"

    def test_get_drive_letter_non_windows(self):
        """Test that non-Windows systems return None."""
        with patch("platform.system", return_value="Linux"):
            path = Path("/home/user/test")
            result = organize_drive.get_drive_letter(path)
            assert result is None


class TestDrivePathValidation:
    """Test drive path validation logic."""

    def test_validate_nonexistent_path(self, tmp_path):
        """Test validation fails for non-existent paths."""
        nonexistent = tmp_path / "does_not_exist"
        is_valid, error_msg = organize_drive.validate_drive_path(nonexistent)
        assert not is_valid
        assert "does not exist" in error_msg.lower()

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_validate_c_drive_direct_rejection(self, tmp_path):
        """Test that C: drive is rejected directly."""
        with patch("platform.system", return_value="Windows"):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "resolve", return_value=Path("C:/Windows/System32")):
                    path = Path("C:/Windows/System32")
                    is_valid, error_msg = organize_drive.validate_drive_path(path)
                    assert not is_valid
                    assert "C: drive is forbidden" in error_msg

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_validate_c_drive_via_symlink(self, tmp_path):
        """Test that C: drive is rejected even when accessed via symbolic link."""
        # Simulate a symlink that resolves to C: drive
        with patch("platform.system", return_value="Windows"):
            with patch.object(Path, "exists", return_value=True):
                # The resolve() call should return the real C: path
                with patch.object(Path, "resolve", return_value=Path("C:/Users/TestUser")):
                    symlink_path = Path("Q:/link_to_c")
                    is_valid, error_msg = organize_drive.validate_drive_path(symlink_path)
                    assert not is_valid
                    assert "C: drive is forbidden" in error_msg

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_validate_c_drive_via_junction(self, tmp_path):
        """Test that C: drive is rejected even when accessed via junction point."""
        # Simulate a junction that resolves to C: drive
        with patch("platform.system", return_value="Windows"):
            with patch.object(Path, "exists", return_value=True):
                # Junction resolves to C: drive
                with patch.object(Path, "resolve", return_value=Path("C:/ProgramData")):
                    junction_path = Path("D:/junction_to_c")
                    is_valid, error_msg = organize_drive.validate_drive_path(junction_path)
                    assert not is_valid
                    assert "C: drive is forbidden" in error_msg

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_validate_valid_q_drive(self):
        """Test that valid Q: drive passes validation."""
        with patch("platform.system", return_value="Windows"):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "resolve", return_value=Path("Q:/litigation/data")):
                    path = Path("Q:/litigation/data")
                    is_valid, error_msg = organize_drive.validate_drive_path(path)
                    assert is_valid
                    assert error_msg == ""

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_validate_valid_d_drive(self):
        """Test that valid D: drive passes validation."""
        with patch("platform.system", return_value="Windows"):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "resolve", return_value=Path("D:/evidence/files")):
                    path = Path("D:/evidence/files")
                    is_valid, error_msg = organize_drive.validate_drive_path(path)
                    assert is_valid
                    assert error_msg == ""

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_validate_valid_z_drive(self):
        """Test that valid Z: drive passes validation."""
        with patch("platform.system", return_value="Windows"):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "resolve", return_value=Path("Z:/backup/docs")):
                    path = Path("Z:/backup/docs")
                    is_valid, error_msg = organize_drive.validate_drive_path(path)
                    assert is_valid
                    assert error_msg == ""

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_validate_invalid_drive_letter(self):
        """Test that non-required drive letters are rejected."""
        with patch("platform.system", return_value="Windows"):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "resolve", return_value=Path("E:/random/path")):
                    path = Path("E:/random/path")
                    is_valid, error_msg = organize_drive.validate_drive_path(path)
                    assert not is_valid
                    assert "not one of the required drives" in error_msg

    def test_validate_non_windows_always_valid(self, tmp_path):
        """Test that non-Windows paths skip drive validation."""
        with patch("platform.system", return_value="Linux"):
            # Create a real directory for Linux
            test_dir = tmp_path / "test"
            test_dir.mkdir()
            is_valid, error_msg = organize_drive.validate_drive_path(test_dir)
            assert is_valid
            assert error_msg == ""

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_validate_custom_required_drives(self):
        """Test validation with custom required drives list."""
        with patch("platform.system", return_value="Windows"):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "resolve", return_value=Path("X:/custom/path")):
                    path = Path("X:/custom/path")
                    # Test with custom drives list
                    is_valid, error_msg = organize_drive.validate_drive_path(path, required_drives=["X", "Y"])
                    assert is_valid
                    assert error_msg == ""


class TestRequiredDrivesCheck:
    """Test checking for required drives presence."""

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_all_required_drives_present(self):
        """Test when all required drives are present."""
        with patch("platform.system", return_value="Windows"):
            # Mock all required drives as existing
            def mock_exists(self):
                drive_letter = str(self).split(":")[0].upper()
                return drive_letter in organize_drive.REQUIRED_DRIVES

            with patch.object(Path, "exists", mock_exists):
                all_present, missing = organize_drive.check_required_drives_exist()
                assert all_present
                assert missing == []

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_missing_single_required_drive(self):
        """Test when one required drive is missing."""
        with patch("platform.system", return_value="Windows"):
            # Mock Q and D as present, Z as missing
            def mock_exists(self):
                drive_letter = str(self).split(":")[0].upper()
                return drive_letter in ["Q", "D"]

            with patch.object(Path, "exists", mock_exists):
                all_present, missing = organize_drive.check_required_drives_exist()
                assert not all_present
                assert "Z" in missing
                assert len(missing) == 1

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_missing_multiple_required_drives(self):
        """Test when multiple required drives are missing."""
        with patch("platform.system", return_value="Windows"):
            # Mock only Q as present
            def mock_exists(self):
                drive_letter = str(self).split(":")[0].upper()
                return drive_letter == "Q"

            with patch.object(Path, "exists", mock_exists):
                all_present, missing = organize_drive.check_required_drives_exist()
                assert not all_present
                assert "D" in missing
                assert "Z" in missing
                assert len(missing) == 2

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_all_required_drives_missing(self):
        """Test when all required drives are missing."""
        with patch("platform.system", return_value="Windows"):
            # Mock all drives as not existing
            with patch.object(Path, "exists", return_value=False):
                all_present, missing = organize_drive.check_required_drives_exist()
                assert not all_present
                assert set(missing) == set(organize_drive.REQUIRED_DRIVES)

    def test_non_windows_skip_check(self):
        """Test that non-Windows systems skip the drive check."""
        with patch("platform.system", return_value="Linux"):
            all_present, missing = organize_drive.check_required_drives_exist()
            assert all_present
            assert missing == []

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_custom_required_drives_list(self):
        """Test checking custom required drives list."""
        with patch("platform.system", return_value="Windows"):
            # Mock only X drive as present
            def mock_exists(self):
                drive_letter = str(self).split(":")[0].upper()
                return drive_letter == "X"

            with patch.object(Path, "exists", mock_exists):
                all_present, missing = organize_drive.check_required_drives_exist(required_drives=["X", "Y"])
                assert not all_present
                assert "Y" in missing
                assert "X" not in missing


class TestMainFunction:
    """Test main function integration with validation."""

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_main_rejects_c_drive(self, capsys):
        """Test that main function rejects C: drive."""
        with patch("platform.system", return_value="Windows"):
            with patch.object(Path, "exists", return_value=True):
                with patch.object(Path, "resolve", return_value=Path("C:/test")):
                    with patch("sys.argv", ["organize_drive.py", "C:/test"]):
                        result = organize_drive.main()
                        captured = capsys.readouterr()
                        assert result == 1
                        assert "C: drive is forbidden" in captured.out

    @pytest.mark.skipif(platform.system() != "Windows", reason="Windows-specific test")
    def test_main_warns_missing_drives(self, capsys):
        """Test that main function warns about missing required drives."""
        with patch("platform.system", return_value="Windows"):
            # Mock Q drive as existing, but D and Z as missing
            def mock_exists(self):
                path_str = str(self)
                if "Q:" in path_str:
                    return True
                return False

            with patch.object(Path, "exists", mock_exists):
                with patch.object(Path, "resolve", return_value=Path("Q:/test")):
                    with patch("sys.argv", ["organize_drive.py", "Q:/test"]):
                        with patch("organize_drive.organize_drive"):  # Mock the actual organization
                            result = organize_drive.main()
                            captured = capsys.readouterr()
                            assert "Missing required drives" in captured.out
                            # Should still complete successfully since Q drive is valid
                            assert result == 0

    def test_main_nonexistent_path(self, capsys, tmp_path):
        """Test that main function handles non-existent paths."""
        nonexistent = tmp_path / "does_not_exist"
        with patch("sys.argv", ["organize_drive.py", str(nonexistent)]):
            result = organize_drive.main()
            captured = capsys.readouterr()
            assert result == 1
            assert "does not exist" in captured.out.lower()


class TestGetCategory:
    """Test file categorization logic."""

    def test_pdf_categorized_as_document(self):
        """Test that PDF files are categorized as Documents."""
        path = Path("test.pdf")
        category = organize_drive.get_category(path)
        assert category == "Documents"

    def test_jpg_categorized_as_image(self):
        """Test that JPG files are categorized as Images."""
        path = Path("photo.jpg")
        category = organize_drive.get_category(path)
        assert category == "Images"

    def test_mp3_categorized_as_music(self):
        """Test that MP3 files are categorized as Music."""
        path = Path("song.mp3")
        category = organize_drive.get_category(path)
        assert category == "Music"

    def test_unknown_extension_categorized_as_other(self):
        """Test that unknown extensions are categorized as Other."""
        path = Path("file.xyz")
        category = organize_drive.get_category(path)
        assert category == "Other"

    def test_case_insensitive_categorization(self):
        """Test that file extension matching is case-insensitive."""
        path = Path("TEST.PDF")
        category = organize_drive.get_category(path)
        assert category == "Documents"
