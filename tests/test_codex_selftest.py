"""Tests for codex_selftest module."""

import os
import subprocess
from unittest.mock import patch


def test_selftest_main_on_codex_branch():
    """Test that selftest works on codex/ branches."""
    with patch("subprocess.check_output") as mock_check_output:
        # Mock git to return a codex branch
        mock_check_output.side_effect = [
            b"codex/test-branch",  # First call for branch name in selftest
            b"codex/test-branch",  # Second call in run_guardian
            b"[core] Test commit",  # Commit message in run_guardian
        ]

        # Import and run the selftest
        from codex_selftest import main

        # Set env var to skip hash checks (since we can't mock the manifest)
        os.environ["CODEX_SKIP_HASH_CHECKS"] = "true"
        try:
            main()
        finally:
            # Clean up
            if "CODEX_SKIP_HASH_CHECKS" in os.environ:
                del os.environ["CODEX_SKIP_HASH_CHECKS"]


def test_selftest_main_on_non_codex_branch():
    """Test that selftest works on non-codex branches."""
    with patch("subprocess.check_output") as mock_check_output:
        # Mock git to return a non-codex branch
        mock_check_output.side_effect = [
            b"copilot/test-branch",  # First call for branch name in selftest
            b"copilot/test-branch",  # Second call in run_guardian
            b"Test commit",  # Commit message in run_guardian
        ]

        # Import and run the selftest
        from codex_selftest import main

        # Clear any existing env vars
        for key in ["CODEX_SKIP_STRICT_CHECKS", "CODEX_SKIP_HASH_CHECKS"]:
            if key in os.environ:
                del os.environ[key]

        try:
            main()
        finally:
            # Clean up
            for key in ["CODEX_SKIP_STRICT_CHECKS", "CODEX_SKIP_HASH_CHECKS"]:
                if key in os.environ:
                    del os.environ[key]


def test_selftest_main_git_fails():
    """Test that selftest handles git failures gracefully."""
    with patch("subprocess.check_output") as mock_check_output:
        # Mock git to fail on first call
        mock_check_output.side_effect = [
            subprocess.CalledProcessError(1, "git"),  # First call fails
            b"main",  # Second call succeeds
            b"Test commit",  # Commit message
        ]

        # Import and run the selftest
        from codex_selftest import main

        # Clear any existing env vars
        for key in ["CODEX_SKIP_STRICT_CHECKS", "CODEX_SKIP_HASH_CHECKS"]:
            if key in os.environ:
                del os.environ[key]

        try:
            main()
        finally:
            # Clean up
            for key in ["CODEX_SKIP_STRICT_CHECKS", "CODEX_SKIP_HASH_CHECKS"]:
                if key in os.environ:
                    del os.environ[key]
