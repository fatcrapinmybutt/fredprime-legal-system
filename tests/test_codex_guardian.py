import pytest

from modules import codex_guardian


def test_verify_commit_message_ok():
    codex_guardian.verify_commit_message("[core] Initial commit")


def test_verify_commit_message_bad():
    with pytest.raises(ValueError):
        codex_guardian.verify_commit_message("add feature TODO")


def test_verify_branch_name():
    assert codex_guardian.verify_branch_name("codex/core-update") is True
    with pytest.raises(ValueError):
        codex_guardian.verify_branch_name("feature/no-prefix")


def test_verify_commit_message_relaxed():
    """Test that commit messages are accepted in relaxed mode."""
    # Messages with banned keywords should be accepted in relaxed mode
    codex_guardian.verify_commit_message("add feature TODO", skip_format_check=True)
    codex_guardian.verify_commit_message("WIP: work in progress", skip_format_check=True)
    # Messages without proper format should be accepted in relaxed mode
    codex_guardian.verify_commit_message("Initial plan", skip_format_check=True)


def test_verify_branch_name_relaxed():
    """Test that branch names are accepted in relaxed mode."""
    # Non-codex branches should be accepted when skip_prefix_check is True
    assert codex_guardian.verify_branch_name("feature/no-prefix", skip_prefix_check=True) is True
    assert codex_guardian.verify_branch_name("copilot/test-branch", skip_prefix_check=True) is True
    # Branches without trigger keywords should be accepted in relaxed mode
    assert codex_guardian.verify_branch_name("codex/random-stuff", skip_prefix_check=True) is True
