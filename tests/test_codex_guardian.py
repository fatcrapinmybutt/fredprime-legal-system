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
