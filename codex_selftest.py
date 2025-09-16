"""Lightweight self-test for Codex guardian."""

from pathlib import Path

from codex_brain import hard_coded_guardian, hash_file


def main() -> None:
    hard_coded_guardian()
    h = hash_file(Path("codex_brain.py"))
    assert len(h) == 64
    print("codex self-test passed")


if __name__ == "__main__":
    main()
