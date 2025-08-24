from pathlib import Path

from modules.hash_utils import hash_file


def test_hash_file(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    p.write_text("content")
    h = hash_file(p)
    assert len(h) == 64
