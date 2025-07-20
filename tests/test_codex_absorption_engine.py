from pathlib import Path
from codex_absorption_engine import get_metadata


def test_get_metadata(tmp_path: Path) -> None:
    test_file = tmp_path / "example.txt"
    test_file.write_text("hello")
    data = get_metadata(str(test_file))
    assert data["sha256"]
    assert data["validated"] is True
