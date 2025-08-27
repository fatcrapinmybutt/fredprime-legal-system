from pathlib import Path

from modules import fts_cli


def test_extract_text(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    content = "lawforge search"
    file_path.write_text(content, encoding="utf-8")
    text = fts_cli.extract_text(str(file_path))
    assert content in text
