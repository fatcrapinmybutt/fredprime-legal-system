from pathlib import Path

from modules import meek_ingest_cli


def test_extract_metadata(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")
    meta = meek_ingest_cli.extract_metadata(str(file_path))
    assert meta["filename"] == "sample.txt"
    assert meta["hash"] == meek_ingest_cli.sha256(str(file_path))
