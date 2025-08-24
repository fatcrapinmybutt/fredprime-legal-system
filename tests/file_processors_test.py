from pathlib import Path

from file_processors import dispatch_get_text


def test_dispatch_get_text_txt(tmp_path: Path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("hello")
    text, source = dispatch_get_text(str(sample))
    assert text.strip() == "hello"
    assert source == "txt"
