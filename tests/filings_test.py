from pathlib import Path

from filings import make_motion_docx


def test_make_motion_docx_insufficient(tmp_path: Path) -> None:
    result = make_motion_docx(
        tmp_path, "Motion to Set Aside / Stay Enforcement", {}, {"materials": []}
    )
    assert result is None
