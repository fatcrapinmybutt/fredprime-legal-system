from pathlib import Path
import zipfile

from scripts.chief_judge_pack import create_docx_from_text


def test_create_docx_from_text(tmp_path: Path) -> None:
    output = tmp_path / "sample.docx"
    create_docx_from_text("sample text", output)
    assert output.exists(), "DOCX file was not created"

    with zipfile.ZipFile(output) as zf:
        with zf.open("word/document.xml") as docx_file:
            content = docx_file.read().decode("utf-8")

    assert "sample text" in content
