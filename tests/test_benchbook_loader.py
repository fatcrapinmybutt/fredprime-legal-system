from pathlib import Path

from pypdf import PdfWriter

from modules.benchbook_loader import load_benchbook_texts


def test_load_benchbook_texts(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)
    texts = load_benchbook_texts(str(tmp_path))
    assert pdf_path.name in texts
