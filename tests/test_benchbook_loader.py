from __future__ import annotations

import logging
from pathlib import Path

import pytest
from PyPDF2 import PdfWriter

from modules.benchbook_loader import load_benchbook_texts


def _create_pdf(path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    with path.open("wb") as f:
        writer.write(f)


def test_valid_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _create_pdf(pdf_path)
    texts = load_benchbook_texts(str(tmp_path))
    assert pdf_path.name in texts


def test_missing_directory() -> None:
    with pytest.raises(FileNotFoundError):
        load_benchbook_texts("missing_dir")


def test_corrupt_pdf(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    good = tmp_path / "good.pdf"
    bad = tmp_path / "bad.pdf"
    _create_pdf(good)
    bad.write_bytes(b"not a pdf")
    with caplog.at_level(logging.WARNING):
        texts = load_benchbook_texts(str(tmp_path))
    assert good.name in texts
    assert bad.name not in texts
    assert any("Failed to read PDF" in r.message for r in caplog.records)
