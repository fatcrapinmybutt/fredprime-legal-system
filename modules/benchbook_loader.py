from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from PyPDF2 import PdfReader


logger = logging.getLogger(__name__)


def load_benchbook_texts(directory: str) -> Dict[str, str]:
    """Load text from all PDF benchbooks in a directory.

    Args:
        directory: Path to a directory containing benchbook PDFs.

    Returns:
        A mapping of PDF file names to their extracted text.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    texts: Dict[str, str] = {}
    for pdf_path in dir_path.glob("*.pdf"):
        try:
            reader = PdfReader(str(pdf_path))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to read PDF %s: %s", pdf_path, exc)
            continue
        content = ""
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                content += page.extract_text() or ""
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to extract text from %s page %s: %s",
                    pdf_path,
                    page_num,
                    exc,
                )
        texts[pdf_path.name] = content
    return texts
