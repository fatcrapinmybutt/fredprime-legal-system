from __future__ import annotations

from pathlib import Path
from typing import Dict

from PyPDF2 import PdfReader

from core.local_llm import analyze_content


def load_benchbook_texts(directory: str) -> Dict[str, Dict[str, object]]:
    """Load and analyze text from all PDF benchbooks in a directory.

    Args:
        directory: Path to a directory containing benchbook PDFs.

    Returns:
        A mapping of PDF file names to analysis results.
    """
    dir_path = Path(directory)
    texts: Dict[str, Dict[str, object]] = {}
    for pdf_path in dir_path.glob("*.pdf"):
        reader = PdfReader(str(pdf_path))
        content = ""
        for page in reader.pages:
            content += page.extract_text() or ""
        texts[pdf_path.name] = analyze_content(content)
    return texts
