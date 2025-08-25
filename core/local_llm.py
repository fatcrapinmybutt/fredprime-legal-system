from __future__ import annotations

import re
from typing import Dict, List


def analyze_content(text: str) -> Dict[str, object]:
    """Analyze text using an offline algorithm.

    The function tokenizes the input and counts words to avoid any
    network-based model calls.

    Args:
        text: Raw text to analyze.

    Returns:
        A dictionary containing the lowercase tokens and total word count.
    """

    tokens: List[str] = re.findall(r"\b\w+\b", text.lower())
    return {"tokens": tokens, "word_count": len(tokens)}
