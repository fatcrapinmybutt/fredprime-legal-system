from pathlib import Path
import re

DEFAULT_PATTERN = re.compile(r"s\/\w+")


def validate_signature(document_path: str, pattern: re.Pattern = DEFAULT_PATTERN) -> bool:
    """Return True if document contains a signature line matching pattern."""
    path = Path(document_path)
    if not path.is_file():
        raise ValueError(f"{document_path} is not a file")

    text = path.read_text(errors="ignore")
    return bool(pattern.search(text))
