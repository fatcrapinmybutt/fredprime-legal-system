from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


def hash_file(path: PathLike) -> str:
    """Return SHA-256 hex digest of the given file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
