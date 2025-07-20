"""Codex manifest verification utilities."""

import hashlib
import json
from pathlib import Path
from typing import Dict

MANIFEST_PATH = Path("codex_manifest.json")


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def verify_all_modules() -> None:
    """Ensure all recorded modules match their stored hash."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")
    data: Dict[str, str] = json.loads(MANIFEST_PATH.read_text())
    for file_str, expected in data.items():
        path = Path(file_str)
        if not path.exists():
            raise FileNotFoundError(f"Missing file listed in manifest: {path}")
        current = _hash_file(path)
        if current != expected:
            raise ValueError(f"Hash mismatch for {path}")


def enforce_final_form_lock() -> None:
    """Placeholder for final form enforcement."""
    pass
