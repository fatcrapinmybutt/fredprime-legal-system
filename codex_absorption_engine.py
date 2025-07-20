"""Scan target directories and build a metadata manifest."""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

from forensic_compliance_validator import (
    classify_legal_function,
    validate_file,
)

TARGET_DIRS = ["F:/", "D:/"]
EXTENSIONS = [".py", ".json", ".txt", ".docx"]
MANIFEST_FILE = Path("codex_manifest.json")
ERROR_LOG = Path("logs/codex_errors.log")

logging.basicConfig(filename=ERROR_LOG, level=logging.ERROR)


def get_metadata(filepath: str) -> Dict[str, Any]:
    """Return metadata for a file, logging errors."""
    try:
        stat = os.stat(filepath)
        sha256 = hashlib.sha256(Path(filepath).read_bytes()).hexdigest()
        return {
            "sha256": sha256,
            "timestamp": time.ctime(stat.st_mtime),
            "source": "absorption_engine",
            "legal_function": classify_legal_function(filepath),
            "validated": validate_file(filepath),
        }
    except Exception as exc:  # pragma: no cover - log and skip
        logging.error("Metadata error for %s: %s", filepath, exc)
        return {}


def run() -> None:
    """Build manifest of files in TARGET_DIRS."""
    manifest: Dict[str, Dict[str, str]] = {}
    for root_dir in TARGET_DIRS:
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if any(file.endswith(ext) for ext in EXTENSIONS):
                    path = os.path.join(subdir, file)
                    manifest[path] = get_metadata(path)
    try:
        with MANIFEST_FILE.open("w") as out:
            json.dump(manifest, out, indent=2)
    except Exception as exc:  # pragma: no cover - log and continue
        logging.error("Failed to write manifest: %s", exc)
    print(f"\u2705 Absorption complete. {len(manifest)} files processed.")


if __name__ == "__main__":
    run()
