# meek_ingest_cli.py
# Purpose: Ingest all files in evidence folders and build metadata records
# Output: JSONL database + indexing map

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

EVIDENCE_FOLDERS: List[str] = ["F:/MEEK1", "F:/MEEK2", "Z:/LAWFORGE_SERVER"]
OUTPUT_PATH = Path("F:/LegalResults/MEEK_DB.jsonl")


def sha256(filepath: str) -> str:
    """Return SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:  # noqa: PTH123
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def auto_tag(filename: str) -> List[str]:
    """Simple keyword-based tagger."""
    tags: List[str] = []
    lower = filename.lower()
    if "custody" in lower:
        tags.append("custody")
    if "ppo" in lower:
        tags.append("ppo")
    if "rent" in lower or "ledger" in lower:
        tags.append("rent")
    if "egle" in lower:
        tags.append("egle")
    if "strike" in lower or "motion" in lower:
        tags.append("motion")
    return tags


def extract_metadata(filepath: str) -> Dict[str, Any]:
    """Gather basic file metadata."""
    path = Path(filepath)
    stat = path.stat()
    return {
        "filename": path.name,
        "path": str(path),
        "size": stat.st_size,
        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "hash": sha256(filepath),
        "tags": auto_tag(path.name),
    }


def ingest_all() -> None:
    results: List[Dict[str, Any]] = []
    for folder in EVIDENCE_FOLDERS:
        for root, _, files in os.walk(folder):
            for name in files:
                full = os.path.join(root, name)
                try:
                    meta = extract_metadata(full)
                    results.append(meta)
                except Exception as exc:  # pragma: no cover - log side effect
                    print(f"[ERROR] Skipping {full}: {exc}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for entry in results:
            json.dump(entry, out)
            out.write("\n")
    print(f"[MEEK] Ingested {len(results)} files into {OUTPUT_PATH}")


if __name__ == "__main__":
    ingest_all()
