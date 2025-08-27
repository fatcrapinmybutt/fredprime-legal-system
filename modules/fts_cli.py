# fts_cli.py
# Purpose: Run full-text keyword search over ingested metadata and file contents

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

DATABASE_PATH = Path("F:/LegalResults/MEEK_DB.jsonl")


def load_index() -> List[Dict[str, Any]]:
    """Load the JSONL metadata index."""
    if not DATABASE_PATH.exists():
        return []
    with DATABASE_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def extract_text(path: str) -> str:
    """Basic plaintext extract (text files only)."""
    try:
        p = Path(path)
        if p.suffix.lower() == ".txt":
            return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        pass
    return ""


def search_index(term: str) -> None:
    """Perform a case-insensitive search across filenames and text content."""
    results: List[Dict[str, Any]] = []
    index = load_index()
    for item in index:
        filename = item.get("filename", "").lower()
        if term.lower() in filename:
            results.append(item)
            continue
        content = extract_text(item.get("path", "")).lower()
        if term.lower() in content:
            results.append(item)
    print(f"\n[FTS] Found {len(results)} matches for '{term}':\n")
    for r in results:
        tags = ", ".join(r.get("tags", []))
        print(f"- {r.get('filename')} ({r.get('path')}) — Tags: {tags}")
    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python fts_cli.py <search term>")
    else:
        search_index(sys.argv[1])
