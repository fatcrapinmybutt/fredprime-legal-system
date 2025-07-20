#!/usr/bin/env python3
"""
FRED PRIME / CODEX SUPREME
Universal Litigation Manifest Generator
---------------------------------------

• Scans entire repo tree (recursive), capturing all .py, .json, .docx (add more as needed)
• Excludes junk: /tests/, /.git/, /__pycache__/, hidden files, and dot-directories
• SHA-256 hashes for full evidentiary/forensic compliance
• Outputs canonical codex_manifest.json (top-level manifest) with file, hash, size, and mtime

Use:
    python generate_manifest.py
    # Or specify output path:
    python generate_manifest.py -o /mnt/data/codex_manifest.json
"""

import hashlib
import json
import os
from pathlib import Path
import argparse
from datetime import datetime

INCLUDE_EXTS = {".py", ".json", ".docx"}
EXCLUDE_DIRS = {"tests", ".git", "__pycache__", ".vscode", ".idea", "output", "build", "dist"}
DEFAULT_OUT = "codex_manifest.json"

def is_valid_file(p: Path) -> bool:
    if not p.is_file():
        return False
    if p.suffix.lower() not in INCLUDE_EXTS:
        return False
    # Exclude any path part matching exclusion
    for part in p.parts:
        if part in EXCLUDE_DIRS or part.startswith("."):
            return False
    return True

def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def scan_tree(root: Path = Path(".")) -> dict:
    manifest = {}
    for p in root.rglob("*"):
        if is_valid_file(p):
            rel = str(p.relative_to(root))
            manifest[rel] = {
                "sha256": hash_file(p),
                "size": p.stat().st_size,
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            }
    return manifest

def main():
    parser = argparse.ArgumentParser(description="Top-level manifest generator for CODEX/FRED PRIME")
    parser.add_argument("-o", "--output", default=DEFAULT_OUT, help="Manifest output file")
    args = parser.parse_args()

    manifest = scan_tree()
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"✅ Manifest written: {os.path.abspath(args.output)} ({len(manifest)} files indexed)")

if __name__ == "__main__":
    main()
