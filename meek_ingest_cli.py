#!/usr/bin/env python3
"""Simple ingestion CLI for the Meek pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional


def ingest_directory(source: Path) -> int:
    """Count files under *source* as a stand-in for ingestion."""
    if not source.exists():
        raise FileNotFoundError(f"{source} not found")
    return sum(1 for p in source.rglob("*") if p.is_file())


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Meek ingestion CLI")
    parser.add_argument("source", nargs="?", default=".", help="Directory to ingest")
    args = parser.parse_args(argv)

    count = ingest_directory(Path(args.source))
    print(f"Ingested {count} files from {args.source}")


if __name__ == "__main__":
    main()
