#!/usr/bin/env python3
"""
CLI entrypoint for generating a manifest using scripts/generate_manifest.py

Usage:
    python cli/generate_manifest.py -o manifest.json
"""
import sys
from pathlib import Path
import argparse

# Add repo root to sys.path for local imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_manifest import generate_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate manifest file")
    parser.add_argument(
        "-o",
        "--output",
        default="manifest.json",
        help="Path to manifest output file",
    )
    args = parser.parse_args()
    try:
        manifest_path = generate_manifest(args.output)
        print(f"✅ Manifest generated: {manifest_path}")
        return 0
    except Exception as e:
        print(f"❌ Error generating manifest: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
