#!/usr/bin/env python
"""CLI entrypoint for generating a manifest."""

import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from scripts.generate_manifest import generate_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate manifest file")
    parser.add_argument("-o", "--output", default="manifest.json", help="Path to manifest output file")
    args = parser.parse_args()
    print(generate_manifest(args.output))


if __name__ == "__main__":
    main()
