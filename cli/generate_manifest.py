#!/usr/bin/env python3
"""
CLI entrypoint for generating a manifest using scripts/generate_manifest.py

Usage:
    python cli/generate_manifest.py -o manifest.json
"""

import sys
import argparse
from pathlib import Path

# --- Import protection and repo-root logic ---
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
GEN_MANIFEST_PATH = REPO_ROOT / "scripts" / "generate_manifest.py"
if not GEN_MANIFEST_PATH.exists():
    print(f"❌ ERROR: Could not find {GEN_MANIFEST_PATH}")
    sys.exit(2)

sys.path.insert(0, str(REPO_ROOT))
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
        print(f"✅ Manifest generated at: {manifest_path}")
        return 0
    except Exception as e:
        print(f"❌ Error generating manifest: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
