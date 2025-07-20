#!/usr/bin/env python3
"""
FRED PRIME Litigation System — Universal Manifest & System Definition Generator

Generates a manifest of the current directory, with SHA-256 and metadata, and (optionally)
creates a system definition JSON for use with the FRED PRIME Litigation OS.

Usage:
    python fredprime_gen.py                   # Outputs manifest as ./manifest.json
    python fredprime_gen.py -o /path/to/out.json
    python fredprime_gen.py --systemdef       # Outputs ./fredprime_litigation_system.json
    python fredprime_gen.py --help
"""

import os
import sys
import json
import hashlib
import argparse
from datetime import datetime

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def scan_dir(root_dir):
    manifest = {}
    for subdir, _, files in os.walk(root_dir):
        for f in files:
            full_path = os.path.join(subdir, f)
            try:
                manifest[os.path.relpath(full_path, root_dir)] = {
                    "sha256": sha256_file(full_path),
                    "size": os.path.getsize(full_path),
                    "mtime": datetime.fromtimestamp(os.path.getmtime(full_path)).isoformat(),
                }
            except Exception as e:
                print(f"Warning: Could not process {full_path}: {e}")
    return manifest

def write_manifest(manifest, out_path):
    with open(out_path, "w") as out:
        json.dump(manifest, out, indent=2)
    print(f"Manifest written to: {out_path}")

def generate_systemdef(manifest, out_path):
    # Minimal stub — customize as needed
    systemdef = {
        "name": "FRED PRIME Litigation System",
        "generated": datetime.now().isoformat(),
        "files_indexed": len(manifest),
        "manifest_ref": out_path,
        "components": [
            "evidence_scan", "timeline_builder", "warboard", "motions", "federal_complaint",
            "patch_manager", "gui", "foia_generator", "contradiction_matrix", "entity_trace"
        ],
        "note": "This file is machine-generated. Always verify contents before court use."
    }
    systemdef_path = os.path.join(os.path.dirname(out_path), "fredprime_litigation_system.json")
    with open(systemdef_path, "w") as sysf:
        json.dump(systemdef, sysf, indent=2)
    print(f"System definition written to: {systemdef_path}")

def print_help():
    print(__doc__)
    print("\nLicense: MIT")
    print("Contact: contact@example.com or open an issue at https://github.com/your-org/your-repo/issues")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-o", "--output", help="Manifest output path", default="manifest.json")
    parser.add_argument("--systemdef", action="store_true", help="Generate system definition as well")
    parser.add_argument("--help", action="store_true")
    args = parser.parse_args()

    if args.help or len(sys.argv) == 1:
        print_help()

    root_dir = os.getcwd()
    manifest = scan_dir(root_dir)
    write_manifest(manifest, args.output)

    if args.systemdef:
        generate_systemdef(manifest, args.output)

if __name__ == "__main__":
    main()
