"""Unified command-line interface for litigation tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from cli.build import run as build_run
from cli.ingest import run as ingest_run
from cli.validate import run as validate_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Litigator command-line interface")
    sub = parser.add_subparsers(dest="command")

    ingest_p = sub.add_parser("ingest", help="Process an evidentiary ZIP archive")
    ingest_p.add_argument("zip_path", type=Path, help="Path to the ZIP archive")
    ingest_p.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Directory for extracted files and logs",
    )

    build_p = sub.add_parser("build", help="Generate litigation system definition JSON")
    build_p.add_argument("output", type=Path, help="Destination for the JSON file")

    validate_p = sub.add_parser(
        "validate", help="Ensure required files exist before zipping"
    )
    validate_p.add_argument(
        "base_path", type=Path, help="Directory containing litigation files"
    )

    args = parser.parse_args()

    if args.command == "ingest":
        ingest_run(args.zip_path, args.base_dir)
    elif args.command == "build":
        path = build_run(args.output)
        print(path)
    elif args.command == "validate":
        validate_run(args.base_path)
    else:
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover - direct execution guard
    main()
