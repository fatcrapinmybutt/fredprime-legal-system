#!/usr/bin/env python3
"""Command-line interface for FRED PRIME helper scripts."""
import argparse
from pathlib import Path
from src.exhibit_labeler import label_exhibits
from src.motion_exhibit_linker import link_motions
from src.signature_validator import validate_signature


def main():
    parser = argparse.ArgumentParser(description="FRED PRIME utilities")
    subparsers = parser.add_subparsers(dest="command")

    label = subparsers.add_parser("label-exhibits", help="Rename exhibits A-Z")
    label.add_argument("exhibit_dir")
    label.add_argument("--output", default="Exhibit_Index.md")

    link = subparsers.add_parser("link-motions", help="Map motions to exhibits")
    link.add_argument("motion_dir")
    link.add_argument("--output", default="Motion_to_Exhibit_Map.md")

    sig = subparsers.add_parser("validate-signature", help="Check document for signature block")
    sig.add_argument("document")

    args = parser.parse_args()

    if args.command == "label-exhibits":
        label_exhibits(args.exhibit_dir, args.output)
    elif args.command == "link-motions":
        link_motions(args.motion_dir, args.output)
    elif args.command == "validate-signature":
        found = validate_signature(args.document)
        if found:
            print("Signature block found")
        else:
            print("Signature block not found")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
