#!/usr/bin/env python3
"""Command-line interface for FRED PRIME helper scripts."""

if __name__ == "__main__":
    import argparse
    from src.exhibit_labeler import label_exhibits
    from src.motion_exhibit_linker import link_motions
    from src.signature_validator import validate_signature

    parser = argparse.ArgumentParser(description="FRED PRIME Litigation Utility Suite")
    subparsers = parser.add_subparsers(dest="command")

    p1 = subparsers.add_parser("label-exhibits")
    p1.add_argument("path")

    p2 = subparsers.add_parser("link-motions")
    p2.add_argument("path")

    p3 = subparsers.add_parser("validate-signature")
    p3.add_argument("path")

    args = parser.parse_args()

    if args.command == "label-exhibits":
        label_exhibits(args.path)
    elif args.command == "link-motions":
        link_motions(args.path)
    elif args.command == "validate-signature":
        print("\u2714 Signature valid" if validate_signature(args.path) else "\u2718 Signature missing")

