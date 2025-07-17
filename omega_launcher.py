"""Omega Litigation Launcher

Utilities for packaging forms, scanning evidence, and generating basic
reports. This script exposes a small subset of functionality described in the
Omega Litigation System overview. Features are intentionally modest and do not
implement any real litigation tactics."""

import argparse
from pathlib import Path
from datetime import datetime
import zipfile
import re

BASE_DIR = Path(__file__).parent


def export_gui(output: Path) -> None:
    """Create a minimal launcher script.

    This simply writes a small Python file that executes :func:`main`. It does
    not build a binary executable but provides a convenient entry point.
    """
    launcher = (
        "#!/usr/bin/env python3\n"
        "from omega_launcher import main\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    output.write_text(launcher)
    output.chmod(0o755)
    print(f"Created launcher script at {output}")


def build_zip(bundle: Path) -> None:
    """Package forms and manifest data into a zip archive."""
    forms_dir = BASE_DIR / "forms"
    manifest = BASE_DIR / "data" / "forms_manifest.json"
    with zipfile.ZipFile(bundle, "w") as zf:
        for path in forms_dir.glob("*.txt"):
            zf.write(path, arcname=path.name)
        if manifest.exists():
            zf.write(manifest, arcname=manifest.name)
    print(f"Created bundle {bundle}")


def run_scan(path: Path) -> None:
    """Scan text files for keywords defined in :mod:`src.constants`."""
    from src.constants import KEYWORDS_TO_FORMS

    hits = []
    for file in path.rglob("*.txt"):
        text = file.read_text(errors="ignore").lower()
        for word in KEYWORDS_TO_FORMS:
            if word in text:
                hits.append((file, word))
    if not hits:
        print("No trigger words found.")
        return
    for file, word in hits:
        print(f"{file}: matched '{word}'")


def launch_canon_system() -> None:
    """Search form text files for the word ``canon``.

    This routine simply prints the file names that contain the term.
    """
    pattern = re.compile(r"canon", re.IGNORECASE)
    forms_dir = BASE_DIR / "forms"
    matches = []
    for file in forms_dir.glob("*.txt"):
        text = file.read_text(errors="ignore")
        if pattern.search(text):
            matches.append(file)
    if matches:
        for m in matches:
            print(f"Potential canon reference in {m}")
    else:
        print("No canon references found in forms.")


def generate_declaration(output: Path) -> None:
    """Write a simple timestamped declaration file."""
    text = (
        "Declaration of Omega Litigation System\n"
        f"Timestamp: {datetime.utcnow().isoformat()}Z\n"
    )
    output.write_text(text)
    print(f"Wrote declaration to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Omega Litigation Launcher")
    sub = parser.add_subparsers(dest="command")

    gui = sub.add_parser("gui", help="Build GUI executable")
    gui.add_argument("output", type=Path, help="Output executable path")

    bundle = sub.add_parser("bundle", help="Create claims bundle zip")
    bundle.add_argument("path", type=Path, help="Output zip path")

    scan = sub.add_parser("scan", help="Scan a directory for trigger points")
    scan.add_argument("directory", type=Path, help="Directory to scan")

    sub.add_parser("canon", help="Launch judicial canon system")

    decl = sub.add_parser("declare", help="Generate lock-in declaration")
    decl.add_argument("output", type=Path, help="Declaration file")

    args = parser.parse_args()

    if args.command == "gui":
        export_gui(args.output)
    elif args.command == "bundle":
        build_zip(args.path)
    elif args.command == "scan":
        run_scan(args.directory)
    elif args.command == "canon":
        launch_canon_system()
    elif args.command == "declare":
        generate_declaration(args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
