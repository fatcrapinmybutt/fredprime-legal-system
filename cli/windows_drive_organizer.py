import argparse
import os
from pathlib import Path
from typing import Iterable, List, Set

DEFAULT_DENY_DIRS = {
    "Windows",
    "Program Files",
    "Program Files (x86)",
    "$Recycle.Bin",
    "System Volume Information",
    "node_modules",
    ".git",
    "__pycache__",
    "venv",
    "dist",
    "build",
}


def _normalize_names(names: Iterable[str]) -> Set[str]:
    return {name.strip().casefold() for name in names if name.strip()}


def _parse_deny_dirs(values: List[str]) -> List[str]:
    parsed: List[str] = []
    for value in values:
        parsed.extend([part.strip() for part in value.split(",") if part.strip()])
    return parsed


def discover_files(root: Path, deny_dirs: Iterable[str]) -> List[Path]:
    deny_names = _normalize_names(deny_dirs)
    candidates: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name for name in dirnames if name.strip().casefold() not in deny_names
        ]
        for filename in filenames:
            candidates.append(Path(dirpath) / filename)
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover files on a Windows drive while skipping risky system folders."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="F:/",
        help="Root path to scan (default F:/)",
    )
    parser.add_argument(
        "--deny-dirs",
        action="append",
        default=[],
        help=(
            "Comma-separated list of directory names to deny. "
            "Use 'none' to disable the default denylist."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extras = _parse_deny_dirs(args.deny_dirs)
    if any(value.casefold() == "none" for value in extras):
        deny_dirs = [value for value in extras if value.casefold() != "none"]
    else:
        deny_dirs = list(DEFAULT_DENY_DIRS) + extras
    root = Path(args.path)
    if not root.exists():
        raise SystemExit(f"Root path does not exist: {root}")
    candidates = discover_files(root, deny_dirs)
    print(f"Discovered {len(candidates)} files under {root}")


if __name__ == "__main__":
    main()
