from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

CODE_EXTENSIONS = {".py", ".ps1", ".sh", ".bat"}
REGISTRY_PATH = Path("output/CODE_KEEPER_REGISTRY.json")
LOG_PATH = Path("output/code_scan_log.csv")


def hash_file(path: Path) -> str:
    """Return SHA-256 for a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:  # noqa: PTH123
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_paths(roots: Iterable[str]) -> List[Dict[str, str]]:
    """Scan roots for code files and return records."""
    records: List[Dict[str, str]] = []
    for root in roots:
        for p in Path(root).rglob("*"):
            if p.is_file() and p.suffix.lower() in CODE_EXTENSIONS:
                records.append(
                    {
                        "path": str(p.resolve()),
                        "hash": hash_file(p),
                        "scanned_at": datetime.utcnow().isoformat(),
                    }
                )
    return records


def write_log(records: List[Dict[str, str]], log_path: Path = LOG_PATH) -> None:
    """Write scan results to a CSV log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as fh:  # noqa: PTH123
        writer = csv.DictWriter(fh, fieldnames=["path", "hash", "scanned_at"])
        writer.writeheader()
        writer.writerows(records)


def update_registry(
    records: List[Dict[str, str]], registry_path: Path = REGISTRY_PATH
) -> None:
    """Update the CODE_KEEPER_REGISTRY.json file."""
    data: Dict[str, Dict[str, str]] = {}
    if registry_path.exists():
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    for rec in records:
        data[rec["path"]] = {"hash": rec["hash"]}
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    roots = [str(Path.cwd())]
    records = scan_paths(roots)
    write_log(records)
    update_registry(records)


if __name__ == "__main__":
    main()
