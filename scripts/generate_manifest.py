import hashlib
import json
import os
from pathlib import Path
from typing import Dict

MANIFEST_PATH = Path("codex_manifest.json")


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def generate_manifest() -> Dict[str, str]:
    manifest: Dict[str, str] = {}
    for root, _, files in os.walk("."):
        for file in files:
            if file.endswith(".py") and "tests" not in root:
                p = Path(root) / file
                manifest[str(p)] = hash_file(p)
    return manifest


def main() -> None:
    manifest = generate_manifest()
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
