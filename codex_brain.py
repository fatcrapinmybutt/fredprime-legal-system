import hashlib
import json
from pathlib import Path

from modules.codex_guardian import run_guardian
from modules.codex_supreme import self_diagnostic

MANIFEST = "codex_manifest.json"


def hash_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def update_manifest():
    manifest = []
    for p in Path(".").rglob("*.py"):
        if p.parts[0].startswith("."):  # skip hidden dirs
            continue
        manifest.append({"module": p.stem, "path": str(p), "hash": hash_file(p)})
    Path(MANIFEST).write_text(json.dumps(manifest, indent=2))


def main() -> None:
    run_guardian()
    update_manifest()
    self_diagnostic()
    print("codex manifest updated")


if __name__ == "__main__":
    main()
