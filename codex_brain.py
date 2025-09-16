import hashlib
import json
from pathlib import Path

from modules.codex_guardian import run_guardian
from modules.codex_supreme import self_diagnostic

MANIFEST = "codex_manifest.json"
BANNED_PATTERNS = [
    "TODO",
    "WIP",
    "temp_var",
    "placeholder",
    "eval(",
    "exec(",
]


def hard_coded_guardian() -> None:
    for path in Path(".").rglob("*.py"):
        if (
            path.match("modules/codex_guardian.py")
            or path.name == "codex_brain.py"
            or path.parts[0] == "tests"
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pat in BANNED_PATTERNS:
            if pat in text:
                raise ValueError(f"Banned pattern '{pat}' found in {path}")


def hash_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def update_manifest() -> None:
    manifest = []
    for p in Path(".").rglob("*.py"):
        if p.parts[0].startswith("."):  # skip hidden dirs
            continue
        manifest.append({"module": p.stem, "path": str(p), "hash": hash_file(p)})
    Path(MANIFEST).write_text(json.dumps(manifest, indent=2))


def main() -> None:
    hard_coded_guardian()
    run_guardian()
    update_manifest()
    self_diagnostic()
    print("codex manifest updated")


if __name__ == "__main__":
    main()
