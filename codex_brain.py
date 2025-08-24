import hashlib
import json
from pathlib import Path
from typing import List

from modules.codex_guardian import run_guardian
from modules.codex_supreme import self_diagnostic

MANIFEST = "codex_manifest.json"

FORBIDDEN_TOKENS = ("TODO", "WIP", "temp_var", "placeholder")
FORBIDDEN_CALLS = ("eval(", "exec(", "subprocess.Popen", "subprocess.call")


def hash_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def enforce_source_hygiene(path: Path) -> None:
    text = path.read_text()
    for token in FORBIDDEN_TOKENS:
        if token in text:
            raise ValueError(f"{path} contains forbidden token '{token}'")
    for call in FORBIDDEN_CALLS:
        if call in text:
            raise ValueError(f"{path} contains forbidden call '{call}'")


def collect_dependencies(path: Path) -> List[str]:
    return [
        line.split("import", 1)[1].strip()
        for line in path.read_text().splitlines()
        if line.startswith("import ") or line.startswith("from ")
    ]


def update_manifest() -> None:
    manifest = []
    for p in Path(".").rglob("*.py"):
        if p.parts[0].startswith("."):
            continue
        enforce_source_hygiene(p)
        manifest.append(
            {
                "module": p.stem,
                "path": str(p),
                "hash": hash_file(p),
                "legal_function": p.stem.replace("_", " ").title(),
                "dependencies": collect_dependencies(p),
            }
        )
    Path(MANIFEST).write_text(json.dumps(manifest, indent=2))


def main() -> None:
    run_guardian()
    update_manifest()
    self_diagnostic()
    print("codex manifest updated")


if __name__ == "__main__":
    main()
