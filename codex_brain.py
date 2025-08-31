import hashlib
import json
from pathlib import Path

from modules.codex_guardian import run_guardian
from modules.codex_supreme import self_diagnostic
import yaml

BANNED_WORDS = ["TODO", "WIP", "temp_var", "placeholder"]
CONFIG_FILE = ".codex_config.yaml"

MANIFEST = "codex_manifest.json"


def hash_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def update_manifest():
    manifest = []
    for p in Path(".").rglob("*.py"):
        if p.parts[0].startswith("."):
            continue
        manifest.append({"module": p.stem, "path": str(p), "hash": hash_file(p)})
    Path(MANIFEST).write_text(json.dumps(manifest, indent=2))


def enforce_source_cleanliness() -> None:
    for path in Path(".").rglob("*.py"):
        if path.parts[0].startswith("."):
            continue
        text = path.read_text(errors="ignore")
        for word in BANNED_WORDS:
            if word in text:
                raise ValueError(f"Banned keyword '{word}' found in {path}")


def load_config() -> dict:
    cfg_path = Path(CONFIG_FILE)
    if not cfg_path.exists():
        raise FileNotFoundError("Missing .codex_config.yaml")
    return yaml.safe_load(cfg_path.read_text())


def main() -> None:
    load_config()
    enforce_source_cleanliness()
    run_guardian()
    update_manifest()
    self_diagnostic()
    print("codex manifest updated")


if __name__ == "__main__":
    main()
