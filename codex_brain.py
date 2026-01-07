import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, cast

from modules.codex_supreme import self_diagnostic

MANIFEST = "codex_manifest.json"
CONFIG_PATH = Path(".codex_config.yaml")


def hash_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def load_config() -> Dict[str, Any]:
    return cast(Dict[str, Any], json.loads(CONFIG_PATH.read_text()))


def enforce_guardian(cfg: Dict[str, Any]) -> None:
    branch = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode()
        .strip()
    )
    message = (
        subprocess.check_output(["git", "log", "-1", "--pretty=%B"])
        .decode()
        .strip()
        .splitlines()[0]
    )
    if not branch.startswith(cfg["branch_prefix"]):
        raise ValueError("branch name must start with codex/")
    if any(keyword in message for keyword in cfg["banned_keywords"]):
        raise ValueError("commit message contains banned keyword")
    if not any(message.startswith(f"[{kind}] ") for kind in cfg["commit_types"]):
        raise ValueError("commit message format invalid")


def update_manifest() -> None:
    manifest = []
    for p in Path(".").rglob("*.py"):
        if p.parts[0].startswith("."):
            continue
        manifest.append({"module": p.stem, "path": str(p), "hash": hash_file(p)})
    Path(MANIFEST).write_text(json.dumps(manifest, indent=2))


def main() -> None:
    cfg = load_config()
    enforce_guardian(cfg)
    update_manifest()
    self_diagnostic()
    print("codex manifest updated")


if __name__ == "__main__":
    main()
