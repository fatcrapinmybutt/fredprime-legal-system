from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, cast

import yaml

MANIFEST_FILE = "codex_manifest.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "branch_prefix": "codex/",
    "banned_keywords": ["TODO", "WIP", "temp_var", "placeholder"],
    "branch_triggers": [
        "core",
        "engine",
        "matrix",
        "protocol",
        "epoch",
        "echelon",
        "patch",
        "hotfix",
    ],
}


def load_config() -> Dict[str, Any]:
    cfg = DEFAULT_CONFIG.copy()
    path = Path(".codex_config.yaml")
    if path.exists():
        data = cast(Dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))
        cfg.update(data)
    return cfg


def get_current_branch() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode()
        .strip()
    )


def get_last_commit_message() -> str:
    return subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode().strip()


def hash_file(path: Path) -> str:
    """Return SHA-256 hash of ``path`` contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> List[Dict[str, Any]]:
    if Path(MANIFEST_FILE).exists():
        data = json.loads(Path(MANIFEST_FILE).read_text(encoding="utf-8"))
        return cast(List[Dict[str, Any]], data)
    return []


def verify_commit_message(msg: str) -> None:
    cfg = load_config()
    banned = cfg.get("banned_keywords", [])
    if any(k in msg for k in banned):
        raise ValueError("Commit message contains banned keyword")
    if not re.match(r"^\[(core|hotfix|docs|merge|patch|engine|matrix|echelon)\] ", msg):
        raise ValueError("Commit message format invalid")


def verify_branch_name(branch: str) -> bool:
    cfg = load_config()
    prefix = cfg.get("branch_prefix", "codex/")
    triggers = cfg.get("branch_triggers", [])
    if not branch.startswith(prefix):
        raise ValueError(f"Branch name must start with '{prefix}'")
    return any(key in branch for key in triggers)


def verify_manifest_hashes() -> None:
    manifest = load_manifest()
    for entry in manifest:
        path = Path(entry["path"])
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        if hash_file(path) != entry["hash"]:
            raise ValueError(f"Hash mismatch for {path}")


def run_guardian() -> bool:
    branch = get_current_branch()
    msg = get_last_commit_message().splitlines()[0]
    triggered = verify_branch_name(branch)
    verify_commit_message(msg)
    if Path(MANIFEST_FILE).exists():
        verify_manifest_hashes()
    return triggered
