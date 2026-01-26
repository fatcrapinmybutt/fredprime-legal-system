import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

MANIFEST_FILE = "codex_manifest.json"
BANNED_KEYWORDS = ["TODO", "WIP", "temp_var", "placeholder"]


def get_current_branch() -> str:
    return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()


def get_last_commit_message() -> str:
    return subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode().strip()


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> list[dict]:
    if Path(MANIFEST_FILE).exists():
        return json.loads(Path(MANIFEST_FILE).read_text())
    return []


def verify_commit_message(msg: str) -> None:
    if any(k in msg for k in BANNED_KEYWORDS):
        raise ValueError("Commit message contains banned keyword")
    if not re.match(r"^\[(core|hotfix|docs|merge|patch|engine|matrix|echelon)\] ", msg):
        raise ValueError("Commit message format invalid")


def verify_branch_name(branch: str) -> bool:
    if not branch.startswith("codex/"):
        raise ValueError("Branch name must start with 'codex/'")
    triggers = [
        "core",
        "engine",
        "matrix",
        "protocol",
        "epoch",
        "echelon",
        "hotfix",
        "merge",
    ]
    if not any(key in branch for key in triggers):
        raise ValueError(
            "Branch name must include a trigger keyword: "
            f"{', '.join(triggers)}"
        )
    return True


def verify_manifest_hashes() -> None:
    manifest = load_manifest()
    for entry in manifest:
        path = Path(entry["path"])
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        if hash_file(path) != entry["hash"]:
            raise ValueError(f"Hash mismatch for {path}")


def run_guardian() -> None:
    branch = get_current_branch()
    msg = get_last_commit_message().splitlines()[0]
    verify_branch_name(branch)
    verify_commit_message(msg)
    if Path(MANIFEST_FILE).exists():
        verify_manifest_hashes()
