import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, cast

from core.local_llm import analyze_content

MANIFEST_FILE = "codex_manifest.json"
BANNED_KEYWORDS = ["TODO", "WIP", "temp_var", "placeholder"]


def get_current_branch() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode()
        .strip()
    )


def get_last_commit_message() -> str:
    return subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode().strip()


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> List[Dict[str, object]]:
    if Path(MANIFEST_FILE).exists():
        data = json.loads(Path(MANIFEST_FILE).read_text())
        return cast(List[Dict[str, object]], data)
    return []


def verify_commit_message(msg: str) -> None:
    analysis = analyze_content(msg)
    tokens = cast(List[str], analysis["tokens"])
    if any(k.lower() in tokens for k in BANNED_KEYWORDS):
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
    return any(key in branch for key in triggers)


def verify_manifest_hashes() -> None:
    manifest = load_manifest()
    for entry in manifest:
        path = Path(cast(str, entry["path"]))
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
