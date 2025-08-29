from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import yaml

from modules.codex_guardian import run_guardian
from modules.codex_supreme import self_diagnostic

MANIFEST = "codex_manifest.json"
BANNED_KEYWORDS: List[str] = ["TODO", "WIP", "temp_var", "placeholder"]


def hard_guardian() -> None:
    branch = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode()
        .strip()
    )
    if not branch.startswith("codex/"):
        raise ValueError("Branch name must start with 'codex/'")
    msg = (
        subprocess.check_output(
            [
                "git",
                "log",
                "-1",
                "--pretty=%B",
            ]
        )
        .decode()
        .strip()
    )
    if any(word in msg for word in BANNED_KEYWORDS):
        raise ValueError("Commit message contains banned keyword")
    cfg = Path(".codex_config.yaml")
    if cfg.exists():
        data: Dict[str, Any] = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        if data.get("banned_keywords") != BANNED_KEYWORDS:
            raise ValueError("Config banned keywords mismatch")


def hash_file(path: Path) -> str:
    """Return SHA-256 hex digest for ``path`` contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def update_manifest() -> None:
    manifest: List[Dict[str, Any]] = []
    for p in Path(".").rglob("*.py"):
        if p.parts[0].startswith("."):
            # skip hidden dirs
            continue
        manifest.append(
            {
                "module": p.stem,
                "path": str(p),
                "hash": hash_file(p),
                "legal_function": "",
                "dependencies": [],
            }
        )
    Path(MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    if os.getenv("CODEX_SKIP_GUARDIAN") != "1":
        hard_guardian()
        run_guardian()
    update_manifest()
    self_diagnostic()
    print("codex manifest updated")


if __name__ == "__main__":
    main()
