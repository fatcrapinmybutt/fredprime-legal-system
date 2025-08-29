from __future__ import annotations

"""Core orchestration for Codex build and guardian enforcement."""

import ast
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, cast

import yaml

from modules.codex_guardian import run_guardian
from modules.codex_supreme import self_diagnostic

MANIFEST = "codex_manifest.json"

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
    """Load repository configuration from ``.codex_config.yaml`` if present."""

    cfg = DEFAULT_CONFIG.copy()
    path = Path(".codex_config.yaml")
    if path.exists():
        data = cast(Dict[str, Any], yaml.safe_load(path.read_text(encoding="utf-8")))
        cfg.update(data)
    return cfg


def hard_guardian() -> None:
    """Enforce branch naming and commit message policies before proceeding."""

    cfg = load_config()
    branch = (
        subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        .decode()
        .strip()
    )
    prefix = cfg.get("branch_prefix", "codex/")
    if not branch.startswith(prefix):
        raise ValueError(f"Branch name must start with '{prefix}'")
    msg = subprocess.check_output(["git", "log", "-1", "--pretty=%B"]).decode().strip()
    banned = cfg.get("banned_keywords", [])
    if any(word in msg for word in banned):
        raise ValueError("Commit message contains banned keyword")
    if cfg.get("banned_keywords") != DEFAULT_CONFIG["banned_keywords"]:
        raise ValueError("Config banned keywords mismatch")


def hash_file(path: Path) -> str:
    """Return SHA-256 hex digest for ``path`` contents."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_dependencies(path: Path) -> List[str]:
    """Extract top-level import dependencies from ``path``."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception:
        return []
    deps: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                deps.append(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            deps.append(node.module.split(".")[0])
    return sorted(set(deps))


def update_manifest() -> None:
    """Regenerate the manifest with hashes and discovered dependencies."""

    manifest: List[Dict[str, Any]] = []
    for p in Path(".").rglob("*.py"):
        if p.parts[0].startswith("."):
            continue  # skip hidden directories
        manifest.append(
            {
                "module": p.stem,
                "path": str(p),
                "hash": hash_file(p),
                "legal_function": "",
                "dependencies": parse_dependencies(p),
                "timestamp": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            }
        )
    Path(MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    """Entry point for manifest updates and guardian enforcement."""

    triggered = False
    if os.getenv("CODEX_SKIP_GUARDIAN") != "1":
        hard_guardian()
        triggered = run_guardian()
    update_manifest()
    if triggered:
        subprocess.run(["python", "ZIP_VALIDATOR.py"], check=False)
    self_diagnostic()
    print("codex manifest updated")


if __name__ == "__main__":
    main()
