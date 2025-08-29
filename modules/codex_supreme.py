from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, cast

PERSISTENT_STATE_FILE = "codex_state.json"
AUDIT_LOG = "audit_chain.log"
MANIFEST_FILE = "codex_manifest.json"
ERROR_LOG = "logs/codex_errors.log"
PATCH_HISTORY = "patch_history.json"


def sha256_file(fpath: str) -> str:
    try:
        return hashlib.sha256(Path(fpath).read_bytes()).hexdigest()
    except Exception:
        return ""


def log_event(event: str, log_file: str = AUDIT_LOG) -> None:
    ts = datetime.datetime.now().isoformat()
    hval = hashlib.sha256(event.encode()).hexdigest()
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"{ts} {hval} {event}\n")


def save_state(state: Dict[str, Any], state_file: str = PERSISTENT_STATE_FILE) -> None:
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def load_state(state_file: str = PERSISTENT_STATE_FILE) -> Dict[str, Any]:
    if os.path.exists(state_file):
        with open(state_file, encoding="utf-8") as f:
            return cast(Dict[str, Any], json.load(f))
    return {}


def self_diagnostic() -> List[str]:
    diagnostics: List[str] = []
    for path in [MANIFEST_FILE, ERROR_LOG, PATCH_HISTORY, PERSISTENT_STATE_FILE]:
        if not os.path.exists(path):
            diagnostics.append(f"Missing critical file: {path}")
        else:
            diagnostics.append(f"OK: {path}")
    manifest: List[Dict[str, Any]] = []
    if os.path.exists(MANIFEST_FILE):
        manifest = json.loads(Path(MANIFEST_FILE).read_text(encoding="utf-8"))
        for entry in manifest:
            p = Path(entry["path"])
            if p.exists() and sha256_file(str(p)) != entry.get("hash"):
                diagnostics.append(f"File hash mismatch: {p}")
    save_state({"last_diagnostic": diagnostics})
    return diagnostics


def forensic_integrity_check() -> List[str]:
    issues: List[str] = []
    if not os.path.exists(MANIFEST_FILE):
        return issues
    manifest: List[Dict[str, Any]] = json.loads(
        Path(MANIFEST_FILE).read_text(encoding="utf-8")
    )
    for entry in manifest:
        path = Path(entry["path"])
        if path.exists() and sha256_file(str(path)) != entry.get("hash"):
            issues.append(f"Tampered: {path}")
    save_state({"last_integrity_check": issues})
    return issues


def timeline_event_matrix() -> List[Dict[str, Any]]:
    timeline: List[Dict[str, Any]] = []
    if os.path.exists(MANIFEST_FILE):
        manifest: List[Dict[str, Any]] = json.loads(
            Path(MANIFEST_FILE).read_text(encoding="utf-8")
        )
        for entry in manifest:
            timeline.append(
                {
                    "file": entry.get("path"),
                    "date": entry.get("timestamp"),
                    "legal_function": entry.get("legal_function"),
                    "validated": entry.get("validated"),
                }
            )
    timeline.sort(key=lambda x: x.get("date") or "")
    save_state({"timeline_matrix": timeline})
    return timeline
