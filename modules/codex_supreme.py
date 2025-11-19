import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import cast

from core.local_llm import analyze_content

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
    if not analyze_content(event)["word_count"]:
        return
    ts = datetime.datetime.now().isoformat()
    hval = hashlib.sha256(event.encode()).hexdigest()
    with open(log_file, "a") as f:
        f.write(f"{ts} {hval} {event}\n")


def save_state(
    state: dict[str, object], state_file: str = PERSISTENT_STATE_FILE
) -> None:
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def load_state(state_file: str = PERSISTENT_STATE_FILE) -> dict[str, object]:
    if os.path.exists(state_file):
        with open(state_file) as f:
            data = json.load(f)
            return cast(dict[str, object], data)
    return {}


def self_diagnostic() -> list[str]:
    diagnostics: list[str] = []
    for path in [MANIFEST_FILE, ERROR_LOG, PATCH_HISTORY, PERSISTENT_STATE_FILE]:
        if not os.path.exists(path):
            diagnostics.append(f"Missing critical file: {path}")
        else:
            diagnostics.append(f"OK: {path}")
    manifest: list[dict[str, object]] = []
    if os.path.exists(MANIFEST_FILE):
        data = json.loads(Path(MANIFEST_FILE).read_text())
        manifest = cast(list[dict[str, object]], data)
        for entry in manifest:
            p = Path(cast(str, entry["path"]))
            if p.exists() and sha256_file(str(p)) != entry.get("hash"):
                diagnostics.append(f"File hash mismatch: {p}")
    save_state({"last_diagnostic": diagnostics})
    return diagnostics


def forensic_integrity_check() -> list[str]:
    issues: list[str] = []
    if not os.path.exists(MANIFEST_FILE):
        return issues
    data = json.loads(Path(MANIFEST_FILE).read_text())
    manifest = cast(list[dict[str, object]], data)
    for entry in manifest:
        path = Path(cast(str, entry["path"]))
        if path.exists() and sha256_file(str(path)) != entry.get("hash"):
            issues.append(f"Tampered: {path}")
    save_state({"last_integrity_check": issues})
    return issues


def timeline_event_matrix() -> list[dict[str, object]]:
    timeline: list[dict[str, object]] = []
    if os.path.exists(MANIFEST_FILE):
        data = json.loads(Path(MANIFEST_FILE).read_text())
        manifest = cast(list[dict[str, object]], data)
        for entry in manifest:
            timeline.append(
                {
                    "file": entry.get("path"),
                    "date": entry.get("timestamp"),
                    "legal_function": entry.get("legal_function"),
                    "validated": entry.get("validated"),
                }
            )
    timeline.sort(key=lambda x: cast(str, x.get("date") or ""))
    save_state({"timeline_matrix": timeline})
    return timeline
