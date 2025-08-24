import json
from pathlib import Path
from typing import Any

from modules import codex_supreme


def test_save_and_load_state(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    codex_supreme.save_state({"x": 1}, state_file=str(state_file))
    state = codex_supreme.load_state(state_file=str(state_file))
    assert state == {"x": 1}


def test_forensic_integrity_check(tmp_path: Path, monkeypatch: Any) -> None:
    p = tmp_path / "file.txt"
    p.write_text("data")
    manifest = [{"path": str(p), "hash": "0" * 64}]
    (tmp_path / codex_supreme.MANIFEST_FILE).write_text(json.dumps(manifest))
    monkeypatch.chdir(tmp_path)
    issues = codex_supreme.forensic_integrity_check()
    assert issues and "Tampered" in issues[0]
