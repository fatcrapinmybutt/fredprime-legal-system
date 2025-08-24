import json
from pathlib import Path
from typing import Any

import codex_brain


def test_update_manifest_creates_entry(tmp_path: Path, monkeypatch: Any) -> None:
    sample = tmp_path / "sample.py"
    sample.write_text("import os\nprint('hi')\n")
    monkeypatch.chdir(tmp_path)
    codex_brain.update_manifest((), ())
    manifest_path = tmp_path / codex_brain.MANIFEST
    data = json.loads(manifest_path.read_text())
    assert data[0]["module"] == "sample"
    assert "timestamp" in data[0]
    assert data[0]["validated"] is False
    assert "os" in data[0]["dependencies"]
