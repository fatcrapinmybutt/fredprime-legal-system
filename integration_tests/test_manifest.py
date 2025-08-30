from __future__ import annotations

import json
from pathlib import Path


def test_manifest_has_entries() -> None:
    """Manifest should exist and contain entries with required keys."""

    manifest = Path("codex_manifest.json")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data, "manifest should not be empty"
    required = {"module", "path", "hash"}
    assert all(required.issubset(entry) for entry in data)
