from pathlib import Path
from codex_manifest import verify_all_modules
import json
import hashlib


def test_verify_all_modules(tmp_path: Path) -> None:
    path = tmp_path / "test.py"
    path.write_text("print('hello')\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {str(path): sha}
    manifest_path = tmp_path / "codex_manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    original = Path("codex_manifest.json")
    original_backup = None
    if original.exists():
        original_backup = original.read_text()
    try:
        original.write_text(manifest_path.read_text())
        verify_all_modules()
    finally:
        if original_backup is not None:
            original.write_text(original_backup)
        else:
            original.unlink()
