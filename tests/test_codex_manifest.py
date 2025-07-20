from pathlib import Path
import json
import hashlib
import shutil
import sys

# Import your core logic (adjust import if you relocate codex_manifest.py)
from modules.codex_manifest import generate_manifest, verify_all_modules

def test_verify_all_modules(tmp_path: Path) -> None:
    """End-to-end test of manifest validation and hash enforcement."""

    # -- SETUP: Create a dummy module and manifest --
    module_file = tmp_path / "example.py"
    module_file.write_text("print('hello')")
    manifest = generate_manifest([
        {
            "path": str(module_file),
            "legal_function": "dummy module",
            "dependencies": [],
        }
    ])
    manifest_path = tmp_path / "codex_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # -- Backup/replace codex_manifest.json if it exists in repo root --
    root_manifest = Path("codex_manifest.json")
    root_backup = None
    if root_manifest.exists():
        root_backup = root_manifest.read_bytes()
    try:
        shutil.copy(str(manifest_path), str(root_manifest))
        # Should pass validation (legal_function, hash, deps present)
        verify_all_modules(manifest)
        print("✅ Baseline module manifest validated.")

        # -- Trigger hash mismatch --
        module_file.write_text("print('changed')")
        try:
            verify_all_modules(manifest)
        except ValueError as e:
            assert "Hash mismatch" in str(e)
            print("✅ Hash mismatch detected as expected.")
        else:
            raise AssertionError("Hash mismatch not detected!")

        # -- Missing legal_function field --
        bad_manifest = manifest.copy()
        key = list(bad_manifest.keys())[0]
        del bad_manifest[key]["legal_function"]
        try:
            verify_all_modules(bad_manifest)
        except ValueError as e:
            assert "legal_function" in str(e)
            print("✅ Missing legal_function field detected.")
        else:
            raise AssertionError("Missing legal_function not detected!")

        # -- Missing dependencies field --
        bad_manifest = manifest.copy()
        key = list(bad_manifest.keys())[0]
        del bad_manifest[key]["dependencies"]
        try:
            verify_all_modules(bad_manifest)
        except ValueError as e:
            assert "dependencies" in str(e)
            print("✅ Missing dependencies field detected.")
        else:
            raise AssertionError("Missing dependencies not detected!")

    finally:
        # Restore or remove root manifest
        if root_backup is not None:
            root_manifest.write_bytes(root_backup)
        elif root_manifest.exists():
            root_manifest.unlink()

if __name__ == "__main__":
    # Standalone invocation for CI or manual run
    import tempfile
    tmp_dir = tempfile.TemporaryDirectory()
    test_verify_all_modules(Path(tmp_dir.name))
    print("All checks passed.")
