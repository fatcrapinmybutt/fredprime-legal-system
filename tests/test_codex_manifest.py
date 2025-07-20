from pathlib import Path
from modules.codex_manifest import generate_manifest, verify_all_modules


def test_generate_and_verify(tmp_path: Path) -> None:
    module_file = tmp_path / "example.py"
    module_file.write_text("print('hello')")

    manifest = generate_manifest(
        [
            {
                "path": str(module_file),
                "legal_function": "example module",
                "dependencies": [],
            }
        ]
    )

    entry = manifest[str(module_file)]
    assert entry["legal_function"] == "example module"
    assert entry["dependencies"] == []

    verify_all_modules(manifest)

    module_file.write_text("print('changed')")
    try:
        verify_all_modules(manifest)
    except ValueError as e:
        assert "Hash mismatch" in str(e)
    else:
        raise AssertionError("Hash mismatch not detected")

    # Missing legal_function should raise error
    bad_manifest = generate_manifest(
        [
            {
                "path": str(module_file),
                "dependencies": [],
            }
        ]
    )
    try:
        verify_all_modules(bad_manifest)
    except ValueError as e:
        assert "legal_function" in str(e)
    else:
        raise AssertionError("Missing legal_function not detected")

    # Explicitly remove the key to test missing field
    del bad_manifest[str(module_file)]["legal_function"]
    try:
        verify_all_modules(bad_manifest)
    except ValueError as e:
        assert "legal_function" in str(e)
    else:
        raise AssertionError("Missing legal_function not detected when key absent")

    # Missing dependencies should raise error
    bad_manifest = generate_manifest(
        [
            {
                "path": str(module_file),
                "legal_function": "example module",
            }
        ]
    )
    del bad_manifest[str(module_file)]["dependencies"]
    try:
        verify_all_modules(bad_manifest)
    except ValueError as e:
        assert "dependencies" in str(e)
    else:
        raise AssertionError("Missing dependencies not detected")

    # Dependencies must be a list
    bad_manifest[str(module_file)]["dependencies"] = "notalist"
    try:
        verify_all_modules(bad_manifest)
    except ValueError as e:
        assert "list" in str(e)
    else:
        raise AssertionError("Non-list dependencies not detected")
