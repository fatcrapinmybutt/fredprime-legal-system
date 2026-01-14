import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_generates_valid_json(tmp_path):
    # repo root
    repo_root = Path(__file__).resolve().parents[1]

    schema_path = repo_root / "schema" / "systemdef.schema.json"
    assert schema_path.exists(), f"schema not found at {schema_path}"

    out_file = tmp_path / "generated.json"

    env = os.environ.copy()
    env["FREDPRIME_SCHEMA"] = str(schema_path)

    # Run the script from the repo root to simulate normal usage, but write output into tmp_path
    script = repo_root / "firstimport.py"
    assert script.exists(), "firstimport.py not found"

    completed = subprocess.run(
        [sys.executable, str(script), "--out", str(out_file)],
        cwd=str(repo_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(f"script failed: {completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")

    assert out_file.exists(), "Output JSON file was not created"

    with open(out_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    # basic sanity checks
    assert data.get("system") is not None
    assert data.get("config_hash") is not None

    # validate against schema using jsonschema if available
    try:
        import jsonschema

        with open(schema_path, "r", encoding="utf-8") as sf:
            schema = json.load(sf)
        jsonschema.validate(instance=data, schema=schema)
    except ModuleNotFoundError:
        pytest.skip("jsonschema not installed; skipping schema validation")
