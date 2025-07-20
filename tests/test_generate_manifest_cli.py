import json
import subprocess
from pathlib import Path


def test_cli_generates_manifest(tmp_path):
    output = tmp_path / "manifest.json"
    subprocess.run([
        "python",
        "cli/generate_manifest.py",
        "-o",
        str(output),
    ], check=True)
    assert output.exists(), "Manifest file was not created"
    data = json.loads(output.read_text())
    assert "files" in data
