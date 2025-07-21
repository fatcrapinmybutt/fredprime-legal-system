import json
import subprocess
import sys


def test_cli_generates_manifest(tmp_path):
    output = tmp_path / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "cli/generate_manifest.py",
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert output.exists()
    assert result.stdout.strip().endswith(str(output))
    data = json.loads(output.read_text())
    assert "files" in data
