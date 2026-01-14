import os
import subprocess
from pathlib import Path

import pytest


def test_make_context7_setup_creates_env(tmp_path, monkeypatch):
    # Run the make target in the repository's tools directory, passing the key
    repo_root = Path(__file__).resolve().parents[1]
    tools_dir = repo_root / "tools"
    assert tools_dir.exists(), "tools directory must exist"

    env_file = tools_dir / ".mcp_env"
    if env_file.exists():
        env_file.unlink()

    # Provide a test key via environment to avoid interactive prompt
    test_key = "testkey_makefile_123"

    env = os.environ.copy()
    env["CONTEXT7_API_KEY"] = test_key

    # Run make in the tools directory
    subprocess.run(["make", "context7-setup"], cwd=str(tools_dir), check=True, env=env)

    assert env_file.exists(), "Make target should create .mcp_env in tools"
    content = env_file.read_text()
    assert f"CONTEXT7_API_KEY={test_key}" in content

    # Cleanup
    try:
        env_file.unlink()
    except Exception:
        pass
