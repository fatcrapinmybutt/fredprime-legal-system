"""Tests for the unified litigator CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ZIP_VALIDATOR import required_files


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            Path(__file__).resolve().parents[1] / "litigator_cli.py",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_build(tmp_path: Path) -> None:
    output = tmp_path / "system.json"
    result = run_cli("build", str(output))
    assert output.exists(), result.stdout
    assert output.read_text().startswith("{")


def test_validate(tmp_path: Path) -> None:
    for file in required_files:
        path = tmp_path / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    result = run_cli("validate", str(tmp_path))
    assert "All required files" in result.stdout
