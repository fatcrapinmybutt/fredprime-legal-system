import json
from pathlib import Path

import pytest

import firstimport


def test_build_systemdef_creates_paths(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    output = base / "output"
    log = base / "logs"

    sysdef = firstimport.build_systemdef(base, output, log)

    assert sysdef["system"] == firstimport.SYSTEM_NAME
    assert "config_hash" in sysdef and isinstance(sysdef["config_hash"], str)
    # output and log directories should exist after build_systemdef
    assert Path(sysdef["output_path"]).exists()
    assert Path(sysdef["log_path"]).exists()


def test_write_systemdef_file_writes_json(tmp_path):
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    json_file = outdir / "sys.json"

    sysdef = firstimport.build_systemdef(outdir, outdir / "output", outdir / "logs")
    firstimport.write_systemdef_file(sysdef, json_file)

    assert json_file.exists()
    with open(json_file, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data.get("system") == firstimport.SYSTEM_NAME
