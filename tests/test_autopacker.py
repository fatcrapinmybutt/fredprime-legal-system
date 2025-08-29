import os
import importlib
import zipfile
from pathlib import Path

import pytest


def test_build_requests_creates_zip_and_hashes(tmp_path: Path) -> None:
    os.environ["LEGAL_RESULTS_DIR"] = str(tmp_path)
    autopacker = importlib.reload(importlib.import_module("foia.autopacker"))
    autopacker.build_requests()
    output_dir = Path(tmp_path) / "FOIA"
    zip_path = output_dir / "FOIA_PACKET_SHADY_OAKS_2025.zip"
    with zipfile.ZipFile(zip_path, "r") as zf:
        assert set(zf.namelist()) == set(autopacker.EXPECTED_HASHES.keys())
    for name, expected in autopacker.EXPECTED_HASHES.items():
        file_path = output_dir / name
        assert autopacker._docx_sha256(str(file_path)) == expected


def test_build_requests_unwritable_output_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    os.environ["LEGAL_RESULTS_DIR"] = str(tmp_path)
    autopacker = importlib.reload(importlib.import_module("foia.autopacker"))

    def raise_perm(*args: object, **kwargs: object) -> None:
        raise PermissionError("cannot create directory")

    monkeypatch.setattr(autopacker.os, "makedirs", raise_perm)
    with pytest.raises(PermissionError):
        autopacker.build_requests()
