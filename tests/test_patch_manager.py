import hashlib
import logging
from pathlib import Path

import codex_patch_manager as cpm


def test_verify_patch_hash(tmp_path: Path) -> None:
    patch = tmp_path / "patch.py"
    patch.write_text("x = 1\n", encoding="utf-8")
    expected = hashlib.sha256(patch.read_bytes()).hexdigest()
    assert cpm.verify_patch_hash(str(patch), expected)
    assert not cpm.verify_patch_hash(str(patch), "0" * 64)


def test_apply_patch_blocks_restricted_modules(tmp_path: Path) -> None:
    cpm.PATCH_HISTORY = str(tmp_path / "history.json")
    cpm.ERROR_LOG = str(tmp_path / "errors.log")
    logging.basicConfig(filename=cpm.ERROR_LOG, level=logging.INFO, force=True)

    target = tmp_path / "target.txt"
    target.write_text("orig", encoding="utf-8")
    patch = tmp_path / "patch.py"
    patch.write_text(
        "import os\nopen(r'{}','w').write('patched')".format(target),
        encoding="utf-8",
    )
    cpm.apply_patch(str(patch), str(target))
    assert target.read_text(encoding="utf-8") == "orig"


def test_apply_patch_success(tmp_path: Path) -> None:
    cpm.PATCH_HISTORY = str(tmp_path / "history.json")
    cpm.ERROR_LOG = str(tmp_path / "errors.log")
    logging.basicConfig(filename=cpm.ERROR_LOG, level=logging.INFO, force=True)

    target = tmp_path / "target.txt"
    target.write_text("orig", encoding="utf-8")
    patch = tmp_path / "patch.py"
    patch.write_text(
        "open(r'{}','w').write('patched')".format(target),
        encoding="utf-8",
    )
    cpm.apply_patch(str(patch), str(target))
    assert target.read_text(encoding="utf-8") == "patched"
