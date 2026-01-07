import os
import sys
import types
import importlib
from pathlib import Path
from types import ModuleType

from pytest import MonkeyPatch


def load_autopacker(monkeypatch: MonkeyPatch) -> ModuleType:
    dummy_docx = types.SimpleNamespace(Document=object)
    monkeypatch.setitem(sys.modules, "docx", dummy_docx)
    monkeypatch.delitem(sys.modules, "foia.autopacker", raising=False)
    return importlib.import_module("foia.autopacker")


def test_custom_base_dir(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LEGAL_RESULTS_DIR", str(tmp_path))
    autopacker = load_autopacker(monkeypatch)
    assert autopacker.BASE_DIR == str(tmp_path)
    monkeypatch.delenv("LEGAL_RESULTS_DIR", raising=False)


def test_default_base_dir(monkeypatch: MonkeyPatch) -> None:
    autopacker = load_autopacker(monkeypatch)
    assert autopacker.BASE_DIR == os.path.join("F:/", "LegalResults")
    monkeypatch.delenv("LEGAL_RESULTS_DIR", raising=False)
