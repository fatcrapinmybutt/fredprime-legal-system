import importlib
import subprocess
from pathlib import Path
from typing import Any


def test_dispatch_task_rejects_invalid_motion(tmp_path: Path, monkeypatch: Any) -> None:
    engine: Any = importlib.import_module("MBP_Omnia_Engine")
    engine.LOG_PATH = tmp_path / "log.txt"
    engine.MODULES_PATH = tmp_path
    calls: list[Any] = []

    def fake_popen(*args: Any, **kwargs: Any) -> None:
        calls.append(args)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    engine.dispatch_task({"motion_type": "NotReal"})
    assert calls == []
    assert "Invalid motion type" in engine.LOG_PATH.read_text()


def test_paths_are_path_objects() -> None:
    engine: Any = importlib.reload(importlib.import_module("MBP_Omnia_Engine"))
    assert isinstance(engine.TASK_QUEUE_PATH, Path)
