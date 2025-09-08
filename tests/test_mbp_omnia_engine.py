import json
import subprocess
import types
import sys
from pathlib import Path

import pytest

# Stub dependency required by MBP_Omnia_Engine at import time
sys.modules.setdefault(
    "EPOCH_UNPACKER_ENGINE_v1", types.SimpleNamespace(run_gui=lambda: None)
)

import MBP_Omnia_Engine as omnia


@pytest.fixture
def temp_paths(tmp_path, monkeypatch):
    task_queue = tmp_path / "task_queue.json"
    log_file = tmp_path / "background.log"
    violation_log = tmp_path / "violations.json"
    scao_map = tmp_path / "forms.json"
    mcr_map = tmp_path / "mcr.json"
    modules_dir = tmp_path / "modules"
    modules_dir.mkdir()

    task_queue.write_text("[]")
    log_file.write_text("")
    scao_map.write_text(json.dumps({"TRO": "form"}))
    mcr_map.write_text(json.dumps({"TRO": "rule"}))

    monkeypatch.setattr(omnia, "TASK_QUEUE_PATH", str(task_queue))
    monkeypatch.setattr(omnia, "LOG_PATH", str(log_file))
    monkeypatch.setattr(omnia, "VIOLATION_LOG_PATH", str(violation_log))
    monkeypatch.setattr(omnia, "SCAO_FORM_MAP", str(scao_map))
    monkeypatch.setattr(omnia, "MCR_RULE_MAP", str(mcr_map))
    monkeypatch.setattr(omnia, "GUI_EXECUTABLE_PATH", str(tmp_path / "missing.exe"))

    handler = modules_dir / "tro_handler.py"
    handler.write_text("pass")

    def dispatch_task(task):
        handler_path = modules_dir / f"{task['motion_type'].lower()}_handler.py"
        if handler_path.exists():
            subprocess.Popen(["python", str(handler_path)], shell=False)

    monkeypatch.setattr(omnia, "dispatch_task", dispatch_task)

    return {
        "task_queue": task_queue,
        "log_file": log_file,
        "violation_log": violation_log,
    }


@pytest.fixture
def mock_violation_log(temp_paths):
    temp_paths["violation_log"].write_text(json.dumps([{"id": 1}]))
    return temp_paths["violation_log"]


def test_scan_for_new_violations_persists_tasks_and_logs(
    temp_paths, mock_violation_log, monkeypatch
):
    calls = []

    def fake_popen(cmd, shell=False):
        calls.append(cmd)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    omnia.scan_for_new_violations()

    assert len(calls) == 1
    assert "tro_handler.py" in calls[0][1]

    queue = json.loads(temp_paths["task_queue"].read_text())
    assert queue and queue[0]["motion_type"] == "TRO"

    log_content = temp_paths["log_file"].read_text()
    assert "TRO queued" in log_content
