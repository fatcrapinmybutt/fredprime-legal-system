# MBP OMNIA INFINITI + X\u2122 SYSTEM — THE LITIGATION SINGULARITY ENGINE
# Simplified background daemon that queues legal filings based on violation logs.

import os
import json
import threading
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from EPOCH_UNPACKER_ENGINE_v1 import run_gui as unpacker_gui

SCAN_INTERVAL_SECONDS = 1800  # 30 minutes

CONFIG_FILE = Path(
    os.environ.get("MBP_PATH_CONFIG", Path(__file__).parent / "config" / "paths.json")
)
with open(CONFIG_FILE, "r", encoding="utf-8") as cfg_file:
    CONFIG = json.load(cfg_file)

TASK_QUEUE_PATH = Path(
    os.environ.get("TASK_QUEUE_PATH", CONFIG["TASK_QUEUE_PATH"])
).expanduser()
VIOLATION_LOG_PATH = Path(
    os.environ.get("VIOLATION_LOG_PATH", CONFIG["VIOLATION_LOG_PATH"])
).expanduser()
AUTO_DOC_OUTPUT_PATH = Path(
    os.environ.get("AUTO_DOC_OUTPUT_PATH", CONFIG["AUTO_DOC_OUTPUT_PATH"])
).expanduser()
GUI_EXECUTABLE_PATH = Path(
    os.environ.get("GUI_EXECUTABLE_PATH", CONFIG["GUI_EXECUTABLE_PATH"])
).expanduser()
LOG_PATH = Path(os.environ.get("LOG_PATH", CONFIG["LOG_PATH"])).expanduser()
MODULES_PATH = Path(os.environ.get("MODULES_PATH", CONFIG["MODULES_PATH"])).expanduser()
CORE_PATH = Path(os.environ.get("CORE_PATH", CONFIG["CORE_PATH"])).expanduser()

VALID_MOTION_TYPES = {
    "tro",
    "affidavit",
    "verifiedcomplaint",
    "canonviolationreport",
    "\u00a71983civilrightsclaim",
    "ricoaction",
    "emergencyinjunction",
    "motiontocompel",
    "motiontodisqualifyjudge",
    "supervisoryappeal",
}


def load_json(path: Path) -> Any:
    if not path.exists():
        return [] if path.suffix == ".json" else {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def dispatch_task(task: Dict[str, Any]) -> None:
    motion = task.get("motion_type", "").lower()
    if motion not in VALID_MOTION_TYPES:
        with open(LOG_PATH, "a", encoding="utf-8") as log:
            log.write(f"{datetime.now().isoformat()} — Invalid motion type: {motion}\n")
        return
    handler = MODULES_PATH / f"{motion}_handler.py"
    if handler.exists():
        try:
            subprocess.Popen(["python", str(handler)], shell=False)
        except Exception as exc:
            with open(LOG_PATH, "a", encoding="utf-8") as log:
                log.write(f"{datetime.now().isoformat()} — Dispatch error: {exc}\n")
    else:
        with open(LOG_PATH, "a", encoding="utf-8") as log:
            log.write(f"{datetime.now().isoformat()} — No handler for {motion}\n")


def is_valid_motion_type(motion: str) -> bool:
    return motion.lower() in VALID_MOTION_TYPES


def scan_for_new_violations() -> None:
    print("[SCAN] Evaluating violation logs and queuing tasks")
    violations = load_json(VIOLATION_LOG_PATH)
    task_queue = load_json(TASK_QUEUE_PATH)
    now = datetime.now().isoformat()

    trigger_motion_types = [
        "TRO",
        "Affidavit",
        "VerifiedComplaint",
        "CanonViolationReport",
        "\u00a71983CivilRightsClaim",
        "RICOAction",
        "EmergencyInjunction",
        "MotionToCompel",
        "MotionToDisqualifyJudge",
        "SupervisoryAppeal",
    ]

    if len(violations) > 500:
        global SCAN_INTERVAL_SECONDS
        SCAN_INTERVAL_SECONDS = 300

    for motion in trigger_motion_types:
        if not is_valid_motion_type(motion):
            continue
        task = {
            "type": "GenerateMotion",
            "motion_type": motion,
            "timestamp": now,
            "source": "OMNIA_BackgroundDaemon",
            "status": "queued",
            "link": VIOLATION_LOG_PATH,
        }
        task_queue.append(task)
        dispatch_task(task)
        with open(LOG_PATH, "a", encoding="utf-8") as log:
            log.write(f"{now} — {motion} queued\n")

    save_json(TASK_QUEUE_PATH, task_queue)

    if GUI_EXECUTABLE_PATH.exists():
        try:
            subprocess.Popen([str(GUI_EXECUTABLE_PATH)], shell=False)
        except Exception as e:
            with open(LOG_PATH, "a", encoding="utf-8") as log:
                log.write(f"{datetime.now().isoformat()} — GUI launch failed: {e}\n")


def background_loop() -> None:
    print("[OMNIA ENGINE] Background daemon running")
    while True:
        scan_for_new_violations()
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    TASK_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    VIOLATION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUTO_DOC_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    for core in [
        "BENCHBOOKS",
        "MCR",
        "FORMS",
        "CANON",
        "EVIDENCE",
        "JUDGES",
        "TRANSCRIPTS",
    ]:
        (CORE_PATH / core).mkdir(parents=True, exist_ok=True)
    MODULES_PATH.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    thread = threading.Thread(target=background_loop, daemon=True)
    thread.start()
    unpacker_gui()
    thread.join()
