#!/usr/bin/env python3
"""
bootstrap_mindseye2_graph_core_os.py

One-shot bootstrap script to create the full MINDSEYE2_GRAPH_CORE_OS
bundle under a chosen root directory (e.g., F:\\XXX).

- Creates required folders: config/, scripts/, graph_data/, neo4j/,
  ui/, ui/cache/, logs_templates/, state/, logs/
- Writes core files: README.txt, MANIFEST.txt, config/*.yml,
  config/requirements.txt, scripts/autopilot.py, ETL/import/layout
  scripts, basic UI, and log templates.

Usage (example):

    cd F:\\XXX
    python bootstrap_mindseye2_graph_core_os.py

or:

    python bootstrap_mindseye2_graph_core_os.py --root F:\\XXX
"""

import argparse
from pathlib import Path


def write_file(root: Path, rel_path: str, content: str) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"[SKIP]   {path} (already exists)")
        return
    path.write_text(content.lstrip("\n"), encoding="utf-8")
    print(f"[CREATE] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap MINDSEYE2_GRAPH_CORE_OS bundle.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Target root directory for the bundle (default: current working directory).",
    )
    args = parser.parse_args()

    if args.root:
        root = Path(args.root).expanduser().resolve()
    else:
        root = Path.cwd().resolve()

    print(f"Using bundle root: {root}")
    root.mkdir(parents=True, exist_ok=True)

    # 1) README
    readme = """\
MINDSEYE2_GRAPH_CORE_OS
=======================

Version: 1.0.0
Bundle Root: .

PURPOSE
-------
This bundle turns your drives (F:\\, D:\\, and any mounted cloud paths)
into a live, directed graph: nodes, edges, and artifacts. It ingests
existing graph exports (JSON, CSV, GraphML, HTML, ZIP), normalizes them
into canonical node/edge tables, updates a Neo4j graph (optional),
computes a "nucleus wheel" layout, and exposes interactive views
through a UI.

Once started, the autopilot can watch your folders for new/changed
files and chain multiple cycles:

    Cycle 0: Intake & Index
    Cycle 1: Parse & Normalize (ETL)
    Cycle 2: Graph Update (Neo4j)
    Cycle 3: Layout & Snapshot
    Cycle 4: UI Snapshot Hook

PREREQUISITES
-------------
1) Python 3.10+ installed and on PATH
2) (Optional) Neo4j 4.x/5.x running locally or remotely
3) (Optional) Node.js 18+ and npm installed if you want to extend the UI
4) (Optional) rclone for mounting Google Drive as a local folder

QUICK START
-----------
1) Ensure this bundle resides under your chosen root directory,
   for example:

       F:\\XXX\\

   with this file at:

       F:\\XXX\\bootstrap_mindseye2_graph_core_os.py

2) Run this script:

       cd F:\\XXX
       python bootstrap_mindseye2_graph_core_os.py

   This will create:

       F:\\XXX\\README.txt
       F:\\XXX\\MANIFEST.txt
       F:\\XXX\\config\\...
       F:\\XXX\\scripts\\...
       F:\\XXX\\graph_data\\...
       F:\\XXX\\neo4j\\...
       F:\\XXX\\ui\\...
       F:\\XXX\\logs_templates\\...
       F:\\XXX\\state\\...
       F:\\XXX\\logs\\...

3) Install Python dependencies:

       python -m pip install -r .\\config\\requirements.txt

4) (Optional) Configure Neo4j connection in:

       config\\graphkit.local.yml

5) Run a basic self-check by starting the autopilot:

       python .\\scripts\\autopilot.py

6) To start the continuous watch + pipeline:

       python .\\scripts\\autopilot.py

CONFIG FILES
------------
- config/graphkit.yml       : Base configuration (paths, sources)
- config/graphkit.local.yml : Local overrides (optional)
- config/autopilot.yml      : Autopilot watcher and scheduling options
- config/requirements.txt   : Python dependencies for all scripts

RUNTIME STATE
-------------
Created automatically under "state/" and "logs/":

- state/pending_files.json
- state/jobs.json
- state/last_etl_watermark.json
- state/last_graph_update.json
- state/last_layout_update.json

- logs/autopilot.log
- logs/etl_cycle.log
- logs/graph_update.log
- logs/layout_cycle.log
- logs/ui_snapshot.log

SAFETY
------
This bundle runs only on your local machine. It does NOT send data
over the network except to Neo4j (as configured), and optionally to
the UI if you run it in a browser on the same machine.

For Google Drive, use rclone mount so that the system only sees a
local folder path; it does not access Drive APIs directly.
"""
    write_file(root, "README.txt", readme)

    # 2) MANIFEST
    manifest = """\
MINDSEYE2_GRAPH_CORE_OS MANIFEST
================================

Bundle: MINDSEYE2_GRAPH_CORE_OS
Version: 1.0.0
Release Channel: internal
Build Layout Root: .

CONTENTS
--------

Root:
- README.txt
- MANIFEST.txt
- config/
- scripts/
- graph_data/
- neo4j/
- ui/
- logs_templates/
- state/         (runtime)
- logs/          (runtime)

config/:
- graphkit.yml
- graphkit.local.yml
- autopilot.yml
- requirements.txt

scripts/:
- autopilot.py
- run_graph_etl.py
- run_neo4j_import.py
- run_graph_layout_and_snapshot.py

graph_data/:
- example_nodes.csv
- example_edges.csv
- example_artifacts.csv

neo4j/:
- import.cypher (template for advanced Cypher workflows)

ui/:
- index.html       (basic graph snapshot viewer)
- cache/           (runtime snapshot store)

logs_templates/:
- etl_log_format.jsonl
- import_log_format.jsonl

RUNTIME-CREATED (NOT IN BUNDLE BY DEFAULT)
-----------------------------------------
- state/pending_files.json
- state/jobs.json
- state/last_etl_watermark.json
- state/last_graph_update.json
- state/last_layout_update.json

- logs/autopilot.log
- logs/etl_cycle.log
- logs/graph_update.log
- logs/layout_cycle.log
- logs/ui_snapshot.log
"""
    write_file(root, "MANIFEST.txt", manifest)

    # 3) config/graphkit.yml
    graphkit_yml = """\
version: 1.0.0

paths:
  bundle_root: "."
  roots:
    - "F:\\\\"
    - "D:\\\\"
    - "F:\\\\GDRIVE_MOUNT"
  graph_data_dir: "graph_data"
  state_dir: "state"
  logs_dir: "logs"

sources:
  include_extensions:
    - ".json"
    - ".csv"
    - ".graphml"
    - ".gexf"
    - ".html"
    - ".zip"
  include_patterns:
    - "mindseye"
    - "graph"
    - "nodes"
    - "edges"
  exclude_patterns:
    - "node_modules"
    - ".git"
    - "venv"
    - "env"
    - "__pycache__"
    - "graph_data"
    - "state"
    - "logs"

neo4j:
  bolt_uri: "bolt://localhost:7687"
  username: "neo4j"
  password: "neo4j"
  database: "neo4j"

graph_schema:
  node_id_field: "id"
  edge_source_field: "source"
  edge_target_field: "target"
  edge_type_field: "type"
  artifact_id_field: "artifact_id"

etl:
  batch_size: 100
  max_files_per_run: 1000
  write_format: "csv"
  dedupe_on:
    - "id"

layout:
  nucleus_selection:
    method: "centrality"
    centrality_metric: "pagerank"
    top_n: 32
  radial_layout:
    radius_min: 1.0
    radius_max: 10.0
  angle_distribution:
    strategy: "cluster"
    cluster_metric: "community"

ui:
  snapshot:
    output_file: "ui/cache/graph_snapshot.json"
    max_nodes: 5000
    max_edges: 5000
"""
    write_file(root, "config/graphkit.yml", graphkit_yml)

    # 4) config/graphkit.local.yml (local override template)
    graphkit_local_yml = """\
# Local overrides for graphkit.yml
# Example:
# paths:
#   roots:
#     - "F:\\\\"
#     - "D:\\\\"
# neo4j:
#   bolt_uri: "bolt://localhost:7687"
#   username: "neo4j"
#   password: "your_password_here"
"""
    write_file(root, "config/graphkit.local.yml", graphkit_local_yml)

    # 5) config/autopilot.yml
    autopilot_yml = """\
autopilot:
  enabled: true
  polling_interval_seconds: 30
  full_sweep_interval_minutes: 60
  max_parallel_jobs: 2
  log_level: "INFO"

watchers:
  use_filesystem_events: true
  roots:
    - "F:\\\\"
    - "D:\\\\"
    - "F:\\\\GDRIVE_MOUNT"
  include_extensions:
    - ".json"
    - ".csv"
    - ".graphml"
    - ".gexf"
    - ".html"
    - ".zip"

cycles:
  cycle_0_intake:
    enabled: true
  cycle_1_etl:
    enabled: true
  cycle_2_graph_update:
    enabled: true
  cycle_3_layout:
    enabled: true
  cycle_4_ui_snapshot:
    enabled: true

  trigger:
    min_new_files_for_etl: 1
    min_etl_updates_for_graph: 1
    min_graph_updates_for_layout: 1
    min_layout_updates_for_snapshot: 1
"""
    write_file(root, "config/autopilot.yml", autopilot_yml)

    # 6) config/requirements.txt
    requirements_txt = """\
pyyaml
watchdog
neo4j
"""
    write_file(root, "config/requirements.txt", requirements_txt)

    # 7) scripts/autopilot.py
    autopilot_py = r'''#!/usr/bin/env python3
"""
autopilot.py

MINDSEYE2_GRAPH_CORE_OS – Autonomous Orchestrator

Purpose:
    - Watch configured roots (F:\\, D:\\, and optional mounted cloud paths)
      for new/changed graph artifacts (.json, .csv, .graphml, .html, .zip).
    - Maintain a queue of jobs in state/jobs.json.
    - Chain multiple cycles:

        Cycle 0: INTAKE        – register files and metadata
        Cycle 1: ETL           – normalize artifacts -> graph_data/*
        Cycle 2: GRAPH_UPDATE  – sync Neo4j from graph_data/*
        Cycle 3: LAYOUT        – compute nucleus wheel layout and snapshot
        Cycle 4: UI_SNAPSHOT   – hook for additional UI work

    - Persist state to state/*.json so work survives restarts.
    - Write detailed logs in logs/autopilot.log.

Expected Layout (relative to bundle root):
    - config/graphkit.yml
    - config/graphkit.local.yml (optional)
    - config/autopilot.yml
    - scripts/run_graph_etl.py
    - scripts/run_neo4j_import.py
    - scripts/run_graph_layout_and_snapshot.py

Dependencies (install via config/requirements.txt):
    pyyaml
    watchdog
    neo4j (indirectly via import script, optional)

Usage:
    cd F:\\XXX
    python .\\scripts\\autopilot.py
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # handled later at runtime

try:
    from watchdog.observers import Observer  # type: ignore
    from watchdog.events import FileSystemEventHandler  # type: ignore
except ImportError:
    Observer = None
    FileSystemEventHandler = object  # type: ignore


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def safe_write_json(path: Path, data: Any) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(path)
    except Exception as e:
        print(f"[autopilot] Failed to write JSON {path}: {e}", file=sys.stderr)


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge_dict(result[k], v)
        else:
            result[k] = v
    return result


def match_extension(path: Path, include_exts: List[str]) -> bool:
    if not include_exts:
        return True
    suffix = path.suffix.lower()
    return suffix in [e.lower() for e in include_exts]


def match_patterns(path: Path, include_patterns: List[str], exclude_patterns: List[str]) -> bool:
    s = str(path).lower()
    if include_patterns:
        if not any(p.lower() in s for p in include_patterns):
            return False
    if exclude_patterns:
        if any(p.lower() in s for p in exclude_patterns):
            return False
    return True


class StateManager:
    def __init__(self, state_dir: Path, logger: logging.Logger) -> None:
        self.state_dir = state_dir
        self.logger = logger
        self.pending_path = self.state_dir / "pending_files.json"
        self.jobs_path = self.state_dir / "jobs.json"
        self.etl_watermark_path = self.state_dir / "last_etl_watermark.json"
        self.graph_update_path = self.state_dir / "last_graph_update.json"
        self.layout_update_path = self.state_dir / "last_layout_update.json"

        self.pending = safe_load_json(self.pending_path, {"files": []})
        self.jobs = safe_load_json(self.jobs_path, {"jobs": []})

    def _find_pending_index(self, file_path: str) -> Optional[int]:
        for idx, entry in enumerate(self.pending.get("files", [])):
            if entry.get("path") == file_path:
                return idx
        return None

    def upsert_pending_file(self, file_path: str, mtime: float, size: int) -> None:
        files = self.pending.setdefault("files", [])
        idx = self._find_pending_index(file_path)
        now = utc_now_iso()
        if idx is None:
            files.append({
                "path": file_path,
                "mtime": mtime,
                "size": size,
                "status": "new",
                "error": None,
                "first_seen": now,
                "last_attempt": None
            })
        else:
            entry = files[idx]
            entry["mtime"] = mtime
            entry["size"] = size
            if entry.get("status") in ("processed", "failed"):
                entry["status"] = "new"
            entry["last_attempt"] = None
            entry["error"] = None
        self.save_pending()

    def mark_pending_status(self, file_path: str, status: str, error: Optional[str] = None) -> None:
        idx = self._find_pending_index(file_path)
        if idx is None:
            return
        entry = self.pending["files"][idx]
        entry["status"] = status
        entry["last_attempt"] = utc_now_iso()
        entry["error"] = error
        self.save_pending()

    def save_pending(self) -> None:
        safe_write_json(self.pending_path, self.pending)

    def _next_job_id(self) -> str:
        max_id = 0
        for job in self.jobs.get("jobs", []):
            jid = job.get("id", "")
            if jid.startswith("job_"):
                try:
                    num = int(jid.split("_", 1)[1])
                    max_id = max(max_id, num)
                except ValueError:
                    continue
        return f"job_{max_id + 1:06d}"

    def add_job(self, job_type: str, payload: Dict[str, Any], unique: bool = False) -> str:
        jobs_list = self.jobs.setdefault("jobs", [])

        if unique:
            for job in jobs_list:
                if job.get("type") == job_type and job.get("status") in ("queued", "running"):
                    return job.get("id", "")

        job_id = self._next_job_id()
        now = utc_now_iso()
        job = {
            "id": job_id,
            "type": job_type,
            "payload": payload,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
            "last_error": None,
            "attempts": 0
        }
        jobs_list.append(job)
        self.save_jobs()
        self.logger.info("Added job %s type=%s payload=%s", job_id, job_type, payload)
        return job_id

    def save_jobs(self) -> None:
        safe_write_json(self.jobs_path, self.jobs)

    def get_next_job(self) -> Optional[Dict[str, Any]]:
        jobs_list = self.jobs.get("jobs", [])
        queued = [j for j in jobs_list if j.get("status") == "queued"]
        if not queued:
            return None
        queued.sort(key=lambda j: j.get("created_at", ""))
        return queued[0]

    def update_job(self, job_id: str, status: str, error: Optional[str] = None) -> None:
        jobs_list = self.jobs.get("jobs", [])
        for job in jobs_list:
            if job.get("id") == job_id:
                job["status"] = status
                job["updated_at"] = utc_now_iso()
                job["last_error"] = error
                if status == "running":
                    job["attempts"] = job.get("attempts", 0) + 1
                break
        self.save_jobs()

    def update_etl_watermark(self, snapshot_id: str) -> None:
        data = {
            "last_run_at": utc_now_iso(),
            "snapshot_id": snapshot_id
        }
        safe_write_json(self.etl_watermark_path, data)

    def get_etl_watermark(self) -> Dict[str, Any]:
        return safe_load_json(self.etl_watermark_path, {})

    def update_graph_update(self, snapshot_id: str) -> None:
        data = {
            "last_run_at": utc_now_iso(),
            "last_snapshot_id": snapshot_id
        }
        safe_write_json(self.graph_update_path, data)

    def get_graph_update(self) -> Dict[str, Any]:
        return safe_load_json(self.graph_update_path, {})

    def update_layout_update(self, layout_version: str, snapshot_id: Optional[str]) -> None:
        data = {
            "last_run_at": utc_now_iso(),
            "layout_version": layout_version,
            "last_graph_update_snapshot": snapshot_id
        }
        safe_write_json(self.layout_update_path, data)

    def get_layout_update(self) -> Dict[str, Any]:
        return safe_load_json(self.layout_update_path, {})


class GraphFileEventHandler(FileSystemEventHandler):
    def __init__(self, autopilot: "Autopilot") -> None:
        super().__init__()
        self.autopilot = autopilot

    def on_created(self, event):  # type: ignore[override]
        if not getattr(event, "is_directory", False):
            self.autopilot.handle_file_event(Path(event.src_path))

    def on_modified(self, event):  # type: ignore[override]
        if not getattr(event, "is_directory", False):
            self.autopilot.handle_file_event(Path(event.src_path))


class Autopilot:
    def __init__(self, bundle_root: Path, args: argparse.Namespace) -> None:
        self.bundle_root = bundle_root
        self.args = args

        self.config_dir = self.bundle_root / "config"
        self.state_dir = self.bundle_root / "state"
        self.logs_dir = self.bundle_root / "logs"

        self.logger = self._setup_logging()

        self.config = self._load_config()
        self.state = StateManager(self.state_dir, self.logger)

        self.observer: Optional[Observer] = None

        autopilot_cfg = self.config.get("autopilot", {})
        watchers_cfg = self.config.get("watchers", {})
        self.enabled = bool(autopilot_cfg.get("enabled", True))
        self.poll_interval = int(autopilot_cfg.get("polling_interval_seconds", 30))
        self.full_sweep_interval = int(autopilot_cfg.get("full_sweep_interval_minutes", 60)) * 60
        self.use_fs_events = bool(watchers_cfg.get("use_filesystem_events", True))
        self.include_extensions = watchers_cfg.get("include_extensions", [])
        self.last_sweep_ts = 0.0

        paths_cfg = self.config.get("paths", {})
        sources_cfg = self.config.get("sources", {})
        self.roots = watchers_cfg.get("roots") or paths_cfg.get("roots") or []
        self.include_patterns = sources_cfg.get("include_patterns", [])
        self.exclude_patterns = sources_cfg.get("exclude_patterns", [])

        self.logger.info("Autopilot initialized. enabled=%s roots=%s", self.enabled, self.roots)

    def _setup_logging(self) -> logging.Logger:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("MINDSEYE2_AUTOPILOT")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ"
        )
        fh = logging.FileHandler(self.logs_dir / "autopilot.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        return logger

    def _load_yaml_file(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        if yaml is None:
            print(f"[autopilot] PyYAML not installed; cannot read {path}", file=sys.stderr)
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[autopilot] Failed to read YAML {path}: {e}", file=sys.stderr)
            return {}

    def _load_config(self) -> Dict[str, Any]:
        graphkit_base = self._load_yaml_file(self.config_dir / "graphkit.yml")
        graphkit_local = self._load_yaml_file(self.config_dir / "graphkit.local.yml")
        autopilot_cfg = self._load_yaml_file(self.config_dir / "autopilot.yml")
        combined = deep_merge_dict(graphkit_base, graphkit_local)
        combined = deep_merge_dict(combined, autopilot_cfg)
        return combined

    def handle_file_event(self, path: Path) -> None:
        try:
            if not path.is_file():
                return
            if not match_extension(path, self.include_extensions):
                return
            if not match_patterns(path, self.include_patterns, self.exclude_patterns):
                return

            stat = path.stat()
            file_path = str(path)
            self.state.upsert_pending_file(file_path, stat.st_mtime, stat.st_size)
            self.state.add_job("INTAKE", {"path": file_path}, unique=False)
            self.logger.info("File event -> queued INTAKE for %s", file_path)
        except Exception as e:
            self.logger.exception("handle_file_event error for %s: %s", path, e)

    def full_sweep(self) -> None:
        self.logger.info("Starting full sweep.")
        for root in self.roots:
            root_path = Path(root)
            if not root_path.exists():
                self.logger.warning("Root does not exist: %s", root_path)
                continue
            for dirpath, _, filenames in os.walk(root_path):
                for name in filenames:
                    candidate = Path(dirpath) / name
                    if not match_extension(candidate, self.include_extensions):
                        continue
                    if not match_patterns(candidate, self.include_patterns, self.exclude_patterns):
                        continue
                    try:
                        stat = candidate.stat()
                    except FileNotFoundError:
                        continue
                    file_path = str(candidate)
                    idx = self.state._find_pending_index(file_path)
                    if idx is not None:
                        entry = self.state.pending["files"][idx]
                        if entry.get("mtime") == stat.st_mtime and entry.get("size") == stat.st_size:
                            continue
                    self.state.upsert_pending_file(file_path, stat.st_mtime, stat.st_size)
                    self.state.add_job("INTAKE", {"path": file_path}, unique=False)
        self.last_sweep_ts = time.time()
        self.logger.info("Full sweep completed.")

    def _run_subprocess(self, cmd: List[str], job_type: str) -> Tuple[bool, str]:
        import subprocess

        self.logger.info("%s: running command: %s", job_type, " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.bundle_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            output_lines: List[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip("\n")
                output_lines.append(line)
                self.logger.info("%s> %s", job_type, line)
            proc.wait()
            success = proc.returncode == 0
            msg = "ok" if success else f"exit code {proc.returncode}"
            return success, msg
        except FileNotFoundError as e:
            msg = f"command not found: {e}"
            self.logger.error("%s: %s", job_type, msg)
            return False, msg
        except Exception as e:
            msg = str(e)
            self.logger.exception("%s: error: %s", job_type, e)
            return False, msg

    def _cycle_intake(self, job_id: str, payload: Dict[str, Any]) -> None:
        file_path = payload.get("path")
        if not file_path:
            self.logger.warning("INTAKE job %s missing 'path' in payload.", job_id)
            return
        path = Path(file_path)
        if not path.exists():
            self.logger.warning("INTAKE job %s: path does not exist: %s", job_id, path)
            self.state.mark_pending_status(file_path, "failed", error="file not found")
            return
        self.state.mark_pending_status(file_path, "processed", None)
        self.state.add_job("ETL", {"trigger": "intake"}, unique=True)
        self.logger.info("INTAKE complete for %s; ETL job queued.", file_path)

    def _cycle_etl(self, job_id: str, payload: Dict[str, Any]) -> None:
        script_path = self.bundle_root / "scripts" / "run_graph_etl.py"
        if not script_path.exists():
            self.logger.warning("ETL script not found at %s; skipping ETL.", script_path)
            return

        success, msg = self._run_subprocess([sys.executable, str(script_path)], "ETL")
        if success:
            snapshot_id = f"etl_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            self.state.update_etl_watermark(snapshot_id)
            self.state.add_job("GRAPH_UPDATE", {"snapshot_id": snapshot_id}, unique=True)
            self.logger.info("ETL completed successfully; GRAPH_UPDATE queued.")
        else:
            self.logger.error("ETL failed: %s", msg)

    def _cycle_graph_update(self, job_id: str, payload: Dict[str, Any]) -> None:
        script_path = self.bundle_root / "scripts" / "run_neo4j_import.py"
        if not script_path.exists():
            self.logger.warning("Graph import script not found at %s; skipping GRAPH_UPDATE.", script_path)
            return

        success, msg = self._run_subprocess([sys.executable, str(script_path)], "GRAPH_UPDATE")
        if success:
            snapshot_id = payload.get("snapshot_id") or self.state.get_etl_watermark().get("snapshot_id") or "unknown"
            self.state.update_graph_update(snapshot_id)
            self.state.add_job("LAYOUT", {"graph_snapshot_id": snapshot_id}, unique=True)
            self.logger.info("GRAPH_UPDATE completed successfully; LAYOUT queued.")
        else:
            self.logger.error("GRAPH_UPDATE failed: %s", msg)

    def _cycle_layout(self, job_id: str, payload: Dict[str, Any]) -> None:
        script_path = self.bundle_root / "scripts" / "run_graph_layout_and_snapshot.py"
        if not script_path.exists():
            self.logger.warning("Layout+snapshot script not found at %s; skipping LAYOUT.", script_path)
            return

        success, msg = self._run_subprocess([sys.executable, str(script_path)], "LAYOUT")
        if success:
            graph_snapshot_id = payload.get("graph_snapshot_id") or self.state.get_graph_update().get("last_snapshot_id")
            self.state.update_layout_update("wheel_v1", graph_snapshot_id)
            self.state.add_job("UI_SNAPSHOT", {"graph_snapshot_id": graph_snapshot_id}, unique=True)
            self.logger.info("LAYOUT completed successfully; UI_SNAPSHOT queued.")
        else:
            self.logger.error("LAYOUT failed: %s", msg)

    def _cycle_ui_snapshot(self, job_id: str, payload: Dict[str, Any]) -> None:
        snapshot_file = self.bundle_root / "ui" / "cache" / "graph_snapshot.json"
        if snapshot_file.exists():
            self.logger.info("UI_SNAPSHOT cycle: snapshot file present at %s", snapshot_file)
        else:
            self.logger.warning("UI_SNAPSHOT cycle: expected snapshot file not found at %s", snapshot_file)

    def _start_watchers_if_available(self) -> None:
        if not self.use_fs_events:
            self.logger.info("File system events disabled; relying on full sweeps only.")
            return
        if Observer is None:
            self.logger.warning("watchdog not installed; cannot use file system events.")
            return

        event_handler = GraphFileEventHandler(self)
        observer = Observer()
        for root in self.roots:
            root_path = Path(root)
            if not root_path.exists():
                self.logger.warning("Watcher root does not exist: %s", root_path)
                continue
            observer.schedule(event_handler, str(root_path), recursive=True)
            self.logger.info("Watcher scheduled for %s", root_path)

        observer.daemon = True
        observer.start()
        self.observer = observer
        self.logger.info("File system watchers started.")

    def _stop_watchers(self) -> None:
        if self.observer is not None:
            self.observer.stop()
            self.observer.join(timeout=5.0)
            self.logger.info("File system watchers stopped.")

    def run(self) -> None:
        if not self.enabled:
            self.logger.warning("Autopilot disabled by config (autopilot.enabled=false). Exiting.")
            return

        if not self.roots:
            self.logger.warning("No roots configured. Exiting.")
            return

        self.logger.info("Starting autopilot main loop.")
        self._start_watchers_if_available()

        try:
            while True:
                job = self.state.get_next_job()
                now = time.time()

                if (not job) and self.full_sweep_interval > 0:
                    if now - self.last_sweep_ts > self.full_sweep_interval:
                        self.full_sweep()

                if not job:
                    time.sleep(self.poll_interval)
                    continue

                job_id = job["id"]
                job_type = job["type"]
                payload = job.get("payload", {})
                self.logger.info("Processing job %s type=%s", job_id, job_type)
                self.state.update_job(job_id, "running")

                try:
                    if job_type == "INTAKE":
                        self._cycle_intake(job_id, payload)
                    elif job_type == "ETL":
                        self._cycle_etl(job_id, payload)
                    elif job_type == "GRAPH_UPDATE":
                        self._cycle_graph_update(job_id, payload)
                    elif job_type == "LAYOUT":
                        self._cycle_layout(job_id, payload)
                    elif job_type == "UI_SNAPSHOT":
                        self._cycle_ui_snapshot(job_id, payload)
                    else:
                        self.logger.warning("Unknown job type %s", job_type)
                    self.state.update_job(job_id, "done")
                except Exception as e:
                    self.logger.exception("Job %s failed: %s", job_id, e)
                    self.state.update_job(job_id, "failed", error=str(e))

        except KeyboardInterrupt:
            self.logger.info("KeyboardInterrupt received, shutting down watchers.")
            self._stop_watchers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MINDSEYE2_GRAPH_CORE_OS Autopilot")
    parser.add_argument(
        "--bundle-root",
        type=str,
        default=None,
        help="Override bundle root directory (default: parent of this script)."
    )
    return parser.parse_args()


def main_autopilot() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    default_bundle_root = script_path.parent.parent
    bundle_root = Path(args.bundle_root).expanduser().resolve() if args.bundle_root else default_bundle_root
    autopilot = Autopilot(bundle_root, args)
    autopilot.run()


if __name__ == "__main__":
    main_autopilot()
'''
    write_file(root, "scripts/autopilot.py", autopilot_py)

    # 8) scripts/run_graph_etl.py
    run_graph_etl_py = r'''#!/usr/bin/env python3
"""
run_graph_etl.py

Basic ETL script for MINDSEYE2_GRAPH_CORE_OS.

Responsibilities:
    - Load config/graphkit.yml (+ local overrides).
    - Scan configured roots for graph-related files.
    - Build simple canonical CSVs in graph_data/:
        - nodes.csv
        - edges.csv
        - artifacts.csv

Generic behavior:
    - CSV files:
        - If columns include ["source", "target"], rows are treated as edges.
        - If columns include ["id"], rows are treated as nodes.
    - JSON files:
        - If a list of dicts with ["source", "target"], treat as edges.
        - If a list of dicts with ["id"], treat as nodes.
        - If a dict with "nodes" / "edges" keys, treat accordingly.

Artifacts are always recorded at file-level.
"""

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        print(f"[ETL] PyYAML not installed; cannot read {path}", file=sys.stderr)
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[ETL] Failed to read YAML {path}: {e}", file=sys.stderr)
        return {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def match_extension(path: Path, include_exts: List[str]) -> bool:
    if not include_exts:
        return True
    suffix = path.suffix.lower()
    return suffix in [e.lower() for e in include_exts]


def match_patterns(path: Path, include_patterns: List[str], exclude_patterns: List[str]) -> bool:
    s = str(path).lower()
    if include_patterns:
        if not any(p.lower() in s for p in include_patterns):
            return False
    if exclude_patterns:
        if any(p.lower() in s for p in exclude_patterns):
            return False
    return True


def scan_files(roots: List[str],
               include_exts: List[str],
               include_patterns: List[str],
               exclude_patterns: List[str],
               bundle_root: Path) -> List[Path]:
    results: List[Path] = []
    for root_str in roots:
        root_path = Path(root_str)
        if not root_path.exists():
            continue
        for dirpath, _, filenames in os.walk(root_path):
            dirpath_path = Path(dirpath)
            try:
                if bundle_root in dirpath_path.parents or dirpath_path == bundle_root:
                    continue
            except Exception:
                pass
            for name in filenames:
                candidate = dirpath_path / name
                if not match_extension(candidate, include_exts):
                    continue
                if not match_patterns(candidate, include_patterns, exclude_patterns):
                    continue
                results.append(candidate)
    return results


def load_csv(path: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        return rows, reader.fieldnames or []


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    script_path = Path(__file__).resolve()
    bundle_root = script_path.parent.parent
    config_dir = bundle_root / "config"

    base_cfg = load_yaml(config_dir / "graphkit.yml")
    local_cfg = load_yaml(config_dir / "graphkit.local.yml")
    cfg = deep_merge(base_cfg, local_cfg)

    paths_cfg = cfg.get("paths", {})
    sources_cfg = cfg.get("sources", {})
    roots = paths_cfg.get("roots") or []
    include_exts = sources_cfg.get("include_extensions") or []
    include_patterns = sources_cfg.get("include_patterns") or []
    exclude_patterns = sources_cfg.get("exclude_patterns") or []

    graph_data_dir = bundle_root / (paths_cfg.get("graph_data_dir") or "graph_data")
    graph_data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ETL] Bundle root: {bundle_root}")
    print(f"[ETL] Roots: {roots}")
    print(f"[ETL] Graph data dir: {graph_data_dir}")

    files = scan_files(roots, include_exts, include_patterns, exclude_patterns, bundle_root)
    print(f"[ETL] Found {len(files)} candidate files.")

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    artifacts: Dict[str, Dict[str, Any]] = {}

    for path in files:
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue

        artifact_id = str(path)
        artifacts[artifact_id] = {
            "artifact_id": artifact_id,
            "path": str(path),
            "ext": path.suffix.lower(),
            "size_bytes": stat.st_size,
        }

        if path.suffix.lower() == ".csv":
            try:
                rows, headers = load_csv(path)
            except Exception as e:
                print(f"[ETL] Failed to read CSV {path}: {e}", file=sys.stderr)
                continue

            header_set = {h.lower() for h in headers}
            is_edge_csv = "source" in header_set and "target" in header_set
            is_node_csv = "id" in header_set

            if is_edge_csv:
                for idx, row in enumerate(rows):
                    src = row.get("source")
                    tgt = row.get("target")
                    if not src or not tgt:
                        continue
                    edge_id = f"{artifact_id}::{idx}"
                    edges.append({
                        "id": edge_id,
                        "source": src,
                        "target": tgt,
                        "type": row.get("type") or "EDGE",
                        "artifact_id": artifact_id,
                    })

            if is_node_csv:
                for row in rows:
                    node_id = row.get("id")
                    if not node_id:
                        continue
                    if node_id not in nodes:
                        nodes[node_id] = {
                            "id": node_id,
                            "label": row.get("label") or node_id,
                            "type": row.get("type") or "NODE",
                            "artifact_id": artifact_id,
                        }

        elif path.suffix.lower() == ".json":
            try:
                data = load_json(path)
            except Exception as e:
                print(f"[ETL] Failed to read JSON {path}: {e}", file=sys.stderr)
                continue

            if isinstance(data, list) and data:
                sample = data[0]
                if isinstance(sample, dict):
                    keys = {k.lower() for k in sample.keys()}
                    if "source" in keys and "target" in keys:
                        for idx, row in enumerate(data):
                            src = row.get("source")
                            tgt = row.get("target")
                            if not src or not tgt:
                                continue
                            edge_id = f"{artifact_id}::json::{idx}"
                            edges.append({
                                "id": edge_id,
                                "source": src,
                                "target": tgt,
                                "type": row.get("type") or "EDGE",
                                "artifact_id": artifact_id,
                            })
                    elif "id" in keys:
                        for row in data:
                            node_id = row.get("id")
                            if not node_id:
                                continue
                            if node_id not in nodes:
                                nodes[node_id] = {
                                    "id": node_id,
                                    "label": row.get("label") or node_id,
                                    "type": row.get("type") or "NODE",
                                    "artifact_id": artifact_id,
                                }
            elif isinstance(data, dict):
                if "nodes" in data or "edges" in data:
                    for row in data.get("nodes", []):
                        if not isinstance(row, dict):
                            continue
                        node_id = row.get("id")
                        if not node_id:
                            continue
                        if node_id not in nodes:
                            nodes[node_id] = {
                                "id": node_id,
                                "label": row.get("label") or node_id,
                                "type": row.get("type") or "NODE",
                                "artifact_id": artifact_id,
                            }
                    for idx, row in enumerate(data.get("edges", [])):
                        if not isinstance(row, dict):
                            continue
                        src = row.get("source")
                        tgt = row.get("target")
                        if not src or not tgt:
                            continue
                        edge_id = row.get("id") or f"{artifact_id}::edges::{idx}"
                        edges.append({
                            "id": edge_id,
                            "source": src,
                            "target": tgt,
                            "type": row.get("type") or "EDGE",
                            "artifact_id": artifact_id,
                        })

    nodes_csv = graph_data_dir / "nodes.csv"
    edges_csv = graph_data_dir / "edges.csv"
    artifacts_csv = graph_data_dir / "artifacts.csv"

    with nodes_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "type", "artifact_id"])
        writer.writeheader()
        for node in nodes.values():
            writer.writerow(node)

    with edges_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "source", "target", "type", "artifact_id"])
        writer.writeheader()
        for edge in edges:
            writer.writerow(edge)

    with artifacts_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["artifact_id", "path", "ext", "size_bytes"])
        writer.writeheader()
        for artifact in artifacts.values():
            writer.writerow(artifact)

    print(f"[ETL] Wrote {len(nodes)} nodes -> {nodes_csv}")
    print(f"[ETL] Wrote {len(edges)} edges -> {edges_csv}")
    print(f"[ETL] Wrote {len(artifacts)} artifacts -> {artifacts_csv}")


if __name__ == "__main__":
    main()
'''
    write_file(root, "scripts/run_graph_etl.py", run_graph_etl_py)

    # 9) scripts/run_neo4j_import.py
    run_neo4j_import_py = r'''#!/usr/bin/env python3
"""
run_neo4j_import.py

Simple Neo4j import script for MINDSEYE2_GRAPH_CORE_OS.

Responsibilities:
    - Load config/graphkit.yml (+ local overrides) for Neo4j connection.
    - Read graph_data/nodes.csv and graph_data/edges.csv.
    - Connect to Neo4j and MERGE nodes/relationships.

Notes:
    - Assumes the Neo4j "neo4j" Python driver is installed.
    - Uses MERGE to keep the graph idempotent.
"""

import csv
import sys
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

try:
    from neo4j import GraphDatabase  # type: ignore
except ImportError:
    GraphDatabase = None


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if yaml is None:
        print(f"[IMPORT] PyYAML not installed; cannot read {path}", file=sys.stderr)
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[IMPORT] Failed to read YAML {path}: {e}", file=sys.stderr)
        return {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def main() -> None:
    script_path = Path(__file__).resolve()
    bundle_root = script_path.parent.parent
    config_dir = bundle_root / "config"
    graph_data_dir = bundle_root / "graph_data"

    base_cfg = load_yaml(config_dir / "graphkit.yml")
    local_cfg = load_yaml(config_dir / "graphkit.local.yml")
    cfg = deep_merge(base_cfg, local_cfg)
    neo_cfg = cfg.get("neo4j", {})

    bolt_uri = neo_cfg.get("bolt_uri")
    username = neo_cfg.get("username")
    password = neo_cfg.get("password")
    database = neo_cfg.get("database") or "neo4j"

    if GraphDatabase is None:
        print("[IMPORT] neo4j driver not installed; skipping import.", file=sys.stderr)
        sys.exit(0)

    if not bolt_uri or not username or not password:
        print("[IMPORT] Neo4j connection not fully configured; skipping import.", file=sys.stderr)
        sys.exit(0)

    nodes_csv = graph_data_dir / "nodes.csv"
    edges_csv = graph_data_dir / "edges.csv"

    if not nodes_csv.exists() or not edges_csv.exists():
        print("[IMPORT] nodes.csv or edges.csv missing; skipping import.", file=sys.stderr)
        sys.exit(0)

    driver = GraphDatabase.driver(bolt_uri, auth=(username, password))
    print(f"[IMPORT] Connecting to Neo4j at {bolt_uri}, database={database}")

    def import_nodes(tx, row):
        tx.run(
            """
            MERGE (n:Node {id: $id})
            SET n.label = $label,
                n.type = $type,
                n.artifact_id = $artifact_id
            """,
            **row,
        )

    def import_edges(tx, row):
        tx.run(
            """
            MATCH (s:Node {id: $source})
            MATCH (t:Node {id: $target})
            MERGE (s)-[r:REL {id: $id}]->(t)
            SET r.type = $type,
                r.artifact_id = $artifact_id
            """,
            **row,
        )

    with driver.session(database=database) as session:
        count_nodes = 0
        with nodes_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                session.execute_write(import_nodes, row)
                count_nodes += 1

        count_edges = 0
        with edges_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                session.execute_write(import_edges, row)
                count_edges += 1

    driver.close()
    print(f"[IMPORT] Imported {count_nodes} nodes and {count_edges} edges into Neo4j.")


if __name__ == "__main__":
    main()
'''
    write_file(root, "scripts/run_neo4j_import.py", run_neo4j_import_py)

    # 10) scripts/run_graph_layout_and_snapshot.py
    run_layout_py = r'''#!/usr/bin/env python3
"""
run_graph_layout_and_snapshot.py

Layout + snapshot script for MINDSEYE2_GRAPH_CORE_OS.

Responsibilities:
    - Read graph_data/nodes.csv and graph_data/edges.csv.
    - Compute a simple radial "nucleus wheel" layout in memory
      (without needing to query Neo4j).
    - Write an interactive-ready JSON snapshot to:

          ui/cache/graph_snapshot.json

The layout is:
    - Nodes grouped by "type".
    - Each type gets a radius band.
    - Nodes within a type are evenly spaced by angle.
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    script_path = Path(__file__).resolve()
    bundle_root = script_path.parent.parent
    graph_data_dir = bundle_root / "graph_data"
    ui_cache_dir = bundle_root / "ui" / "cache"
    ui_cache_dir.mkdir(parents=True, exist_ok=True)

    nodes_csv = graph_data_dir / "nodes.csv"
    edges_csv = graph_data_dir / "edges.csv"

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    if nodes_csv.exists():
        with nodes_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                nodes.append(dict(row))

    if edges_csv.exists():
        with edges_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                edges.append(dict(row))

    if not nodes:
        snapshot = {
            "generated_at": None,
            "node_count": 0,
            "edge_count": len(edges),
            "nodes": [],
            "edges": edges,
        }
        out_path = ui_cache_dir / "graph_snapshot.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2)
        print(f"[LAYOUT] No nodes found; wrote empty snapshot to {out_path}")
        return

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for node in nodes:
        node_type = node.get("type") or "NODE"
        groups.setdefault(node_type, []).append(node)

    type_list = sorted(groups.keys())
    radius_min = 1.0
    radius_step = 1.5

    positioned_nodes: List[Dict[str, Any]] = []
    for idx, node_type in enumerate(type_list):
        group = groups[node_type]
        radius = radius_min + radius_step * idx
        n = len(group)
        for i, node in enumerate(group):
            angle = 2.0 * math.pi * (i / max(n, 1))
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            node_copy = dict(node)
            node_copy["radius"] = radius
            node_copy["angle"] = angle
            node_copy["x"] = x
            node_copy["y"] = y
            positioned_nodes.append(node_copy)

    snapshot = {
        "generated_at": None,
        "node_count": len(positioned_nodes),
        "edge_count": len(edges),
        "nodes": positioned_nodes,
        "edges": edges,
    }

    out_path = ui_cache_dir / "graph_snapshot.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print(f"[LAYOUT] Wrote snapshot with {len(positioned_nodes)} nodes and {len(edges)} edges to {out_path}")


if __name__ == "__main__":
    main()
'''
    write_file(root, "scripts/run_graph_layout_and_snapshot.py", run_layout_py)

    # 11) graph_data example CSVs
    example_nodes = """\
id,label,type,artifact_id
A,A,AUTHORITY,example
B,B,AUTHORITY,example
C,C,NODE,example
"""
    example_edges = """\
id,source,target,type,artifact_id
e1,A,B,EDGE,example
e2,B,C,EDGE,example
"""
    example_artifacts = """\
artifact_id,path,ext,size_bytes
example,example_source.csv,.csv,0
"""
    write_file(root, "graph_data/example_nodes.csv", example_nodes)
    write_file(root, "graph_data/example_edges.csv", example_edges)
    write_file(root, "graph_data/example_artifacts.csv", example_artifacts)

    # 12) neo4j/import.cypher template
    import_cypher = """\
// Example Cypher for advanced Neo4j import workflows.
// The Python import script already uses MERGE via the neo4j driver.
// You can place additional Cypher here for manual batch imports or GDS operations.
"""
    write_file(root, "neo4j/import.cypher", import_cypher)

    # 13) ui/index.html – simple snapshot viewer
    index_html = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>MINDSEYE2 Graph Snapshot Viewer</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 1rem; background: #050510; color: #f2f2ff; }
    h1 { margin-bottom: 0.5rem; }
    #summary { margin-bottom: 1rem; font-weight: 600; }
    pre  { background: #0b0b1a; color: #e0e0ff; padding: 0.75rem; overflow: auto; border-radius: 8px; }
    a { color: #8fe4ff; }
  </style>
</head>
<body>
  <h1>MINDSEYE2 Graph Snapshot Viewer</h1>
  <div id="summary">Loading snapshot...</div>
  <pre id="preview"></pre>

  <script>
    async function loadSnapshot() {
      const summary = document.getElementById('summary');
      const preview = document.getElementById('preview');
      try {
        const res = await fetch('cache/graph_snapshot.json?_=' + Date.now());
        if (!res.ok) {
          summary.textContent = 'Snapshot not found yet. Run the ETL + layout cycles (autopilot).';
          return;
        }
        const data = await res.json();
        const nodes = data.nodes || [];
        const edges = data.edges || [];
        const nodeCount = data.node_count || nodes.length;
        const edgeCount = data.edge_count || edges.length;

        summary.textContent =
          'Nodes: ' + nodeCount +
          ' | Edges: ' + edgeCount +
          ' | Sample below (first 10 of each).';

        const sample = {
          nodes_sample: nodes.slice(0, 10),
          edges_sample: edges.slice(0, 10)
        };
        preview.textContent = JSON.stringify(sample, null, 2);
      } catch (err) {
        summary.textContent = 'Error loading snapshot: ' + err;
      }
    }
    loadSnapshot();
  </script>
</body>
</html>
"""
    write_file(root, "ui/index.html", index_html)

    # 14) logs_templates
    etl_log_template = """\
# ETL log format (JSON Lines)
# Each line can represent an event, e.g.:
# {"level": "INFO", "event": "file_processed", "path": "F:/...", "nodes": 10, "edges": 5}
"""
    import_log_template = """\
# Import log format (JSON Lines)
# Each line can represent an event, e.g.:
# {"level": "INFO", "event": "neo4j_import", "nodes": 100, "edges": 200}
"""
    write_file(root, "logs_templates/etl_log_format.jsonl", etl_log_template)
    write_file(root, "logs_templates/import_log_format.jsonl", import_log_template)

    print("\nBootstrap complete.")
    print("Next steps:")
    print(f"  1) python -m pip install -r {root / 'config' / 'requirements.txt'}")
    print(f"  2) python {root / 'scripts' / 'autopilot.py'}")


if __name__ == "__main__":
    main()
