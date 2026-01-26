from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

VERSION = "2.0.0"


@dataclass(frozen=True)
class Config:
    sources: list[Path]
    dest_root: Path
    move_mode: bool
    dedupe: bool
    dedupe_action: str
    skip_reparse: bool
    compute_sha256: bool
    exclude_roots: list[Path]
    self_test: bool
    self_test_runs: int
    self_test_only: bool
    catalog_db: Path | None
    mermaid_path: Path | None
    erd_path: Path | None
    esd_path: Path | None
    forensic_path: Path | None
    stratus_path: Path | None
    graph_output_dir: Path | None
    logs_dir: Path


@dataclass
class Counts:
    copied: int = 0
    moved: int = 0
    dup_skipped: int = 0
    dup_quarantined: int = 0
    failed: int = 0
    source_missing: int = 0


@dataclass
class LogPaths:
    csv_log: Path
    jsonl_log: Path
    index_csv: Path


@dataclass
class RunState:
    counts: Counts = field(default_factory=Counts)
    hash_index: dict[str, Path] = field(default_factory=dict)
    logs: LogPaths | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_atomic(path: Path, contents: str) -> None:
    ensure_dir(path.parent)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(contents, encoding="utf-8")
    temp_path.replace(path)


def normalize_root(path: Path) -> Path:
    resolved = path.resolve()
    return resolved


def build_exclude_list(roots: Iterable[Path]) -> list[Path]:
    return [normalize_root(root) for root in roots]


def is_excluded(path: Path, exclude_roots: Iterable[Path]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    for root in exclude_roots:
        try:
            if resolved == root or root in resolved.parents:
                return True
        except RuntimeError:
            if str(resolved).lower().startswith(str(root).lower()):
                return True
    return False


def get_ext_bucket(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return ext if ext else "_no_extension"


def get_string_md5(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def get_file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def unique_dest_name(path: Path, src_leaf: str) -> str:
    path_hash = get_string_md5(str(path))[:10]
    return f"{src_leaf}__{path_hash}__{path.name}"


def copy_or_move(src: Path, dst: Path, move_mode: bool) -> bool:
    try:
        ensure_dir(dst.parent)
        if move_mode:
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
        return True
    except OSError:
        return False


def get_files_safe(
    root: Path, skip_reparse: bool, exclude_roots: Iterable[Path]
) -> Iterator[Path]:
    if is_excluded(root, exclude_roots):
        return
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    entry_path = Path(entry.path)
                    if is_excluded(entry_path, exclude_roots):
                        continue
                    if skip_reparse and entry.is_symlink():
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(entry_path)
                    elif entry.is_file(follow_symlinks=False):
                        yield entry_path
        except (OSError, PermissionError):
            continue


def write_row_csv(path: Path, row: dict[str, object]) -> None:
    ensure_dir(path.parent)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_row_jsonl(path: Path, row: dict[str, object]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def init_catalog(path: Path) -> None:
    ensure_dir(path.parent)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS catalog (
                id INTEGER PRIMARY KEY,
                time_utc TEXT NOT NULL,
                action TEXT NOT NULL,
                source_root TEXT,
                source TEXT,
                dest TEXT,
                ext TEXT,
                bytes INTEGER,
                sha256 TEXT,
                note TEXT,
                error TEXT
            )
            """
        )
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")


def catalog_row(path: Path, row: dict[str, object]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            INSERT INTO catalog
            (time_utc, action, source_root, source, dest, ext, bytes, sha256, note, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("time_utc"),
                row.get("action"),
                row.get("source_root"),
                row.get("source"),
                row.get("dest"),
                row.get("ext"),
                row.get("bytes"),
                row.get("sha256"),
                row.get("note"),
                row.get("error"),
            ),
        )


def record_event(row: dict[str, object], state: RunState, catalog_db: Path | None) -> None:
    if state.logs is None:
        raise RuntimeError("Logs not initialized")
    write_row_csv(state.logs.csv_log, row)
    write_row_jsonl(state.logs.jsonl_log, row)
    if catalog_db:
        catalog_row(catalog_db, row)


def generate_mermaid(path: Path, config: Config, state: RunState) -> None:
    contents = [
        "flowchart TD",
        "  A[Harvest: Scan Files] --> B[Deduplicate + Hash]",
        "  B --> C[Bucket by Extension]",
        "  C --> D[Copy/Move]",
        "  D --> E[Index + Logs]",
        "  E --> F[Catalog (SQLite ACID)]",
        "  E --> G[Duplicate Quarantine]",
        f"  F --> H[Summary: copied={state.counts.copied} moved={state.counts.moved}]",
        f"  H --> I[Destination: {config.dest_root.as_posix()}]",
        f"  I --> J[Mode: {'MOVE' if config.move_mode else 'COPY'}]",
        f"  J --> K[Dedupe: {config.dedupe_action if config.dedupe else 'OFF'}]",
    ]
    write_atomic(path, "\n".join(contents) + "\n")


def generate_erd_blueprint(path: Path) -> None:
    contents = [
        "erDiagram",
        "  RUN ||--o{ CATALOG_EVENT : records",
        "  RUN ||--o{ EXT_BUCKET : aggregates",
        "  RUN ||--o{ RUN_LOG : logs",
        "  RUN {",
        "    string time_utc",
        "    string dest_root",
        "    string logs_dir",
        "    string mode_react",
        "    string mode_reflexion",
        "  }",
        "  RUN_LOG {",
        "    string time_utc",
        "    string csv_log",
        "    string jsonl_log",
        "    string index_csv",
        "  }",
        "  CATALOG_EVENT {",
        "    string time_utc",
        "    string action",
        "    string source_root",
        "    string source",
        "    string dest",
        "    string ext",
        "    int bytes",
        "    string sha256",
        "    string note",
        "    string error",
        "  }",
        "  EXT_BUCKET {",
        "    string ext_bucket",
        "    int file_count",
        "    int total_bytes",
        "    string folder",
        "  }",
    ]
    write_atomic(path, "\n".join(contents) + "\n")


def generate_esd_blueprint(path: Path) -> None:
    contents = [
        "flowchart TD",
        "  ROOT[ESD Blueprint: Unified Attachments + Evidence]",
        "  ROOT --> L0[Blueprint Stack]",
        "  ROOT --> ERD[Core ERD / Spine]",
        "  ROOT --> AUTH[Authority ERD]",
        "  ROOT --> AUTO[Automation ERD]",
        "  ROOT --> EVID[Evidence ERD]",
        "  ROOT --> INTEROP[Interop ERD]",
        "  L0 --> L1[L0 Storage Eligibility + Roots]",
        "  L1 --> L2[L1 Intake + Delta Harvest]",
        "  L2 --> L3[L2 EvidenceAtoms + Provenance]",
        "  L3 --> L4[L3 ChronoDB + QuoteDB]",
        "  L4 --> L5[L4 AuthoritySnapshot + Vehicles]",
        "  L5 --> L6[L5 Contracts (C2→C3)]",
        "  L6 --> L7[L6 Scoring + Gates]",
        "  L7 --> L8[L7 Actions + Automation]",
        "  L8 --> L9[L8 Packaging + Filing Packs]",
        "  ERD --> ERD1[Run → Artifact → Evidence/Authority]",
        "  ERD --> ERD2[Vehicle → Proof Obligation → Gate Result]",
        "  AUTH --> AUTH1[Proposition ⇄ AuthoritySnapshot]",
        "  AUTH --> AUTH2[VehiclePropositionLink]",
        "  AUTO --> AUTO1[Run → RunStep → RunEvent]",
        "  AUTO --> AUTO2[Schedule/Task → Agent]",
        "  EVID --> EVID1[EvidenceAtom ⇄ EvidenceItem]",
        "  EVID --> EVID2[Exhibit ⇄ EvidenceItemExhibitLink]",
        "  INTEROP --> INT1[BagitManifest + CloudEvent]",
        "  INTEROP --> INT2[OpenLineageRun + OTEL Span/Metric]",
        "  INTEROP --> INT3[Provenance Entity/Relation]",
        "  INTEROP --> INT4[SLSA Provenance]",
        "  INTEROP --> INT5[Catalog Events + Log Index]",
        "  EVID --> EVID3[Duplicate Quarantine Store]",
        "  EVID --> EVID4[Extension Buckets]",
    ]
    write_atomic(path, "\n".join(contents) + "\n")


def generate_forensic_inventory(path: Path, state: RunState) -> None:
    inventory = {
        "generated_utc": utc_now_iso(),
        "counts": {
            "copied": state.counts.copied,
            "moved": state.counts.moved,
            "dup_skipped": state.counts.dup_skipped,
            "dup_quarantined": state.counts.dup_quarantined,
            "failed": state.counts.failed,
            "source_missing": state.counts.source_missing,
        },
        "logs": {
            "csv": str(state.logs.csv_log) if state.logs else None,
            "jsonl": str(state.logs.jsonl_log) if state.logs else None,
            "index": str(state.logs.index_csv) if state.logs else None,
        },
    }
    write_atomic(path, json.dumps(inventory, indent=2) + "\n")


def generate_stratus_overview(path: Path) -> None:
    contents = [
        "flowchart TD",
        "  STRATUS[Stratus Overview]",
        "  STRATUS --> CELL[Per-Cell Tranche Inventory]",
        "  STRATUS --> LATTICE[Kosahedron Plane Lattice Torus Overlay]",
        "  CELL --> C1[Cell: L0 Storage]",
        "  CELL --> C2[Cell: L1 Intake]",
        "  CELL --> C3[Cell: L2 EvidenceAtoms]",
        "  CELL --> C4[Cell: L3 Chrono/Quote]",
        "  CELL --> C5[Cell: L4 Authority/Vehicles]",
        "  CELL --> C6[Cell: L5 Contracts]",
        "  CELL --> C7[Cell: L6 Scoring/Gates]",
        "  CELL --> C8[Cell: L7 Actions]",
        "  CELL --> C9[Cell: L8 Packaging]",
        "  LATTICE --> L1[Kosahedron Plane]",
        "  LATTICE --> L2[Lattice Mesh]",
        "  LATTICE --> L3[Torus Tranche Overlay]",
    ]
    write_atomic(path, "\n".join(contents) + "\n")


def load_index_rows(index_csv: Path | None) -> list[dict[str, str]]:
    if not index_csv or not index_csv.exists():
        return []
    with index_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def build_graph_payload(index_rows: list[dict[str, str]], config: Config) -> dict[str, object]:
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []

    root_id = "run_root"
    nodes.append(
        {
            "data": {
                "id": root_id,
                "label": f"Run: {config.dest_root}",
                "type": "run",
            }
        }
    )

    logs_id = "run_logs"
    nodes.append(
        {
            "data": {
                "id": logs_id,
                "label": "Logs + Index",
                "type": "log",
            }
        }
    )
    edges.append(
        {
            "data": {
                "id": f"edge::{root_id}::{logs_id}",
                "source": root_id,
                "target": logs_id,
                "label": "emits",
            }
        }
    )

    if config.dedupe and config.dedupe_action == "Quarantine":
        dup_id = "dup_quarantine"
        nodes.append(
            {
                "data": {
                    "id": dup_id,
                    "label": "Duplicates Quarantine",
                    "type": "duplicate",
                }
            }
        )
        edges.append(
            {
                "data": {
                    "id": f"edge::{root_id}::{dup_id}",
                    "source": root_id,
                    "target": dup_id,
                    "label": "quarantines",
                }
            }
        )

    if config.catalog_db:
        catalog_id = "catalog_db"
        nodes.append(
            {
                "data": {
                    "id": catalog_id,
                    "label": "SQLite Catalog",
                    "type": "catalog",
                }
            }
        )
        edges.append(
            {
                "data": {
                    "id": f"edge::{root_id}::{catalog_id}",
                    "source": root_id,
                    "target": catalog_id,
                    "label": "records",
                }
            }
        )

    for row in index_rows:
        ext_bucket = row.get("ext_bucket", "_unknown")
        node_id = f"ext::{ext_bucket}"
        nodes.append(
            {
                "data": {
                    "id": node_id,
                    "label": f"{ext_bucket} ({row.get('file_count', '0')})",
                    "type": "extension",
                    "file_count": row.get("file_count", "0"),
                    "total_bytes": row.get("total_bytes", "0"),
                    "folder": row.get("folder", ""),
                }
            }
        )
        edges.append(
            {
                "data": {
                    "id": f"edge::{root_id}::{node_id}",
                    "source": root_id,
                    "target": node_id,
                    "label": "contains",
                }
            }
        )

    return {"nodes": nodes, "edges": edges}


def write_graph_exports(output_dir: Path, payload: dict[str, object]) -> None:
    ensure_dir(output_dir)
    graph_json = json.dumps(payload, indent=2)
    write_atomic(output_dir / "graph.json", graph_json + "\n")

    nodes = payload.get("nodes", [])
    edges = payload.get("edges", [])
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Executive Suite Graph</title>
    <style>
      body {{ margin: 0; font-family: Arial, sans-serif; }}
      #cy {{ width: 100vw; height: 100vh; }}
      .panel {{
        position: absolute; top: 12px; left: 12px; z-index: 10;
        background: rgba(255,255,255,0.9); padding: 10px; border-radius: 6px;
      }}
    </style>
    <script src="https://unpkg.com/cytoscape@3.27.0/dist/cytoscape.min.js"></script>
  </head>
  <body>
    <div class="panel">
      <strong>Executive Suite Graph</strong>
      <div>Interactive nodes + zoom + pan</div>
    </div>
    <div id="cy"></div>
    <script>
      const elements = {{"nodes": {json.dumps(nodes)}, "edges": {json.dumps(edges)}}};
      const cy = cytoscape({{
        container: document.getElementById("cy"),
        elements,
        layout: {{ name: "cose" }},
        style: [
          {{ selector: "node", style: {{ "label": "data(label)", "background-color": "#4b8bbe", "color": "#111", "text-outline-width": 1, "text-outline-color": "#fff" }} }},
          {{ selector: "edge", style: {{ "width": 1.5, "line-color": "#999", "target-arrow-shape": "triangle", "target-arrow-color": "#999" }} }},
          {{ selector: "node[type = 'run']", style: {{ "background-color": "#6c5ce7", "shape": "round-rectangle" }} }},
          {{ selector: "node[type = 'log']", style: {{ "background-color": "#00b894", "shape": "diamond" }} }},
          {{ selector: "node[type = 'duplicate']", style: {{ "background-color": "#fdcb6e", "shape": "hexagon" }} }},
          {{ selector: "node[type = 'catalog']", style: {{ "background-color": "#e17055", "shape": "ellipse" }} }}
        ]
      }});
    </script>
  </body>
</html>
"""
    write_atomic(output_dir / "graph.html", html)

    node_rows = ["id,label,type,file_count,total_bytes,folder"]
    for node in nodes:
        data = node.get("data", {})
        node_rows.append(
            ",".join(
                [
                    str(data.get("id", "")),
                    str(data.get("label", "")),
                    str(data.get("type", "")),
                    str(data.get("file_count", "")),
                    str(data.get("total_bytes", "")),
                    str(data.get("folder", "")),
                ]
            )
        )
    write_atomic(output_dir / "neo4j_nodes.csv", "\n".join(node_rows) + "\n")

    edge_rows = ["id,source,target,label"]
    for edge in edges:
        data = edge.get("data", {})
        edge_rows.append(
            ",".join(
                [
                    str(data.get("id", "")),
                    str(data.get("source", "")),
                    str(data.get("target", "")),
                    str(data.get("label", "")),
                ]
            )
        )
    write_atomic(output_dir / "neo4j_edges.csv", "\n".join(edge_rows) + "\n")


def index_by_extension(dest_root: Path, logs_dir: Path) -> Path:
    ensure_dir(logs_dir)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    index_path = logs_dir / f"index_by_ext_{ts}.csv"
    rows: list[dict[str, object]] = []
    for child in dest_root.iterdir():
        if not child.is_dir():
            continue
        if child.name in {"__LOGS", "__DUPLICATES"}:
            continue
        total_bytes = 0
        file_count = 0
        for file_path in child.rglob("*"):
            if file_path.is_file():
                try:
                    total_bytes += file_path.stat().st_size
                except OSError:
                    continue
                file_count += 1
        rows.append(
            {
                "ext_bucket": child.name,
                "file_count": file_count,
                "total_bytes": total_bytes,
                "folder": str(child),
            }
        )
    rows.sort(key=lambda row: row["file_count"], reverse=True)
    if rows:
        with index_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        index_path.write_text("ext_bucket,file_count,total_bytes,folder\n", encoding="utf-8")
    return index_path


def prepare_logs(dest_root: Path, logs_dir: Path) -> LogPaths:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_log = logs_dir / f"organize_by_ext_{ts}.csv"
    jsonl_log = logs_dir / f"organize_by_ext_{ts}.jsonl"
    index_csv = logs_dir / f"index_by_ext_{ts}.csv"
    ensure_dir(logs_dir)
    return LogPaths(csv_log=csv_log, jsonl_log=jsonl_log, index_csv=index_csv)


def organize_by_extension(config: Config, state: RunState) -> None:
    logs = prepare_logs(config.dest_root, config.logs_dir)
    state.logs = logs
    if config.catalog_db:
        init_catalog(config.catalog_db)

    duplicates_root = config.dest_root / "__DUPLICATES"
    ensure_dir(config.dest_root)
    ensure_dir(config.logs_dir)
    if config.dedupe and config.dedupe_action == "Quarantine":
        ensure_dir(duplicates_root)

    for src_root in config.sources:
        if not src_root.exists():
            state.counts.source_missing += 1
            row = {
                "time_utc": utc_now_iso(),
                "action": "SOURCE_MISSING",
                "source_root": str(src_root),
                "source": None,
                "dest": None,
                "ext": None,
                "bytes": None,
                "sha256": None,
                "note": None,
                "error": None,
            }
            record_event(row, state, config.catalog_db)
            continue

        src_leaf = src_root.name or "source"
        for file_path in get_files_safe(src_root, config.skip_reparse, config.exclude_roots):
            ext_bucket = get_ext_bucket(file_path)
            ext_dir = config.dest_root / ext_bucket
            dest_name = unique_dest_name(file_path, src_leaf)
            dest_path = ext_dir / dest_name

            sha = None
            if config.compute_sha256 or config.dedupe:
                try:
                    sha = get_file_sha256(file_path)
                except OSError:
                    sha = None

            if config.dedupe and sha and sha in state.hash_index:
                if config.dedupe_action == "Skip":
                    state.counts.dup_skipped += 1
                    row = {
                        "time_utc": utc_now_iso(),
                        "action": "DUPLICATE_SKIPPED",
                        "source_root": str(src_root),
                        "source": str(file_path),
                        "dest": None,
                        "ext": ext_bucket,
                        "bytes": safe_stat_size(file_path),
                        "sha256": sha,
                        "note": f"primary={state.hash_index[sha]}",
                        "error": None,
                    }
                    record_event(row, state, config.catalog_db)
                    continue
                if config.dedupe_action == "Quarantine":
                    q_dir = duplicates_root / ext_bucket
                    q_path = q_dir / dest_name
                    ok = copy_or_move(file_path, q_path, config.move_mode)
                    if ok:
                        state.counts.dup_quarantined += 1
                        row = {
                            "time_utc": utc_now_iso(),
                            "action": "DUPLICATE_MOVED_TO_QUARANTINE"
                            if config.move_mode
                            else "DUPLICATE_COPIED_TO_QUARANTINE",
                            "source_root": str(src_root),
                            "source": str(file_path),
                            "dest": str(q_path),
                            "ext": ext_bucket,
                            "bytes": safe_stat_size(file_path),
                            "sha256": sha,
                            "note": f"primary={state.hash_index[sha]}",
                            "error": None,
                        }
                        record_event(row, state, config.catalog_db)
                        continue
                    state.counts.failed += 1
                    row = {
                        "time_utc": utc_now_iso(),
                        "action": "FAILED_DUP_QUARANTINE",
                        "source_root": str(src_root),
                        "source": str(file_path),
                        "dest": str(q_path),
                        "ext": ext_bucket,
                        "bytes": safe_stat_size(file_path),
                        "sha256": sha,
                        "note": f"primary={state.hash_index[sha]}",
                        "error": "copy_or_move failed",
                    }
                    record_event(row, state, config.catalog_db)
                    continue

            ok = copy_or_move(file_path, dest_path, config.move_mode)
            if ok:
                if config.move_mode:
                    state.counts.moved += 1
                else:
                    state.counts.copied += 1
                if config.dedupe and sha and sha not in state.hash_index:
                    state.hash_index[sha] = dest_path
                row = {
                    "time_utc": utc_now_iso(),
                    "action": "MOVED" if config.move_mode else "COPIED",
                    "source_root": str(src_root),
                    "source": str(file_path),
                    "dest": str(dest_path),
                    "ext": ext_bucket,
                    "bytes": safe_stat_size(file_path),
                    "sha256": sha,
                    "note": None,
                    "error": None,
                }
                record_event(row, state, config.catalog_db)
            else:
                state.counts.failed += 1
                row = {
                    "time_utc": utc_now_iso(),
                    "action": "FAILED",
                    "source_root": str(src_root),
                    "source": str(file_path),
                    "dest": str(dest_path),
                    "ext": ext_bucket,
                    "bytes": safe_stat_size(file_path),
                    "sha256": sha,
                    "note": None,
                    "error": "copy_or_move failed",
                }
                record_event(row, state, config.catalog_db)

    state.logs.index_csv = index_by_extension(config.dest_root, config.logs_dir)


def safe_stat_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def remove_empty_dirs(roots: Iterable[Path], exclude_roots: Iterable[Path]) -> None:
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            current = Path(dirpath)
            if is_excluded(current, exclude_roots):
                continue
            try:
                if not any(Path(dirpath).iterdir()):
                    current.rmdir()
            except OSError:
                continue


def run_self_test() -> None:
    base = Path(os.getenv("TEMP", "/tmp")) / f"OrgByExt_SelfTest_{int(time.time())}"
    src1 = base / "src1"
    src2 = base / "src2"
    dst = base / "dst"
    ensure_dir(src1 / "nested")
    ensure_dir(src2 / "nested")
    ensure_dir(dst)

    (src1 / "a.txt").write_text("same-content", encoding="utf-8")
    (src2 / "b.txt").write_text("same-content", encoding="utf-8")
    (src1 / "c.pdf").write_text("pdf-content", encoding="utf-8")
    (src2 / "d").write_text("noext", encoding="utf-8")
    (src1 / "nested" / "e.jpg").write_text("img", encoding="utf-8")
    (src2 / "nested" / "f.jpg").write_text("img2", encoding="utf-8")

    config = Config(
        sources=[src1, src2],
        dest_root=dst,
        move_mode=False,
        dedupe=True,
        dedupe_action="Quarantine",
        skip_reparse=True,
        compute_sha256=True,
        exclude_roots=[],
        self_test=False,
        self_test_runs=1,
        self_test_only=False,
        catalog_db=None,
        mermaid_path=None,
        erd_path=None,
        esd_path=None,
        forensic_path=None,
        stratus_path=None,
        graph_output_dir=None,
        logs_dir=dst / "__LOGS",
    )
    state = RunState()
    organize_by_extension(config, state)

    assert (dst / "txt").exists(), "SelfTest: missing txt folder"
    assert (dst / "pdf").exists(), "SelfTest: missing pdf folder"
    assert (dst / "jpg").exists(), "SelfTest: missing jpg folder"
    assert (dst / "_no_extension").exists(), "SelfTest: missing _no_extension folder"
    assert (dst / "__DUPLICATES" / "txt").exists(), "SelfTest: missing dup quarantine txt folder"
    primary_txt = list((dst / "txt").glob("*"))
    dup_txt = list((dst / "__DUPLICATES" / "txt").glob("*"))
    assert len(primary_txt) == 1, "SelfTest: expected 1 primary txt file"
    assert len(dup_txt) == 1, "SelfTest: expected 1 quarantined dup txt file"
    shutil.rmtree(base, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "High-signal extension organizer with ACID catalog, dedupe, quarantine, "
            "and mermaid lakehouse graph outputs."
        )
    )
    parser.add_argument("--sources", nargs="+", required=True, help="Source roots to scan")
    parser.add_argument("--dest-root", required=True, help="Destination root")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying")
    parser.add_argument(
        "--dedupe-action",
        choices=["Quarantine", "Skip", "KeepAll"],
        default="Quarantine",
        help="Duplicate handling mode",
    )
    parser.add_argument(
        "--no-dedupe", action="store_true", help="Disable SHA-256 deduplication"
    )
    parser.add_argument(
        "--skip-reparse",
        action="store_true",
        help="Skip symlinks and reparse points",
    )
    parser.add_argument(
        "--no-sha256", action="store_true", help="Disable SHA-256 computation"
    )
    parser.add_argument(
        "--exclude-root",
        action="append",
        default=[],
        help="Exclude root (repeatable)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run self-test before processing sources",
    )
    parser.add_argument(
        "--self-test-runs",
        type=int,
        default=1,
        help="Number of self-test runs (default: 1)",
    )
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        help="Exit after completing self-test runs",
    )
    parser.add_argument(
        "--catalog-db",
        default=None,
        help="Optional SQLite catalog path (ACID)",
    )
    parser.add_argument(
        "--mermaid",
        default=None,
        help="Optional path to write mermaid flowchart",
    )
    parser.add_argument(
        "--erd-blueprint",
        default=None,
        help="Optional path to write ERD-style blueprint",
    )
    parser.add_argument(
        "--esd-blueprint",
        default=None,
        help="Optional path to write unified ESD-style blueprint",
    )
    parser.add_argument(
        "--forensic-inventory",
        default=None,
        help="Optional path to write forensic inventory JSON",
    )
    parser.add_argument(
        "--stratus-overview",
        default=None,
        help="Optional path to write stratus overview mermaid",
    )
    parser.add_argument(
        "--graph-output-dir",
        default=None,
        help="Optional directory to write graph artifacts (json/html/csv)",
    )
    parser.add_argument(
        "--logs-dir",
        default="__LOGS",
        help="Directory name for logs (inside dest root)",
    )
    parser.add_argument("--version", action="version", version=VERSION)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    sources = [Path(item).expanduser() for item in args.sources]
    dest_root = Path(args.dest_root).expanduser()
    exclude_roots = [Path(item).expanduser() for item in args.exclude_root]
    logs_dir = dest_root / args.logs_dir
    return Config(
        sources=sources,
        dest_root=dest_root,
        move_mode=args.move,
        dedupe=not args.no_dedupe,
        dedupe_action=args.dedupe_action,
        skip_reparse=args.skip_reparse,
        compute_sha256=not args.no_sha256,
        exclude_roots=build_exclude_list(exclude_roots),
        self_test=args.self_test,
        self_test_runs=args.self_test_runs,
        self_test_only=args.self_test_only,
        catalog_db=Path(args.catalog_db).expanduser() if args.catalog_db else None,
        mermaid_path=Path(args.mermaid).expanduser() if args.mermaid else None,
        erd_path=Path(args.erd_blueprint).expanduser()
        if args.erd_blueprint
        else None,
        esd_path=Path(args.esd_blueprint).expanduser()
        if args.esd_blueprint
        else None,
        forensic_path=Path(args.forensic_inventory).expanduser()
        if args.forensic_inventory
        else None,
        stratus_path=Path(args.stratus_overview).expanduser()
        if args.stratus_overview
        else None,
        graph_output_dir=Path(args.graph_output_dir).expanduser()
        if args.graph_output_dir
        else None,
        logs_dir=logs_dir,
    )


def validate_config(config: Config) -> None:
    if config.dedupe_action not in {"Quarantine", "Skip", "KeepAll"}:
        raise ValueError("Invalid dedupe action")
    if config.self_test_runs < 1:
        raise ValueError("self_test_runs must be >= 1")


def main() -> int:
    args = parse_args()
    config = build_config(args)
    validate_config(config)
    if config.self_test:
        for _ in range(config.self_test_runs):
            run_self_test()
        if config.self_test_only:
            return 0

    state = RunState()
    organize_by_extension(config, state)

    if config.mermaid_path:
        generate_mermaid(config.mermaid_path, config, state)
    if config.erd_path:
        generate_erd_blueprint(config.erd_path)
    if config.esd_path:
        generate_esd_blueprint(config.esd_path)
    if config.forensic_path:
        generate_forensic_inventory(config.forensic_path, state)
    if config.stratus_path:
        generate_stratus_overview(config.stratus_path)
    if config.graph_output_dir:
        index_rows = load_index_rows(state.logs.index_csv if state.logs else None)
        graph_payload = build_graph_payload(index_rows, config)
        write_graph_exports(config.graph_output_dir, graph_payload)

    summary = {
        "time_utc": utc_now_iso(),
        "destination": str(config.dest_root),
        "index_csv": str(state.logs.index_csv) if state.logs else None,
        "csv_log": str(state.logs.csv_log) if state.logs else None,
        "jsonl_log": str(state.logs.jsonl_log) if state.logs else None,
        "counts": {
            "copied": state.counts.copied,
            "moved": state.counts.moved,
            "dup_skipped": state.counts.dup_skipped,
            "dup_quarantined": state.counts.dup_quarantined,
            "failed": state.counts.failed,
            "source_missing": state.counts.source_missing,
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
