#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERD_NEO4J_GLOSSARY_BINDER_SUITE
Executive, suite-grade binder builder for Neo4j / ERD graph artifact ecosystems.

What it does (high signal):
- Scans one or more filesystem roots (Windows drive letters supported) and optional ZIP packs.
- Selects "graph artifacts" (CSV, JSON, HTML, Cypher, DOT, GraphML, GEXF, etc).
- Builds an extreme glossary index binder (machine readable + human readable).
- Performs content deduplication (two-phase: quick signature then SHA-256 confirmation).
- Appends new Artifact nodes and edges into an existing Blueprint graph CSV pair
  (graph/neo4j_nodes.csv and graph/neo4j_edges.csv from your Blueprint pack).
- Creates an organized Vault copy of canonical artifacts (append-only; does not mutate originals).
- Emits: Run ledger (JSONL), provenance index, validation report, and import-ready CSVs.

Design goals:
- Not a hairball: "Artifact-first" minimal node set with optional Schema nodes.
- Dense, condensed, deterministic, append-only outputs.
- Safe defaults: no deletes; no in-place moves; explicit flags for any destructive action (not implemented).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

APP_ID = "ERD_NEO4J_GLOSSARY_BINDER_SUITE"
APP_VER = "v2026-01-27.0"

DEFAULT_WINDOWS_DRIVES = ["C:\\", "E:\\", "F:\\", "H:\\", "J:\\"]
DEFAULT_EXTS = {
    ".csv", ".tsv", ".json", ".jsonl", ".ndjson",
    ".graphml", ".gexf", ".gml", ".rdf", ".ttl", ".nt", ".nq", ".owl",
    ".dot", ".gv", ".cypher", ".cql", ".cy", ".txt",
    ".html", ".htm", ".svg",
    ".zip",
    ".png", ".jpg", ".jpeg", ".webp",
    ".yaml", ".yml", ".xml",
}

DEFAULT_EXCLUDE_DIR_NAMES = {
    "$recycle.bin", "system volume information", "windows", "program files",
    "program files (x86)", "programdata", "appdata", "recovery",
    ".git", ".svn", ".hg", "node_modules", "__pycache__", ".venv", "venv",
    "data", "transactions", "logs",
}

DEFAULT_EXCLUDE_PATH_PATTERNS = [
    r"\\Windows\\WinSxS\\",
    r"\\Windows\\Installer\\",
    r"\\Windows\\SoftwareDistribution\\",
    r"\\Program Files\\WindowsApps\\",
]

BUCKETS_MAX_15 = [
    ("csv", {".csv", ".tsv"}),
    ("json", {".json", ".jsonl", ".ndjson"}),
    ("html", {".html", ".htm"}),
    ("cypher", {".cypher", ".cql", ".cy", ".txt"}),
    ("graph", {".graphml", ".gexf", ".gml", ".dot", ".gv"}),
    ("rdf", {".rdf", ".ttl", ".nt", ".nq", ".owl"}),
    ("svg", {".svg"}),
    ("images", {".png", ".jpg", ".jpeg", ".webp"}),
    ("zip", {".zip"}),
    ("yaml", {".yaml", ".yml"}),
    ("xml", {".xml"}),
    ("other", set()),
]


def utc_now_iso() -> str:
    return (
        dt.datetime.now(dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def is_windows() -> bool:
    return os.name == "nt"


def norm_case(s: str) -> str:
    return s.lower() if is_windows() else s


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_small_bytes(path: Path, n: int) -> bytes:
    with path.open("rb") as f:
        return f.read(n)


def b32_16(b: bytes) -> str:
    import base64
    return base64.b32encode(b).decode("ascii").rstrip("=")


def stable_id_from_text(text: str) -> str:
    h = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=10).digest()
    return b32_16(h)


def sha256_hex_stream(stream, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    while True:
        b = stream.read(chunk)
        if not b:
            break
        h.update(b)
    return h.hexdigest()


def sha256_hex_file(path: Path, chunk: int = 1024 * 1024) -> str:
    with path.open("rb") as f:
        return sha256_hex_stream(f, chunk=chunk)


def quick_sig_file(path: Path, head: int = 65536, tail: int = 65536) -> Tuple[int, str, str]:
    size = path.stat().st_size
    with path.open("rb") as f:
        head_bytes = f.read(min(head, size))
        if size > tail:
            f.seek(max(0, size - tail))
        tail_bytes = f.read(min(tail, size))
    h1 = hashlib.sha1(head_bytes).hexdigest()
    h2 = hashlib.sha1(tail_bytes).hexdigest()
    return (size, h1, h2)


def quick_sig_zip_member(
    zf: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    head: int = 65536,
    tail: int = 65536,
) -> Tuple[int, str, str]:
    size = member.file_size
    with zf.open(member, "r") as f:
        head_bytes = f.read(min(head, size))
        if size <= head + tail:
            tail_bytes = b""
        else:
            discard = size - tail - len(head_bytes)
            while discard > 0:
                chunk = f.read(min(1024 * 1024, discard))
                if not chunk:
                    break
                discard -= len(chunk)
            tail_bytes = f.read(min(tail, size))
    h1 = hashlib.sha1(head_bytes).hexdigest()
    h2 = hashlib.sha1(tail_bytes).hexdigest()
    return (size, h1, h2)


def bucket_for_ext(ext: str) -> str:
    e = ext.lower()
    for name, exts in BUCKETS_MAX_15:
        if e in exts:
            return name
    return "other"


def should_exclude_dir(dirname: str) -> bool:
    return norm_case(dirname) in DEFAULT_EXCLUDE_DIR_NAMES


def should_exclude_path(path_str: str) -> bool:
    ps = path_str
    for pat in DEFAULT_EXCLUDE_PATH_PATTERNS:
        if re.search(pat, ps, flags=re.IGNORECASE):
            return True
    return False


def is_graph_artifact_ext(ext: str, allowed_exts: set) -> bool:
    return ext.lower() in allowed_exts


def sniff_csv_schema_bytes(data: bytes, max_rows: int = 200) -> Dict[str, object]:
    text = data.decode("utf-8", errors="replace")
    lines = [ln for ln in text.splitlines() if ln.strip()][:max_rows + 1]
    if not lines:
        return {"kind": "csv", "columns": [], "delimiter": None, "sample_rows": 0}
    sample = "\n".join(lines)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        delim = dialect.delimiter
    except Exception:
        delim = ","
    reader = csv.reader(io.StringIO(sample), delimiter=delim)
    rows = list(reader)
    cols = rows[0] if rows else []
    return {
        "kind": "csv",
        "columns": cols,
        "delimiter": delim,
        "sample_rows": max(0, len(rows) - 1),
    }


def sniff_json_schema_bytes(data: bytes, max_keys: int = 60) -> Dict[str, object]:
    text = data.decode("utf-8", errors="replace").strip()
    if not text:
        return {"kind": "json", "mode": "empty", "keys": []}
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 2:
        try:
            json.loads(text)
        except Exception:
            try:
                first = json.loads(lines[0])
                if isinstance(first, dict):
                    keys = sorted(list(first.keys()))[:max_keys]
                else:
                    keys = []
                return {
                    "kind": "json",
                    "mode": "jsonl",
                    "keys": keys,
                    "sample_lines": min(len(lines), 50),
                }
            except Exception:
                pass
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            keys = sorted(list(obj.keys()))[:max_keys]
            return {"kind": "json", "mode": "object", "keys": keys}
        if isinstance(obj, list):
            keys = []
            if obj and isinstance(obj[0], dict):
                keys = sorted(list(obj[0].keys()))[:max_keys]
            return {
                "kind": "json",
                "mode": "array",
                "keys": keys,
                "sample_len": min(len(obj), 5000),
            }
        return {"kind": "json", "mode": type(obj).__name__, "keys": []}
    except Exception:
        return {"kind": "json", "mode": "unparseable", "keys": []}


def sniff_html_meta_bytes(data: bytes, max_len: int = 200000) -> Dict[str, object]:
    text = data[:max_len].decode("utf-8", errors="replace")
    title = None
    m = re.search(r"<title>(.*?)</title>", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
    hints = []
    lower = text.lower()
    for token in ["reactflow", "cytoscape", "d3.", "vis-network", "sigma", "graphviz"]:
        if token in lower:
            hints.append(token)
    return {"kind": "html", "title": title, "hints": hints[:12]}


@dataclass(frozen=True)
class ArtifactRef:
    container: str
    path: str
    member: Optional[str]
    ext: str
    bytes: int
    mtime_utc: Optional[str]
    bucket: str
    basename: str


@dataclass
class ArtifactRecord:
    artifact_id: str
    stable_id: str
    ref: ArtifactRef
    kind: str
    schema: Dict[str, object]
    quick_sig: Tuple[int, str, str]
    sha256: Optional[str]
    canonical_id: Optional[str]
    notes: str


def build_artifact_id_from_quick(
    container: str,
    p: str,
    member: Optional[str],
    quick_sig: Tuple[int, str, str],
) -> str:
    size, h1, h2 = quick_sig
    base = f"{container}|{p}|{member or ''}|{size}|{h1}|{h2}"
    return "Q:" + stable_id_from_text(base)


def build_stable_node_id(artifact_id: str) -> str:
    return "ARTIFACT:" + artifact_id


def iter_filesystem_artifacts(
    root: Path,
    allowed_exts: set,
    follow_symlinks: bool = False,
) -> Iterator[ArtifactRef]:
    root = root.resolve()
    if not root.exists():
        return
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            if should_exclude_path(str(d)):
                continue
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=follow_symlinks):
                            if should_exclude_dir(entry.name):
                                continue
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=follow_symlinks):
                            p = Path(entry.path)
                            ext = p.suffix.lower()
                            if not is_graph_artifact_ext(ext, allowed_exts):
                                continue
                            if should_exclude_path(str(p)):
                                continue
                            st = p.stat()
                            mtime = (
                                dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc)
                                .replace(microsecond=0)
                                .isoformat()
                                .replace("+00:00", "Z")
                            )
                            yield ArtifactRef(
                                container="filesystem",
                                path=str(p),
                                member=None,
                                ext=ext,
                                bytes=st.st_size,
                                mtime_utc=mtime,
                                bucket=bucket_for_ext(ext),
                                basename=p.name,
                            )
                    except PermissionError:
                        continue
                    except OSError:
                        continue
        except PermissionError:
            continue
        except OSError:
            continue


def iter_zip_artifacts(zip_path: Path, allowed_exts: set) -> Iterator[ArtifactRef]:
    if not zip_path.exists():
        return
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                ext = (Path(name).suffix or "").lower()
                if not is_graph_artifact_ext(ext, allowed_exts):
                    continue
                try:
                    mdt = dt.datetime(*info.date_time, tzinfo=dt.timezone.utc).replace(
                        microsecond=0
                    )
                    mtime = mdt.isoformat().replace("+00:00", "Z")
                except Exception:
                    mtime = None
                yield ArtifactRef(
                    container="zip",
                    path=str(zip_path),
                    member=name,
                    ext=ext,
                    bytes=info.file_size,
                    mtime_utc=mtime,
                    bucket=bucket_for_ext(ext),
                    basename=Path(name).name,
                )
    except zipfile.BadZipFile:
        return


def build_artifact_record(
    ref: ArtifactRef,
    allowed_exts: set,
    zip_cache: Dict[str, zipfile.ZipFile],
) -> ArtifactRecord:
    schema: Dict[str, object] = {}
    kind = "other"
    sha256_full: Optional[str] = None

    if ref.container == "filesystem":
        p = Path(ref.path)
        quick = quick_sig_file(p)
        artifact_id = build_artifact_id_from_quick(ref.container, ref.path, None, quick)
        if ref.ext in {".csv", ".tsv"}:
            kind = "csv"
            data = read_small_bytes(p, 512 * 1024)
            schema = sniff_csv_schema_bytes(data)
        elif ref.ext in {".json", ".jsonl", ".ndjson"}:
            kind = "json"
            data = read_small_bytes(p, 512 * 1024)
            schema = sniff_json_schema_bytes(data)
        elif ref.ext in {".html", ".htm", ".svg"}:
            kind = "html"
            data = read_small_bytes(p, 512 * 1024)
            schema = sniff_html_meta_bytes(data)
        else:
            kind = ref.ext.lstrip(".") or "other"
            schema = {"kind": kind}
    else:
        zp = ref.path
        if zp not in zip_cache:
            zip_cache[zp] = zipfile.ZipFile(zp, "r")
        zf = zip_cache[zp]
        info = zf.getinfo(ref.member or "")
        quick = quick_sig_zip_member(zf, info)
        artifact_id = build_artifact_id_from_quick(ref.container, ref.path, ref.member, quick)
        try:
            with zf.open(info, "r") as f:
                data = f.read(512 * 1024)
        except Exception:
            data = b""
        if ref.ext in {".csv", ".tsv"}:
            kind = "csv"
            schema = sniff_csv_schema_bytes(data)
        elif ref.ext in {".json", ".jsonl", ".ndjson"}:
            kind = "json"
            schema = sniff_json_schema_bytes(data)
        elif ref.ext in {".html", ".htm", ".svg"}:
            kind = "html"
            schema = sniff_html_meta_bytes(data)
        else:
            kind = ref.ext.lstrip(".") or "other"
            schema = {"kind": kind}

    return ArtifactRecord(
        artifact_id=artifact_id,
        stable_id=build_stable_node_id(artifact_id),
        ref=ref,
        kind=kind,
        schema=schema,
        quick_sig=quick,
        sha256=sha256_full,
        canonical_id=None,
        notes="",
    )


def compute_sha256_for_record(
    rec: ArtifactRecord,
    zip_cache: Dict[str, zipfile.ZipFile],
) -> str:
    if rec.sha256:
        return rec.sha256
    if rec.ref.container == "filesystem":
        rec.sha256 = sha256_hex_file(Path(rec.ref.path))
        return rec.sha256
    zp = rec.ref.path
    if zp not in zip_cache:
        zip_cache[zp] = zipfile.ZipFile(zp, "r")
    zf = zip_cache[zp]
    info = zf.getinfo(rec.ref.member or "")
    with zf.open(info, "r") as f:
        rec.sha256 = sha256_hex_stream(f)
    return rec.sha256


def deduplicate(
    records: List[ArtifactRecord],
    zip_cache: Dict[str, zipfile.ZipFile],
    compute_full_hash: bool = True,
) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    buckets: Dict[Tuple[int, str, str], List[ArtifactRecord]] = {}
    for r in records:
        buckets.setdefault(r.quick_sig, []).append(r)

    duplicates_by_canonical: Dict[str, List[str]] = {}
    canonical_of: Dict[str, str] = {}

    for group in buckets.values():
        if len(group) == 1:
            canonical_of[group[0].artifact_id] = group[0].artifact_id
            continue

        if compute_full_hash:
            by_hash: Dict[str, List[ArtifactRecord]] = {}
            for r in group:
                h = compute_sha256_for_record(r, zip_cache)
                by_hash.setdefault(h, []).append(r)
            for g2 in by_hash.values():
                if len(g2) == 1:
                    canonical_of[g2[0].artifact_id] = g2[0].artifact_id
                    continue

                def canon_key(x: ArtifactRecord) -> Tuple[str, str, str]:
                    return (
                        x.ref.container,
                        norm_case(x.ref.path),
                        norm_case(x.ref.member or ""),
                    )

                g2_sorted = sorted(g2, key=canon_key)
                canon = g2_sorted[0]
                canonical_of[canon.artifact_id] = canon.artifact_id
                dups = [x.artifact_id for x in g2_sorted[1:]]
                for d in dups:
                    canonical_of[d] = canon.artifact_id
                duplicates_by_canonical.setdefault(canon.artifact_id, []).extend(dups)
        else:
            def canon_key(x: ArtifactRecord) -> Tuple[str, str, str]:
                return (
                    x.ref.container,
                    norm_case(x.ref.path),
                    norm_case(x.ref.member or ""),
                )

            group_sorted = sorted(group, key=canon_key)
            canon = group_sorted[0]
            canonical_of[canon.artifact_id] = canon.artifact_id
            dups = [x.artifact_id for x in group_sorted[1:]]
            for d in dups:
                canonical_of[d] = canon.artifact_id
            duplicates_by_canonical.setdefault(canon.artifact_id, []).extend(dups)

    for r in records:
        canon = canonical_of.get(r.artifact_id, r.artifact_id)
        r.canonical_id = canon
        if canon != r.artifact_id:
            r.notes = "duplicate"
    return duplicates_by_canonical, canonical_of


NODES_HEADER = [
    "stable_id",
    "labels",
    "torus_x",
    "torus_y",
    "name",
    "version",
    "source_pdf",
    "plane_id",
    "table_id",
    "category",
    "desc",
    "field_name",
    "field_type",
    "notes",
    "missing_reason",
    "path",
    "ext",
    "bytes",
]
EDGES_HEADER = ["src", "dst", "type", "reason", "via_field", "to_field"]


def write_csv(path: Path, header: List[str], rows: Iterable[List[str]]) -> None:
    safe_mkdir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def read_existing_nodes_edges(blueprint_dir: Path) -> Tuple[List[List[str]], List[List[str]]]:
    nodes_path = blueprint_dir / "graph" / "neo4j_nodes.csv"
    edges_path = blueprint_dir / "graph" / "neo4j_edges.csv"
    nodes: List[List[str]] = []
    edges: List[List[str]] = []
    if nodes_path.exists():
        with nodes_path.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.reader(f)
            _ = next(rdr, None)
            for row in rdr:
                nodes.append(row)
    if edges_path.exists():
        with edges_path.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.reader(f)
            _ = next(rdr, None)
            for row in rdr:
                edges.append(row)
    return nodes, edges


def extract_blueprint_zip(zip_path: Path, out_dir: Path) -> Path:
    safe_mkdir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir


def merge_nodes_edges(
    existing_nodes: List[List[str]],
    existing_edges: List[List[str]],
    append_nodes: List[List[str]],
    append_edges: List[List[str]],
) -> Tuple[List[List[str]], List[List[str]]]:
    existing_ids = set()
    for row in existing_nodes:
        if row:
            existing_ids.add(row[0])
    merged_nodes = list(existing_nodes)
    for row in append_nodes:
        if not row:
            continue
        sid = row[0]
        if sid in existing_ids:
            continue
        existing_ids.add(sid)
        merged_nodes.append(row)
    existing_e = set()
    for r in existing_edges:
        existing_e.add(tuple(r))
    merged_edges = list(existing_edges)
    for r in append_edges:
        t = tuple(r)
        if t in existing_e:
            continue
        existing_e.add(t)
        merged_edges.append(r)
    return merged_nodes, merged_edges


def record_to_compact_json(rec: ArtifactRecord) -> Dict[str, object]:
    ref = rec.ref
    return {
        "id": rec.artifact_id,
        "sid": rec.stable_id,
        "c": ref.container[0],
        "p": ref.path,
        "m": ref.member,
        "e": ref.ext,
        "b": ref.bytes,
        "t": ref.mtime_utc,
        "bk": ref.bucket,
        "k": rec.kind,
        "sh": rec.schema,
        "canon": rec.canonical_id,
        "h": rec.sha256,
        "n": rec.notes,
    }


def write_jsonl(path: Path, objs: Iterable[Dict[str, object]]) -> int:
    safe_mkdir(path.parent)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for o in objs:
            f.write(
                json.dumps(o, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            )
            f.write("\n")
            n += 1
    return n


def write_index_csv(path: Path, records: List[ArtifactRecord]) -> None:
    header = [
        "artifact_id",
        "canonical_id",
        "container",
        "path",
        "member",
        "ext",
        "bytes",
        "mtime_utc",
        "bucket",
        "kind",
        "sha256",
        "schema_keys",
        "schema_cols",
    ]
    rows: List[List[str]] = []
    for r in records:
        schema_keys = ",".join(sorted([k for k in r.schema.keys() if isinstance(k, str)])[:30])
        schema_cols = ""
        if r.kind == "csv":
            cols = r.schema.get("columns") or []
            schema_cols = "|".join([str(c) for c in cols][:200])
        rows.append(
            [
                r.artifact_id,
                r.canonical_id or r.artifact_id,
                r.ref.container,
                r.ref.path,
                r.ref.member or "",
                r.ref.ext,
                str(r.ref.bytes),
                r.ref.mtime_utc or "",
                r.ref.bucket,
                r.kind,
                r.sha256 or "",
                schema_keys,
                schema_cols,
            ]
        )
    write_csv(path, header, rows)


def write_dedup_map_csv(path: Path, duplicates_by_canonical: Dict[str, List[str]]) -> None:
    header = ["canonical_artifact_id", "duplicate_artifact_id"]
    rows = []
    for canon, dups in sorted(duplicates_by_canonical.items()):
        for d in sorted(dups):
            rows.append([canon, d])
    write_csv(path, header, rows)


def build_neo4j_append_rows(
    model_id: str,
    records: List[ArtifactRecord],
    duplicates_by_canonical: Dict[str, List[str]],
) -> Tuple[List[List[str]], List[List[str]]]:
    nodes: List[List[str]] = []
    edges: List[List[str]] = []

    for r in records:
        if (r.canonical_id or r.artifact_id) != r.artifact_id:
            continue
        labels = "Artifact;GraphArtifact"
        desc = ""
        if r.kind == "csv":
            cols = r.schema.get("columns") or []
            desc = f"CSV cols={len(cols)} delim={r.schema.get('delimiter')}"
        elif r.kind == "json":
            keys = r.schema.get("keys") or []
            desc = f"JSON mode={r.schema.get('mode')} keys={len(keys)}"
        elif r.kind == "html":
            title = r.schema.get("title") or ""
            hints = r.schema.get("hints") or []
            desc = f"HTML title={title} hints={','.join(hints)}"
        else:
            desc = f"{r.kind}"

        nodes.append(
            [
                r.stable_id,
                labels,
                "",
                "",
                r.ref.basename,
                APP_VER,
                "",
                "",
                "",
                r.ref.bucket,
                desc,
                "",
                "",
                r.notes or "",
                "",
                (r.ref.path + (("::" + r.ref.member) if r.ref.member else "")),
                r.ref.ext,
                str(r.ref.bytes),
            ]
        )
        edges.append([model_id, r.stable_id, "HAS_ARTIFACT", "scan_append", "", ""])

    for canon, dups in duplicates_by_canonical.items():
        canon_sid = build_stable_node_id(canon)
        for d in dups:
            dup_sid = build_stable_node_id(d)
            edges.append([dup_sid, canon_sid, "DUPLICATE_OF", "sha256", "", ""])
    return nodes, edges


def write_markdown_binder(
    path: Path,
    model_id: str,
    records: List[ArtifactRecord],
    duplicates_by_canonical: Dict[str, List[str]],
    stats: Dict[str, object],
    sources: Dict[str, object],
) -> None:
    safe_mkdir(path.parent)
    lines: List[str] = []
    lines.append("# ERD Graph Artifact Glossary Binder")
    lines.append(f"- app_id: {APP_ID}")
    lines.append(f"- app_ver: {APP_VER}")
    lines.append(f"- model_id: {model_id}")
    lines.append(f"- generated_utc: {utc_now_iso()}")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- artifacts_scanned: {stats.get('artifacts_scanned')}")
    lines.append(f"- canonical_artifacts: {stats.get('canonical_artifacts')}")
    lines.append(f"- duplicates_total: {stats.get('duplicates_total')}")
    lines.append(
        f"- by_bucket: {json.dumps(stats.get('by_bucket'), ensure_ascii=False, separators=(',',':'))}"
    )
    lines.append("")
    lines.append("## Sources")
    lines.append("```json")
    lines.append(json.dumps(sources, ensure_ascii=False, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Canonical Artifacts (compact)")
    canon = [r for r in records if (r.canonical_id or r.artifact_id) == r.artifact_id]
    canon_sorted = sorted(
        canon,
        key=lambda r: (
            r.ref.bucket,
            norm_case(r.ref.basename),
            norm_case(r.ref.path),
            norm_case(r.ref.member or ""),
        ),
    )
    for r in canon_sorted:
        loc = r.ref.path + (("::" + r.ref.member) if r.ref.member else "")
        if r.kind == "csv":
            cols = r.schema.get("columns") or []
            schema_note = f"csv_cols={len(cols)}"
        elif r.kind == "json":
            keys = r.schema.get("keys") or []
            schema_note = f"json_mode={r.schema.get('mode')} keys={len(keys)}"
        elif r.kind == "html":
            schema_note = f"html_title={r.schema.get('title') or ''}"
        else:
            schema_note = r.kind
        lines.append(
            f"- {r.artifact_id} | {r.ref.bucket} | {r.ref.ext} | {r.ref.bytes}B | {schema_note} | {loc}"
        )
    lines.append("")
    lines.append("## Dedup (canonical -> duplicates count)")
    for c, dups in sorted(duplicates_by_canonical.items()):
        lines.append(f"- {c} -> {len(dups)}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def vault_copy(
    records: List[ArtifactRecord],
    vault_dir: Path,
    zip_cache: Dict[str, zipfile.ZipFile],
) -> Dict[str, object]:
    safe_mkdir(vault_dir)
    copied = 0
    skipped = 0
    errors = 0
    for r in records:
        if (r.canonical_id or r.artifact_id) != r.artifact_id:
            continue
        bucket = r.ref.bucket
        out_bucket = vault_dir / bucket
        safe_mkdir(out_bucket)
        sid_tail = r.artifact_id[-10:]
        base = Path(r.ref.basename).stem
        ext = r.ref.ext
        out_name = f"{base}__{sid_tail}{ext}"
        out_path = out_bucket / out_name
        if out_path.exists() and out_path.stat().st_size == r.ref.bytes:
            skipped += 1
            continue
        try:
            if r.ref.container == "filesystem":
                shutil.copy2(r.ref.path, out_path)
            else:
                zp = r.ref.path
                if zp not in zip_cache:
                    zip_cache[zp] = zipfile.ZipFile(zp, "r")
                zf = zip_cache[zp]
                with zf.open(r.ref.member or "", "r") as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
            copied += 1
        except Exception:
            errors += 1
    return {"copied": copied, "skipped": skipped, "errors": errors}


def validate_outputs(out_dir: Path, required_relpaths: List[str]) -> Tuple[bool, List[str]]:
    missing = []
    for rp in required_relpaths:
        p = out_dir / rp
        if not p.exists():
            missing.append(rp)
        else:
            if p.is_file() and p.stat().st_size == 0:
                missing.append(rp + " (zero bytes)")
    return (len(missing) == 0), missing


def run(args: argparse.Namespace) -> int:
    start = time.time()
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%SZ")
    out_root = Path(args.out_root).resolve()
    run_dir = out_root / f"RUN_{APP_ID}_{run_id}"
    safe_mkdir(run_dir)

    ledger_path = run_dir / "run_ledger.jsonl"
    prov_path = run_dir / "provenance_index.json"

    def ledger(event: str, payload: Dict[str, object]) -> None:
        rec = {
            "ts": utc_now_iso(),
            "app_id": APP_ID,
            "app_ver": APP_VER,
            "event": event,
            "payload": payload,
        }
        with ledger_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
            f.write("\n")

    ledger("run_start", {"out_root": str(out_root), "run_dir": str(run_dir)})

    allowed_exts = set([e.strip().lower() for e in args.exts.split(",") if e.strip()])
    if not allowed_exts:
        allowed_exts = set(DEFAULT_EXTS)

    roots: List[Path] = []
    if args.roots:
        for r in args.roots:
            roots.append(Path(r))
    else:
        if is_windows():
            roots = [Path(r) for r in DEFAULT_WINDOWS_DRIVES]
        else:
            roots = [Path(".")]

    seed_zips: List[Path] = []
    for zp in args.seed_zips:
        seed_zips.append(Path(zp))

    blueprint_zip = Path(args.blueprint_zip).resolve() if args.blueprint_zip else None
    blueprint_dir: Optional[Path] = None
    if blueprint_zip and blueprint_zip.exists():
        blueprint_dir = run_dir / "blueprint_seed"
        extract_blueprint_zip(blueprint_zip, blueprint_dir)
        ledger(
            "blueprint_zip_extracted",
            {"blueprint_zip": str(blueprint_zip), "blueprint_dir": str(blueprint_dir)},
        )

    zip_cache: Dict[str, zipfile.ZipFile] = {}
    refs: List[ArtifactRef] = []
    for root in roots:
        for ref in iter_filesystem_artifacts(root, allowed_exts, follow_symlinks=args.follow_symlinks):
            refs.append(ref)
    for zp in seed_zips:
        for ref in iter_zip_artifacts(zp, allowed_exts):
            refs.append(ref)

    ledger("scan_done", {"refs": len(refs)})

    records: List[ArtifactRecord] = []

    def build_one(ref: ArtifactRef) -> ArtifactRecord:
        return build_artifact_record(ref, allowed_exts, zip_cache)

    with cf.ThreadPoolExecutor(max_workers=max(4, args.workers)) as ex:
        for rec in ex.map(build_one, refs, chunksize=128):
            records.append(rec)

    ledger("records_built", {"records": len(records)})

    dups_by_canon, canon_of = deduplicate(
        records,
        zip_cache,
        compute_full_hash=not args.dedup_quick_only,
    )
    ledger(
        "dedup_done",
        {
            "canon": len(set(canon_of.values())),
            "dups_total": sum(len(v) for v in dups_by_canon.values()),
        },
    )

    by_bucket: Dict[str, int] = {}
    canon_records = [r for r in records if (r.canonical_id or r.artifact_id) == r.artifact_id]
    for r in canon_records:
        by_bucket[r.ref.bucket] = by_bucket.get(r.ref.bucket, 0) + 1
    stats = {
        "artifacts_scanned": len(records),
        "canonical_artifacts": len(canon_records),
        "duplicates_total": sum(len(v) for v in dups_by_canon.values()),
        "by_bucket": dict(sorted(by_bucket.items(), key=lambda kv: (-kv[1], kv[0]))),
    }

    sources = {
        "roots": [str(p) for p in roots],
        "seed_zips": [str(p) for p in seed_zips],
        "blueprint_zip": str(blueprint_zip) if blueprint_zip else None,
    }

    binder_dir = run_dir / "binder"
    safe_mkdir(binder_dir)
    index_compact = binder_dir / "INDEX.compact.jsonl"
    index_verbose = binder_dir / "INDEX.verbose.jsonl"
    index_csv = binder_dir / "INDEX.csv"
    dedup_csv = binder_dir / "DEDUP.csv"
    binder_md = binder_dir / "BINDER.md"

    if args.hash_canon_sha256:
        for r in canon_records:
            try:
                compute_sha256_for_record(r, zip_cache)
            except Exception:
                pass

    write_jsonl(index_compact, (record_to_compact_json(r) for r in records))

    def record_to_verbose(rec: ArtifactRecord) -> Dict[str, object]:
        ref = rec.ref
        return {
            "artifact_id": rec.artifact_id,
            "stable_id": rec.stable_id,
            "canonical_id": rec.canonical_id,
            "container": ref.container,
            "path": ref.path,
            "member": ref.member,
            "ext": ref.ext,
            "bytes": ref.bytes,
            "mtime_utc": ref.mtime_utc,
            "bucket": ref.bucket,
            "kind": rec.kind,
            "schema": rec.schema,
            "quick_sig": {
                "size": rec.quick_sig[0],
                "head_sha1": rec.quick_sig[1],
                "tail_sha1": rec.quick_sig[2],
            },
            "sha256": rec.sha256,
            "notes": rec.notes,
        }

    write_jsonl(index_verbose, (record_to_verbose(r) for r in records))
    write_index_csv(index_csv, records)
    write_dedup_map_csv(dedup_csv, dups_by_canon)
    model_id = args.model_id or f"MODEL:{APP_ID}.{APP_VER}"
    write_markdown_binder(binder_md, model_id, records, dups_by_canon, stats, sources)
    ledger("binder_written", {"binder_dir": str(binder_dir)})

    vault_dir = run_dir / "vault"
    vault_stats = {"disabled": True}
    if not args.no_vault:
        vault_stats = vault_copy(records, vault_dir, zip_cache)
    ledger("vault_done", vault_stats)

    append_nodes, append_edges = build_neo4j_append_rows(model_id, records, dups_by_canon)
    neo4j_dir = run_dir / "neo4j_append"
    safe_mkdir(neo4j_dir)
    append_nodes_csv = neo4j_dir / "neo4j_nodes_append.csv"
    append_edges_csv = neo4j_dir / "neo4j_edges_append.csv"
    write_csv(append_nodes_csv, NODES_HEADER, append_nodes)
    write_csv(append_edges_csv, EDGES_HEADER, append_edges)
    ledger("neo4j_append_written", {"nodes": len(append_nodes), "edges": len(append_edges)})

    merged_dir = None
    if blueprint_dir:
        existing_nodes, existing_edges = read_existing_nodes_edges(blueprint_dir)
        merged_nodes, merged_edges = merge_nodes_edges(
            existing_nodes,
            existing_edges,
            append_nodes,
            append_edges,
        )
        merged_dir = run_dir / "blueprint_merged"
        safe_mkdir(merged_dir / "graph")
        write_csv(merged_dir / "graph" / "neo4j_nodes.csv", NODES_HEADER, merged_nodes)
        write_csv(merged_dir / "graph" / "neo4j_edges.csv", EDGES_HEADER, merged_edges)
        ledger(
            "blueprint_merged_written",
            {"merged_dir": str(merged_dir), "nodes": len(merged_nodes), "edges": len(merged_edges)},
        )

    prov = {
        "app_id": APP_ID,
        "app_ver": APP_VER,
        "generated_utc": utc_now_iso(),
        "run_dir": str(run_dir),
        "inputs": sources,
        "outputs": {
            "binder": {
                "INDEX.compact.jsonl": str(index_compact),
                "INDEX.verbose.jsonl": str(index_verbose),
                "INDEX.csv": str(index_csv),
                "DEDUP.csv": str(dedup_csv),
                "BINDER.md": str(binder_md),
            },
            "vault_dir": str(vault_dir) if not args.no_vault else None,
            "neo4j_append": {
                "neo4j_nodes_append.csv": str(append_nodes_csv),
                "neo4j_edges_append.csv": str(append_edges_csv),
            },
            "blueprint_merged": str(merged_dir) if merged_dir else None,
            "run_ledger.jsonl": str(ledger_path),
        },
        "stats": stats,
    }
    prov_path.write_text(
        json.dumps(prov, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )

    required = [
        "binder/INDEX.compact.jsonl",
        "binder/INDEX.verbose.jsonl",
        "binder/INDEX.csv",
        "binder/DEDUP.csv",
        "binder/BINDER.md",
        "neo4j_append/neo4j_nodes_append.csv",
        "neo4j_append/neo4j_edges_append.csv",
        "run_ledger.jsonl",
        "provenance_index.json",
    ]
    ok, missing = validate_outputs(run_dir, required)
    vreport = {
        "ok": ok,
        "missing_or_empty": missing,
        "stats": stats,
        "vault": vault_stats,
        "elapsed_sec": round(time.time() - start, 3),
    }
    (run_dir / "validation_report.json").write_text(
        json.dumps(vreport, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
        newline="\n",
    )
    ledger("validation_done", vreport)

    if args.package_zip:
        zip_out = out_root / f"{APP_ID}_{APP_VER}_{run_id}.zip"
        if zip_out.exists():
            zip_out.unlink()
        with zipfile.ZipFile(
            zip_out,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as zf:
            for p in run_dir.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(run_dir)
                    zf.write(p, arcname=str(Path(run_dir.name) / rel))
        ledger("package_zip_written", {"zip": str(zip_out), "bytes": zip_out.stat().st_size})
        print(str(zip_out))

    for zf in zip_cache.values():
        try:
            zf.close()
        except Exception:
            pass

    print(str(run_dir))
    return 0 if ok else 2


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=APP_ID,
        description=(
            "Scan drives and ZIPs for Neo4j/ERD graph artifacts, dedup, build binder, and append to blueprint."
        ),
    )
    ap.add_argument(
        "--out-root",
        default=".",
        help="Output root directory (default: current directory).",
    )
    ap.add_argument(
        "--roots",
        nargs="*",
        default=None,
        help="Filesystem roots to scan. Default: C:,E:,F:,H:,J: on Windows; current directory otherwise.",
    )
    ap.add_argument(
        "--seed-zips",
        nargs="*",
        default=[],
        help="Additional ZIP packs to scan as artifact sources.",
    )
    ap.add_argument(
        "--blueprint-zip",
        default=None,
        help=(
            "Existing blueprint pack ZIP containing graph/neo4j_nodes.csv and graph/neo4j_edges.csv."
        ),
    )
    ap.add_argument(
        "--model-id",
        default=None,
        help="Model node stable_id to attach artifacts to. Default is suite model id.",
    )
    ap.add_argument(
        "--exts",
        default=",".join(sorted(DEFAULT_EXTS)),
        help="Comma-separated extension allowlist.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Worker threads for record building.",
    )
    ap.add_argument(
        "--follow-symlinks",
        action="store_true",
        help="Follow symlinks during scan (default: false).",
    )
    ap.add_argument(
        "--dedup-quick-only",
        action="store_true",
        help="Skip SHA-256 confirm step; quick signature only.",
    )
    ap.add_argument(
        "--hash-canon-sha256",
        action="store_true",
        help="Compute SHA-256 for canonical artifacts (for stronger provenance).",
    )
    ap.add_argument(
        "--no-vault",
        action="store_true",
        help="Disable vault copy of canonical artifacts.",
    )
    ap.add_argument(
        "--package-zip",
        action="store_true",
        help="Create a ZIP of the run output under out-root.",
    )
    return ap


def main() -> int:
    ap = build_argparser()
    args = ap.parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
