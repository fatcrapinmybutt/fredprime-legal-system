"""Harvest ERD/graph glossary artifacts into an appended blueprint package."""

import argparse
import csv
import datetime as _dt
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

APP_NAME = "LITIGATIONOS_ERD_GLOSSARY_HARVESTER"
APP_VERSION = "v2026-01-26.1"
DEFAULT_ROOTS = ["C:", "E:", "H:", "J:", "F:"]
DEFAULT_OUT_ROOT = Path.cwd() / "OUT_ERD_GLOSSARY_HARVEST"
CANDIDATE_EXTS = {
    ".csv", ".tsv", ".psv",
    ".json", ".jsonl", ".ndjson",
    ".cypher", ".cql", ".cyp",
    ".graphml", ".gml", ".dot",
    ".html", ".htm", ".svg",
    ".ttl", ".rdf", ".nt",
    ".parquet", ".feather",
    ".md",
}
MEDIA_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf"}
DEFAULT_EXCLUDE_DIR_NAMES = {
    "$recycle.bin",
    "system volume information",
    "windows",
    "program files",
    "program files (x86)",
    "programdata",
    "recovery",
    "msocache",
    "appdata",
    "node_modules",
    ".git",
    ".svn",
    ".hg",
    ".idea",
    ".vscode",
    "__pycache__",
    ".cache",
    "temp",
    "tmp",
}
DEFAULT_EXCLUDE_PATH_FRAGMENTS = {
    r"\windows\winsxs\\",
    r"\windows\installer\\",
    r"\$recycle.bin\\",
    r"\system volume information\\",
}
QF_CHUNK = 64 * 1024
CSV_SAMPLE_MAX_BYTES = 5_000_000
JSON_SAMPLE_MAX_BYTES = 5_000_000
HTML_SAMPLE_MAX_BYTES = 2_000_000
DOT_SAMPLE_MAX_BYTES = 2_000_000
SVG_SAMPLE_MAX_BYTES = 2_000_000
MD_SAMPLE_MAX_BYTES = 2_000_000
NEO4J_DELIM = ","


def utc_now() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def norm_lower(s: str) -> str:
    return s.strip().lower()


def safe_relpath(p: Path) -> str:
    return str(p).replace("\\", "/")


def read_small_bytes(path: Path, max_bytes: int) -> bytes:
    with path.open("rb") as f:
        return f.read(max_bytes)


def blake2b_hex_stream(path: Path, chunk_size: int = 1024 * 1024, digest_size: int = 32) -> str:
    h = hashlib.blake2b(digest_size=digest_size)
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def quick_fingerprint(path: Path) -> str:
    """
    Fast fingerprint for dedup candidate grouping:
    size + hash(first 64KiB) + hash(last 64KiB).
    """
    st = path.stat()
    size = int(st.st_size)
    h1 = hashlib.blake2b(digest_size=16)
    h2 = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        head = f.read(QF_CHUNK)
        h1.update(head)
        if size > QF_CHUNK:
            try:
                f.seek(max(0, size - QF_CHUNK))
            except Exception:
                pass
            tail = f.read(QF_CHUNK)
            h2.update(tail)
        else:
            h2.update(head)
    return f"{size}:{h1.hexdigest()}:{h2.hexdigest()}"


def stable_artifact_id(content_hash: str) -> str:
    return f"ART_{content_hash[:20]}" if content_hash else "ART_NOHASH"


def stable_occurrence_id(origin_path: str, size: int, mtime_utc: str) -> str:
    # Deterministic per-path snapshot key, not content key.
    s = f"{origin_path}|{size}|{mtime_utc}"
    h = hashlib.blake2b(s.encode("utf-8", errors="ignore"), digest_size=16).hexdigest()
    return f"OCC_{h}"


def guess_delimiter(sample: str) -> str:
    candidates = [",", "\t", "|", ";"]
    scores = []
    rows = sample.splitlines()[:50]
    for d in candidates:
        counts = [len(r.split(d)) for r in rows if r.strip()]
        score = (sum(1 for c in counts if c > 1), sum(counts))
        scores.append((score, d))
    scores.sort(reverse=True)
    return scores[0][1] if scores else ","


def infer_scalar_type(v: str) -> str:
    s = v.strip()
    if s == "":
        return "empty"
    if re.fullmatch(r"[-+]?\d+", s):
        return "int"
    if re.fullmatch(r"[-+]?\d*\.\d+([eE][-+]?\d+)?", s) or re.fullmatch(r"[-+]?\d+[eE][-+]?\d+", s):
        return "float"
    if s.lower() in {"true", "false", "yes", "no"}:
        return "bool"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", s):
        return "date"
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        return "json"
    return "str"


def merge_type_counts(type_counts: Dict[str, int], t: str) -> None:
    type_counts[t] = type_counts.get(t, 0) + 1


def summarize_csv(path: Path) -> Dict[str, object]:
    st = path.stat()
    data = read_small_bytes(path, CSV_SAMPLE_MAX_BYTES) if st.st_size > CSV_SAMPLE_MAX_BYTES else path.read_bytes()
    truncated = st.st_size > CSV_SAMPLE_MAX_BYTES
    try:
        txt = data.decode("utf-8", errors="replace")
    except Exception:
        txt = data.decode("latin-1", errors="replace")
    delim = guess_delimiter(txt)
    reader = csv.reader(io.StringIO(txt), delimiter=delim)
    header: List[str] = []
    samples: List[List[str]] = []
    for i, row in enumerate(reader):
        if i == 0:
            header = row
            continue
        if i <= 50:
            samples.append(row)
        if i > 200:
            break
    ncols = len(header) if header else max((len(r) for r in samples), default=0)
    col_types: List[Dict[str, int]] = [{} for _ in range(ncols)]
    for row in samples:
        for j in range(ncols):
            v = row[j] if j < len(row) else ""
            merge_type_counts(col_types[j], infer_scalar_type(v))
    return {
        "kind": "csv",
        "path": safe_relpath(path),
        "bytes": int(st.st_size),
        "delimiter": "\\t" if delim == "\t" else delim,
        "header": header[:200],
        "ncols": ncols,
        "sample_rows": len(samples),
        "type_hints": col_types[:200],
        "truncated_bytes": bool(truncated),
    }


def summarize_json(path: Path) -> Dict[str, object]:
    st = path.stat()
    data = read_small_bytes(path, JSON_SAMPLE_MAX_BYTES) if st.st_size > JSON_SAMPLE_MAX_BYTES else path.read_bytes()
    truncated = st.st_size > JSON_SAMPLE_MAX_BYTES
    ext = path.suffix.lower()
    if ext in {".jsonl", ".ndjson"}:
        lines = data.splitlines()
        objs = []
        for ln in lines[:200]:
            ln = ln.strip()
            if not ln:
                continue
            try:
                objs.append(json.loads(ln.decode("utf-8", errors="replace")))
            except Exception:
                try:
                    objs.append(json.loads(ln.decode("latin-1", errors="replace")))
                except Exception:
                    continue
        key_counts: Dict[str, int] = {}
        for o in objs:
            if isinstance(o, dict):
                for k in o.keys():
                    key_counts[str(k)] = key_counts.get(str(k), 0) + 1
        top_keys = sorted(key_counts.items(), key=lambda x: (-x[1], x[0]))[:200]
        return {
            "kind": "jsonl",
            "path": safe_relpath(path),
            "bytes": int(st.st_size),
            "objects_sampled": len(objs),
            "top_keys": top_keys,
            "truncated_bytes": bool(truncated),
        }

    def try_load(b: bytes) -> object:
        try:
            return json.loads(b.decode("utf-8", errors="replace"))
        except Exception:
            return json.loads(b.decode("latin-1", errors="replace"))

    try:
        obj = try_load(data)
        if isinstance(obj, dict):
            return {
                "kind": "json",
                "path": safe_relpath(path),
                "bytes": int(st.st_size),
                "root_type": "object",
                "keys": list(map(str, obj.keys()))[:500],
                "truncated_bytes": bool(truncated),
            }
        if isinstance(obj, list):
            sample = obj[:50]
            key_counts: Dict[str, int] = {}
            for it in sample:
                if isinstance(it, dict):
                    for k in it.keys():
                        key_counts[str(k)] = key_counts.get(str(k), 0) + 1
            top_keys = sorted(key_counts.items(), key=lambda x: (-x[1], x[0]))[:200]
            return {
                "kind": "json",
                "path": safe_relpath(path),
                "bytes": int(st.st_size),
                "root_type": "array",
                "items_sampled": len(sample),
                "top_keys": top_keys,
                "truncated_bytes": bool(truncated),
            }
        return {
            "kind": "json",
            "path": safe_relpath(path),
            "bytes": int(st.st_size),
            "root_type": type(obj).__name__,
            "truncated_bytes": bool(truncated),
        }
    except Exception as e:
        return {
            "kind": "json",
            "path": safe_relpath(path),
            "bytes": int(st.st_size),
            "parse_error": str(e),
            "truncated_bytes": bool(truncated),
        }


def summarize_html(path: Path) -> Dict[str, object]:
    st = path.stat()
    data = read_small_bytes(path, HTML_SAMPLE_MAX_BYTES) if st.st_size > HTML_SAMPLE_MAX_BYTES else path.read_bytes()
    truncated = st.st_size > HTML_SAMPLE_MAX_BYTES
    txt = data.decode("utf-8", errors="replace")
    title = ""
    m = re.search(r"<title>(.*?)</title>", txt, flags=re.IGNORECASE | re.DOTALL)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()[:300]
    markers = {
        "visjs": bool(re.search(r"\bvis\.js\b|\bvis-network\b", txt, re.IGNORECASE)),
        "d3": bool(re.search(r"\bd3(\.min)?\.js\b", txt, re.IGNORECASE)),
        "cytoscape": bool(re.search(r"\bcytoscape(\.min)?\.js\b", txt, re.IGNORECASE)),
        "neo4j": bool(re.search(r"\bneo4j\b", txt, re.IGNORECASE)),
        "graph_keywords": bool(re.search(r"\b(graph|nodes|edges|relationship|cypher)\b", txt, re.IGNORECASE)),
    }
    return {
        "kind": "html",
        "path": safe_relpath(path),
        "bytes": int(st.st_size),
        "title": title,
        "markers": markers,
        "truncated_bytes": bool(truncated),
    }


def summarize_dot(path: Path) -> Dict[str, object]:
    st = path.stat()
    data = read_small_bytes(path, DOT_SAMPLE_MAX_BYTES) if st.st_size > DOT_SAMPLE_MAX_BYTES else path.read_bytes()
    truncated = st.st_size > DOT_SAMPLE_MAX_BYTES
    txt = data.decode("utf-8", errors="replace")
    node_like = len(re.findall(r"\b\[label=", txt))
    edge_like = len(re.findall(r"->", txt))
    return {
        "kind": "dot",
        "path": safe_relpath(path),
        "bytes": int(st.st_size),
        "approx_nodes": node_like,
        "approx_edges": edge_like,
        "truncated_bytes": bool(truncated),
    }


def summarize_svg(path: Path) -> Dict[str, object]:
    st = path.stat()
    data = read_small_bytes(path, SVG_SAMPLE_MAX_BYTES) if st.st_size > SVG_SAMPLE_MAX_BYTES else path.read_bytes()
    truncated = st.st_size > SVG_SAMPLE_MAX_BYTES
    txt = data.decode("utf-8", errors="replace")
    width = ""
    height = ""
    viewbox = ""
    m = re.search(r"<svg[^>]*>", txt, flags=re.IGNORECASE)
    if m:
        tag = m.group(0)
        mw = re.search(r'width="([^"]+)"', tag, flags=re.IGNORECASE)
        mh = re.search(r'height="([^"]+)"', tag, flags=re.IGNORECASE)
        mv = re.search(r'viewBox="([^"]+)"', tag, flags=re.IGNORECASE)
        width = mw.group(1)[:80] if mw else ""
        height = mh.group(1)[:80] if mh else ""
        viewbox = mv.group(1)[:120] if mv else ""
    return {
        "kind": "svg",
        "path": safe_relpath(path),
        "bytes": int(st.st_size),
        "width": width,
        "height": height,
        "viewBox": viewbox,
        "truncated_bytes": bool(truncated),
    }


def summarize_md(path: Path) -> Dict[str, object]:
    st = path.stat()
    data = read_small_bytes(path, MD_SAMPLE_MAX_BYTES) if st.st_size > MD_SAMPLE_MAX_BYTES else path.read_bytes()
    truncated = st.st_size > MD_SAMPLE_MAX_BYTES
    txt = data.decode("utf-8", errors="replace")
    heads = []
    for ln in txt.splitlines()[:400]:
        if ln.startswith("#"):
            heads.append(re.sub(r"\s+", " ", ln.strip())[:200])
    return {
        "kind": "md",
        "path": safe_relpath(path),
        "bytes": int(st.st_size),
        "headings_sample": heads[:100],
        "truncated_bytes": bool(truncated),
    }


def infer_schema_summary(path: Path) -> Optional[Dict[str, object]]:
    ext = path.suffix.lower()
    try:
        if ext in {".csv", ".tsv", ".psv"}:
            return summarize_csv(path)
        if ext in {".json", ".jsonl", ".ndjson"}:
            return summarize_json(path)
        if ext in {".html", ".htm"}:
            return summarize_html(path)
        if ext == ".dot":
            return summarize_dot(path)
        if ext == ".svg":
            return summarize_svg(path)
        if ext == ".md":
            return summarize_md(path)
        return None
    except Exception as e:
        return {"kind": "summary_error", "path": safe_relpath(path), "error": str(e)}


def classify_role(path: Path, schema_summary: Optional[Dict[str, object]]) -> str:
    ext = path.suffix.lower()
    name = path.name.lower()
    if ext in {".cypher", ".cql", ".cyp"}:
        return "cypher"
    if ext == ".dot":
        return "dot_model"
    if ext in {".graphml", ".gml"}:
        return "graph_serialization"
    if ext in {".csv", ".tsv", ".psv"}:
        if any(k in name for k in ["node", "nodes", "vertex", "vertices"]):
            return "tabular_nodes"
        if any(k in name for k in ["edge", "edges", "rel", "rels", "relationship", "relationships"]):
            return "tabular_relationships"
        if any(k in name for k in ["field", "fields", "schema", "dictionary", "dict", "erd"]):
            return "tabular_schema"
        return "tabular_generic"
    if ext in {".json", ".jsonl", ".ndjson"}:
        if any(k in name for k in ["schema", "erd", "dictionary", "index", "manifest", "map"]):
            return "json_schema_or_index"
        return "json_generic"
    if ext in {".html", ".htm"}:
        if schema_summary and isinstance(schema_summary.get("markers"), dict):
            mk = schema_summary["markers"]
            if mk.get("neo4j"):
                return "html_neo4j_related"
            if mk.get("graph_keywords"):
                return "html_graph_viewer"
        return "html_generic"
    if ext == ".svg":
        return "svg_diagram"
    if ext in {".ttl", ".rdf", ".nt"}:
        return "rdf_serialization"
    if ext in {".parquet", ".feather"}:
        return "columnar_table"
    if ext == ".md":
        return "markdown_doc"
    if ext in MEDIA_EXTS:
        return "media_asset"
    return "other"


def bucket_for_ext(ext: str) -> str:
    ext = ext.lower()
    if ext in {".csv", ".tsv", ".psv", ".parquet", ".feather"}:
        return "tables"
    if ext in {".json", ".jsonl", ".ndjson"}:
        return "json"
    if ext in {".html", ".htm", ".svg"}:
        return "viewers"
    if ext in {".dot", ".graphml", ".gml"}:
        return "models"
    if ext in {".cypher", ".cql", ".cyp"}:
        return "queries"
    if ext in {".ttl", ".rdf", ".nt"}:
        return "rdf"
    if ext == ".md":
        return "docs"
    if ext in MEDIA_EXTS:
        return "media"
    return "other"


def sanitize_name(name: str, max_len: int = 180) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:max_len]


def determine_blueprint_target(blueprint_root: Path, role: str, ext: str, content_hash: str, original_name: str) -> Path:
    """
    Deterministic canonicalization path inside blueprint root.
    Append-only: never mutates existing top-level blueprint structures.
    """
    date_bucket = _dt.datetime.utcnow().strftime("%Y%m%d")
    safe_name = sanitize_name(original_name)
    h2 = content_hash[:2] if content_hash else "NO"
    h10 = content_hash[:10] if content_hash else "NOHASH"
    return blueprint_root / "APPENDS" / date_bucket / bucket_for_ext(ext) / role / h2 / h10 / safe_name


def copy_or_link(src: Path, dst: Path, mode: str) -> str:
    ensure_dir(dst.parent)
    if dst.exists():
        return "SKIPPED_EXISTS"
    if mode == "hardlink":
        try:
            os.link(str(src), str(dst))
            return "HARDLINKED"
        except Exception:
            shutil.copy2(str(src), str(dst))
            return "COPIED_FALLBACK"
    if mode == "symlink":
        try:
            os.symlink(str(src), str(dst))
            return "SYMLINKED"
        except Exception:
            shutil.copy2(str(src), str(dst))
            return "COPIED_FALLBACK"
    shutil.copy2(str(src), str(dst))
    return "COPIED"


def unzip_to_dir(zip_path: Path, out_dir: Path) -> None:
    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            name = info.filename
            if not name or name.endswith("/") or name == "/":
                continue
            dest = out_dir / name
            dest_parent = dest.parent.resolve()
            if not str(dest_parent).startswith(str(out_dir.resolve())):
                continue
            ensure_dir(dest.parent)
            with z.open(info) as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    ensure_dir(zip_path.parent)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in src_dir.rglob("*"):
            if p.is_dir():
                continue
            rel = p.relative_to(src_dir)
            z.write(p, arcname=str(rel).replace("\\", "/"))


def is_drive_token(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]:", s.strip()))


def root_path_from_token(token: str) -> Tuple[str, Path]:
    """
    Returns (label, root_path).
    - For "C:" -> ("C:", Path("C:\\"))
    - For absolute/relative paths -> (normalized label, Path(token))
    """
    t = token.strip().rstrip("\\/")
    if is_drive_token(t):
        return (t.upper(), Path(t.upper() + "\\"))
    p = Path(t).expanduser()
    label = sanitize_name(str(p.resolve())) if p.exists() else sanitize_name(str(p))
    return (label, p)


@dataclass(frozen=True)
class ArtifactHit:
    path: Path
    root_label: str
    ext: str
    size: int
    mtime_utc: str


def should_exclude_dir(dirpath: Path, dirnames: List[str], exclude_dir_names: set, exclude_path_frags: set) -> None:
    """
    Mutates dirnames in-place to control os.walk descent (topdown=True).
    """
    keep = []
    for d in dirnames:
        if norm_lower(d) in exclude_dir_names:
            continue
        keep.append(d)
    dirnames[:] = keep
    dp = str(dirpath).lower().replace("/", "\\")
    for frag in exclude_path_frags:
        if frag.lower() in dp:
            dirnames[:] = []
            return


def iter_candidate_files(
    roots: Sequence[Tuple[str, Path]],
    include_media: bool,
    exclude_dir_names: set,
    exclude_path_frags: set,
    max_files: int,
    follow_symlinks: bool,
) -> Iterator[ArtifactHit]:
    count = 0
    for root_label, root_path in roots:
        if not root_path.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root_path, topdown=True, followlinks=follow_symlinks):
            dp = Path(dirpath)
            should_exclude_dir(dp, dirnames, exclude_dir_names, exclude_path_frags)
            for fn in filenames:
                p = dp / fn
                try:
                    ext = p.suffix.lower()
                    if ext in CANDIDATE_EXTS or (include_media and ext in MEDIA_EXTS):
                        st = p.stat()
                        yield ArtifactHit(
                            path=p,
                            root_label=root_label,
                            ext=ext,
                            size=int(st.st_size),
                            mtime_utc=_dt.datetime.utcfromtimestamp(st.st_mtime)
                            .replace(microsecond=0)
                            .isoformat()
                            + "Z",
                        )
                        count += 1
                        if max_files > 0 and count >= max_files:
                            return
                except (PermissionError, FileNotFoundError, OSError):
                    continue


class StateDB:
    """
    Persistent state store (SQLite).
    Purpose: skip reprocessing unchanged origin paths and track canonical artifacts.
    """

    def __init__(self, db_path: Path) -> None:
        ensure_dir(db_path.parent)
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init()

    def _init(self) -> None:
        self.conn.execute(
            """
        CREATE TABLE IF NOT EXISTS occurrences (
            origin_path TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL,
            mtime_utc TEXT NOT NULL,
            quick_fp TEXT,
            content_hash TEXT,
            artifact_id TEXT,
            last_seen_ts TEXT NOT NULL
        );
        """
        )
        self.conn.execute(
            """
        CREATE TABLE IF NOT EXISTS artifacts (
            content_hash TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL,
            ext TEXT,
            role TEXT,
            bucket TEXT,
            bytes INTEGER,
            blueprint_path TEXT,
            first_seen_ts TEXT NOT NULL,
            last_seen_ts TEXT NOT NULL
        );
        """
        )
        self.conn.execute(
            """
        CREATE TABLE IF NOT EXISTS blueprint_qf_cache (
            quick_fp TEXT PRIMARY KEY,
            paths_json TEXT NOT NULL,
            last_built_ts TEXT NOT NULL
        );
        """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.commit()
        self.conn.close()

    def get_occurrence(self, origin_path: str) -> Optional[Dict[str, object]]:
        cur = self.conn.execute(
            "SELECT origin_path,size_bytes,mtime_utc,quick_fp,content_hash,artifact_id,last_seen_ts "
            "FROM occurrences WHERE origin_path=?",
            (origin_path,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "origin_path": row[0],
            "size_bytes": int(row[1]),
            "mtime_utc": row[2],
            "quick_fp": row[3],
            "content_hash": row[4],
            "artifact_id": row[5],
            "last_seen_ts": row[6],
        }

    def upsert_occurrence(
        self,
        origin_path: str,
        size_bytes: int,
        mtime_utc: str,
        quick_fp: str,
        content_hash: str,
        artifact_id: str,
        last_seen_ts: str,
    ) -> None:
        self.conn.execute(
            """
        INSERT INTO occurrences(origin_path,size_bytes,mtime_utc,quick_fp,content_hash,artifact_id,last_seen_ts)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(origin_path) DO UPDATE SET
            size_bytes=excluded.size_bytes,
            mtime_utc=excluded.mtime_utc,
            quick_fp=excluded.quick_fp,
            content_hash=excluded.content_hash,
            artifact_id=excluded.artifact_id,
            last_seen_ts=excluded.last_seen_ts
        """,
            (origin_path, int(size_bytes), mtime_utc, quick_fp, content_hash, artifact_id, last_seen_ts),
        )
        self.conn.commit()

    def get_artifact(self, content_hash: str) -> Optional[Dict[str, object]]:
        cur = self.conn.execute(
            "SELECT content_hash,artifact_id,ext,role,bucket,bytes,blueprint_path,first_seen_ts,last_seen_ts "
            "FROM artifacts WHERE content_hash=?",
            (content_hash,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return {
            "content_hash": row[0],
            "artifact_id": row[1],
            "ext": row[2],
            "role": row[3],
            "bucket": row[4],
            "bytes": int(row[5]) if row[5] is not None else None,
            "blueprint_path": row[6],
            "first_seen_ts": row[7],
            "last_seen_ts": row[8],
        }

    def upsert_artifact(
        self,
        content_hash: str,
        artifact_id: str,
        ext: str,
        role: str,
        bucket: str,
        bytes_: int,
        blueprint_path: str,
        now_ts: str,
    ) -> None:
        existing = self.get_artifact(content_hash)
        if existing is None:
            self.conn.execute(
                """
            INSERT INTO artifacts(content_hash,artifact_id,ext,role,bucket,bytes,blueprint_path,first_seen_ts,last_seen_ts)
            VALUES(?,?,?,?,?,?,?,?,?)
            """,
                (content_hash, artifact_id, ext, role, bucket, int(bytes_), blueprint_path, now_ts, now_ts),
            )
        else:
            self.conn.execute(
                """
            UPDATE artifacts SET last_seen_ts=?, blueprint_path=COALESCE(NULLIF(?,''), blueprint_path)
            WHERE content_hash=?
            """,
                (now_ts, blueprint_path, content_hash),
            )
        self.conn.commit()


def append_jsonl(path: Path, obj: Dict[str, object]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False, sort_keys=True))
        f.write("\n")


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8", newline="\n")


def write_csv_rows(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", delimiter=",")
        w.writeheader()
        for r in rows:
            out = {}
            for k in fieldnames:
                v = r.get(k)
                if isinstance(v, (dict, list)):
                    out[k] = json.dumps(v, ensure_ascii=False)
                else:
                    out[k] = v
            w.writerow(out)


def build_neo4j_import_tables(run_dir: Path, artifacts: List[Dict[str, object]], occurrences: List[Dict[str, object]]) -> None:
    """
    Neo4j import-ready CSVs.
    Headers follow neo4j-admin conventions (:ID, :START_ID, :END_ID, :TYPE).
    Source: Neo4j Operations Manual neo4j-admin database import.
    """
    neo = run_dir / "NEO4J_IMPORT"
    ensure_dir(neo)
    # Artifact nodes
    nodes_art = neo / "nodes_Artifact.csv"
    art_fields = [
        "artifact_id:ID(Artifact)",
        "content_hash",
        "ext",
        "role",
        "bucket",
        "bytes:long",
        "blueprint_path",
        "first_seen_ts",
        "last_seen_ts",
    ]
    with nodes_art.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=NEO4J_DELIM)
        w.writerow(art_fields)
        for a in artifacts:
            w.writerow(
                [
                    a.get("artifact_id", ""),
                    a.get("content_hash", ""),
                    a.get("ext", ""),
                    a.get("role", ""),
                    a.get("bucket", ""),
                    a.get("bytes", ""),
                    a.get("blueprint_path", ""),
                    a.get("first_seen_ts", ""),
                    a.get("last_seen_ts", ""),
                ]
            )
    # Occurrence nodes
    nodes_occ = neo / "nodes_Occurrence.csv"
    occ_fields = [
        "occurrence_id:ID(Occurrence)",
        "origin_path",
        "root_label",
        "size_bytes:long",
        "mtime_utc",
        "quick_fp",
        "seen_ts",
    ]
    with nodes_occ.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=NEO4J_DELIM)
        w.writerow(occ_fields)
        for o in occurrences:
            w.writerow(
                [
                    o.get("occurrence_id", ""),
                    o.get("origin_path", ""),
                    o.get("root_label", ""),
                    o.get("size_bytes", ""),
                    o.get("mtime_utc", ""),
                    o.get("quick_fp", ""),
                    o.get("seen_ts", ""),
                ]
            )
    # Relationships: Occurrence -[:IS_OCCURRENCE_OF]-> Artifact
    rels = neo / "rels_IS_OCCURRENCE_OF.csv"
    rel_fields = [":START_ID(Occurrence)", ":END_ID(Artifact)", ":TYPE", "seen_ts"]
    with rels.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=NEO4J_DELIM)
        w.writerow(rel_fields)
        for o in occurrences:
            w.writerow(
                [
                    o.get("occurrence_id", ""),
                    o.get("artifact_id", ""),
                    "IS_OCCURRENCE_OF",
                    o.get("seen_ts", ""),
                ]
            )


def append_blueprint_graph_tables(
    blueprint_root: Path, artifacts: List[Dict[str, object]], occurrences: List[Dict[str, object]]
) -> None:
    """
    Append to blueprint "map" without mutating existing modeling files:
    emits GRAPH/APPENDS/<date>/ nodes and rels tables.
    """
    date_bucket = _dt.datetime.utcnow().strftime("%Y%m%d")
    out_dir = blueprint_root / "GRAPH" / "APPENDS" / date_bucket
    ensure_dir(out_dir)
    nodes_path = out_dir / "nodes_Artifact.csv"
    rels_path = out_dir / "rels_IS_OCCURRENCE_OF.csv"
    occ_path = out_dir / "nodes_Occurrence.csv"
    write_csv_rows(
        nodes_path,
        artifacts,
        fieldnames=[
            "artifact_id",
            "content_hash",
            "ext",
            "role",
            "bucket",
            "bytes",
            "blueprint_path",
            "first_seen_ts",
            "last_seen_ts",
        ],
    )
    write_csv_rows(
        occ_path,
        occurrences,
        fieldnames=[
            "occurrence_id",
            "artifact_id",
            "origin_path",
            "root_label",
            "size_bytes",
            "mtime_utc",
            "quick_fp",
            "seen_ts",
        ],
    )
    # Relationship rows
    rel_rows = [
        {
            "occurrence_id": o["occurrence_id"],
            "artifact_id": o["artifact_id"],
            "type": "IS_OCCURRENCE_OF",
            "seen_ts": o["seen_ts"],
        }
        for o in occurrences
    ]
    write_csv_rows(rels_path, rel_rows, fieldnames=["occurrence_id", "artifact_id", "type", "seen_ts"])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=APP_NAME, add_help=True)
    p.add_argument(
        "--roots",
        nargs="*",
        default=DEFAULT_ROOTS,
        help="Root tokens to scan. Accepts drive letters (C:) and/or directory paths.",
    )
    p.add_argument("--include-media", action="store_true", help="Also index media assets (pdf/png/jpg/etc). Default off.")
    p.add_argument("--exclude-dir", nargs="*", default=[], help="Additional directory names to exclude (case-insensitive).")
    p.add_argument(
        "--exclude-frag",
        nargs="*",
        default=[],
        help="Additional path fragments to exclude (case-insensitive, use backslashes).",
    )
    p.add_argument("--max-files", type=int, default=0, help="Stop after indexing this many candidates (0 = no limit).")
    p.add_argument("--follow-symlinks", action="store_true", help="Follow symlinks in traversal.")
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Output root folder.")
    p.add_argument("--blueprint", required=True, help="Existing blueprint pack path (.zip or folder). Script creates an appended copy.")
    p.add_argument("--seed-zip", nargs="*", default=[], help="Seed zip bundles to unpack into blueprint APPENDS/SEEDS/ (optional).")
    p.add_argument("--copy-mode", choices=["copy", "hardlink", "symlink"], default="copy", help="How to materialize uniques in blueprint.")
    p.add_argument("--watch", action="store_true", help="Run multiple cycles until stable (convergence) based on no new uniques.")
    p.add_argument("--interval", type=int, default=0, help="Watch interval seconds between cycles.")
    p.add_argument("--stable-n", type=int, default=2, help="Convergence requires this many stable cycles.")
    p.add_argument("--max-cycles", type=int, default=10, help="Max cycles in watch mode.")
    p.add_argument("--simulate", action="store_true", help="Index only: do not copy/link into blueprint.")
    p.add_argument("--skip-unchanged", action="store_true", help="Use STATE db to skip unchanged origin paths (recommended).")
    p.add_argument("--full-hash-max-bytes", type=int, default=200_000_000, help="Max bytes to full-hash per file (default 200MB).")
    p.add_argument("--hash-all", action="store_true", help="Full-hash even above full-hash-max-bytes.")
    p.add_argument("--no-summaries", action="store_true", help="Disable schema summaries (faster).")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out_root).resolve()
    ensure_dir(out_root)
    run_ts = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%SZ")
    run_dir = out_root / f"RUN_{APP_NAME}_{APP_VERSION}_{run_ts}"
    ensure_dir(run_dir)
    run_ledger = run_dir / "RUN" / "run_ledger.jsonl"
    ensure_dir(run_ledger.parent)

    def log_event(event: Dict[str, object]) -> None:
        event = dict(event)
        event.setdefault("ts", utc_now())
        append_jsonl(run_ledger, event)

    log_event({"event": "RUN_START", "app": APP_NAME, "version": APP_VERSION, "args": vars(args)})
    # Blueprint stage
    blueprint_in = Path(args.blueprint).resolve()
    if not blueprint_in.exists():
        log_event({"event": "ERROR", "error": "BLUEPRINT_NOT_FOUND", "path": str(blueprint_in)})
        return 2
    stage_dir = run_dir / "BLUEPRINT_STAGE"
    ensure_dir(stage_dir)
    blueprint_was_zip = blueprint_in.suffix.lower() == ".zip"
    if blueprint_was_zip:
        log_event({"event": "BLUEPRINT_UNZIP", "zip": str(blueprint_in), "dest": str(stage_dir)})
        unzip_to_dir(blueprint_in, stage_dir)
    else:
        log_event({"event": "BLUEPRINT_COPYTREE", "src": str(blueprint_in), "dest": str(stage_dir)})
        shutil.copytree(blueprint_in, stage_dir, dirs_exist_ok=True)
    # Seed zips into blueprint APPENDS/SEEDS
    if args.seed_zip:
        for zt in args.seed_zip:
            zpath = Path(zt).resolve()
            if not zpath.exists():
                log_event({"event": "SEED_ZIP_MISSING", "zip": str(zpath)})
                continue
            dest = stage_dir / "APPENDS" / "SEEDS" / sanitize_name(zpath.stem)
            ensure_dir(dest)
            unzip_to_dir(zpath, dest)
            log_event({"event": "SEED_ZIP_UNPACKED", "zip": str(zpath), "dest": str(dest)})
    # Binder directories inside blueprint
    binder_dir = stage_dir / "BINDER"
    schema_dir = binder_dir / "schema_summaries"
    ensure_dir(schema_dir)
    # Persistent state lives in the blueprint stage (becomes part of appended blueprint)
    state_dir = stage_dir / "STATE"
    ensure_dir(state_dir)
    state_db_path = state_dir / "harvester_state.sqlite"
    state_db = StateDB(state_db_path)
    # Exclusions
    exclude_dir_names = set(DEFAULT_EXCLUDE_DIR_NAMES)
    exclude_dir_names.update({norm_lower(x) for x in args.exclude_dir if x})
    exclude_path_frags = set(DEFAULT_EXCLUDE_PATH_FRAGMENTS)
    exclude_path_frags.update({x.lower() for x in args.exclude_frag if x})
    # Roots
    roots: List[Tuple[str, Path]] = []
    for t in args.roots:
        label, p = root_path_from_token(t)
        roots.append((label, p))
    log_event({"event": "ROOTS_RESOLVED", "roots": [{"label": lb, "path": str(p)} for lb, p in roots]})
    # Blueprint quick_fp index (for fast matching)
    blueprint_qf: Dict[str, List[str]] = {}
    blueprint_files_scanned = 0

    def consider_blueprint_file(p: Path) -> bool:
        ext = p.suffix.lower()
        return (ext in CANDIDATE_EXTS) or (args.include_media and ext in MEDIA_EXTS)

    for p in stage_dir.rglob("*"):
        if p.is_dir():
            continue
        if not consider_blueprint_file(p):
            continue
        try:
            qf = quick_fingerprint(p)
            blueprint_qf.setdefault(qf, []).append(str(p))
            blueprint_files_scanned += 1
        except Exception:
            continue
    log_event(
        {"event": "BLUEPRINT_QF_INDEX_BUILT", "files_scanned": blueprint_files_scanned, "unique_qf": len(blueprint_qf)}
    )
    # Watch cycles
    total_cycles = args.max_cycles if args.watch else 1
    stable_cycles = 0
    prior_unique_artifacts = None
    # Run-scoped collections (for outputs and graphs)
    run_artifacts: Dict[str, Dict[str, object]] = {}
    run_occurrences: List[Dict[str, object]] = []
    counters_total = {
        "candidates_scanned": 0,
        "skipped_unchanged": 0,
        "permission_skips": 0,
        "hashed_full": 0,
        "hashed_qf_only": 0,
        "uniques": 0,
        "duplicates": 0,
        "errors": 0,
    }
    for cycle in range(1, total_cycles + 1):
        cycle_start = utc_now()
        log_event({"event": "CYCLE_START", "cycle": cycle, "cycle_start": cycle_start})
        counters = dict(counters_total)
        counters.update(
            {
                "candidates_scanned": 0,
                "skipped_unchanged": 0,
                "permission_skips": 0,
                "hashed_full": 0,
                "hashed_qf_only": 0,
                "uniques": 0,
                "duplicates": 0,
                "errors": 0,
            }
        )
        for hit in iter_candidate_files(
            roots=roots,
            include_media=args.include_media,
            exclude_dir_names=exclude_dir_names,
            exclude_path_frags=exclude_path_frags,
            max_files=args.max_files,
            follow_symlinks=args.follow_symlinks,
        ):
            counters["candidates_scanned"] += 1
            origin_path = str(hit.path)
            size_bytes = hit.size
            mtime_utc = hit.mtime_utc
            try:
                prior = state_db.get_occurrence(origin_path) if args.skip_unchanged else None
                if prior and prior.get("size_bytes") == size_bytes and prior.get("mtime_utc") == mtime_utc:
                    counters["skipped_unchanged"] += 1
                    state_db.upsert_occurrence(
                        origin_path=origin_path,
                        size_bytes=size_bytes,
                        mtime_utc=mtime_utc,
                        quick_fp=str(prior.get("quick_fp") or ""),
                        content_hash=str(prior.get("content_hash") or ""),
                        artifact_id=str(prior.get("artifact_id") or ""),
                        last_seen_ts=utc_now(),
                    )
                    continue
                qf = quick_fingerprint(hit.path)
                # Decide hash mode
                do_full = args.hash_all or (size_bytes <= int(args.full_hash_max_bytes))
                if do_full:
                    content_hash = blake2b_hex_stream(hit.path)
                    counters["hashed_full"] += 1
                else:
                    content_hash = ""
                    counters["hashed_qf_only"] += 1
                # If full hash not computed, attempt blueprint duplicate check by qf only (best-effort)
                is_duplicate = False
                canonical_blueprint_path = ""
                if qf in blueprint_qf:
                    if content_hash:
                        # Confirm by full-hash against blueprint candidates with same qf
                        for bp in blueprint_qf.get(qf, []):
                            bpp = Path(bp)
                            try:
                                bp_hash = blake2b_hex_stream(bpp)
                                if bp_hash == content_hash:
                                    is_duplicate = True
                                    canonical_blueprint_path = str(bpp).replace("\\", "/")
                                    break
                            except Exception:
                                continue
                    else:
                        # qf-only: treat as likely duplicate (risk: extremely low but non-zero collisions)
                        is_duplicate = True
                        canonical_blueprint_path = str(Path(blueprint_qf[qf][0])).replace("\\", "/")
                # Resolve artifact identity
                if content_hash:
                    artifact_id = stable_artifact_id(content_hash)
                else:
                    artifact_id = f"ARTQF_{hashlib.blake2b(qf.encode('utf-8', errors='ignore'), digest_size=10).hexdigest()}"
                ext = hit.ext
                schema_summary = None if args.no_summaries else infer_schema_summary(hit.path)
                role = classify_role(hit.path, schema_summary)
                bucket = bucket_for_ext(ext)
                # Canonical materialization path
                target_path = determine_blueprint_target(stage_dir, role, ext, content_hash or qf, hit.path.name)
                blueprint_path_for_artifact = str(target_path).replace("\\", "/")
                # Artifact record (normalized per content hash; qf-only records are tracked but flagged)
                now_ts = utc_now()
                artifact_key = content_hash if content_hash else f"QFONLY:{qf}"
                if content_hash:
                    existing_art = run_artifacts.get(artifact_key) or state_db.get_artifact(content_hash)
                else:
                    existing_art = None
                if isinstance(existing_art, dict):
                    # refresh last_seen
                    first_seen_ts = existing_art.get("first_seen_ts", now_ts)
                    last_seen_ts = now_ts
                else:
                    first_seen_ts = now_ts
                    last_seen_ts = now_ts
                # Store schema summary file keyed by artifact_id
                schema_ref = ""
                if schema_summary is not None:
                    sum_name = sanitize_name(f"{artifact_id}_{role}_{ext.lstrip('.').lower()}.json", max_len=220)
                    sum_path = schema_dir / sum_name
                    write_json(sum_path, schema_summary)
                    schema_ref = str(sum_path.relative_to(stage_dir)).replace("\\", "/")
                # Determine write action
                blueprint_write_action = "SKIPPED_SIMULATE"
                if (not is_duplicate) and (not args.simulate):
                    blueprint_write_action = copy_or_link(hit.path, target_path, args.copy_mode)
                if is_duplicate:
                    counters["duplicates"] += 1
                else:
                    counters["uniques"] += 1
                    # add to blueprint qf index so within-run uniques dedup properly
                    blueprint_qf.setdefault(qf, []).append(str(target_path))
                # Upsert artifacts and occurrences into state db
                if content_hash:
                    state_db.upsert_artifact(
                        content_hash=content_hash,
                        artifact_id=artifact_id,
                        ext=ext,
                        role=role,
                        bucket=bucket,
                        bytes_=size_bytes,
                        blueprint_path=blueprint_path_for_artifact if not is_duplicate else canonical_blueprint_path,
                        now_ts=now_ts,
                    )
                state_db.upsert_occurrence(
                    origin_path=origin_path,
                    size_bytes=size_bytes,
                    mtime_utc=mtime_utc,
                    quick_fp=qf,
                    content_hash=content_hash,
                    artifact_id=artifact_id,
                    last_seen_ts=now_ts,
                )
                # Append binder JSONL rows (append-only)
                # Artifacts.jsonl: only when first observed in this run (per artifact_key)
                if artifact_key not in run_artifacts:
                    run_artifacts[artifact_key] = {
                        "artifact_id": artifact_id,
                        "content_hash": content_hash,
                        "hash_mode": "full" if content_hash else "qf_only",
                        "quick_fp": qf,
                        "ext": ext,
                        "role": role,
                        "bucket": bucket,
                        "bytes": size_bytes,
                        "blueprint_path": canonical_blueprint_path
                        if (is_duplicate and canonical_blueprint_path)
                        else blueprint_path_for_artifact,
                        "schema_summary_ref": schema_ref,
                        "first_seen_ts": first_seen_ts,
                        "last_seen_ts": last_seen_ts,
                    }
                    append_jsonl(binder_dir / "artifacts.jsonl", run_artifacts[artifact_key])
                # Occurrences.jsonl: one row per processed origin path snapshot
                occ_id = stable_occurrence_id(origin_path, size_bytes, mtime_utc)
                occ_row = {
                    "occurrence_id": occ_id,
                    "artifact_id": artifact_id,
                    "origin_path": origin_path,
                    "root_label": hit.root_label,
                    "size_bytes": size_bytes,
                    "mtime_utc": mtime_utc,
                    "quick_fp": qf,
                    "seen_ts": now_ts,
                    "blueprint_write_action": blueprint_write_action,
                    "is_duplicate": bool(is_duplicate),
                }
                run_occurrences.append(occ_row)
                append_jsonl(binder_dir / "occurrences.jsonl", occ_row)
            except PermissionError:
                counters["permission_skips"] += 1
                continue
            except FileNotFoundError:
                continue
            except OSError:
                counters["errors"] += 1
                continue
            except Exception:
                counters["errors"] += 1
                continue
        # Emit per-cycle neo4j import tables
        artifacts_list = list(run_artifacts.values())
        build_neo4j_import_tables(run_dir, artifacts_list, run_occurrences)
        append_blueprint_graph_tables(stage_dir, artifacts_list, run_occurrences)
        log_event(
            {
                "event": "CYCLE_END",
                "cycle": cycle,
                "counters": counters,
                "run_artifacts_unique": len(artifacts_list),
                "run_occurrences": len(run_occurrences),
            }
        )
        # Convergence check
        if args.watch:
            current_unique_artifacts = len(artifacts_list)
            if prior_unique_artifacts is not None and current_unique_artifacts == prior_unique_artifacts and counters[
                "uniques"
            ] == 0:
                stable_cycles += 1
            else:
                stable_cycles = 0
            prior_unique_artifacts = current_unique_artifacts
            if stable_cycles >= int(args.stable_n):
                log_event({"event": "CONVERGENCE_TRUE", "stable_cycles": stable_cycles})
                break
            if int(args.interval) > 0:
                time.sleep(int(args.interval))
        # reset per-cycle counters base
        for k in counters_total.keys():
            counters_total[k] += counters.get(k, 0)
    # Provenance index
    provenance = {
        "ts": utc_now(),
        "app": APP_NAME,
        "version": APP_VERSION,
        "blueprint_input": str(blueprint_in),
        "blueprint_staged_root": str(stage_dir),
        "roots": [{"label": lb, "path": str(p)} for lb, p in roots],
        "include_media": bool(args.include_media),
        "candidate_exts": sorted(CANDIDATE_EXTS | (MEDIA_EXTS if args.include_media else set())),
        "exclude_dir_names": sorted(exclude_dir_names),
        "exclude_path_fragments": sorted(exclude_path_frags),
        "counts": {
            "run_artifacts_unique": len(run_artifacts),
            "run_occurrences": len(run_occurrences),
        },
        "neo4j_import_dir": str((run_dir / "NEO4J_IMPORT").resolve()),
    }
    write_json(run_dir / "RUN" / "provenance_index.json", provenance)
    # Convenience CSV exports (current run scope)
    artifacts_list = list(run_artifacts.values())
    write_csv_rows(
        run_dir / "BINDER_export_artifacts.csv",
        artifacts_list,
        fieldnames=[
            "artifact_id",
            "content_hash",
            "hash_mode",
            "quick_fp",
            "ext",
            "role",
            "bucket",
            "bytes",
            "blueprint_path",
            "schema_summary_ref",
            "first_seen_ts",
            "last_seen_ts",
        ],
    )
    write_csv_rows(
        run_dir / "BINDER_export_occurrences.csv",
        run_occurrences,
        fieldnames=[
            "occurrence_id",
            "artifact_id",
            "origin_path",
            "root_label",
            "size_bytes",
            "mtime_utc",
            "quick_fp",
            "seen_ts",
            "blueprint_write_action",
            "is_duplicate",
        ],
    )
    # Emit appended blueprint zip
    appended_zip = out_root / f"{blueprint_in.stem}_APPENDED_{run_ts}.zip"
    zip_dir(stage_dir, appended_zip)
    log_event({"event": "BLUEPRINT_APPENDED_ZIP_WRITTEN", "path": str(appended_zip)})
    summary = {
        "ts": utc_now(),
        "app": APP_NAME,
        "version": APP_VERSION,
        "run_dir": str(run_dir),
        "appended_blueprint_zip": str(appended_zip),
        "run_artifacts_unique": len(run_artifacts),
        "run_occurrences": len(run_occurrences),
    }
    write_json(run_dir / "RUN_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))
    state_db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
