"""Utility helpers for Golden Litigator OS."""

from __future__ import annotations

import json
import hashlib
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, cast


def load_config(path: str = "config.json") -> Dict[str, Any]:
    """Load JSON configuration file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return cast(Dict[str, Any], data)


def sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def excerpt(text: str, n: int = 1000) -> str:
    """Return first ``n`` characters of ``text`` collapsed to single spaces."""
    return re.sub(r"\s+", " ", text or "").strip()[:n]


def ensure_dirs(results_dir: Path) -> None:
    """Create required output subdirectories."""
    for sub in [
        "",
        "Motions",
        "Narratives",
        "Transcripts",
        "Exhibits",
        "Binder",
        "Forms",
        "Bundles",
        "Judges",
    ]:
        (results_dir / sub).mkdir(parents=True, exist_ok=True)


def is_excluded_dir(root: str, excludes: set[str]) -> bool:
    """Return True if ``root`` is an excluded path on Windows."""
    return os.name == "nt" and any(root.startswith(e) for e in excludes)


def db_conn(db_path: str) -> sqlite3.Connection:
    """Open connection to SQLite database at ``db_path``."""
    return sqlite3.connect(db_path)


SCHEMA: Dict[str, str] = {
    "evidence": """
    CREATE TABLE IF NOT EXISTS evidence (
      id INTEGER PRIMARY KEY,
      sha256 TEXT UNIQUE,
      filename TEXT, filepath TEXT, ext TEXT,
      size_bytes INTEGER, modified_ts TEXT,
      content_excerpt TEXT,
      party TEXT,
      parties_json TEXT, claims_json TEXT,
      statutes_json TEXT, court_rules_json TEXT,
      relevance_score REAL,
      timeline_refs_json TEXT,
      exhibit_tag TEXT, exhibit_label TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    "exhibits": """
    CREATE TABLE IF NOT EXISTS exhibits (
      id INTEGER PRIMARY KEY,
      evidence_sha256 TEXT, label TEXT, title TEXT,
      description TEXT, page_refs_json TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    "timelines": """
    CREATE TABLE IF NOT EXISTS timelines (
      id INTEGER PRIMARY KEY,
      evidence_sha256 TEXT, event_dt TEXT,
      actor TEXT, action TEXT, location TEXT, details TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    "filings": """
    CREATE TABLE IF NOT EXISTS filings (
      id INTEGER PRIMARY KEY,
      filing_type TEXT, title TEXT,
      court_name TEXT, case_number TEXT,
      body_path TEXT, status TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    "parties": """
    CREATE TABLE IF NOT EXISTS parties (
      id INTEGER PRIMARY KEY,
      role TEXT, name TEXT, contact_json TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    "sources": """
    CREATE TABLE IF NOT EXISTS sources (
      id INTEGER PRIMARY KEY,
      evidence_sha256 TEXT, source_type TEXT, meta_json TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    "case_meta": """
    CREATE TABLE IF NOT EXISTS case_meta (
      id INTEGER PRIMARY KEY,
      court_name TEXT, case_number TEXT,
      caption_plaintiff TEXT, caption_defendant TEXT,
      judge TEXT, jurisdiction TEXT, division TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
    "code_registry": """
    CREATE TABLE IF NOT EXISTS code_registry (
      id INTEGER PRIMARY KEY,
      sha256 TEXT UNIQUE, filename TEXT, filepath TEXT, ext TEXT,
      size_bytes INTEGER, modified_ts TEXT, header_preview TEXT,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""",
}


def init_db(db_path: str) -> None:
    """Create tables if they do not already exist."""
    conn = db_conn(db_path)
    cur = conn.cursor()
    for stmt in SCHEMA.values():
        cur.execute(stmt)
    conn.commit()
    conn.close()


def insert_source(
    db_path: str, sha: str, source_type: str, meta: Dict[str, Any]
) -> None:
    conn = db_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO sources (evidence_sha256, source_type, meta_json)
                   VALUES (?, ?, ?)""",
        (sha, source_type, json.dumps(meta, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


def insert_evidence(db_path: str, rec: Dict[str, Any]) -> None:
    conn = db_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        """
      INSERT OR IGNORE INTO evidence (
        sha256, filename, filepath, ext, size_bytes, modified_ts,
        content_excerpt, party, parties_json, claims_json, statutes_json,
        court_rules_json, relevance_score, timeline_refs_json, exhibit_tag, exhibit_label
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            rec["sha256"],
            rec["filename"],
            rec["filepath"],
            rec["ext"],
            rec["size_bytes"],
            rec["modified_ts"],
            rec.get("content_excerpt", ""),
            rec.get("party", ""),
            json.dumps(rec.get("parties", []), ensure_ascii=False),
            json.dumps(rec.get("claims", []), ensure_ascii=False),
            json.dumps(rec.get("statutes", []), ensure_ascii=False),
            json.dumps(rec.get("court_rules", []), ensure_ascii=False),
            rec.get("relevance_score", 0.0),
            json.dumps(rec.get("timeline_refs", []), ensure_ascii=False),
            rec.get("exhibit_tag", ""),
            rec.get("exhibit_label", ""),
        ),
    )
    conn.commit()
    conn.close()


def insert_timeline(db_path: str, sha: str, ev: Dict[str, Any]) -> None:
    conn = db_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO timelines (evidence_sha256, event_dt, actor, action, location, details)
                   VALUES (?, ?, ?, ?, ?, ?)""",
        (
            sha,
            ev.get("date", ""),
            ev.get("actor", ""),
            ev.get("action", ""),
            ev.get("location", ""),
            ev.get("details", ""),
        ),
    )
    conn.commit()
    conn.close()


def insert_exhibit(db_path: str, sha: str, ex: Dict[str, Any]) -> None:
    conn = db_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO exhibits (evidence_sha256, label, title, description, page_refs_json)
                   VALUES (?, ?, ?, ?, ?)""",
        (
            sha,
            ex.get("label", ""),
            ex.get("title", ""),
            ex.get("description", ""),
            json.dumps(ex.get("pages", []), ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()


def upsert_case_meta(db_path: str, meta: Dict[str, Any]) -> None:
    needed = {
        "court_name",
        "case_number",
        "caption_plaintiff",
        "caption_defendant",
        "jurisdiction",
    }
    if not needed.issubset(meta.keys()):
        return
    conn = db_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        (
            "INSERT INTO case_meta (court_name, case_number, caption_plaintiff, "
            "caption_defendant, judge, jurisdiction, division) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        ),
        (
            meta.get("court_name", ""),
            meta.get("case_number", ""),
            meta.get("caption_plaintiff", ""),
            meta.get("caption_defendant", ""),
            meta.get("judge", ""),
            meta.get("jurisdiction", ""),
            meta.get("division", ""),
        ),
    )
    conn.commit()
    conn.close()


def register_code_file(db_path: str, path: Path, sha: str) -> None:
    try:
        preview = excerpt(path.read_text(encoding="utf-8", errors="ignore"), 500)
    except Exception:
        preview = ""
    stat = path.stat()
    conn = db_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        """INSERT OR IGNORE INTO code_registry
                   (sha256, filename, filepath, ext, size_bytes, modified_ts, header_preview)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            sha,
            path.name,
            str(path),
            path.suffix.lower(),
            stat.st_size,
            datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            preview,
        ),
    )
    conn.commit()
    conn.close()


def safe_rename_done(path: Path, tag: str) -> None:
    """Rename ``path`` by appending ``tag`` before the extension."""
    try:
        new_path = path.with_name(f"{path.stem}{tag}{path.suffix}")
        if not new_path.exists():
            path.rename(new_path)
    except Exception:
        pass
