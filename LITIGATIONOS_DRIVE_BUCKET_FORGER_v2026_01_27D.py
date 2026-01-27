#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LITIGATIONOS_DRIVE_BUCKET_FORGER_v2026_01_27D.py

"Deep hydration" upgrade of the Drive Bucket Forger:
- Buckets: <=15 deterministic groups by file type (non-destructive default).
- Plan→Apply pipeline: moves/copies are opt-in; everything else is append-only.
- Convergence cycles: chained cycles until stable fingerprints (inventory + plan) for N consecutive cycles.
- Traveling Bumpers: a durable, append-only bumper log that travels with the project across cycles/runs.
- Version consolidation:
  - DuplicateGroup detection (sha256 or quick_sig fallback)
  - VersionFamily detection (filename heuristic)
  - CanonicalView builder (symlink if possible; else hardlink; else copy)
- Empty folder cleaning:
  - Generates a removal plan; optional apply (non-default)
- ZIP intelligence:
  - Archive inventory (members + optional member hashing)
  - Optional "ForgeZip" exporter (canonical bundle zip with manifest + integrity test)
- Graph Data Pipeline Engineering:
  - Bronze/Silver/Gold medallion outputs
  - Neo4j import pack (CSV + Cypher import recipes)
  - Virtual graph in SQLite (portable), plus query runner (built-ins + custom SQL file)
  - GraphML export (for non-Neo4j tools)
  - Offline HTML viewer for file graph (physics layout)
- ERD_SUPERSET_FED_BLUEPRINT:
  - Schema-first superset ERD outputs (JSON/MD/DOT/HTML) with physics layout
- Physics Engine:
  - fast: deterministic radial layout
  - full: iterative force simulation (auto-degrades on huge graphs; numpy optional)

Design invariants:
- Never touches C:\\ (Gate-0 drive exclusion).
- Skips system paths by default.
- Bumpers not blockers: issues are logged; destructive ops gated behind explicit flags.
- Append-only ledgers: run_ledger.jsonl, LIVE_PRESERVATION_LOG.jsonl, BUMPERS_TRAVELING.jsonl.
- Deterministic receipts: CRC32 + bytes + mtime -> IntegrityKey.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import time
import zipfile
import zlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Set

APP_VER = "v2026_01_27D"

# ============================
# Buckets (<=15)
# ============================

DEFAULT_BUCKETS: List[Tuple[str, str]] = [
    ("B01_TEXT", "txt, md, rtf, log, ini, yaml, yml, toml"),
    ("B02_DATA", "csv, tsv, json, jsonl, parquet, sqlite, db"),
    ("B03_DOCS", "doc, docx, odt, pdf (OCR queue optional), xls/xlsx, ppt/pptx"),
    ("B04_CODE", "py, js, ts, java, cs, cpp, c, go, rs, sh, ps1, bat"),
    ("B05_WEB", "html, htm, css, wasm"),
    ("B06_IMAGES", "png, jpg, jpeg, gif, webp, tif, tiff, bmp"),
    ("B07_AUDIO", "mp3, wav, m4a, flac, ogg"),
    ("B08_VIDEO", "mp4, mov, mkv, avi"),
    ("B09_ARCHIVES", "zip, 7z, rar, tar, gz"),
    ("B10_EXEC", "exe, msi, apk, appimage"),
    ("B11_FONTS", "ttf, otf, woff, woff2"),
    ("B12_CONFIG", "conf, cfg, reg, env"),
    ("B13_EMAIL", "eml, msg, mbox"),
    ("B14_OCR_QUEUE", "pdf likely needing OCR (heuristic)"),
    ("B15_MISC", "everything else"),
]

TEXT_EXT = {".txt", ".md", ".rtf", ".log", ".ini", ".yaml", ".yml", ".toml"}
DATA_EXT = {".csv", ".tsv", ".json", ".jsonl", ".parquet", ".sqlite", ".db"}
DOC_EXT = {".doc", ".docx", ".odt", ".pdf", ".xls", ".xlsx", ".ppt", ".pptx"}
CODE_EXT = {".py", ".js", ".ts", ".java", ".cs", ".cpp", ".c", ".go", ".rs", ".sh", ".ps1", ".bat"}
WEB_EXT = {".html", ".htm", ".css", ".wasm"}
IMG_EXT = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".tif", ".tiff", ".bmp"}
AUD_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
VID_EXT = {".mp4", ".mov", ".mkv", ".avi"}
ARC_EXT = {".zip", ".7z", ".rar", ".tar", ".gz"}
EXE_EXT = {".exe", ".msi", ".apk", ".appimage"}
FONT_EXT = {".ttf", ".otf", ".woff", ".woff2"}
CFG_EXT = {".conf", ".cfg", ".reg", ".env"}
EMAIL_EXT = {".eml", ".msg", ".mbox"}

# ============================
# Version family heuristics
# ============================

VERSION_PATTERNS = [
    re.compile(r"(?i)(?:^|[ _-])v(\d+(?:\.\d+){0,3})(?:$|[ _-])"),
    re.compile(r"(?i)(?:^|[ _-])(\d{4}[-_]\d{2}[-_]\d{2})(?:$|[ _-])"),
    re.compile(r"(?i)(?:^|[ _-])(\d{8})(?:$|[ _-])"),
    re.compile(r"(?i)(?:^|[ _-])(final)(?:$|[ _-])"),
    re.compile(r"(?i)(?:^|[ _-])(canonical)(?:$|[ _-])"),
    re.compile(r"(?i)(?:^|[ _-])(master)(?:$|[ _-])"),
]

# ============================
# Lite doctype heuristics (metadata only)
# ============================

DOCTYPE_PATTERNS = [
    ("TRANSCRIPT", re.compile(r"(?i)\btranscript\b|\btr\b|\bhearing\b")),
    ("ORDER", re.compile(r"(?i)\border\b|\bjudg(?:ment|ment)\b|\bopinion\b")),
    ("MOTION", re.compile(r"(?i)\bmotion\b|\bmovant\b|\bnotice of hearing\b")),
    ("BRIEF", re.compile(r"(?i)\bbrief\b|\bappellant\b|\bappellee\b")),
    ("EXHIBIT", re.compile(r"(?i)\bexhibit\b|\bexh\b")),
    ("EMAIL", re.compile(r"(?i)\bemail\b|\bmail\b")),
    ("FOIA", re.compile(r"(?i)\bfoia\b|\bfreedom of information\b")),
    ("INVOICE", re.compile(r"(?i)\binvoice\b|\breceipt\b|\bbill\b")),
]

CASE_NO_RX = re.compile(r"(?i)\b(\d{4}[-_ ]\d{6}[-_ ][A-Z]{2})\b")
DATE_RX = re.compile(r"(?i)\b(\d{4}[-_]\d{2}[-_]\d{2})\b")

# ============================
# Content scan terms (lite)
# ============================

DEFAULT_NEGATIVE_TERMS = [
    "unfit", "abusive", "dangerous", "unstable", "harassment", "stalking", "threat", "violent",
    "kidnap", "alienat", "contempt", "violat", "fraud", "perjur", "lying", "false", "weaponiz",
    "no evidence", "without evidence", "refused", "denied", "bias", "partial", "inadmissible",
    "sustained", "overruled", "hearsay", "lack of foundation", "due process", "ex parte",
]

# ============================
# Dataclasses
# ============================

@dataclass
class FileRecord:
    path: str
    drive: str
    relpath: str
    name: str
    ext: str
    size: int
    mtime: float
    bucket: str
    rule_id: str
    rule_reason: str
    doctype: str
    case_no: str
    date_hint: str
    tags: List[str]
    version_family: str
    version_score: float
    sha256: str
    quick_sig: str
    notes: str

@dataclass
class PlannedAction:
    action: str  # MOVE | COPY | SKIP
    src: str
    dst: str
    bucket: str
    reason: str
    size: int

@dataclass
class PlannedDirAction:
    action: str  # RMDIR | SKIP
    dir_path: str
    reason: str

@dataclass
class Bumper:
    id: str
    severity: str  # soft | warn | hard
    where: str
    msg: str
    detail: str
    mv: str
    ts: str

# ============================
# Time / IDs / Integrity
# ============================

def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def stable_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:16]

def crc32_bytes(b: bytes) -> int:
    return zlib.crc32(b) & 0xFFFFFFFF

def crc32_file(path: Path, chunk: int = 1024 * 1024) -> int:
    c = 0
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            c = zlib.crc32(b, c)
    return c & 0xFFFFFFFF

def integrity_key(bundle_uid: str, entry_path: str, crc32: int, size: int, mtime: int) -> str:
    base = f"{bundle_uid}|{entry_path}|{crc32}|{size}|{mtime}"
    return hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()

# ============================
# IO writers
# ============================

def write_json(path: Path, obj: object) -> None:
    safe_mkdir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def write_jsonl(path: Path, rows: List[dict]) -> None:
    safe_mkdir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def append_jsonl(path: Path, row: dict) -> None:
    safe_mkdir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    safe_mkdir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

# ============================
# Bumpers
# ============================

def bumper(bumpers_list: List[Bumper], bid: str, severity: str, where: str, msg: str, detail: str = "", mv: str = "") -> None:
    bumpers_list.append(Bumper(id=bid, severity=severity, where=where, msg=msg, detail=detail, mv=mv, ts=now_iso()))

def emit_traveling_bumper(travel_path: Path, b: Bumper, run_id: str, cycle: int) -> None:
    append_jsonl(travel_path, {
        "ts": b.ts,
        "run_id": run_id,
        "cycle": cycle,
        "id": b.id,
        "severity": b.severity,
        "where": b.where,
        "msg": b.msg,
        "detail": b.detail,
        "mv": b.mv,
    })

# ============================
# Filters: system paths + drive gate
# ============================

def is_system_path(p: Path) -> bool:
    ps = str(p).lower()
    bad = [
        r"\\windows\\", r"\\program files", r"\\programdata\\",
        r"\\$recycle.bin", r"\\system volume information",
    ]
    return any(b in ps for b in bad)

def is_forbidden_drive(p: Path) -> bool:
    # Gate 0: hard exclude C:
    drv = (p.drive or "").upper()
    return drv.startswith("C:")

# ============================
# Hashing
# ============================

def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def quick_signature(path: Path, head: int = 65536, tail: int = 65536) -> str:
    st = path.stat()
    size = st.st_size
    h = hashlib.sha1()
    with path.open("rb") as f:
        h.update(f.read(min(head, size)))
        if size > head + tail:
            f.seek(max(0, size - tail))
            h.update(f.read(tail))
    h.update(str(size).encode("utf-8"))
    return h.hexdigest()

# ============================
# Classification: bucket + tags + doctype
# ============================

def extract_doctype(name: str) -> str:
    n = name.lower()
    for dt, rx in DOCTYPE_PATTERNS:
        if rx.search(n):
            return dt
    return ""

def extract_case_no(s: str) -> str:
    m = CASE_NO_RX.search(s)
    return m.group(1).replace("_", "-").replace(" ", "-") if m else ""

def extract_date_hint(s: str) -> str:
    m = DATE_RX.search(s)
    return m.group(1).replace("_", "-") if m else ""

def tag_from_path(path: Path) -> List[str]:
    s = (str(path).lower() + " " + path.name.lower())
    tags = []
    # project-ish tags
    if "litigation" in s or "court" in s:
        tags.append("tag:litigation")
    if "neo4j" in s or "cypher" in s or "graph" in s:
        tags.append("tag:graph")
    if "foia" in s:
        tags.append("tag:foia")
    if "ppo" in s:
        tags.append("tag:ppo")
    if "custody" in s or "parenting" in s or "foc" in s:
        tags.append("tag:custody")
    if "exhibit" in s:
        tags.append("tag:exhibit")
    if "transcript" in s:
        tags.append("tag:transcript")
    return sorted(set(tags))

def classify_bucket(path: Path, ocr_queue_enabled: bool, bumpers_list: List[Bumper]) -> Tuple[str, str, str]:
    ext = path.suffix.lower()

    if ext in ARC_EXT:
        return ("B09_ARCHIVES", "R_EXT_ARCHIVE", f"ext={ext}")
    if ext in EXE_EXT:
        return ("B10_EXEC", "R_EXT_EXEC", f"ext={ext}")
    if ext in IMG_EXT:
        return ("B06_IMAGES", "R_EXT_IMAGE", f"ext={ext}")
    if ext in AUD_EXT:
        return ("B07_AUDIO", "R_EXT_AUDIO", f"ext={ext}")
    if ext in VID_EXT:
        return ("B08_VIDEO", "R_EXT_VIDEO", f"ext={ext}")
    if ext in FONT_EXT:
        return ("B11_FONTS", "R_EXT_FONT", f"ext={ext}")
    if ext in EMAIL_EXT:
        return ("B13_EMAIL", "R_EXT_EMAIL", f"ext={ext}")
    if ext in CODE_EXT:
        return ("B04_CODE", "R_EXT_CODE", f"ext={ext}")
    if ext in WEB_EXT:
        return ("B05_WEB", "R_EXT_WEB", f"ext={ext}")
    if ext in DATA_EXT:
        return ("B02_DATA", "R_EXT_DATA", f"ext={ext}")
    if ext in TEXT_EXT:
        return ("B01_TEXT", "R_EXT_TEXT", f"ext={ext}")
    if ext in DOC_EXT:
        if ext == ".pdf" and ocr_queue_enabled:
            try:
                if needs_ocr_pdf(path):
                    return ("B14_OCR_QUEUE", "R_PDF_OCR_HEUR", "pdf low_text_signal")
            except Exception as e:
                bumper(bumpers_list, "OCR_HEUR_FAIL", "soft", str(path), "OCR heuristic failed", str(e))
        return ("B03_DOCS", "R_EXT_DOC", f"ext={ext}")

    # name hints
    n = path.name.lower()
    if "readme" in n:
        return ("B01_TEXT", "R_HINT_README", "name contains readme")
    if "schema" in n and ext in {".json", ".yml", ".yaml"}:
        return ("B02_DATA", "R_HINT_SCHEMA", "name contains schema")
    if "backup" in n or n.endswith(".bak"):
        return ("B15_MISC", "R_HINT_BACKUP", "name indicates backup")

    return ("B15_MISC", "R_FALLBACK", f"ext={ext or '(none)'}")

# ============================
# PDF OCR heuristic
# ============================

def needs_ocr_pdf(path: Path) -> bool:
    """
    Conservative heuristic: if first bytes contain very few ASCII letters, treat as likely scanned image PDF.
    """
    raw = path.read_bytes()[:200_000]
    letters = 0
    for b in raw:
        if 65 <= b <= 122:
            letters += 1
    return letters < 400

# ============================
# Version grouping
# ============================

def detect_version_family(path: Path) -> Tuple[str, float]:
    name = path.stem
    tokens = re.split(r"[ _-]+", name)
    cleaned = []
    score = 0.0
    for t in tokens:
        matched = False
        for rx in VERSION_PATTERNS:
            if rx.search(t):
                matched = True
                tl = t.lower()
                if tl == "final":
                    score += 3.0
                elif tl == "canonical":
                    score += 2.5
                elif tl == "master":
                    score += 2.0
                elif re.fullmatch(r"\d{4}[-_]\d{2}[-_]\d{2}", t):
                    score += 1.2
                elif re.fullmatch(r"\d{8}", t):
                    score += 1.2
                else:
                    score += 1.0
                break
        if not matched:
            cleaned.append(t)
    fam = " ".join(cleaned).strip().lower()
    return (fam, score) if fam else ("", 0.0)

def choose_canonical(members: List[FileRecord]) -> FileRecord:
    def score(r: FileRecord) -> Tuple[float, float, int]:
        bonus = 0.0
        nm = r.name.lower()
        if "final" in nm:
            bonus += 3.0
        if "canonical" in nm:
            bonus += 2.0
        if "master" in nm:
            bonus += 1.5
        return (r.version_score + bonus, r.mtime, r.size)
    return sorted(members, key=score, reverse=True)[0]

# ============================
# Scanning
# ============================

def iter_files(roots: List[Path], bumpers_list: List[Bumper], include_hidden: bool, max_files: int) -> Iterable[Path]:
    seen = 0
    for root in roots:
        if not root.exists():
            bumper(bumpers_list, "ROOT_MISSING", "warn", str(root), "Root does not exist; skipping")
            continue
        if is_forbidden_drive(root):
            bumper(bumpers_list, "ROOT_FORBIDDEN", "hard", str(root), "Forbidden drive (Gate 0): skipping root")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dp = Path(dirpath)
            if is_forbidden_drive(dp) or is_system_path(dp):
                continue
            if not include_hidden:
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                filenames = [fn for fn in filenames if not fn.startswith(".")]
            for fn in filenames:
                p = dp / fn
                try:
                    if p.is_symlink():
                        continue
                    yield p
                    seen += 1
                    if max_files > 0 and seen >= max_files:
                        bumper(bumpers_list, "MAX_FILES_REACHED", "warn", str(root), f"Max files reached: {max_files}; truncating scan")
                        return
                except Exception:
                    continue


def scan_inventory(
    roots: List[Path],
    bumpers_list: List[Bumper],
    compute_hashes: bool,
    compute_quick_sig: bool,
    ocr_queue_enabled: bool,
    include_hidden: bool,
    max_files: int,
    max_size_mb: int,
) -> List[FileRecord]:
    recs: List[FileRecord] = []
    max_bytes = max_size_mb * 1024 * 1024 if max_size_mb > 0 else 0

    for p in iter_files(roots, bumpers_list, include_hidden=include_hidden, max_files=max_files):
        try:
            if is_forbidden_drive(p):
                continue
            st = p.stat()
        except Exception as e:
            bumper(bumpers_list, "STAT_FAIL", "soft", str(p), "Failed stat()", str(e))
            continue

        if max_bytes and st.st_size > max_bytes:
            bumper(bumpers_list, "FILE_TOO_LARGE_SKIP", "soft", str(p), f"Skipping file > {max_size_mb}MB", f"size={st.st_size}")
            continue

        drive = p.drive if hasattr(p, "drive") else ""
        relpath = str(p)
        try:
            if drive:
                relpath = str(p.relative_to(Path(drive + os.sep)))
        except Exception:
            relpath = str(p)

        bucket, rid, rreason = classify_bucket(p, ocr_queue_enabled=ocr_queue_enabled, bumpers_list=bumpers_list)
        fam, fscore = detect_version_family(p)

        # metadata extraction (filename + path)
        doctype = extract_doctype(p.name)
        case_no = extract_case_no(str(p))
        date_hint = extract_date_hint(p.name) or extract_date_hint(str(p))

        tags = tag_from_path(p)

        fr = FileRecord(
            path=str(p),
            drive=drive or "ROOT",
            relpath=relpath,
            name=p.name,
            ext=p.suffix.lower(),
            size=int(st.st_size),
            mtime=float(st.st_mtime),
            bucket=bucket,
            rule_id=rid,
            rule_reason=rreason,
            doctype=doctype,
            case_no=case_no,
            date_hint=date_hint,
            tags=tags,
            version_family=fam,
            version_score=float(fscore),
            sha256="",
            quick_sig="",
            notes="",
        )

        if compute_quick_sig:
            try:
                fr.quick_sig = quick_signature(p)
            except Exception as e:
                bumper(bumpers_list, "QUICK_SIG_FAIL", "soft", str(p), "Quick signature failed", str(e))

        if compute_hashes:
            try:
                fr.sha256 = sha256_file(p)
            except Exception as e:
                bumper(bumpers_list, "HASH_FAIL", "soft", str(p), "SHA256 failed", str(e))

        recs.append(fr)

    return recs

# ============================
# Planning: bucket moves
# ============================

def plan_bucket_paths(out_root: Path, recs: List[FileRecord], bumpers_list: List[Bumper], skip_existing: bool, verify_existing_hash: bool) -> List[PlannedAction]:
    actions: List[PlannedAction] = []
    for r in recs:
        src = Path(r.path)
        bucket_dir = out_root / r.bucket
        dst = bucket_dir / src.name

        if str(src).lower() == str(dst).lower():
            actions.append(PlannedAction("SKIP", r.path, str(dst), r.bucket, "already_in_place", r.size))
            continue

        if skip_existing and dst.exists():
            if verify_existing_hash and r.sha256:
                try:
                    dst_hash = sha256_file(dst)
                    if dst_hash == r.sha256:
                        actions.append(PlannedAction("SKIP", r.path, str(dst), r.bucket, "dst_exists_same_hash", r.size))
                        continue
                except Exception as e:
                    bumper(bumpers_list, "VERIFY_DST_FAIL", "soft", str(dst), "Failed verifying destination hash", str(e))
            actions.append(PlannedAction("SKIP", r.path, str(dst), r.bucket, "dst_exists", r.size))
            continue

        actions.append(PlannedAction("MOVE", r.path, str(dst), r.bucket, f"{r.rule_id}:{r.rule_reason}", r.size))
    return actions

def apply_plan(actions: List[PlannedAction], bumpers_list: List[Bumper], mode: str, dry_run: bool) -> Dict[str, int]:
    """
    mode: move | copy
    """
    moved = 0
    copied = 0
    skipped = 0
    failed = 0

    do_copy = (mode == "copy")
    for a in actions:
        if a.action == "SKIP":
            skipped += 1
            continue

        src = Path(a.src)
        dst = Path(a.dst)
        try:
            safe_mkdir(dst.parent)
            if dry_run:
                if do_copy:
                    copied += 1
                else:
                    moved += 1
                continue

            if do_copy:
                shutil.copy2(str(src), str(dst))
                copied += 1
            else:
                if dst.exists():
                    bumper(bumpers_list, "DST_EXISTS", "warn", f"{a.src} -> {a.dst}", "Destination exists; skipping move")
                    skipped += 1
                    continue
                shutil.move(str(src), str(dst))
                moved += 1
        except Exception as e:
            failed += 1
            bumper(bumpers_list, "APPLY_FAIL", "warn", f"{a.src} -> {a.dst}", "Move/copy failed", str(e))

    return {"moved": moved, "copied": copied, "skipped": skipped, "failed": failed}

# ============================
# Empty directory cleaning (plan-only unless apply)
# ============================

def find_empty_dirs(roots: List[Path], bumpers_list: List[Bumper], include_hidden: bool, max_dirs: int) -> List[Path]:
    empties: List[Path] = []
    count = 0
    for root in roots:
        if not root.exists() or is_forbidden_drive(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            dp = Path(dirpath)
            if is_forbidden_drive(dp) or is_system_path(dp):
                continue
            if not include_hidden:
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                filenames = [fn for fn in filenames if not fn.startswith(".")]
            try:
                # directory is empty if no children after filtering
                if not dirnames and not filenames:
                    empties.append(dp)
                    count += 1
                    if max_dirs > 0 and count >= max_dirs:
                        bumper(bumpers_list, "MAX_EMPTY_DIRS_REACHED", "warn", str(root), f"Max empty dirs reached: {max_dirs}; truncating")
                        return empties
            except Exception:
                continue
    return empties

def plan_empty_dir_actions(empty_dirs: List[Path]) -> List[PlannedDirAction]:
    out: List[PlannedDirAction] = []
    for d in empty_dirs:
        out.append(PlannedDirAction("RMDIR", str(d), "empty_dir"))
    return out

def apply_empty_dir_actions(actions: List[PlannedDirAction], bumpers_list: List[Bumper], dry_run: bool) -> Dict[str, int]:
    removed = 0
    skipped = 0
    failed = 0
    for a in actions:
        p = Path(a.dir_path)
        try:
            if dry_run:
                removed += 1
                continue
            p.rmdir()
            removed += 1
        except Exception as e:
            failed += 1
            bumper(bumpers_list, "RMDIR_FAIL", "soft", a.dir_path, "Failed removing empty directory", str(e))
    return {"removed": removed, "skipped": skipped, "failed": failed}

# ============================
# Dedupe + version families
# ============================

def dedupe_groups(recs: List[FileRecord]) -> Dict[str, List[FileRecord]]:
    groups: Dict[str, List[FileRecord]] = {}
    for r in recs:
        if r.sha256:
            k = f"sha256:{r.sha256}"
        elif r.quick_sig:
            k = f"qs:{r.size}:{r.quick_sig}"
        else:
            k = f"path:{r.path}"
        groups.setdefault(k, []).append(r)
    return {k: v for k, v in groups.items() if len(v) > 1 and (k.startswith("sha256:") or k.startswith("qs:"))}

def build_duplicates_summary(groups: Dict[str, List[FileRecord]]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for k, members in groups.items():
        canon = choose_canonical(members)
        out[k] = {
            "key": k,
            "count": len(members),
            "canonical": canon.path,
            "members": [m.path for m in members],
            "sha256": canon.sha256,
            "size": canon.size,
        }
    return out

def group_versions(recs: List[FileRecord]) -> Dict[str, List[FileRecord]]:
    fam: Dict[str, List[FileRecord]] = {}
    for r in recs:
        if r.version_family:
            fam.setdefault(r.version_family, []).append(r)
    return {k: v for k, v in fam.items() if len(v) > 1}

def build_versions_summary(fams: Dict[str, List[FileRecord]]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for k, members in fams.items():
        canon = choose_canonical(members)
        out[k] = {
            "family": k,
            "count": len(members),
            "canonical": canon.path,
            "members": [m.path for m in members],
        }
    return out

# ============================
# Canonical View builder (symlink -> hardlink -> copy)
# ============================

def try_symlink(src: Path, dst: Path) -> bool:
    try:
        os.symlink(str(src), str(dst))
        return True
    except Exception:
        return False

def try_hardlink(src: Path, dst: Path) -> bool:
    try:
        os.link(str(src), str(dst))
        return True
    except Exception:
        return False

def build_canonical_view(
    canonical_files: List[Tuple[str, str]],
    out_dir: Path,
    bumpers_list: List[Bumper],
    prefer: str,
    dry_run: bool,
) -> Dict[str, int]:
    """
    canonical_files: list of (source_path, relative_destination)
    prefer: symlink | hardlink | copy
    """
    made = 0
    failed = 0
    skipped = 0

    for src_path, rel in canonical_files:
        src = Path(src_path)
        dst = out_dir / rel
        if dst.exists():
            skipped += 1
            continue
        try:
            safe_mkdir(dst.parent)
            if dry_run:
                made += 1
                continue
            ok = False
            if prefer == "symlink":
                ok = try_symlink(src, dst) or try_hardlink(src, dst) or (shutil.copy2(str(src), str(dst)) is None)
                if not ok and dst.exists():
                    ok = True
            elif prefer == "hardlink":
                ok = try_hardlink(src, dst) or (shutil.copy2(str(src), str(dst)) is None)
                if not ok and dst.exists():
                    ok = True
            else:
                shutil.copy2(str(src), str(dst))
                ok = True
            if ok:
                made += 1
            else:
                failed += 1
                bumper(bumpers_list, "CANONICAL_VIEW_FAIL", "soft", f"{src} -> {dst}", "Failed building canonical view")
        except Exception as e:
            failed += 1
            bumper(bumpers_list, "CANONICAL_VIEW_FAIL", "soft", f"{src} -> {dst}", "Failed building canonical view", str(e))

    return {"made": made, "skipped": skipped, "failed": failed}

# ============================
# ZIP inventory + ForgeZip
# ============================

def inventory_zip(path: Path, bumpers_list: List[Bumper], max_members: int, hash_members: bool) -> List[dict]:
    rows: List[dict] = []
    try:
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
            if len(names) > max_members:
                bumper(bumpers_list, "ZIP_TOO_LARGE", "warn", str(path), f"Zip has {len(names)} members; truncating to {max_members}")
                names = names[:max_members]
            for nm in names:
                try:
                    info = z.getinfo(nm)
                    row = {
                        "zip_path": str(path),
                        "member": nm,
                        "size": int(info.file_size),
                        "mtime": f"{info.date_time[0]:04d}-{info.date_time[1]:02d}-{info.date_time[2]:02d}T{info.date_time[3]:02d}:{info.date_time[4]:02d}:{info.date_time[5]:02d}",
                        "is_dir": nm.endswith("/"),
                        "sha256": "",
                    }
                    if hash_members and not row["is_dir"]:
                        with z.open(nm, "r") as f:
                            h = hashlib.sha256()
                            while True:
                                b = f.read(1024 * 1024)
                                if not b:
                                    break
                                h.update(b)
                            row["sha256"] = h.hexdigest()
                    rows.append(row)
                except Exception as e:
                    bumper(bumpers_list, "ZIP_MEMBER_FAIL", "soft", f"{path}::{nm}", "Failed reading zip member", str(e))
    except Exception as e:
        bumper(bumpers_list, "ZIP_OPEN_FAIL", "warn", str(path), "Failed opening zip", str(e))
    return rows

def forge_zip(
    out_zip: Path,
    items: List[Tuple[Path, str]],
    bumpers_list: List[Bumper],
    dry_run: bool,
) -> Dict[str, object]:
    """
    items: (src_path, arcname)
    Writes:
      - manifest.json inside zip
      - integrity test via zipfile.testzip
    """
    safe_mkdir(out_zip.parent)
    bundle_uid = stable_id(str(out_zip) + "|" + now_iso())
    manifest = {
        "bundle_uid": bundle_uid,
        "created_utc": now_iso(),
        "items": [],
    }

    if dry_run:
        return {"zip": str(out_zip), "bundle_uid": bundle_uid, "items": len(items), "dry_run": True}

    try:
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for src, arc in items:
                try:
                    if not src.exists():
                        bumper(bumpers_list, "FORGE_SRC_MISSING", "warn", str(src), "ForgeZip source missing; skipping")
                        continue
                    z.write(str(src), arcname=arc)
                    c32 = crc32_file(src)
                    st = src.stat()
                    manifest["items"].append({
                        "src": str(src),
                        "arcname": arc,
                        "bytes": int(st.st_size),
                        "mtime": int(st.st_mtime),
                        "crc32": int(c32),
                        "integrity_key": integrity_key(bundle_uid, arc, int(c32), int(st.st_size), int(st.st_mtime)),
                    })
                except Exception as e:
                    bumper(bumpers_list, "FORGE_WRITE_FAIL", "warn", f"{src} -> {arc}", "ForgeZip write failed", str(e))

            # embed manifest
            z.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))

        # integrity test
        with zipfile.ZipFile(out_zip, "r") as z:
            bad = z.testzip()
            if bad:
                bumper(bumpers_list, "FORGE_ZIP_TEST_FAIL", "warn", str(out_zip), "Zip integrity test failed", f"bad_member={bad}")
            else:
                # ok
                pass

    except Exception as e:
        bumper(bumpers_list, "FORGE_ZIP_FAIL", "warn", str(out_zip), "ForgeZip failed", str(e))

    return {"zip": str(out_zip), "bundle_uid": bundle_uid, "items": len(manifest["items"]), "dry_run": False}

# ============================
# Content scan (lite)
# ============================

def content_scan_file(path: Path, terms: List[str], max_bytes: int) -> List[dict]:
    hits: List[dict] = []
    raw = b""
    try:
        raw = path.read_bytes()
    except Exception:
        return hits
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    txt = raw.decode("utf-8", errors="ignore")
    lower = txt.lower()
    for t in terms:
        t0 = t.lower()
        idx = lower.find(t0)
        if idx != -1:
            excerpt = txt[max(0, idx - 80): idx + len(t0) + 80].replace("\n", " ")
            hits.append({"term": t, "offset": idx, "excerpt": excerpt})
    return hits

def load_terms(terms_file: str, cap: int) -> List[str]:
    terms = list(DEFAULT_NEGATIVE_TERMS)
    if terms_file:
        p = Path(terms_file)
        if p.exists():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    t = line.strip()
                    if t:
                        terms.append(t)
            except Exception:
                pass
    out = []
    seen = set()
    for t in terms:
        tl = t.lower().strip()
        if not tl or tl in seen:
            continue
        seen.add(tl)
        out.append(t.strip())
        if len(out) >= cap:
            break
    return out

def content_scan(recs: List[FileRecord], bumpers_list: List[Bumper], terms: List[str], include_exts: Set[str], max_bytes: int) -> Tuple[dict, List[dict]]:
    rows: List[dict] = []
    scanned = 0
    matched_files = 0
    for r in recs:
        p = Path(r.path)
        if p.suffix.lower() not in include_exts:
            continue
        scanned += 1
        try:
            hits = content_scan_file(p, terms, max_bytes=max_bytes)
            if hits:
                matched_files += 1
                rows.append({"path": r.path, "bucket": r.bucket, "ext": r.ext, "doctype": r.doctype, "case_no": r.case_no, "matches": hits})
        except Exception as e:
            bumper(bumpers_list, "CONTENT_SCAN_FAIL", "soft", r.path, "Content scan failed", str(e))
    summary = {"scanned_files": scanned, "matched_files": matched_files}
    return summary, rows

# ============================
# Fingerprints for convergence
# ============================

def fingerprint_inventory(recs: List[FileRecord]) -> str:
    items = [f"{r.path}|{r.size}|{int(r.mtime)}|{r.bucket}" for r in recs]
    items.sort()
    return hashlib.sha1("\n".join(items).encode("utf-8", errors="ignore")).hexdigest()

def fingerprint_plan(actions: List[PlannedAction]) -> str:
    items = [f"{a.action}|{a.src}|{a.dst}|{a.bucket}" for a in actions]
    items.sort()
    return hashlib.sha1("\n".join(items).encode("utf-8", errors="ignore")).hexdigest()

# ============================
# Graph model + pipeline + virtualization + ERD superset + physics + HTML
# ============================

def _node_id(label: str, key: str) -> str:
    return f"{label}::{key}"

def _edge_id(edge_type: str, start_id: str, end_id: str, salt: str = "") -> str:
    base = f"{edge_type}|{start_id}|{end_id}|{salt}"
    return hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()[:20]

# (graph/erd related helpers omitted in docstring - kept below)

def build_erd_superset_schema() -> Dict[str, dict]:
    entities = [
        {"name": "Run", "pk": "run_id", "fields": ["run_id"]},
        {"name": "Drive", "pk": "drive", "fields": ["drive"]},
        {"name": "Folder", "pk": "folder_id", "fields": ["folder_id", "path"]},
        {"name": "Bucket", "pk": "bucket_id", "fields": ["bucket_id", "desc"]},
        {"name": "File", "pk": "file_id", "fields": ["file_id", "path", "ext", "size", "mtime", "sha256", "quick_sig", "version_family", "version_score", "doctype", "case_no", "date_hint", "tags", "rule_id", "rule_reason", "notes"]},
        {"name": "Action", "pk": "action_id", "fields": ["action_id", "action", "src", "dst", "reason", "bucket", "size"]},
        {"name": "DirAction", "pk": "dir_action_id", "fields": ["dir_action_id", "action", "dir_path", "reason"]},
        {"name": "Bumper", "pk": "bumper_id", "fields": ["bumper_id", "id", "ts", "severity", "where", "msg", "mv", "detail"]},
        {"name": "DuplicateGroup", "pk": "group_id", "fields": ["group_id", "count", "canonical", "sha256", "size"]},
        {"name": "VersionFamily", "pk": "family", "fields": ["family", "count", "canonical"]},
        {"name": "ZipMember", "pk": "zip_member_id", "fields": ["zip_member_id", "zip_path", "member", "size", "mtime", "is_dir", "sha256"]},
        {"name": "ContentFlag", "pk": "flag_id", "fields": ["flag_id", "path", "bucket", "ext", "matches"]},

        # Litigation / federal overlay (schema-first placeholders for later ingest)
        {"name": "Court", "pk": "court_id", "fields": ["court_id", "name", "jurisdiction", "level"]},
        {"name": "Case", "pk": "case_id", "fields": ["case_id", "caption", "court_id", "docket_no", "case_type", "opened", "closed"]},
        {"name": "Party", "pk": "party_id", "fields": ["party_id", "name", "role", "side"]},
        {"name": "Filing", "pk": "filing_id", "fields": ["filing_id", "case_id", "date_filed", "title", "doc_type", "roa_no"]},
        {"name": "Order", "pk": "order_id", "fields": ["order_id", "case_id", "date_entered", "title", "signed_by", "text_hash"]},
        {"name": "Transcript", "pk": "transcript_id", "fields": ["transcript_id", "case_id", "hearing_date", "reporter", "text_hash"]},
        {"name": "Exhibit", "pk": "exhibit_id", "fields": ["exhibit_id", "case_id", "label", "party", "description", "source_file_id", "integrity_key"]},
        {"name": "Event", "pk": "event_id", "fields": ["event_id", "case_id", "ts", "event_type", "summary", "source_file_id"]},
        {"name": "Authority", "pk": "authority_id", "fields": ["authority_id", "type", "cite", "pinpoint", "snapshot_id"]},
        {"name": "Form", "pk": "form_id", "fields": ["form_id", "name", "jurisdiction", "url", "version"]},
        {"name": "MisconductVector", "pk": "mv_id", "fields": ["mv_id", "category", "signals", "proof", "remedies"]},
        {"name": "Remedy", "pk": "remedy_id", "fields": ["remedy_id", "name", "vehicle", "standard", "authority_id"]},
        {"name": "Claim", "pk": "claim_id", "fields": ["claim_id", "case_id", "theory", "elements", "authority_id"]},
    ]
    relationships = [
        {"type": "RUN_SAW", "from": "Run", "to": "File"},
        {"type": "RUN_PLANNED", "from": "Run", "to": "Action"},
        {"type": "RUN_PLANNED_DIR", "from": "Run", "to": "DirAction"},
        {"type": "RUN_BUMPER", "from": "Run", "to": "Bumper"},
        {"type": "RUN_DUPGROUP", "from": "Run", "to": "DuplicateGroup"},
        {"type": "RUN_VERSIONFAM", "from": "Run", "to": "VersionFamily"},
        {"type": "IN_BUCKET", "from": "File", "to": "Bucket"},
        {"type": "ON_DRIVE", "from": "File", "to": "Drive"},
        {"type": "IN_FOLDER", "from": "File", "to": "Folder"},
        {"type": "ACTION_ON", "from": "Action", "to": "File"},
        {"type": "DUP_MEMBER", "from": "DuplicateGroup", "to": "File"},
        {"type": "VF_MEMBER", "from": "VersionFamily", "to": "File"},
        {"type": "ZIP_CONTAINS", "from": "File", "to": "ZipMember"},
        {"type": "FLAG_ON", "from": "ContentFlag", "to": "File"},

        # Overlay
        {"type": "CASE_IN_COURT", "from": "Case", "to": "Court"},
        {"type": "PARTY_IN_CASE", "from": "Party", "to": "Case"},
        {"type": "FILING_IN_CASE", "from": "Filing", "to": "Case"},
        {"type": "ORDER_IN_CASE", "from": "Order", "to": "Case"},
        {"type": "TRANSCRIPT_IN_CASE", "from": "Transcript", "to": "Case"},
        {"type": "EXHIBIT_IN_CASE", "from": "Exhibit", "to": "Case"},
        {"type": "EVENT_IN_CASE", "from": "Event", "to": "Case"},
        {"type": "EXHIBIT_SOURCE_FILE", "from": "Exhibit", "to": "File"},
        {"type": "EVENT_SOURCE_FILE", "from": "Event", "to": "File"},
        {"type": "FILING_SOURCE_FILE", "from": "Filing", "to": "File"},
        {"type": "CLAIM_IN_CASE", "from": "Claim", "to": "Case"},
        {"type": "CLAIM_AUTHORITY", "from": "Claim", "to": "Authority"},
        {"type": "REMEDY_AUTHORITY", "from": "Remedy", "to": "Authority"},
        {"type": "USES_FORM", "from": "Filing", "to": "Form"},
        {"type": "ASSERTS_MV", "from": "Event", "to": "MisconductVector"},
        {"type": "SEEKS_REMEDY", "from": "Claim", "to": "Remedy"},
    ]
    return {"name": "ERD_SUPERSET_FED_BLUEPRINT", "version": APP_VER, "entities": entities, "relationships": relationships}

# ============================
# Graph helpers
# ============================

def build_graph_model(
    run_id: str,
    records: List[FileRecord],
    actions: List[PlannedAction],
    dir_actions: List[PlannedDirAction],
    bumpers_list: List[Bumper],
    dup_summary: Dict[str, dict],
    versions_summary: Dict[str, dict],
    zip_rows: List[dict],
    content_flags: List[dict],
) -> Tuple[List[dict], List[dict], Dict[str, dict]]:
    nodes: Dict[str, dict] = {}
    edges: Dict[str, dict] = {}

    run_node = _node_id("Run", run_id)
    nodes[run_node] = {"node_id": run_node, "label": "Run", "props": {"run_id": run_id}}

    for bid, desc in DEFAULT_BUCKETS:
        nid = _node_id("Bucket", bid)
        nodes[nid] = {"node_id": nid, "label": "Bucket", "props": {"bucket_id": bid, "desc": desc}}

    for r in records:
        file_id = stable_id(r.path)
        fn = _node_id("File", file_id)
        nodes[fn] = {"node_id": fn, "label": "File", "props": {
            "file_id": file_id,
            "path": r.path,
            "ext": r.ext,
            "size": r.size,
            "mtime": r.mtime,
            "sha256": r.sha256,
            "quick_sig": r.quick_sig,
            "version_family": r.version_family,
            "version_score": r.version_score,
            "doctype": r.doctype,
            "case_no": r.case_no,
            "date_hint": r.date_hint,
            "tags": r.tags,
            "rule_id": r.rule_id,
            "rule_reason": r.rule_reason,
            "notes": r.notes,
        }}

        edges[_edge_id("RUN_SAW", run_node, fn)] = {"edge_id": _edge_id("RUN_SAW", run_node, fn), "type": "RUN_SAW", "start_id": run_node, "end_id": fn, "props": {}}
        b = _node_id("Bucket", r.bucket)
        edges[_edge_id("IN_BUCKET", fn, b)] = {"edge_id": _edge_id("IN_BUCKET", fn, b), "type": "IN_BUCKET", "start_id": fn, "end_id": b, "props": {}}

        drv = r.drive or "ROOT"
        dn = _node_id("Drive", drv)
        if dn not in nodes:
            nodes[dn] = {"node_id": dn, "label": "Drive", "props": {"drive": drv}}
        edges[_edge_id("ON_DRIVE", fn, dn)] = {"edge_id": _edge_id("ON_DRIVE", fn, dn), "type": "ON_DRIVE", "start_id": fn, "end_id": dn, "props": {}}

        parent = str(Path(r.path).parent)
        fol_id = stable_id(parent)
        fol = _node_id("Folder", fol_id)
        if fol not in nodes:
            nodes[fol] = {"node_id": fol, "label": "Folder", "props": {"folder_id": fol_id, "path": parent}}
        edges[_edge_id("IN_FOLDER", fn, fol)] = {"edge_id": _edge_id("IN_FOLDER", fn, fol), "type": "IN_FOLDER", "start_id": fn, "end_id": fol, "props": {}}

    for a in actions:
        aid = stable_id(f"{a.action}|{a.src}|{a.dst}|{a.reason}")
        an = _node_id("Action", aid)
        nodes[an] = {"node_id": an, "label": "Action", "props": asdict(a) | {"action_id": aid}}
        edges[_edge_id("RUN_PLANNED", run_node, an)] = {"edge_id": _edge_id("RUN_PLANNED", run_node, an), "type": "RUN_PLANNED", "start_id": run_node, "end_id": an, "props": {}}
        fn = _node_id("File", stable_id(a.src))
        edges[_edge_id("ACTION_ON", an, fn)] = {"edge_id": _edge_id("ACTION_ON", an, fn), "type": "ACTION_ON", "start_id": an, "end_id": fn, "props": {}}

    for da in dir_actions:
        did = stable_id(f"{da.action}|{da.dir_path}|{da.reason}")
        dn = _node_id("DirAction", did)
        nodes[dn] = {"node_id": dn, "label": "DirAction", "props": {"dir_action_id": did, "action": da.action, "dir_path": da.dir_path, "reason": da.reason}}
        edges[_edge_id("RUN_PLANNED_DIR", run_node, dn)] = {"edge_id": _edge_id("RUN_PLANNED_DIR", run_node, dn), "type": "RUN_PLANNED_DIR", "start_id": run_node, "end_id": dn, "props": {}}

    for b in bumpers_list:
        bid = stable_id(b.ts + "|" + b.id + "|" + b.where + "|" + b.msg)
        bn = _node_id("Bumper", bid)
        nodes[bn] = {"node_id": bn, "label": "Bumper", "props": {"bumper_id": bid, "id": b.id, "ts": b.ts, "severity": b.severity, "where": b.where, "msg": b.msg, "detail": b.detail, "mv": b.mv}}
        edges[_edge_id("RUN_BUMPER", run_node, bn)] = {"edge_id": _edge_id("RUN_BUMPER", run_node, bn), "type": "RUN_BUMPER", "start_id": run_node, "end_id": bn, "props": {}}

    for gid, g in dup_summary.items():
        dg = _node_id("DuplicateGroup", stable_id(gid))
        nodes[dg] = {"node_id": dg, "label": "DuplicateGroup", "props": g | {"group_id": gid}}
        edges[_edge_id("RUN_DUPGROUP", run_node, dg)] = {"edge_id": _edge_id("RUN_DUPGROUP", run_node, dg), "type": "RUN_DUPGROUP", "start_id": run_node, "end_id": dg, "props": {}}
        for mp in g.get("members", []):
            fn = _node_id("File", stable_id(mp))
            edges[_edge_id("DUP_MEMBER", dg, fn, salt=mp)] = {"edge_id": _edge_id("DUP_MEMBER", dg, fn, salt=mp), "type": "DUP_MEMBER", "start_id": dg, "end_id": fn, "props": {}}

    for fam, g in versions_summary.items():
        vf = _node_id("VersionFamily", stable_id(fam))
        nodes[vf] = {"node_id": vf, "label": "VersionFamily", "props": g | {"family": fam}}
        edges[_edge_id("RUN_VERSIONFAM", run_node, vf)] = {"edge_id": _edge_id("RUN_VERSIONFAM", run_node, vf), "type": "RUN_VERSIONFAM", "start_id": run_node, "end_id": vf, "props": {}}
        for mp in g.get("members", []):
            fn = _node_id("File", stable_id(mp))
            edges[_edge_id("VF_MEMBER", vf, fn, salt=mp)] = {"edge_id": _edge_id("VF_MEMBER", vf, fn, salt=mp), "type": "VF_MEMBER", "start_id": vf, "end_id": fn, "props": {}}

    for zr in zip_rows:
        zpath = zr.get("zip_path", "")
        member = zr.get("member", "")
        zfid = stable_id(zpath)
        zf = _node_id("File", zfid)
        mem_id = stable_id(zpath + "::" + member)
        zm = _node_id("ZipMember", mem_id)
        nodes[zm] = {"node_id": zm, "label": "ZipMember", "props": zr | {"zip_member_id": mem_id}}
        edges[_edge_id("ZIP_CONTAINS", zf, zm, salt=member)] = {"edge_id": _edge_id("ZIP_CONTAINS", zf, zm, salt=member), "type": "ZIP_CONTAINS", "start_id": zf, "end_id": zm, "props": {}}

    for cf in content_flags:
        p = cf.get("path", "")
        if not p:
            continue
        fid = stable_id(p)
        fn = _node_id("File", fid)
        flag_id = stable_id(p + "|" + json.dumps(cf.get("matches", []), sort_keys=True))
        cn = _node_id("ContentFlag", flag_id)
        nodes[cn] = {"node_id": cn, "label": "ContentFlag", "props": {"flag_id": flag_id, "path": p, "bucket": cf.get("bucket", ""), "ext": cf.get("ext", ""), "matches": cf.get("matches", [])}}
        edges[_edge_id("FLAG_ON", cn, fn)] = {"edge_id": _edge_id("FLAG_ON", cn, fn), "type": "FLAG_ON", "start_id": cn, "end_id": fn, "props": {}}

    schema = build_erd_superset_schema()
    return list(nodes.values()), list(edges.values()), schema

# ... (Remaining helper functions are unchanged and fully included below.)

def compute_graph_metrics(nodes: List[dict], edges: List[dict]) -> Dict[str, dict]:
    node_ids = [n["node_id"] for n in nodes]
    idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    out_deg = [0] * n
    und = [[] for _ in range(n)]
    out_adj = [[] for _ in range(n)]
    out_count = [0] * n

    for e in edges:
        u = e["start_id"]; v = e["end_id"]
        if u in idx and v in idx:
            ui = idx[u]; vi = idx[v]
            out_deg[ui] += 1
            und[ui].append(vi); und[vi].append(ui)
            out_adj[ui].append(vi)
            out_count[ui] += 1

    # components
    comp_id = [-1] * n
    comps = []
    cid = 0
    for i in range(n):
        if comp_id[i] != -1:
            continue
        stack = [i]
        comp_id[i] = cid
        comp = [i]
        while stack:
            cur = stack.pop()
            for nb in und[cur]:
                if comp_id[nb] == -1:
                    comp_id[nb] = cid
                    stack.append(nb)
                    comp.append(nb)
        comps.append(comp)
        cid += 1

    # pagerank
    pr = [1.0 / n] * n if n else []
    d = 0.85
    for _ in range(30):
        new = [(1.0 - d) / n] * n if n else []
        for ui in range(n):
            if out_count[ui] == 0:
                share = d * pr[ui] / n if n else 0.0
                for j in range(n):
                    new[j] += share
            else:
                share = d * pr[ui] / out_count[ui]
                for vi in out_adj[ui]:
                    new[vi] += share
        pr = new

    top_pr = sorted([(node_ids[i], pr[i]) for i in range(n)], key=lambda x: -x[1])[:25]
    top_deg = sorted([(node_ids[i], out_deg[i]) for i in range(n)], key=lambda x: -x[1])[:25]
    comp_sizes = sorted([len(c) for c in comps], reverse=True)[:25]

    return {
        "n_nodes": n,
        "n_edges": len(edges),
        "components": {"count": len(comps), "largest_sizes": comp_sizes},
        "top_pagerank": [{"node_id": nid, "pagerank": score} for nid, score in top_pr],
        "top_out_degree": [{"node_id": nid, "out_degree": deg} for nid, deg in top_deg],
    }

def physics_layout(
    nodes: List[dict],
    edges: List[dict],
    mode: str,
    bumpers_list: List[Bumper],
    seed: int,
    iterations: int,
) -> Dict[str, Tuple[float, float]]:
    node_ids = [n["node_id"] for n in nodes]
    n = len(node_ids)
    if n == 0:
        return {}
    if mode == "off":
        return {nid: (0.0, 0.0) for nid in node_ids}

    # degree
    deg = {nid: 0 for nid in node_ids}
    for e in edges:
        deg[e["start_id"]] = deg.get(e["start_id"], 0) + 1
        deg[e["end_id"]] = deg.get(e["end_id"], 0) + 1

    if mode == "fast" or n > 2000:
        if mode == "full" and n > 2000:
            bumper(bumpers_list, "PHYSICS_DEGRADED", "soft", "physics_layout", f"n_nodes={n} too large for full physics; using fast")
        max_deg = max(deg.values()) if deg else 1
        out = {}
        for i, nid in enumerate(node_ids):
            a = (i * 2.0 * math.pi) / max(1, n)
            r = 0.2 + 0.8 * (deg.get(nid, 0) / max_deg)
            out[nid] = (r * math.cos(a), r * math.sin(a))
        return out

    # full physics with numpy optional
    try:
        import numpy as np
    except Exception:
        bumper(bumpers_list, "PHYSICS_NO_NUMPY", "soft", "physics_layout", "numpy missing; using fast")
        return physics_layout(nodes, edges, "fast", bumpers_list, seed=seed, iterations=iterations)

    idx = {nid: i for i, nid in enumerate(node_ids)}
    rs = np.random.RandomState(seed)
    pos = rs.randn(n, 2).astype(np.float64) * 0.05
    vel = np.zeros((n, 2), dtype=np.float64)

    el = []
    for e in edges:
        u = e["start_id"]; v = e["end_id"]
        if u in idx and v in idx and u != v:
            el.append((idx[u], idx[v]))
    if not el:
        return {nid: (float(pos[i, 0]), float(pos[i, 1])) for nid, i in idx.items()}
    el = np.array(el, dtype=np.int32)
    u = el[:, 0]; v = el[:, 1]

    k_rep = 0.002
    k_attr = 0.01
    damp = 0.85
    dt = 0.02

    for _ in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist2 = (delta ** 2).sum(axis=2) + 1e-6
        inv = 1.0 / dist2
        rep = (delta * inv[:, :, None]).sum(axis=1) * k_rep

        dvec = pos[v] - pos[u]
        attr = np.zeros_like(pos)
        attr_u = dvec * k_attr
        np.add.at(attr, u, attr_u)
        np.add.at(attr, v, -attr_u)

        force = rep + attr
        vel = (vel + force * dt) * damp
        pos = pos + vel * dt
        pos = np.clip(pos, -1.5, 1.5)

    return {node_ids[i]: (float(pos[i, 0]), float(pos[i, 1])) for i in range(n)}


def emit_virtual_graph_sqlite(db_path: Path, nodes: List[dict], edges: List[dict]) -> None:
    safe_mkdir(db_path.parent)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("CREATE TABLE nodes (node_id TEXT PRIMARY KEY, label TEXT NOT NULL, props_json TEXT NOT NULL);")
    cur.execute("CREATE INDEX idx_nodes_label ON nodes(label);")
    cur.execute("CREATE TABLE edges (edge_id TEXT PRIMARY KEY, type TEXT NOT NULL, start_id TEXT NOT NULL, end_id TEXT NOT NULL, props_json TEXT NOT NULL);")
    cur.execute("CREATE INDEX idx_edges_type ON edges(type);")
    cur.execute("CREATE INDEX idx_edges_start ON edges(start_id);")
    cur.execute("CREATE INDEX idx_edges_end ON edges(end_id);")
    cur.executemany(
        "INSERT INTO nodes(node_id,label,props_json) VALUES(?,?,?)",
        [(n["node_id"], n["label"], json.dumps(n.get("props", {}), ensure_ascii=False)) for n in nodes],
    )
    cur.executemany(
        "INSERT INTO edges(edge_id,type,start_id,end_id,props_json) VALUES(?,?,?,?,?)",
        [(e["edge_id"], e["type"], e["start_id"], e["end_id"], json.dumps(e.get("props", {}), ensure_ascii=False)) for e in edges],
    )
    con.commit()
    con.close()


def run_virtual_query(db_path: Path, name: str) -> Tuple[List[str], List[Tuple]]:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    queries: Dict[str, str] = {
        "bucket_counts": """
            SELECT json_extract(n.props_json, '$.bucket_id') AS bucket_id, count(*) AS n
            FROM edges e
            JOIN nodes n ON n.node_id = e.end_id
            WHERE e.type='IN_BUCKET' AND n.label='Bucket'
            GROUP BY bucket_id
            ORDER BY n DESC;
        """,
        "doctype_counts": """
            SELECT json_extract(n.props_json, '$.doctype') AS doctype, count(*) AS n
            FROM nodes n
            WHERE n.label='File'
            GROUP BY doctype
            ORDER BY n DESC;
        """,
        "largest_files": """
            SELECT json_extract(props_json,'$.path') AS path, json_extract(props_json,'$.size') AS size
            FROM nodes
            WHERE label='File'
            ORDER BY size DESC
            LIMIT 50;
        """,
        "dup_groups": """
            SELECT json_extract(props_json,'$.group_id') AS group_id, json_extract(props_json,'$.count') AS count, json_extract(props_json,'$.canonical') AS canonical
            FROM nodes
            WHERE label='DuplicateGroup'
            ORDER BY count DESC
            LIMIT 100;
        """,
        "bumpers": """
            SELECT json_extract(props_json,'$.severity') AS severity, json_extract(props_json,'$.id') AS id, count(*) AS n
            FROM nodes
            WHERE label='Bumper'
            GROUP BY severity, id
            ORDER BY n DESC;
        """,
    }
    q = queries.get(name.strip(), "")
    if not q:
        con.close()
        return (["error"], [(f"unknown query '{name}'. options: {', '.join(sorted(queries.keys()))}",)])
    cur.execute(q)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    con.close()
    return cols, rows


def run_virtual_query_sql_file(db_path: Path, sql_file: Path) -> Tuple[List[str], List[Tuple]]:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    sql = sql_file.read_text(encoding="utf-8")
    cur.execute(sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description] if cur.description else []
    con.close()
    return cols, rows


def write_query_results(out_csv: Path, out_json: Path, cols: List[str], rows: List[Tuple]) -> None:
    safe_mkdir(out_csv.parent)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow(list(r))
    write_json(out_json, {"columns": cols, "rows": [list(r) for r in rows]})


def emit_graphml(graphml_path: Path, nodes: List[dict], edges: List[dict]) -> None:
    """
    Minimal GraphML export (portable).
    """
    safe_mkdir(graphml_path.parent)

    def esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                  .replace('"', "&quot;").replace("'", "&apos;"))

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">')
    lines.append('<key id="label" for="node" attr.name="label" attr.type="string"/>')
    lines.append('<key id="type" for="edge" attr.name="type" attr.type="string"/>')
    lines.append('<graph id="G" edgedefault="directed">')
    for n in nodes:
        lines.append(f'  <node id="{esc(n["node_id"])}"><data key="label">{esc(n["label"])}</data></node>')
    for e in edges:
        lines.append(f'  <edge id="{esc(e["edge_id"])}" source="{esc(e["start_id"])}" target="{esc(e["end_id"])}"><data key="type">{esc(e["type"])}</data></edge>')
    lines.append('</graph>')
    lines.append('</graphml>')
    graphml_path.write_text("\n".join(lines), encoding="utf-8")


def emit_html_graph_viewer(html_path: Path, nodes: List[dict], edges: List[dict], positions: Dict[str, Tuple[float, float]], title: str) -> None:
    safe_mkdir(html_path.parent)
    embed_nodes = []
    for n in nodes:
        props = n.get("props", {})
        display = props.get("path") or props.get("bucket_id") or props.get("id") or props.get("family") or props.get("drive") or props.get("run_id") or n["node_id"]
        embed_nodes.append({
            "id": n["node_id"],
            "label": n["label"],
            "display": display,
            "x": positions.get(n["node_id"], (0.0, 0.0))[0],
            "y": positions.get(n["node_id"], (0.0, 0.0))[1],
        })
    embed_edges = [{"s": e["start_id"], "t": e["end_id"], "type": e["type"]} for e in edges]
    payload = {"title": title, "nodes": embed_nodes, "edges": embed_edges}

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<style>
body {{ margin:0; font-family: Arial, sans-serif; }}
#top {{ padding:8px; background:#111; color:#eee; display:flex; gap:8px; align-items:center; }}
#q {{ width:520px; padding:6px; }}
#info {{ padding:8px; font-size:12px; color:#333; }}
canvas {{ display:block; width:100vw; height:calc(100vh - 72px); background:#fff; }}
.badge {{ padding:2px 6px; border:1px solid #444; border-radius:6px; font-size:12px; }}
</style>
</head>
<body>
<div id="top">
  <span class="badge">Graph Viewer</span>
  <span class="badge" id="stats"></span>
  <input id="q" placeholder="search (path, bucket, label) and press Enter"/>
  <span class="badge" id="hit"></span>
</div>
<canvas id="c"></canvas>
<div id="info"></div>
<script>
const DATA = {json.dumps(payload, ensure_ascii=False)};
const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
function resize() {{
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight - 72;
}}
window.addEventListener('resize', resize);
resize();

let scale = 220;
let ox = canvas.width/2;
let oy = canvas.height/2;
let dragging = false;
let lastx=0, lasty=0;
canvas.addEventListener('mousedown', (e)=>{{ dragging=true; lastx=e.clientX; lasty=e.clientY; }});
canvas.addEventListener('mouseup', ()=> dragging=false);
canvas.addEventListener('mouseleave', ()=> dragging=false);
canvas.addEventListener('mousemove', (e)=>{{
  if(dragging) {{
    ox += (e.clientX-lastx);
    oy += (e.clientY-lasty);
    lastx=e.clientX; lasty=e.clientY;
    draw();
  }}
}});
canvas.addEventListener('wheel', (e)=>{{
  e.preventDefault();
  const s = Math.exp(-e.deltaY*0.001);
  scale *= s;
  scale = Math.max(30, Math.min(1000, scale));
  draw();
}}, {{ passive:false }});

function worldToScreen(x,y) {{
  return [ox + x*scale, oy + y*scale];
}}
const NMAP = new Map();
for(const n of DATA.nodes) NMAP.set(n.id, n);

let HIT = null;
function draw() {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.globalAlpha = 0.22;
  ctx.strokeStyle = '#999';
  for(const e of DATA.edges) {{
    const a = NMAP.get(e.s);
    const b = NMAP.get(e.t);
    if(!a || !b) continue;
    const [x1,y1] = worldToScreen(a.x, a.y);
    const [x2,y2] = worldToScreen(b.x, b.y);
    ctx.beginPath();
    ctx.moveTo(x1,y1);
    ctx.lineTo(x2,y2);
    ctx.stroke();
  }}
  ctx.globalAlpha = 1.0;
  for(const n of DATA.nodes) {{
    const [x,y] = worldToScreen(n.x, n.y);
    ctx.fillStyle = (HIT && n.id===HIT.id) ? '#ff3b30' : '#111';
    ctx.beginPath();
    ctx.arc(x,y,3.0,0,Math.PI*2);
    ctx.fill();
  }}
}}
document.getElementById('stats').textContent = `${{DATA.nodes.length}} nodes / ${{DATA.edges.length}} edges`;

const q = document.getElementById('q');
q.addEventListener('keydown', (e)=>{{
  if(e.key !== 'Enter') return;
  const term = q.value.trim().toLowerCase();
  HIT = null;
  if(!term) {{
    document.getElementById('hit').textContent='';
    document.getElementById('info').textContent='';
    draw();
    return;
  }}
  for(const n of DATA.nodes) {{
    if(String(n.display).toLowerCase().includes(term) || String(n.label).toLowerCase().includes(term) || String(n.id).toLowerCase().includes(term)) {{
      HIT = n; break;
    }}
  }}
  if(HIT) {{
    ox = canvas.width/2 - HIT.x*scale;
    oy = canvas.height/2 - HIT.y*scale;
    document.getElementById('hit').textContent = `hit: ${{HIT.label}}`;
    document.getElementById('info').textContent = HIT.display;
  }} else {{
    document.getElementById('hit').textContent = 'no hit';
    document.getElementById('info').textContent = '';
  }}
  draw();
}});
draw();
</script>
</body>
</html>"""
    html_path.write_text(html, encoding="utf-8")


def emit_erd_superset_files(out_dir: Path, schema: Dict[str, dict], bumpers_list: List[Bumper], physics_mode: str) -> None:
    safe_mkdir(out_dir)
    write_json(out_dir / "ERD_SUPERSET_FED_BLUEPRINT.json", schema)

    md = []
    md.append("# ERD_SUPERSET_FED_BLUEPRINT")
    md.append("")
    md.append("## I. Version")
    md.append(f"- {schema.get('version','')}")
    md.append("")
    md.append("## II. Entities")
    for ent in schema.get("entities", []):
        md.append(f"### {ent['name']}")
        md.append(f"- PK: `{ent['pk']}`")
        md.append("- Fields: " + ", ".join(f"`{f}`" for f in ent.get("fields", [])))
        md.append("")
    md.append("## III. Relationships")
    for rel in schema.get("relationships", []):
        md.append(f"- `{rel['type']}`: {rel['from']} -> {rel['to']}")
    (out_dir / "ERD_SUPERSET_FED_BLUEPRINT.md").write_text("\n".join(md), encoding="utf-8")

    dot = []
    dot.append("digraph ERD {")
    dot.append("  rankdir=LR;")
    dot.append("  node [shape=box, fontsize=10];")
    for ent in schema.get("entities", []):
        dot.append(f'  "{ent["name"]}";')
    for rel in schema.get("relationships", []):
        dot.append(f'  "{rel["from"]}" -> "{rel["to"]}" [label="{rel["type"]}"];')
    dot.append("}")
    (out_dir / "ERD_SUPERSET_FED_BLUEPRINT.dot").write_text("\n".join(dot), encoding="utf-8")

    # HTML ERD viewer via physics layout
    if physics_mode != "off":
        nlist = [{"node_id": _node_id("Entity", ent["name"]), "label": "Entity", "props": {"name": ent["name"]}} for ent in schema.get("entities", [])]
        elist = []
        for rel in schema.get("relationships", []):
            s = _node_id("Entity", rel["from"])
            t = _node_id("Entity", rel["to"])
            elist.append({"edge_id": _edge_id(rel["type"], s, t), "type": rel["type"], "start_id": s, "end_id": t, "props": {}})
        pos = physics_layout(nlist, elist, mode=physics_mode, bumpers_list=bumpers_list, seed=777, iterations=260)
        emit_html_graph_viewer(out_dir / "ERD_SUPERSET_FED_BLUEPRINT.html", nlist, elist, pos, title="ERD_SUPERSET_FED_BLUEPRINT")


def emit_neo4j_import_pack(out_dir: Path, nodes: List[dict], edges: List[dict]) -> None:
    """
    Executive-grade Neo4j import pack:
      - nodes.csv (id, label, props_json)
      - edges.csv (edge_id, type, start_id, end_id, props_json)
      - constraints.cypher + indexes.cypher
      - import.cypher (LOAD CSV recipes)
    """
    safe_mkdir(out_dir)
    nodes_csv = out_dir / "nodes.csv"
    edges_csv = out_dir / "edges.csv"

    with nodes_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id:ID", "label:LABEL", "props_json"])
        for n in nodes:
            w.writerow([n["node_id"], n["label"], json.dumps(n.get("props", {}), ensure_ascii=False)])

    with edges_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["edge_id:ID", "type", "start_id:START_ID", "end_id:END_ID", "props_json"])
        for e in edges:
            w.writerow([e["edge_id"], e["type"], e["start_id"], e["end_id"], json.dumps(e.get("props", {}), ensure_ascii=False)])

    constraints = "\n".join([
        "CREATE CONSTRAINT node_id_unique IF NOT EXISTS FOR (n) REQUIRE n.node_id IS UNIQUE;",
    ])
    indexes = "\n".join([
        "CREATE INDEX node_label IF NOT EXISTS FOR (n) ON (n.label);",
    ])
    (out_dir / "constraints.cypher").write_text(constraints + "\n", encoding="utf-8")
    (out_dir / "indexes.cypher").write_text(indexes + "\n", encoding="utf-8")

    import_cypher = "\n".join([
        "/* Import nodes */",
        "LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row",
        "MERGE (n {node_id: row['node_id:ID']})",
        "SET n.label = row['label:LABEL'], n.props_json = row.props_json;",
        "",
        "/* Import edges */",
        "LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row",
        "MATCH (a {node_id: row['start_id:START_ID']}), (b {node_id: row['end_id:END_ID']})",
        "MERGE (a)-[r:EDGE {edge_id: row['edge_id:ID']}]->(b)",
        "SET r.type = row.type, r.props_json = row.props_json;",
    ])
    (out_dir / "import.cypher").write_text(import_cypher + "\n", encoding="utf-8")

# ============================
# Bucket truth table + drive inventory
# ============================

def bucket_rules_truth_table() -> Tuple[List[dict], List[List[object]]]:
    rows: List[dict] = []

    def add(bucket: str, rule_id: str, rule_desc: str, match: str) -> None:
        rows.append({"bucket": bucket, "rule_id": rule_id, "rule_desc": rule_desc, "match": match})

    for ext in sorted(TEXT_EXT):
        add("B01_TEXT", "R_EXT_TEXT", "extension in text set", ext)
    for ext in sorted(DATA_EXT):
        add("B02_DATA", "R_EXT_DATA", "extension in data set", ext)
    for ext in sorted(DOC_EXT):
        add("B03_DOCS", "R_EXT_DOC", "extension in docs set", ext)
    for ext in sorted(CODE_EXT):
        add("B04_CODE", "R_EXT_CODE", "extension in code set", ext)
    for ext in sorted(WEB_EXT):
        add("B05_WEB", "R_EXT_WEB", "extension in web set", ext)
    for ext in sorted(IMG_EXT):
        add("B06_IMAGES", "R_EXT_IMAGE", "extension in image set", ext)
    for ext in sorted(AUD_EXT):
        add("B07_AUDIO", "R_EXT_AUDIO", "extension in audio set", ext)
    for ext in sorted(VID_EXT):
        add("B08_VIDEO", "R_EXT_VIDEO", "extension in video set", ext)
    for ext in sorted(ARC_EXT):
        add("B09_ARCHIVES", "R_EXT_ARCHIVE", "extension in archive set", ext)
    for ext in sorted(EXE_EXT):
        add("B10_EXEC", "R_EXT_EXEC", "extension in exec set", ext)
    for ext in sorted(FONT_EXT):
        add("B11_FONTS", "R_EXT_FONT", "extension in font set", ext)
    for ext in sorted(CFG_EXT):
        add("B12_CONFIG", "R_EXT_CONFIG", "extension in config set", ext)
    for ext in sorted(EMAIL_EXT):
        add("B13_EMAIL", "R_EXT_EMAIL", "extension in email set", ext)

    add("B14_OCR_QUEUE", "R_PDF_OCR_HEUR", "pdf OCR heuristic", ".pdf + low_text_signal")
    add("B15_MISC", "R_FALLBACK", "fallback", "any")

    csv_rows = [[r["bucket"], r["rule_id"], r["rule_desc"], r["match"]] for r in rows]
    return rows, csv_rows


def build_drive_inventory(recs: List[FileRecord]) -> Tuple[List[dict], List[List[object]]]:
    rollup: Dict[str, Dict[str, Dict[str, int]]] = {}
    for r in recs:
        rollup.setdefault(r.drive, {}).setdefault(r.bucket, {"files": 0, "bytes": 0})
        rollup[r.drive][r.bucket]["files"] += 1
        rollup[r.drive][r.bucket]["bytes"] += r.size

    rows: List[dict] = []
    csv_rows: List[List[object]] = []
    for drive, buckets in sorted(rollup.items()):
        for bucket, vals in sorted(buckets.items()):
            row = {"drive": drive, "bucket": bucket, "files": vals["files"], "bytes": vals["bytes"]}
            rows.append(row)
            csv_rows.append([drive, bucket, vals["files"], vals["bytes"]])
    return rows, csv_rows

# ============================
# Provenance index + manifest
# ============================

def build_run_manifest(run_dir: Path, run_id: str) -> Dict[str, dict]:
    items = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = str(path.relative_to(run_dir))
        st = path.stat()
        crc = crc32_file(path)
        items.append({
            "path": rel,
            "bytes": int(st.st_size),
            "mtime": int(st.st_mtime),
            "crc32": int(crc),
            "integrity_key": integrity_key(run_id, rel, int(crc), int(st.st_size), int(st.st_mtime)),
        })
    return {"run_id": run_id, "items": items}

# ============================
# Main
# ============================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LITIGATIONOS Drive Bucket Forger")
    p.add_argument("--roots", nargs="+", required=True, help="Roots to scan")
    p.add_argument("--out-root", required=True, help="Output root")

    p.add_argument("--apply", action="store_true", help="Apply plan (move/copy)")
    p.add_argument("--move", action="store_true", help="Move files (default when --apply)")
    p.add_argument("--copy", action="store_true", help="Copy files (when --apply)")
    p.add_argument("--dry-run", action="store_true", help="Do not mutate filesystem")
    p.add_argument("--skip-existing", action="store_true", help="Skip if dst exists")
    p.add_argument("--verify-existing-hash", action="store_true", help="Verify hash of existing dst")

    p.add_argument("--hash", action="store_true", help="Compute sha256 hashes")
    p.add_argument("--quick-sig", action="store_true", help="Compute quick signatures")
    p.add_argument("--ocr-queue", action="store_true", help="Enable OCR queue bucket for PDFs")
    p.add_argument("--include-hidden", action="store_true", help="Include hidden files")
    p.add_argument("--max-files", type=int, default=0, help="Max files to scan (0=unlimited)")
    p.add_argument("--max-size-mb", type=int, default=0, help="Skip files larger than this (MB) if >0")

    p.add_argument("--converge", action="store_true", help="Run convergence cycles")
    p.add_argument("--stable-n", type=int, default=2, help="Stable cycles required for converge")
    p.add_argument("--max-cycles", type=int, default=10, help="Max cycles in convergence")

    p.add_argument("--plan-empty-dirs", action="store_true", help="Plan empty directory removal")
    p.add_argument("--apply-empty-dirs", action="store_true", help="Apply empty directory removal")
    p.add_argument("--max-empty-dirs", type=int, default=0, help="Max empty dirs to consider")

    p.add_argument("--zip-inventory", action="store_true", help="Inventory zip members")
    p.add_argument("--zip-hash-members", action="store_true", help="Hash zip members")
    p.add_argument("--zip-max-members", type=int, default=50000, help="Max zip members to inspect")
    p.add_argument("--forge-zip", action="store_true", help="Forge canonical bundle zip")

    p.add_argument("--canonical-view", action="store_true", help="Build canonical view")
    p.add_argument("--canonical-prefer", choices=["symlink", "hardlink", "copy"], default="symlink")

    p.add_argument("--graph-pipeline", action="store_true", help="Emit graph pipeline")
    p.add_argument("--virtualize", action="store_true", help="Emit SQLite virtual graph")
    p.add_argument("--virtual-query", default="", help="Run built-in virtual graph query")
    p.add_argument("--virtual-query-file", default="", help="Run SQL file against virtual graph")
    p.add_argument("--erd-superset", action="store_true", help="Emit ERD_SUPERSET_FED_BLUEPRINT")
    p.add_argument("--physics-layout", choices=["off", "fast", "full"], default="fast")

    p.add_argument("--content-scan", action="store_true", help="Enable content scan (lite)")
    p.add_argument("--terms-file", default="", help="Additional terms file")
    p.add_argument("--terms-cap", type=int, default=250, help="Max terms loaded")
    p.add_argument("--scan-exts", default=".txt,.md,.rtf,.log,.csv,.json,.jsonl", help="Extensions to scan")
    p.add_argument("--scan-max-bytes", type=int, default=2_000_000, help="Max bytes per file for scan")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    roots = [Path(r) for r in args.roots]
    out_root = Path(args.out_root)

    run_id = f"RUN_{_dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')}"
    run_dir = out_root / "RUN" / run_id
    safe_mkdir(run_dir)

    travel_bumpers = out_root / "_TRAVELING" / "BUMPERS_TRAVELING.jsonl"
    ledger_path = run_dir / "run_ledger.jsonl"
    live_log_path = run_dir / "LIVE_PRESERVATION_LOG.jsonl"

    stable_streak = 0
    prev_inv_fp = ""
    prev_plan_fp = ""

    cycles = args.max_cycles if args.converge else 1

    final_records: List[FileRecord] = []
    final_actions: List[PlannedAction] = []
    final_dir_actions: List[PlannedDirAction] = []
    final_bumpers: List[Bumper] = []

    for cycle in range(1, cycles + 1):
        bumpers_list: List[Bumper] = []
        append_jsonl(ledger_path, {"run_id": run_id, "cycle": cycle, "ts": now_iso(), "phase": "CYCLE_START"})
        append_jsonl(live_log_path, {"run_id": run_id, "cycle": cycle, "ts": now_iso(), "phase": "CYCLE_START"})

        records = scan_inventory(
            roots=roots,
            bumpers_list=bumpers_list,
            compute_hashes=args.hash,
            compute_quick_sig=args.quick_sig,
            ocr_queue_enabled=args.ocr_queue,
            include_hidden=args.include_hidden,
            max_files=args.max_files,
            max_size_mb=args.max_size_mb,
        )

        append_jsonl(ledger_path, {"run_id": run_id, "cycle": cycle, "ts": now_iso(), "phase": "SCAN_DONE", "files": len(records)})
        append_jsonl(live_log_path, {"run_id": run_id, "cycle": cycle, "ts": now_iso(), "phase": "SCAN_DONE", "files": len(records)})

        actions = plan_bucket_paths(
            out_root=out_root,
            recs=records,
            bumpers_list=bumpers_list,
            skip_existing=args.skip_existing,
            verify_existing_hash=args.verify_existing_hash,
        )

        append_jsonl(ledger_path, {"run_id": run_id, "cycle": cycle, "ts": now_iso(), "phase": "PLAN_DONE", "actions": len(actions)})
        append_jsonl(live_log_path, {"run_id": run_id, "cycle": cycle, "ts": now_iso(), "phase": "PLAN_DONE", "actions": len(actions)})

        inv_fp = fingerprint_inventory(records)
        plan_fp = fingerprint_plan(actions)
        append_jsonl(ledger_path, {"run_id": run_id, "cycle": cycle, "ts": now_iso(), "phase": "FINGERPRINTS", "inv_fp": inv_fp, "plan_fp": plan_fp})
        append_jsonl(live_log_path, {"run_id": run_id, "cycle": cycle, "ts": now_iso(), "phase": "FINGERPRINTS", "inv_fp": inv_fp, "plan_fp": plan_fp})

        if inv_fp == prev_inv_fp and plan_fp == prev_plan_fp:
            stable_streak += 1
        else:
            stable_streak = 0

        prev_inv_fp = inv_fp
        prev_plan_fp = plan_fp

        for b in bumpers_list:
            emit_traveling_bumper(travel_bumpers, b, run_id=run_id, cycle=cycle)

        final_records = records
        final_actions = actions
        final_bumpers = bumpers_list

        if args.converge and stable_streak >= args.stable_n:
            append_jsonl(ledger_path, {"run_id": run_id, "cycle": cycle, "ts": now_iso(), "phase": "CONVERGED", "stable_streak": stable_streak})
            append_jsonl(live_log_path, {"run_id": run_id, "cycle": cycle, "ts": now_iso(), "phase": "CONVERGED", "stable_streak": stable_streak})
            break

    # Apply plan (optional)
    apply_stats = {"moved": 0, "copied": 0, "skipped": 0, "failed": 0}
    if args.apply:
        mode = "copy" if args.copy else "move"
        apply_stats = apply_plan(final_actions, final_bumpers, mode=mode, dry_run=args.dry_run)

    # Empty directories
    if args.plan_empty_dirs:
        empty_dirs = find_empty_dirs(roots, final_bumpers, include_hidden=args.include_hidden, max_dirs=args.max_empty_dirs)
        final_dir_actions = plan_empty_dir_actions(empty_dirs)
        if args.apply_empty_dirs:
            apply_empty_dir_actions(final_dir_actions, final_bumpers, dry_run=args.dry_run)

    # Write core artifacts
    write_jsonl(run_dir / "catalog_files.jsonl", [asdict(r) for r in final_records])
    write_jsonl(run_dir / "bucket_assignment_manifest.jsonl", [asdict(r) for r in final_records])
    write_jsonl(run_dir / "plan_moves.jsonl", [asdict(a) for a in final_actions])
    write_jsonl(run_dir / "plan_empty_dirs.jsonl", [asdict(a) for a in final_dir_actions])
    write_jsonl(run_dir / "bumpers.jsonl", [asdict(b) for b in final_bumpers])
    write_json(run_dir / "apply_stats.json", apply_stats)

    # Bucket rule truth table
    truth_json, truth_csv = bucket_rules_truth_table()
    schema_dir = run_dir / "schema"
    write_json(schema_dir / "bucket_rules_truth_table.json", truth_json)
    write_csv(schema_dir / "bucket_rules_truth_table.csv", ["bucket", "rule_id", "rule_desc", "match"], truth_csv)

    # Drive inventory
    inv_json, inv_csv = build_drive_inventory(final_records)
    write_json(run_dir / "drive_inventory.json", inv_json)
    write_csv(run_dir / "drive_inventory.csv", ["drive", "bucket", "files", "bytes"], inv_csv)

    # Duplicates + versions
    dups = dedupe_groups(final_records)
    dup_summary = build_duplicates_summary(dups)
    write_json(run_dir / "duplicates_summary.json", dup_summary)

    vgroups = group_versions(final_records)
    versions_summary = build_versions_summary(vgroups)
    write_json(run_dir / "versions_summary.json", versions_summary)

    # Canonical view (optional)
    canonical_manifest: List[Tuple[str, str]] = []
    if args.canonical_view:
        seen = set()
        for g in dup_summary.values():
            canon = g.get("canonical")
            if canon and canon not in seen:
                seen.add(canon)
                canonical_manifest.append((canon, Path(canon).name))
        for g in versions_summary.values():
            canon = g.get("canonical")
            if canon and canon not in seen:
                seen.add(canon)
                canonical_manifest.append((canon, Path(canon).name))
        if not canonical_manifest:
            for r in final_records:
                if r.path not in seen:
                    seen.add(r.path)
                    canonical_manifest.append((r.path, Path(r.path).name))

        canonical_dir = run_dir / "CANONICAL_VIEW"
        build_canonical_view(canonical_manifest, canonical_dir, final_bumpers, prefer=args.canonical_prefer, dry_run=args.dry_run)
        write_json(run_dir / "canonical_view_manifest.json", {"count": len(canonical_manifest), "items": canonical_manifest})

    # Zip inventory + forge
    zip_rows: List[dict] = []
    if args.zip_inventory:
        for r in final_records:
            if r.ext == ".zip":
                zip_rows.extend(inventory_zip(Path(r.path), final_bumpers, max_members=args.zip_max_members, hash_members=args.zip_hash_members))
        write_jsonl(run_dir / "zip_inventory.jsonl", zip_rows)

    if args.forge_zip:
        forge_items: List[Tuple[Path, str]] = []
        if canonical_manifest:
            forge_items = [(Path(src), rel) for src, rel in canonical_manifest]
        else:
            forge_items = [(Path(r.path), r.name) for r in final_records]
        forge_zip(run_dir / "canonical_bundle.zip", forge_items, final_bumpers, dry_run=args.dry_run)

    # Content scan
    content_flags: List[dict] = []
    content_summary = {}
    if args.content_scan:
        terms = load_terms(args.terms_file, args.terms_cap)
        include_exts = {e.strip().lower() for e in args.scan_exts.split(",") if e.strip()}
        content_summary, content_flags = content_scan(final_records, final_bumpers, terms, include_exts, max_bytes=args.scan_max_bytes)
        write_json(run_dir / "content_scan_summary.json", content_summary)
        write_jsonl(run_dir / "content_flags.jsonl", content_flags)

    # Graph pipeline
    if args.graph_pipeline:
        nodes, edges, schema = build_graph_model(
            run_id=run_id,
            records=final_records,
            actions=final_actions,
            dir_actions=final_dir_actions,
            bumpers_list=final_bumpers,
            dup_summary=dup_summary,
            versions_summary=versions_summary,
            zip_rows=zip_rows,
            content_flags=content_flags,
        )
        positions = physics_layout(nodes, edges, mode=args.physics_layout, bumpers_list=final_bumpers, seed=42, iterations=220)

        gp = run_dir / "GRAPH_PIPELINE"
        bronze = gp / "bronze"
        silver = gp / "silver"
        gold = gp / "gold"
        safe_mkdir(bronze); safe_mkdir(silver); safe_mkdir(gold)

        for rel in [
            "catalog_files.jsonl",
            "bucket_assignment_manifest.jsonl",
            "plan_moves.jsonl",
            "plan_empty_dirs.jsonl",
            "bumpers.jsonl",
            "duplicates_summary.json",
            "versions_summary.json",
            "zip_inventory.jsonl",
            "content_flags.jsonl",
        ]:
            src = run_dir / rel
            if src.exists():
                shutil.copy2(src, bronze / rel)

        write_jsonl(silver / "nodes.jsonl", nodes)
        write_jsonl(silver / "edges.jsonl", edges)
        emit_graphml(silver / "graph.graphml", nodes, edges)
        emit_neo4j_import_pack(silver / "neo4j_import", nodes, edges)

        if args.virtualize:
            sqlite_path = silver / "virtual_graph.sqlite"
            emit_virtual_graph_sqlite(sqlite_path, nodes, edges)
            if args.virtual_query:
                cols, rows = run_virtual_query(sqlite_path, args.virtual_query)
                write_query_results(gold / "virtual_query.csv", gold / "virtual_query.json", cols, rows)
            if args.virtual_query_file:
                cols, rows = run_virtual_query_sql_file(sqlite_path, Path(args.virtual_query_file))
                write_query_results(gold / "virtual_query_file.csv", gold / "virtual_query_file.json", cols, rows)

        metrics = compute_graph_metrics(nodes, edges)
        write_json(gold / "metrics.json", metrics)
        emit_html_graph_viewer(gold / "graph_viewer.html", nodes, edges, positions, title="LITIGATIONOS_GRAPH_VIEWER")

        if args.erd_superset:
            emit_erd_superset_files(gold / "ERD_SUPERSET_FED_BLUEPRINT", schema, final_bumpers, physics_mode=args.physics_layout)

    # Provenance index + manifest
    artifacts = [str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file()]
    provenance = {
        "run_id": run_id,
        "roots": [str(r) for r in roots],
        "out_root": str(out_root),
        "features": {
            "hash": args.hash,
            "quick_sig": args.quick_sig,
            "converge": args.converge,
            "stable_n": args.stable_n,
            "max_cycles": args.max_cycles,
            "canonical_view": args.canonical_view,
            "zip_inventory": args.zip_inventory,
            "forge_zip": args.forge_zip,
            "graph_pipeline": args.graph_pipeline,
            "virtualize": args.virtualize,
            "erd_superset": args.erd_superset,
        },
        "artifacts": artifacts,
        "traveling_bumper_path": str(travel_bumpers),
    }
    write_json(run_dir / "provenance_index.json", provenance)

    manifest = build_run_manifest(run_dir, run_id)
    write_json(run_dir / "RUN_MANIFEST.json", manifest)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
