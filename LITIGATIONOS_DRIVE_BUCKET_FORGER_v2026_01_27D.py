#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LITIGATIONOS_DRIVE_BUCKET_FORGER_v2026_01_27D.py

"Deep Hydration" upgrade of the Drive Bucket Forger spine tool.

WHAT’S NEW vs 01_27C (high-signal upgrades)
1) Graph Data Pipeline Engineering (Bronze/Silver/Gold) hardened
   - Silver emits JSONL + CSV + SQLite DDL + Neo4j CSV + LOAD.cypher
   - Gold emits:
     - virtual_graph.sqlite (graph virtualization) with indices + built-in queries + query pack export
     - graph_viewer.html (offline) with deterministic physics layout
     - dashboard_index.html linking to key artifacts

2) Graph Data Virtualization (practical, local-first)
   - SQLite “virtual graph” database:
       nodes(node_id,label,props_json)
       edges(edge_id,type,start_id,end_id,props_json)
   - Built-in query runner (--vquery) + query pack emitter RUN/GRAPH_PIPELINE/gold/queries/*.sql

3) ERD_SUPERSET_FED_BLUEPRINT (schema-first superset)
   - Emits JSON + MD + DOT + HTML ERD viewer
   - Adds JSON-Schema for nodes/edges
   - Adds SQLite DDL for overlay entities (future ingestion), without requiring them now

4) Physics Engine (deterministic)
   - fast: O(n) degree-radial placement (default)
   - full: iterative force simulation (uses numpy if present; else degrades to grid-approx)
   - grid: O(n) repulsion approximation, dependency-free; used automatically when numpy missing

5) Convergence cycles (watch mode) now emits:
   - RUN/DELTA/graph_delta.json and inventory_delta.json per cycle
   - fingerprints + stabilized cycles counter

6) “Adversarial / rights / negative statement” harvesting (lightweight, dependency-free)
   - content scan can emit:
       RUN/adversarial_hits.jsonl : normalized hits with EvidenceAtom-style pointers (EAID)
       RUN/GRAPH_PIPELINE/silver/content_flags.jsonl : graph-ready flags
   - No invented facts: it only reports literal term hits + excerpts + offsets.

7) Transactional apply (move/copy) with resumable apply log
   - RUN/APPLY/apply_log.jsonl + apply_stats.json
   - Safe defaults: plan-only unless you opt-in to apply
   - “Bumpers not blockers”: apply proceeds while logging WARN/HARD bumpers; only stops on explicit --halt-on-hard

8) Empty folder cleanup planning (never deletes unless you ask)
   - RUN/cleanup_empty_folders_plan.jsonl (plan)
   - --apply-cleanup to remove only folders verified empty after operations (optional)

9) CyclePack forging (portable run bundle)
   - --forge-cyclepack emits RUN/CYCLEPACK_<run_id>.zip with manifests + key outputs

Design invariants (kept):
- Non-destructive by default (plan-only).
- Append-only logs (JSONL) for reproducibility.
- Deterministic classification and graph IDs.
- Hard exclusion: C:\\ and OS system paths are never scanned by default.

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import zipfile
import zlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Set, Any

APP_VER = "v2026_01_27D"

# -------------------------
# Buckets (<=15)
# -------------------------
DEFAULT_BUCKETS: List[Tuple[str, str]] = [
    ("B01_TEXT", "txt, md, rtf, log, ini, yaml, yml, toml"),
    ("B02_DATA", "csv, tsv, json, jsonl, parquet, sqlite, db"),
    ("B03_DOCS", "doc, docx, odt, pdf (non-OCR isolated separately), xls/xlsx, ppt/pptx"),
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
    ("B14_OCR_QUEUE", "pdf needing OCR (detected/flagged)"),
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

# -------------------------
# Version family heuristics
# -------------------------
VERSION_PATTERNS = [
    re.compile(r"(?i)(?:^|[ _-])v(\d+(?:\.\d+){0,3})(?:$|[ _-])"),
    re.compile(r"(?i)(?:^|[ _-])(\d{4}[-_]\d{2}[-_]\d{2})(?:$|[ _-])"),
    re.compile(r"(?i)(?:^|[ _-])(\d{8})(?:$|[ _-])"),
    re.compile(r"(?i)(?:^|[ _-])(final)(?:$|[ _-])"),
]

# -------------------------
# Misconduct Vectors (seed)
# -------------------------
DEFAULT_MV_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "MV01": {
        "name": "Bias/Partiality",
        "signals": ["Asymmetric rulings", "Credibility favoritism", "Unexplained deviations", "Outcome-driven findings"],
        "proof": ["ContradictionMap", "AuthorityTriples", "Order-vs-Record deltas", "Transcript omissions"],
        "remedies": ["MotionToDisqualify", "MotionForReconsideration", "AppealIssue", "JTCComplaint"],
    },
    "MV02": {
        "name": "Weaponized_PPO",
        "signals": ["Ex parte overreach", "Scope inflation", "Evidentiary asymmetry", "Parenting-time interference"],
        "proof": ["PPO record gaps", "Statutory noncompliance", "Hearing timing defects"],
        "remedies": ["MotionToQuashOrModify", "Appeal", "CollateralRecord", "JTCComplaint"],
    },
    "MV03": {
        "name": "Retaliatory_Contempt",
        "signals": ["Contempt after protected filings", "Vague orders", "Impossible compliance"],
        "proof": ["Order ambiguity", "Compliance timeline", "Authority mismatch"],
        "remedies": ["MotionToVacate", "Stay", "Appeal", "DueProcessObjections"],
    },
}

DEFAULT_EVENT_TO_MV_MAP: Dict[str, List[str]] = {
    "EVIDENCE_EXCLUDED": ["MV01"],
    "CROSS_EXAM_LIMITED": ["MV01"],
    "EX_PARTE_PPO_SCOPE": ["MV02"],
    "PPO_USED_TO_BLOCK_PT": ["MV02"],
    "CONTEMPT_AFTER_FILING": ["MV03"],
    "AMBIGUOUS_ORDER": ["MV03"],
}

DEFAULT_BUMPER_QUERY_PACK: Dict[str, Any] = {
    "queries": [
        {"id": "BQ01", "name": "High severity bumpers", "match": {"severity": ["warn", "hard"]}, "group_by": ["id", "where"], "limit": 200},
        {"id": "BQ02", "name": "Bumpers mapped to MV02", "match": {"mv": ["MV02"]}, "group_by": ["id", "where"], "limit": 500},
    ]
}

# -------------------------
# Adversarial lexicon (term hits only)
# -------------------------
DEFAULT_ADVERSARIAL_LEXICON: Dict[str, List[str]] = {
    "NEGATIVE_CHARACTERIZATION": [
        "unfit", "abusive", "dangerous", "unstable", "harassment", "stalking", "threat", "violent",
        "kidnap", "alienat", "contempt", "violat", "fraud", "perjur", "lying", "false", "weaponiz"
    ],
    "RIGHTS_PROCESS_VIOLATION": [
        "due process", "no notice", "without notice", "denied hearing", "refused evidence", "excluded evidence",
        "not allowed", "muted", "bias", "partial", "ex parte", "no opportunity", "no counsel"
    ],
    "RECORD_DEFECT": [
        "no transcript", "missing transcript", "off the record", "not recorded", "inaudible", "omitted",
        "order does not reflect", "discrepancy", "record mismatch"
    ],
}

DEFAULT_VIRTUAL_QUERY_PACK: Dict[str, str] = {
    "bucket_counts": """
        SELECT json_extract(n.props_json, '$.bucket_id') AS bucket_id, count(*) AS n
        FROM edges e
        JOIN nodes n ON n.node_id = e.end_id
        WHERE e.type='IN_BUCKET' AND n.label='Bucket'
        GROUP BY bucket_id
        ORDER BY n DESC;
    """,
    "bumper_counts": """
        SELECT json_extract(props_json, '$.id') AS bumper_id, json_extract(props_json, '$.mv') AS mv, count(*) AS n
        FROM nodes
        WHERE label='Bumper'
        GROUP BY bumper_id, mv
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
        LIMIT 200;
    """,
    "version_families": """
        SELECT json_extract(props_json,'$.family') AS family, json_extract(props_json,'$.count') AS count, json_extract(props_json,'$.canonical') AS canonical
        FROM nodes
        WHERE label='VersionFamily'
        ORDER BY count DESC
        LIMIT 200;
    """,
}

# -------------------------
# Dataclasses
# -------------------------
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
    version_family: Optional[str] = None
    version_score: Optional[float] = None
    sha256: Optional[str] = None
    quick_sig: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class PlannedAction:
    action: str  # MOVE | COPY | SKIP
    src: str
    dst: str
    bucket: str
    reason: str
    size: int


@dataclass
class Bumper:
    id: str
    severity: str  # soft | warn | hard
    where: str
    msg: str
    detail: Optional[str] = None
    mv: Optional[str] = None
    ts: str = ""


# -------------------------
# Utilities
# -------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    safe_mkdir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, s: str) -> None:
    safe_mkdir(path.parent)
    path.write_text(s, encoding="utf-8")


def write_jsonl(path: Path, rows: List[dict]) -> None:
    safe_mkdir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    safe_mkdir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def stable_id_for_path(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:16]


def integrity_key(path: Path) -> str:
    """
    Durable-ish integrity key (not a cryptographic commitment): BundleUID+EntryPath+CRC32+bytes+mtime
    CRC32 is over first+last bytes (quick), not full file.
    """
    try:
        size = path.stat().st_size
        mtime = int(path.stat().st_mtime)
        raw = path.read_bytes()
        sample = raw[:65536] + raw[-65536:] if len(raw) > 131072 else raw
        crc = zlib.crc32(sample) & 0xffffffff
        return f"IK|{path.name}|{crc:08x}|{size}|{mtime}"
    except Exception:
        return f"IK|{path.name}|ERROR"


def is_system_path(p: Path) -> bool:
    ps = str(p).lower()
    bad = [
        r"\windows\\", r"\program files", r"\programdata\\",
        r"\$recycle.bin", r"\system volume information",
        r"\appdata\\local\\temp", r"\appdata\\local\\microsoft\\windows\\inetcache",
    ]
    return any(b in ps for b in bad)


def bumper(bumpers_list: List[Bumper], bid: str, severity: str, where: str, msg: str, detail: Optional[str] = None, mv: Optional[str] = None) -> None:
    bumpers_list.append(Bumper(id=bid, severity=severity, where=where, msg=msg, detail=detail, mv=mv, ts=now_iso()))


# -------------------------
# Classification rules
# -------------------------
def classify_bucket(path: Path) -> Tuple[str, str, str]:
    ext = path.suffix.lower()
    name = path.name.lower()

    # Extension-first
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
        return ("B03_DOCS", "R_EXT_DOC", f"ext={ext}")

    # Name hints
    if "readme" in name:
        return ("B01_TEXT", "R_HINT_README", "name contains readme")
    if "schema" in name and ext in {".json", ".yml", ".yaml"}:
        return ("B02_DATA", "R_HINT_SCHEMA", "name contains schema")
    if "backup" in name or name.endswith(".bak"):
        return ("B15_MISC", "R_HINT_BACKUP", "name indicates backup")

    return ("B15_MISC", "R_FALLBACK", f"ext={ext or '(none)'}")


def detect_version_family(path: Path) -> Tuple[Optional[str], Optional[float]]:
    name = path.stem
    tokens = re.split(r"[ _-]+", name)
    cleaned = []
    score = 0.0
    for t in tokens:
        matched = False
        for rx in VERSION_PATTERNS:
            if rx.search(t):
                matched = True
                if t.lower() == "final":
                    score += 2.0
                elif re.fullmatch(r"\d{4}[-_]\d{2}[-_]\d{2}", t):
                    score += 1.0
                elif re.fullmatch(r"\d{8}", t):
                    score += 1.0
                else:
                    score += 0.8
                break
        if not matched:
            cleaned.append(t)
    family = " ".join(cleaned).strip().lower()
    if not family:
        return (None, None)
    return (family, score)


# -------------------------
# Scanning
# -------------------------
def iter_files(roots: List[Path], bumpers_list: List[Bumper], include_hidden: bool = False) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            bumper(bumpers_list, "ROOT_MISSING", "warn", str(root), "Root does not exist; skipping")
            continue
        # Gate 0: exclude C:\ unless explicitly passed
        if os.name == "nt":
            if str(root).lower().startswith("c:\\"):
                bumper(bumpers_list, "GATE0_C_EXCLUDED", "hard", str(root), "C:\\ excluded by policy; remove from roots")
                continue

        for dirpath, dirnames, filenames in os.walk(root):
            dp = Path(dirpath)
            if is_system_path(dp):
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
                except Exception:
                    continue


def scan_inventory(roots: List[Path], bumpers_list: List[Bumper], compute_hashes: bool, compute_quick_sig: bool) -> List[FileRecord]:
    recs: List[FileRecord] = []
    for p in iter_files(roots, bumpers_list):
        try:
            st = p.stat()
        except Exception as e:
            bumper(bumpers_list, "STAT_FAIL", "soft", str(p), "Failed stat()", str(e))
            continue

        drive = p.drive if hasattr(p, "drive") else ""
        relpath = str(p.relative_to(Path(drive + os.sep))) if drive else str(p)

        bucket, rid, rreason = classify_bucket(p)
        fam, fscore = detect_version_family(p)

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
            version_family=fam,
            version_score=fscore,
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


# -------------------------
# Planning
# -------------------------
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
            if verify_existing_hash:
                try:
                    # if source hash not computed, compute quick sha256 now (soft bumper if fails)
                    sh = r.sha256
                    if not sh:
                        sh = sha256_file(src)
                    dh = sha256_file(dst)
                    if sh == dh:
                        actions.append(PlannedAction("SKIP", r.path, str(dst), r.bucket, "dst_exists_same_hash", r.size))
                        continue
                    else:
                        bumper(bumpers_list, "DST_EXISTS_HASH_DIFF", "warn", f"{src} -> {dst}", "Destination exists with different hash; leaving in place")
                        actions.append(PlannedAction("SKIP", r.path, str(dst), r.bucket, "dst_exists_hash_diff", r.size))
                        continue
                except Exception as e:
                    bumper(bumpers_list, "DST_EXISTS_VERIFY_FAIL", "soft", f"{src} -> {dst}", "Failed verifying existing destination", str(e))
                    actions.append(PlannedAction("SKIP", r.path, str(dst), r.bucket, "dst_exists_verify_fail", r.size))
                    continue
            actions.append(PlannedAction("SKIP", r.path, str(dst), r.bucket, "dst_exists_skip_existing", r.size))
            continue

        actions.append(PlannedAction("MOVE", r.path, str(dst), r.bucket, f"{r.rule_id}:{r.rule_reason}", r.size))
    return actions


# -------------------------
# Apply plan (transactional logs, resumable)
# -------------------------
def apply_plan(actions: List[PlannedAction], bumpers_list: List[Bumper], do_copy: bool, dry_run: bool, apply_dir: Path, halt_on_hard: bool) -> Dict[str, int]:
    safe_mkdir(apply_dir)
    log_path = apply_dir / "apply_log.jsonl"

    moved = 0
    copied = 0
    skipped = 0
    failed = 0

    # hard bumpers check (optional)
    if halt_on_hard:
        hard = [b for b in bumpers_list if b.severity == "hard"]
        if hard:
            bumper(bumpers_list, "HALT_ON_HARD", "hard", "apply_plan", "Halting apply due to hard bumpers", detail=f"count={len(hard)}")
            return {"moved": 0, "copied": 0, "skipped": 0, "failed": 0}

    for a in actions:
        if a.action == "SKIP":
            skipped += 1
            append_jsonl(log_path, {"ts": now_iso(), "action": "SKIP", "src": a.src, "dst": a.dst, "reason": a.reason})
            continue

        src = Path(a.src)
        dst = Path(a.dst)

        try:
            safe_mkdir(dst.parent)
            if dry_run:
                if do_copy:
                    copied += 1
                    append_jsonl(log_path, {"ts": now_iso(), "action": "DRY_COPY", "src": a.src, "dst": a.dst, "bucket": a.bucket})
                else:
                    moved += 1
                    append_jsonl(log_path, {"ts": now_iso(), "action": "DRY_MOVE", "src": a.src, "dst": a.dst, "bucket": a.bucket})
                continue

            if dst.exists():
                bumper(bumpers_list, "DST_EXISTS", "warn", f"{a.src} -> {a.dst}", "Destination exists; skipping")
                skipped += 1
                append_jsonl(log_path, {"ts": now_iso(), "action": "SKIP_DST_EXISTS", "src": a.src, "dst": a.dst})
                continue

            if do_copy:
                shutil.copy2(str(src), str(dst))
                copied += 1
                append_jsonl(log_path, {"ts": now_iso(), "action": "COPY", "src": a.src, "dst": a.dst, "bucket": a.bucket})
            else:
                shutil.move(str(src), str(dst))
                moved += 1
                append_jsonl(log_path, {"ts": now_iso(), "action": "MOVE", "src": a.src, "dst": a.dst, "bucket": a.bucket})

        except Exception as e:
            failed += 1
            bumper(bumpers_list, "APPLY_FAIL", "warn", f"{a.src} -> {a.dst}", "Move/copy failed", str(e))
            append_jsonl(log_path, {"ts": now_iso(), "action": "FAIL", "src": a.src, "dst": a.dst, "error": str(e)})

    stats = {"moved": moved, "copied": copied, "skipped": skipped, "failed": failed}
    write_json(apply_dir / "apply_stats.json", stats)
    return stats


# -------------------------
# Dedup + version consolidation
# -------------------------
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


def choose_canonical(paths: List[FileRecord]) -> FileRecord:
    def score(r: FileRecord) -> Tuple[float, float, float]:
        vs = r.version_score or 0.0
        name = Path(r.path).name.lower()
        bonus = 0.0
        if "final" in name:
            bonus += 3.0
        if "canonical" in name:
            bonus += 2.0
        if "master" in name:
            bonus += 1.5
        return (vs + bonus, r.mtime, r.size)
    return sorted(paths, key=score, reverse=True)[0]


def build_duplicates_summary(groups: Dict[str, List[FileRecord]]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for k, members in groups.items():
        canon = choose_canonical(members)
        out[k] = {
            "key": k,
            "count": len(members),
            "canonical": canon.path,
            "members": [m.path for m in members],
            "sha256": canon.sha256 or "",
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
        out[k] = {"family": k, "count": len(members), "canonical": canon.path, "members": [m.path for m in members]}
    return out


# -------------------------
# ZIP inventory
# -------------------------
def inventory_zip(path: Path, bumpers_list: List[Bumper], max_members: int = 50000, hash_members: bool = False) -> List[dict]:
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


# -------------------------
# OCR queue heuristic (conservative)
# -------------------------
def needs_ocr_pdf(path: Path) -> bool:
    try:
        raw = path.read_bytes()[:250_000]
        letters = sum(1 for b in raw if 65 <= b <= 122)
        if letters < 450:
            return True
    except Exception:
        return False
    return False


# -------------------------
# Content scan + adversarial hit normalization
# -------------------------
def load_terms_from_file(p: Path, cap: int) -> List[str]:
    out: List[str] = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        t = line.strip()
        if t:
            out.append(t)
        if len(out) >= cap:
            break
    return out


def content_read_text_best_effort(path: Path, max_bytes: int) -> str:
    raw = path.read_bytes()
    if len(raw) > max_bytes:
        raw = raw[:max_bytes]
    return raw.decode("utf-8", errors="ignore")


def find_term_hits(text: str, term: str, max_hits: int = 25) -> List[int]:
    hits: List[int] = []
    lower = text.lower()
    t = term.lower()
    start = 0
    while len(hits) < max_hits:
        i = lower.find(t, start)
        if i == -1:
            break
        hits.append(i)
        start = i + max(1, len(t))
    return hits


def make_eaid(path: str, offset: int, term: str) -> str:
    base = f"{path}|{offset}|{term}"
    return "EAID_" + hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()[:16]


def content_scan_and_adversarial(
    recs: List[FileRecord],
    bumpers_list: List[Bumper],
    include_exts: Set[str],
    max_bytes: int,
    lexicon: Dict[str, List[str]],
    extra_terms: List[str],
) -> Tuple[dict, List[dict], List[dict]]:
    """
    Returns:
      summary, content_flags_rows, adversarial_hits_rows
    """
    content_flags_rows: List[dict] = []
    adversarial_hits_rows: List[dict] = []
    scanned = 0
    matched_files = 0

    # Build flattened term list per category (plus extras under CUSTOM)
    cat_terms: Dict[str, List[str]] = {}
    for cat, terms in lexicon.items():
        cat_terms[cat] = list(terms)
    if extra_terms:
        cat_terms.setdefault("CUSTOM", [])
        cat_terms["CUSTOM"].extend(extra_terms)

    # de-dup terms per category
    for cat in list(cat_terms.keys()):
        seen = set()
        uniq = []
        for t in cat_terms[cat]:
            tl = t.lower().strip()
            if not tl or tl in seen:
                continue
            seen.add(tl)
            uniq.append(t)
        cat_terms[cat] = uniq

    for r in recs:
        p = Path(r.path)
        if p.suffix.lower() not in include_exts:
            continue
        scanned += 1
        try:
            text = content_read_text_best_effort(p, max_bytes=max_bytes)
        except Exception as e:
            bumper(bumpers_list, "CONTENT_READ_FAIL", "soft", r.path, "Content read failed", str(e))
            continue

        file_any = False
        file_flags: List[dict] = []
        for cat, terms in cat_terms.items():
            for t in terms:
                for idx in find_term_hits(text, t, max_hits=20):
                    file_any = True
                    excerpt = text[max(0, idx - 90): idx + len(t) + 90].replace("\n", " ")
                    file_flags.append({"category": cat, "term": t, "offset": idx, "excerpt": excerpt})
                    adversarial_hits_rows.append({
                        "eaid": make_eaid(r.path, idx, t),
                        "path": r.path,
                        "bucket": r.bucket,
                        "ext": r.ext,
                        "category": cat,
                        "term": t,
                        "offset": idx,
                        "excerpt": excerpt,
                        "integrity_key": integrity_key(p),
                        "resolver": {"path": r.path, "offset": idx, "hint": "byte-offset in decoded utf-8 best-effort sample"},
                    })
        if file_any:
            matched_files += 1
            content_flags_rows.append({"path": r.path, "bucket": r.bucket, "ext": r.ext, "matches": file_flags})

    summary = {"scanned_files": scanned, "matched_files": matched_files}
    return summary, content_flags_rows, adversarial_hits_rows


# -------------------------
# Cleanup empty folders (plan/apply optional)
# -------------------------
def plan_empty_folder_cleanup(roots: List[Path], bumpers_list: List[Bumper]) -> List[dict]:
    """
    Plan: list empty directories under roots excluding system paths.
    """
    empties: List[dict] = []
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            dp = Path(dirpath)
            if is_system_path(dp):
                continue
            try:
                # Consider empty if no files and no subdirs (after traversal)
                if not list(dp.iterdir()):
                    empties.append({"path": str(dp), "reason": "empty_dir"})
            except Exception as e:
                bumper(bumpers_list, "EMPTY_CHECK_FAIL", "soft", str(dp), "Failed checking emptiness", str(e))
    return empties


def apply_empty_folder_cleanup(plan: List[dict], bumpers_list: List[Bumper], dry_run: bool, log_path: Path) -> Dict[str, int]:
    removed = 0
    skipped = 0
    failed = 0
    for row in plan:
        p = Path(row["path"])
        try:
            if not p.exists():
                skipped += 1
                append_jsonl(log_path, {"ts": now_iso(), "action": "SKIP_MISSING_DIR", "path": str(p)})
                continue
            if list(p.iterdir()):
                skipped += 1
                append_jsonl(log_path, {"ts": now_iso(), "action": "SKIP_NOT_EMPTY", "path": str(p)})
                continue
            if dry_run:
                removed += 1
                append_jsonl(log_path, {"ts": now_iso(), "action": "DRY_REMOVE_DIR", "path": str(p)})
                continue
            p.rmdir()
            removed += 1
            append_jsonl(log_path, {"ts": now_iso(), "action": "REMOVE_DIR", "path": str(p)})
        except Exception as e:
            failed += 1
            bumper(bumpers_list, "REMOVE_DIR_FAIL", "warn", str(p), "Failed removing empty dir", str(e))
            append_jsonl(log_path, {"ts": now_iso(), "action": "FAIL_REMOVE_DIR", "path": str(p), "error": str(e)})
    return {"removed": removed, "skipped": skipped, "failed": failed}


# ============================
# Graph model + pipeline + virtualization + physics + ERD
# ============================
def read_jsonl(path: Path, max_rows: Optional[int] = None) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _node_id(label: str, key: str) -> str:
    return f"{label}::{key}"


def _edge_id(edge_type: str, start_id: str, end_id: str, salt: str = "") -> str:
    base = f"{edge_type}|{start_id}|{end_id}|{salt}"
    return hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()[:20]


def build_erd_superset_schema() -> Dict[str, Any]:
    entities = [
        {"name": "Run", "pk": "run_id", "fields": ["run_id"]},
        {"name": "Drive", "pk": "drive", "fields": ["drive"]},
        {"name": "Folder", "pk": "folder_id", "fields": ["folder_id", "path"]},
        {"name": "Bucket", "pk": "bucket_id", "fields": ["bucket_id", "desc"]},
        {"name": "File", "pk": "file_id", "fields": ["file_id", "path", "ext", "size", "mtime", "sha256", "quick_sig", "version_family", "version_score", "rule_id", "rule_reason", "notes"]},
        {"name": "Action", "pk": "action_id", "fields": ["action_id", "action", "src", "dst", "reason", "bucket", "size"]},
        {"name": "Bumper", "pk": "bumper_id", "fields": ["bumper_id", "id", "ts", "severity", "where", "msg", "mv", "detail"]},
        {"name": "DuplicateGroup", "pk": "group_id", "fields": ["group_id", "count", "canonical", "sha256", "size"]},
        {"name": "VersionFamily", "pk": "family", "fields": ["family", "count", "canonical"]},
        {"name": "ZipMember", "pk": "zip_member_id", "fields": ["zip_member_id", "zip_path", "member", "size", "mtime", "is_dir", "sha256"]},
        {"name": "ContentFlag", "pk": "flag_id", "fields": ["flag_id", "path", "bucket", "ext", "matches"]},

        # Superset overlays (schema-first)
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
        {"type": "RUN_BUMPER", "from": "Run", "to": "Bumper"},
        {"type": "RUN_DUPGROUP", "from": "Run", "to": "DuplicateGroup"},
        {"type": "RUN_VERSIONFAM", "from": "Run", "to": "VersionFamily"},
        {"type": "IN_BUCKET", "from": "File", "to": "Bucket"},
        {"type": "ON_DRIVE", "from": "File", "to": "Drive"},
        {"type": "IN_FOLDER", "from": "File", "to": "Folder"},
        {"type": "FOLDER_ON_DRIVE", "from": "Folder", "to": "Drive"},
        {"type": "ACTION_ON", "from": "Action", "to": "File"},
        {"type": "BUMPER_ON", "from": "Bumper", "to": "File"},
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


def build_graph_model(
    run_id: str,
    records: List[FileRecord],
    actions: List[PlannedAction],
    bumpers_list: List[Bumper],
    dup_summary: Dict[str, dict],
    versions_summary: Dict[str, dict],
    zip_rows: Optional[List[dict]],
    content_flags: Optional[List[dict]],
) -> Tuple[List[dict], List[dict], Dict[str, Any]]:
    nodes: Dict[str, dict] = {}
    edges: Dict[str, dict] = {}

    run_node_id = _node_id("Run", run_id)
    nodes[run_node_id] = {"node_id": run_node_id, "label": "Run", "props": {"run_id": run_id}}

    for bid, desc in DEFAULT_BUCKETS:
        nid = _node_id("Bucket", bid)
        nodes[nid] = {"node_id": nid, "label": "Bucket", "props": {"bucket_id": bid, "desc": desc}}

    # Drives + folders + files
    for r in records:
        file_id = stable_id_for_path(r.path)
        f_nid = _node_id("File", file_id)
        nodes[f_nid] = {"node_id": f_nid, "label": "File", "props": {
            "file_id": file_id,
            "path": r.path,
            "ext": r.ext,
            "size": r.size,
            "mtime": r.mtime,
            "sha256": r.sha256 or "",
            "quick_sig": r.quick_sig or "",
            "version_family": r.version_family or "",
            "version_score": float(r.version_score or 0.0),
            "rule_id": r.rule_id,
            "rule_reason": r.rule_reason,
            "notes": r.notes or "",
        }}

        edges[_edge_id("RUN_SAW", run_node_id, f_nid)] = {"edge_id": _edge_id("RUN_SAW", run_node_id, f_nid), "type": "RUN_SAW", "start_id": run_node_id, "end_id": f_nid, "props": {}}
        b_nid = _node_id("Bucket", r.bucket)
        edges[_edge_id("IN_BUCKET", f_nid, b_nid)] = {"edge_id": _edge_id("IN_BUCKET", f_nid, b_nid), "type": "IN_BUCKET", "start_id": f_nid, "end_id": b_nid, "props": {}}

        drv = r.drive or "ROOT"
        d_nid = _node_id("Drive", drv)
        if d_nid not in nodes:
            nodes[d_nid] = {"node_id": d_nid, "label": "Drive", "props": {"drive": drv}}
        edges[_edge_id("ON_DRIVE", f_nid, d_nid)] = {"edge_id": _edge_id("ON_DRIVE", f_nid, d_nid), "type": "ON_DRIVE", "start_id": f_nid, "end_id": d_nid, "props": {}}

        parent = str(Path(r.path).parent)
        folder_id = stable_id_for_path(parent)
        fol_nid = _node_id("Folder", folder_id)
        if fol_nid not in nodes:
            nodes[fol_nid] = {"node_id": fol_nid, "label": "Folder", "props": {"folder_id": folder_id, "path": parent}}
        edges[_edge_id("IN_FOLDER", f_nid, fol_nid)] = {"edge_id": _edge_id("IN_FOLDER", f_nid, fol_nid), "type": "IN_FOLDER", "start_id": f_nid, "end_id": fol_nid, "props": {}}
        edges[_edge_id("FOLDER_ON_DRIVE", fol_nid, d_nid)] = {"edge_id": _edge_id("FOLDER_ON_DRIVE", fol_nid, d_nid), "type": "FOLDER_ON_DRIVE", "start_id": fol_nid, "end_id": d_nid, "props": {}}

    # Actions
    for a in actions:
        act_id = stable_id_for_path(f"{a.src}->{a.dst}|{a.action}|{a.reason}")
        act_nid = _node_id("Action", act_id)
        nodes[act_nid] = {"node_id": act_nid, "label": "Action", "props": {
            "action_id": act_id, "action": a.action, "src": a.src, "dst": a.dst,
            "reason": a.reason, "bucket": a.bucket, "size": a.size
        }}
        edges[_edge_id("RUN_PLANNED", run_node_id, act_nid)] = {"edge_id": _edge_id("RUN_PLANNED", run_node_id, act_nid), "type": "RUN_PLANNED", "start_id": run_node_id, "end_id": act_nid, "props": {}}
        src_file_id = stable_id_for_path(a.src)
        f_nid = _node_id("File", src_file_id)
        edges[_edge_id("ACTION_ON", act_nid, f_nid)] = {"edge_id": _edge_id("ACTION_ON", act_nid, f_nid), "type": "ACTION_ON", "start_id": act_nid, "end_id": f_nid, "props": {}}

    # Bumpers
    for b in bumpers_list:
        bkey = stable_id_for_path(b.ts + "|" + b.id + "|" + b.where + "|" + (b.msg or ""))
        b_nid = _node_id("Bumper", bkey)
        nodes[b_nid] = {"node_id": b_nid, "label": "Bumper", "props": {
            "bumper_id": bkey, "id": b.id, "ts": b.ts, "severity": b.severity, "where": b.where,
            "msg": b.msg, "mv": b.mv or "", "detail": b.detail or ""
        }}
        edges[_edge_id("RUN_BUMPER", run_node_id, b_nid)] = {"edge_id": _edge_id("RUN_BUMPER", run_node_id, b_nid), "type": "RUN_BUMPER", "start_id": run_node_id, "end_id": b_nid, "props": {}}

        # Link bumper to known file if possible
        known_paths = {r.path for r in records}
        fp = None
        if b.where in known_paths:
            fp = b.where
        elif " -> " in b.where:
            left = b.where.split(" -> ")[0].strip()
            if left in known_paths:
                fp = left
        elif "::" in b.where:
            left = b.where.split("::")[0].strip()
            if left in known_paths:
                fp = left
        if fp:
            f_nid = _node_id("File", stable_id_for_path(fp))
            edges[_edge_id("BUMPER_ON", b_nid, f_nid)] = {"edge_id": _edge_id("BUMPER_ON", b_nid, f_nid), "type": "BUMPER_ON", "start_id": b_nid, "end_id": f_nid, "props": {}}

    # Duplicate groups
    for gid, g in dup_summary.items():
        dg_nid = _node_id("DuplicateGroup", stable_id_for_path(gid))
        nodes[dg_nid] = {"node_id": dg_nid, "label": "DuplicateGroup", "props": {
            "group_id": gid, "count": int(g.get("count", 0)), "canonical": g.get("canonical", ""),
            "sha256": g.get("sha256", ""), "size": int(g.get("size", 0))
        }}
        edges[_edge_id("RUN_DUPGROUP", run_node_id, dg_nid)] = {"edge_id": _edge_id("RUN_DUPGROUP", run_node_id, dg_nid), "type": "RUN_DUPGROUP", "start_id": run_node_id, "end_id": dg_nid, "props": {}}
        for mp in g.get("members", []):
            f_nid = _node_id("File", stable_id_for_path(mp))
            edges[_edge_id("DUP_MEMBER", dg_nid, f_nid, salt=mp)] = {"edge_id": _edge_id("DUP_MEMBER", dg_nid, f_nid, salt=mp), "type": "DUP_MEMBER", "start_id": dg_nid, "end_id": f_nid, "props": {}}

    # Version families
    for fam, g in versions_summary.items():
        vf_nid = _node_id("VersionFamily", stable_id_for_path(fam))
        nodes[vf_nid] = {"node_id": vf_nid, "label": "VersionFamily", "props": {
            "family": fam, "count": int(g.get("count", 0)), "canonical": g.get("canonical", "")
        }}
        edges[_edge_id("RUN_VERSIONFAM", run_node_id, vf_nid)] = {"edge_id": _edge_id("RUN_VERSIONFAM", run_node_id, vf_nid), "type": "RUN_VERSIONFAM", "start_id": run_node_id, "end_id": vf_nid, "props": {}}
        for mp in g.get("members", []):
            f_nid = _node_id("File", stable_id_for_path(mp))
            edges[_edge_id("VF_MEMBER", vf_nid, f_nid, salt=mp)] = {"edge_id": _edge_id("VF_MEMBER", vf_nid, f_nid, salt=mp), "type": "VF_MEMBER", "start_id": vf_nid, "end_id": f_nid, "props": {}}

    # Zip members
    if zip_rows:
        for zr in zip_rows:
            zpath = zr.get("zip_path", "")
            member = zr.get("member", "")
            zid = stable_id_for_path(zpath)
            zf_nid = _node_id("File", zid)
            mem_id = stable_id_for_path(zpath + "::" + member)
            zm_nid = _node_id("ZipMember", mem_id)
            nodes[zm_nid] = {"node_id": zm_nid, "label": "ZipMember", "props": {
                "zip_member_id": mem_id, "zip_path": zpath, "member": member,
                "size": int(zr.get("size", 0)), "mtime": zr.get("mtime", ""),
                "is_dir": bool(zr.get("is_dir", False)), "sha256": zr.get("sha256", "")
            }}
            edges[_edge_id("ZIP_CONTAINS", zf_nid, zm_nid, salt=member)] = {"edge_id": _edge_id("ZIP_CONTAINS", zf_nid, zm_nid, salt=member), "type": "ZIP_CONTAINS", "start_id": zf_nid, "end_id": zm_nid, "props": {}}

    # Content flags
    if content_flags:
        for cf in content_flags:
            p = cf.get("path", "")
            if not p:
                continue
            fid = stable_id_for_path(p)
            f_nid = _node_id("File", fid)
            flag_id = stable_id_for_path(p + "|" + json.dumps(cf.get("matches", []), sort_keys=True))
            c_nid = _node_id("ContentFlag", flag_id)
            nodes[c_nid] = {"node_id": c_nid, "label": "ContentFlag", "props": {
                "flag_id": flag_id, "path": p, "bucket": cf.get("bucket", ""), "ext": cf.get("ext", ""),
                "matches": cf.get("matches", [])
            }}
            edges[_edge_id("FLAG_ON", c_nid, f_nid)] = {"edge_id": _edge_id("FLAG_ON", c_nid, f_nid), "type": "FLAG_ON", "start_id": c_nid, "end_id": f_nid, "props": {}}

    schema = build_erd_superset_schema()
    return list(nodes.values()), list(edges.values()), schema


def compute_graph_metrics(nodes: List[dict], edges: List[dict]) -> Dict[str, Any]:
    node_ids = [n["node_id"] for n in nodes]
    idx = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)
    out_deg = [0] * n
    und = [[] for _ in range(n)]
    out_adj = [[] for _ in range(n)]
    out_count = [0] * n

    for e in edges:
        u = e["start_id"]
        v = e["end_id"]
        if u in idx and v in idx:
            ui = idx[u]
            vi = idx[v]
            out_deg[ui] += 1
            und[ui].append(vi)
            und[vi].append(ui)
            out_adj[ui].append(vi)
            out_count[ui] += 1

    # Components
    comp_id = [-1] * n
    comps: List[List[int]] = []
    cid = 0
    for i in range(n):
        if comp_id[i] != -1:
            continue
        q = [i]
        comp_id[i] = cid
        comp = [i]
        while q:
            cur = q.pop()
            for nb in und[cur]:
                if comp_id[nb] == -1:
                    comp_id[nb] = cid
                    q.append(nb)
                    comp.append(nb)
        comps.append(comp)
        cid += 1

    # PageRank
    pr = [1.0 / n] * n if n else []
    d = 0.85
    for _ in range(30):
        new = [(1.0 - d) / n] * n if n else []
        for ui in range(n):
            if out_count[ui] == 0:
                share = d * pr[ui] / n if n else 0
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


def math_sin(x: float) -> float:
    import math
    return math.sin(x)


def math_cos(x: float) -> float:
    import math
    return math.cos(x)


def physics_layout(
    nodes: List[dict],
    edges: List[dict],
    mode: str,
    bumpers_list: List[Bumper],
    seed: int = 1337,
    iterations: int = 400,
) -> Dict[str, Tuple[float, float]]:
    """
    Deterministic layout:
      fast: degree-based radial placement
      full: numpy force simulation if available; else degrades to grid approximation
      grid: dependency-free approximate repulsion (bin grid)
    """
    import random
    random.seed(seed)

    node_ids = [n["node_id"] for n in nodes]
    n = len(node_ids)
    if n == 0:
        return {}

    # degrees
    deg = {nid: 0 for nid in node_ids}
    for e in edges:
        deg[e["start_id"]] = deg.get(e["start_id"], 0) + 1
        deg[e["end_id"]] = deg.get(e["end_id"], 0) + 1

    if mode == "off":
        return {nid: (0.0, 0.0) for nid in node_ids}

    if mode == "fast":
        max_deg = max(deg.values()) if deg else 1
        out: Dict[str, Tuple[float, float]] = {}
        for i, nid in enumerate(node_ids):
            a = (i * 2.0 * 3.1415926535) / max(1, n)
            r = 0.25 + 0.75 * (deg.get(nid, 0) / max_deg)
            jitter = 0.03 * (random.random() - 0.5)
            out[nid] = (r * (1.0 + jitter) * float(math_cos(a)), r * (1.0 + jitter) * float(math_sin(a)))
        return out

    if mode == "full":
        if n > 2500:
            bumper(bumpers_list, "PHYSICS-DEGRADED", "soft", "physics_layout", f"n_nodes={n} too large for full; using grid")
            mode = "grid"
        else:
            try:
                import numpy as np
                idx = {nid: i for i, nid in enumerate(node_ids)}
                pos = np.random.RandomState(seed).randn(n, 2).astype(np.float64) * 0.08
                vel = np.zeros((n, 2), dtype=np.float64)

                el = []
                for e in edges:
                    u = e["start_id"]
                    v = e["end_id"]
                    if u in idx and v in idx and u != v:
                        el.append((idx[u], idx[v]))
                if not el:
                    return {nid: (float(pos[i, 0]), float(pos[i, 1])) for nid, i in idx.items()}
                el = np.array(el, dtype=np.int32)

                k_rep = 0.003
                k_attr = 0.02
                damp = 0.86
                dt = 0.02

                for _ in range(iterations):
                    delta = pos[:, None, :] - pos[None, :, :]
                    dist2 = (delta ** 2).sum(axis=2) + 1e-6
                    inv = 1.0 / dist2
                    rep = (delta * inv[:, :, None]).sum(axis=1) * k_rep

                    u = el[:, 0]
                    v = el[:, 1]
                    dxy = pos[v] - pos[u]
                    attr = np.zeros_like(pos)
                    attr_u = dxy * k_attr
                    np.add.at(attr, u, attr_u)
                    np.add.at(attr, v, -attr_u)

                    force = rep + attr
                    vel = (vel + force * dt) * damp
                    pos = pos + vel * dt
                    pos = np.clip(pos, -1.6, 1.6)

                return {node_ids[i]: (float(pos[i, 0]), float(pos[i, 1])) for i in range(n)}
            except Exception:
                bumper(bumpers_list, "PHYSICS-NO-NUMPY", "soft", "physics_layout", "numpy missing or failed; using grid")
                mode = "grid"

    # grid mode: bin-based approximate repulsion + edge attraction (dependency-free)
    # Initialize fast positions
    pos = {nid: (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)) for nid in node_ids}
    # adjacency for attraction
    adj: Dict[str, List[str]] = {}
    for e in edges:
        adj.setdefault(e["start_id"], []).append(e["end_id"])
        adj.setdefault(e["end_id"], []).append(e["start_id"])

    def clamp(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    cell = 0.15
    k_rep = 0.008
    k_attr = 0.010

    for _ in range(min(iterations, 260)):
        # Build grid bins
        bins: Dict[Tuple[int, int], List[str]] = {}
        for nid, (x, y) in pos.items():
            gx = int(x / cell)
            gy = int(y / cell)
            bins.setdefault((gx, gy), []).append(nid)

        # Repulsion approximate: only neighbor cells
        new_pos = {}
        for nid, (x, y) in pos.items():
            gx = int(x / cell)
            gy = int(y / cell)
            fx = 0.0
            fy = 0.0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for other in bins.get((gx + dx, gy + dy), []):
                        if other == nid:
                            continue
                        ox, oy = pos[other]
                        rx = x - ox
                        ry = y - oy
                        d2 = rx * rx + ry * ry + 1e-4
                        inv = 1.0 / d2
                        fx += rx * inv
                        fy += ry * inv
            # Attraction to neighbors
            for other in adj.get(nid, []):
                ox, oy = pos.get(other, (x, y))
                fx += (ox - x) * k_attr
                fy += (oy - y) * k_attr

            nx = clamp(x + fx * k_rep, -1.5, 1.5)
            ny = clamp(y + fy * k_rep, -1.5, 1.5)
            new_pos[nid] = (nx, ny)
        pos = new_pos

    return pos


def emit_virtual_graph_sqlite(db_path: Path, nodes: List[dict], edges: List[dict]) -> None:
    import sqlite3
    safe_mkdir(db_path.parent)
    if db_path.exists():
        db_path.unlink()

    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    cur.execute("""
        CREATE TABLE nodes (
            node_id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            props_json TEXT NOT NULL
        );
    """)
    cur.execute("CREATE INDEX idx_nodes_label ON nodes(label);")

    cur.execute("""
        CREATE TABLE edges (
            edge_id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            start_id TEXT NOT NULL,
            end_id TEXT NOT NULL,
            props_json TEXT NOT NULL
        );
    """)
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
    import sqlite3
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    q = DEFAULT_VIRTUAL_QUERY_PACK.get(name.strip(), "")
    if not q:
        con.close()
        return (["error"], [(f"unknown query '{name}'. options: {', '.join(sorted(DEFAULT_VIRTUAL_QUERY_PACK.keys()))}",)])
    cur.execute(q)
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
            "x": float(positions.get(n["node_id"], (0.0, 0.0))[0]),
            "y": float(positions.get(n["node_id"], (0.0, 0.0))[1]),
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
#info {{ padding:8px; font-size:12px; color:#222; }}
canvas {{ display:block; width:100vw; height:calc(100vh - 72px); background:#fff; }}
.badge {{ padding:2px 6px; border:1px solid #444; border-radius:6px; font-size:12px; }}
</style>
</head>
<body>
<div id="top">
  <span class="badge">Graph Viewer</span>
  <span class="badge" id="stats"></span>
  <input id="q" placeholder="search display/path/bucket (press Enter)"/>
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
  scale = Math.max(30, Math.min(900, scale));
  draw();
}}, {{ passive:false }});

function worldToScreen(x,y) {{
  return [ox + x*scale, oy + y*scale];
}}

const NMAP = new Map();
for(const n of DATA.nodes) NMAP.set(n.id, n);
document.getElementById('stats').textContent = `${{DATA.nodes.length}} nodes / ${{DATA.edges.length}} edges`;

let HIT = null;
function draw() {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  // edges
  ctx.globalAlpha = 0.28;
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
  // nodes
  ctx.globalAlpha = 1.0;
  for(const n of DATA.nodes) {{
    const [x,y] = worldToScreen(n.x, n.y);
    ctx.fillStyle = (HIT && n.id===HIT.id) ? '#ff3b30' : '#111';
    ctx.beginPath();
    ctx.arc(x,y,3.2,0,Math.PI*2);
    ctx.fill();
  }}
}}
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


def emit_json_schema_nodes_edges(out_dir: Path) -> None:
    safe_mkdir(out_dir)
    node_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "VirtualGraphNode",
        "type": "object",
        "required": ["node_id", "label", "props"],
        "properties": {
            "node_id": {"type": "string"},
            "label": {"type": "string"},
            "props": {"type": "object"},
        }
    }
    edge_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "VirtualGraphEdge",
        "type": "object",
        "required": ["edge_id", "type", "start_id", "end_id", "props"],
        "properties": {
            "edge_id": {"type": "string"},
            "type": {"type": "string"},
            "start_id": {"type": "string"},
            "end_id": {"type": "string"},
            "props": {"type": "object"},
        }
    }
    write_json(out_dir / "nodes.schema.json", node_schema)
    write_json(out_dir / "edges.schema.json", edge_schema)


def emit_sqlite_ddl_overlay(out_dir: Path) -> None:
    ddl = """
-- ERD_SUPERSET_FED_BLUEPRINT overlay DDL (schema-first). Optional future ingestion.
CREATE TABLE IF NOT EXISTS Court (court_id TEXT PRIMARY KEY, name TEXT, jurisdiction TEXT, level TEXT);
CREATE TABLE IF NOT EXISTS Case_ (case_id TEXT PRIMARY KEY, caption TEXT, court_id TEXT, docket_no TEXT, case_type TEXT, opened TEXT, closed TEXT);
CREATE TABLE IF NOT EXISTS Party (party_id TEXT PRIMARY KEY, name TEXT, role TEXT, side TEXT);
CREATE TABLE IF NOT EXISTS Filing (filing_id TEXT PRIMARY KEY, case_id TEXT, date_filed TEXT, title TEXT, doc_type TEXT, roa_no TEXT);
CREATE TABLE IF NOT EXISTS Order_ (order_id TEXT PRIMARY KEY, case_id TEXT, date_entered TEXT, title TEXT, signed_by TEXT, text_hash TEXT);
CREATE TABLE IF NOT EXISTS Transcript (transcript_id TEXT PRIMARY KEY, case_id TEXT, hearing_date TEXT, reporter TEXT, text_hash TEXT);
CREATE TABLE IF NOT EXISTS Exhibit (exhibit_id TEXT PRIMARY KEY, case_id TEXT, label TEXT, party TEXT, description TEXT, source_file_id TEXT, integrity_key TEXT);
CREATE TABLE IF NOT EXISTS Event (event_id TEXT PRIMARY KEY, case_id TEXT, ts TEXT, event_type TEXT, summary TEXT, source_file_id TEXT);
CREATE TABLE IF NOT EXISTS Authority (authority_id TEXT PRIMARY KEY, type TEXT, cite TEXT, pinpoint TEXT, snapshot_id TEXT);
CREATE TABLE IF NOT EXISTS Form (form_id TEXT PRIMARY KEY, name TEXT, jurisdiction TEXT, url TEXT, version TEXT);
CREATE TABLE IF NOT EXISTS MisconductVector (mv_id TEXT PRIMARY KEY, category TEXT, signals_json TEXT, proof_json TEXT, remedies_json TEXT);
CREATE TABLE IF NOT EXISTS Remedy (remedy_id TEXT PRIMARY KEY, name TEXT, vehicle TEXT, standard TEXT, authority_id TEXT);
CREATE TABLE IF NOT EXISTS Claim (claim_id TEXT PRIMARY KEY, case_id TEXT, theory TEXT, elements_json TEXT, authority_id TEXT);
"""
    write_text(out_dir / "ERD_SUPERSET_overlay_ddl.sql", ddl.strip() + "\n")


def emit_erd_superset_files(out_dir: Path, schema: Dict[str, Any], bumpers_list: List[Bumper], physics_mode: str) -> None:
    safe_mkdir(out_dir)
    write_json(out_dir / "ERD_SUPERSET_FED_BLUEPRINT.json", schema)

    md_lines: List[str] = []
    md_lines.append("# ERD_SUPERSET_FED_BLUEPRINT")
    md_lines.append("")
    md_lines.append("## I. Version")
    md_lines.append(f"- {schema.get('version','')}")
    md_lines.append("")
    md_lines.append("## II. Entities")
    for ent in schema.get("entities", []):
        md_lines.append(f"### {ent['name']}")
        md_lines.append(f"- PK: `{ent['pk']}`")
        md_lines.append(f"- Fields: {', '.join('`'+f+'`' for f in ent.get('fields', []))}")
        md_lines.append("")
    md_lines.append("## III. Relationships")
    for rel in schema.get("relationships", []):
        md_lines.append(f"- `{rel['type']}`: {rel['from']} → {rel['to']}")
    write_text(out_dir / "ERD_SUPERSET_FED_BLUEPRINT.md", "\n".join(md_lines))

    dot = []
    dot.append("digraph ERD {")
    dot.append("  rankdir=LR;")
    dot.append("  node [shape=box, fontsize=10];")
    for ent in schema.get("entities", []):
        dot.append(f'  "{ent["name"]}";')
    for rel in schema.get("relationships", []):
        dot.append(f'  "{rel["from"]}" -> "{rel["to"]}" [label="{rel["type"]}"];')
    dot.append("}")
    write_text(out_dir / "ERD_SUPERSET_FED_BLUEPRINT.dot", "\n".join(dot))

    emit_json_schema_nodes_edges(out_dir / "json_schema")
    emit_sqlite_ddl_overlay(out_dir)

    # HTML ERD viewer (entity graph)
    if physics_mode != "off":
        nlist = [{"node_id": _node_id("Entity", ent["name"]), "label": "Entity", "props": {"name": ent["name"]}} for ent in schema.get("entities", [])]
        elist = []
        for rel in schema.get("relationships", []):
            s = _node_id("Entity", rel["from"])
            t = _node_id("Entity", rel["to"])
            elist.append({"edge_id": _edge_id(rel["type"], s, t), "type": rel["type"], "start_id": s, "end_id": t, "props": {}})
        pos = physics_layout(nlist, elist, mode=("fast" if physics_mode == "grid" else physics_mode), bumpers_list=bumpers_list, seed=777, iterations=260)
        emit_html_graph_viewer(out_dir / "ERD_SUPERSET_FED_BLUEPRINT.html", nlist, elist, pos, title="ERD_SUPERSET_FED_BLUEPRINT")


def emit_csv_nodes_edges(out_dir: Path, nodes: List[dict], edges: List[dict]) -> None:
    safe_mkdir(out_dir)
    # nodes.csv
    with (out_dir / "nodes.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["node_id", "label", "props_json"])
        for n in nodes:
            w.writerow([n["node_id"], n["label"], json.dumps(n.get("props", {}), ensure_ascii=False)])
    # edges.csv
    with (out_dir / "edges.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["edge_id", "type", "start_id", "end_id", "props_json"])
        for e in edges:
            w.writerow([e["edge_id"], e["type"], e["start_id"], e["end_id"], json.dumps(e.get("props", {}), ensure_ascii=False)])


def emit_neo4j_import_pack(out_dir: Path, nodes: List[dict], edges: List[dict]) -> None:
    """
    Neo4j CSV import pack for consistent loading.
    """
    safe_mkdir(out_dir)
    nodes_path = out_dir / "neo4j_nodes.csv"
    edges_path = out_dir / "neo4j_edges.csv"
    with nodes_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id:ID", "label:LABEL", "props_json"])
        for n in nodes:
            w.writerow([n["node_id"], n["label"], json.dumps(n.get("props", {}), ensure_ascii=False)])
    with edges_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([":START_ID", ":END_ID", ":TYPE", "edge_id", "props_json"])
        for e in edges:
            w.writerow([e["start_id"], e["end_id"], e["type"], e["edge_id"], json.dumps(e.get("props", {}), ensure_ascii=False)])

    load_cypher = """// LOAD.cypher (Cypher-based CSV load) - requires APOC for JSON parsing if you want props expanded.
// Minimal load creates nodes with labels + props_json string (expand later if desired).
// Example:
// :param nodes => 'file:///neo4j_nodes.csv';
// :param edges => 'file:///neo4j_edges.csv';
//
// LOAD CSV WITH HEADERS FROM $nodes AS row
// WITH row
// CALL {
//   WITH row
//   MERGE (n:Virtual {id: row.`id:ID`})
//   SET n: `+row.`label:LABEL`+`
//   SET n.props_json = row.props_json
// } IN TRANSACTIONS OF 5000 ROWS;
//
// LOAD CSV WITH HEADERS FROM $edges AS row
// WITH row
// CALL {
//   WITH row
//   MATCH (a:Virtual {id: row.`:START_ID`})
//   MATCH (b:Virtual {id: row.`:END_ID`})
//   MERGE (a)-[r:VREL {edge_id: row.edge_id}]->(b)
//   SET r.type = row.`:TYPE`
//   SET r.props_json = row.props_json
// } IN TRANSACTIONS OF 5000 ROWS;
"""
    write_text(out_dir / "LOAD.cypher", load_cypher)

    constraints = """CREATE CONSTRAINT v_id IF NOT EXISTS FOR (n:Virtual) REQUIRE n.id IS UNIQUE;"""
    indexes = """CREATE INDEX v_props IF NOT EXISTS FOR (n:Virtual) ON (n.props_json);"""
    write_text(out_dir / "constraints.cypher", constraints)
    write_text(out_dir / "indexes.cypher", indexes)


def emit_dashboard_index(out_path: Path, links: Dict[str, str], title: str) -> None:
    safe_mkdir(out_path.parent)
    items = "\n".join([f'<li><a href="{href}">{name}</a></li>' for name, href in links.items()])
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{title}</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 18px; }}
    h1 {{ margin: 0 0 10px; }}
    ul {{ padding-left: 18px; }}
    code {{ background: #f3f3f3; padding: 2px 4px; border-radius: 4px; }}
    a {{ color: #1b5fd6; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p>Quick links to core artifacts.</p>
  <ul>
    {items}
  </ul>
</body>
</html>"""
    write_text(out_path, html)
