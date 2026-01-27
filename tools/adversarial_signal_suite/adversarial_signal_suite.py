#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LITIGATIONOS_ADVERSARIAL_SIGNAL_SUITE
Version: v2.2

Purpose
-------
Non-destructive, append-only adversarial signal detector for litigation corpora. It scans
supported document types, extracts text (without OCR by default), runs configurable regex
patterns, emits append-only JSONL events, and writes summaries. Optional Neo4j import CSVs
are generated for new events in a run. A bootstrap bundle (schema + queries + config)
can be emitted to any output directory.

Key guardrails
-------------
- Detector only: pattern matches are not findings or assertions.
- Append-only: source files are never modified.
- Deterministic: pattern expansion is repeatable for the same inputs.

Typical runs
------------
  python adversarial_signal_suite.py bootstrap --out ./BOOTSTRAP_BUNDLE
  python adversarial_signal_suite.py scan --roots /data --out /data/harvest/ADV
  python adversarial_signal_suite.py watch --roots /data --out /data/harvest/ADV --poll-seconds 5
"""

from __future__ import annotations

import argparse
import bisect
import csv
import datetime as _dt
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional parsers (best-effort). These are not required to run.
_HAS_FITZ = False
_HAS_PDFPLUMBER = False
_HAS_DOCX = False
_HAS_BS4 = False

SEGMENT_SIZE = 800
SEGMENT_OVERLAP = 200

if importlib.util.find_spec("fitz"):
    fitz = importlib.import_module("fitz")  # type: ignore[assignment]
    _HAS_FITZ = True

if importlib.util.find_spec("pdfplumber"):
    pdfplumber = importlib.import_module("pdfplumber")  # type: ignore[assignment]
    _HAS_PDFPLUMBER = True

if importlib.util.find_spec("docx"):
    docx = importlib.import_module("docx")  # type: ignore[assignment]
    _HAS_DOCX = True

if importlib.util.find_spec("bs4"):
    BeautifulSoup = importlib.import_module("bs4").BeautifulSoup  # type: ignore[assignment]
    _HAS_BS4 = True


# -----------------------------
# SCHEMA: doctypes + buckets
# -----------------------------

DOCTYPE_REGISTRY: Dict[str, Dict[str, Any]] = {
    ".txt": {"doctype": "TEXT", "parser": "read_text", "content": "plain"},
    ".md": {"doctype": "MARKDOWN", "parser": "read_text", "content": "plain"},
    ".log": {"doctype": "LOG", "parser": "read_text", "content": "plain"},
    ".json": {"doctype": "JSON", "parser": "read_json_strings", "content": "structured"},
    ".jsonl": {"doctype": "JSONL", "parser": "read_jsonl_strings", "content": "structured"},
    ".csv": {"doctype": "CSV", "parser": "read_csv_strings", "content": "structured"},
    ".tsv": {"doctype": "TSV", "parser": "read_csv_strings", "content": "structured"},
    ".pdf": {"doctype": "PDF", "parser": "read_pdf_text", "content": "paged"},
    ".docx": {"doctype": "DOCX", "parser": "read_docx_text", "content": "paragraphs"},
    ".rtf": {"doctype": "RTF", "parser": "read_rtf_text", "content": "plain"},
    ".html": {"doctype": "HTML", "parser": "read_html_text", "content": "dom"},
    ".htm": {"doctype": "HTML", "parser": "read_html_text", "content": "dom"},
    ".zip": {"doctype": "ZIP", "parser": "inventory_zip", "content": "archive"},
    ".7z": {"doctype": "7Z", "parser": "deferred_archive", "content": "archive"},
    ".rar": {"doctype": "RAR", "parser": "deferred_archive", "content": "archive"},
}

BUCKET_RULES: Dict[str, Any] = {
    "version": "bucket_rules.v1",
    "max_buckets": 15,
    "buckets": [
        {"bucket": "B01_TEXT", "ext": [".txt", ".md", ".log"]},
        {"bucket": "B02_PDF", "ext": [".pdf"]},
        {"bucket": "B03_DOCX", "ext": [".docx"]},
        {"bucket": "B04_RTF", "ext": [".rtf"]},
        {"bucket": "B05_HTML", "ext": [".html", ".htm"]},
        {"bucket": "B06_JSON", "ext": [".json", ".jsonl"]},
        {"bucket": "B07_CSV", "ext": [".csv", ".tsv"]},
        {"bucket": "B08_IMAGES", "ext": [".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"]},
        {"bucket": "B09_AUDIO", "ext": [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"]},
        {"bucket": "B10_VIDEO", "ext": [".mp4", ".mov", ".mkv", ".avi", ".webm"]},
        {"bucket": "B11_ARCHIVES", "ext": [".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz"]},
        {
            "bucket": "B12_CODE",
            "ext": [
                ".py",
                ".ps1",
                ".js",
                ".ts",
                ".tsx",
                ".java",
                ".cs",
                ".cpp",
                ".c",
                ".h",
                ".rs",
                ".go",
                ".sql",
                ".yml",
                ".yaml",
                ".toml",
                ".ini",
                ".cfg",
                ".bat",
                ".sh",
            ],
        },
        {"bucket": "B13_OFFICE_MISC", "ext": [".ppt", ".pptx", ".xls", ".xlsx"]},
        {"bucket": "B14_EMAIL", "ext": [".eml", ".msg"]},
        {"bucket": "B15_OTHER", "ext": ["*"]},
    ],
}


# -----------------------------
# Misconduct Vector (MV) mapping
# -----------------------------

EVENT_TO_MV_MAP: Dict[str, Any] = {
    "version": "event_to_mv_map.v1",
    "mv": {
        "MV01": {"name": "Bias/Partiality"},
        "MV02": {"name": "Weaponized_PPO"},
        "MV03": {"name": "Retaliatory_Contempt"},
        "MV04": {"name": "Due_Process/Notice_Defect"},
        "MV05": {"name": "Evidentiary_Exclusion/Record_Suppression"},
        "MV06": {"name": "Barrier/Paywall/Access_To_Court"},
        "MV07": {"name": "Mental_Health_Label_Weaponization"},
        "MV08": {"name": "Defamation/Character_Assassination"},
        "MV09": {"name": "Parenting_Time_Interference/Alienation_Signal"},
    },
    "map": {
        "BIAS_OR_PARTIALITY": [{"mv": "MV01", "w": 0.9}],
        "CREDIBILITY_ATTACK": [{"mv": "MV08", "w": 0.6}, {"mv": "MV01", "w": 0.4}],
        "SUBSTANCE_ALLEGATION": [{"mv": "MV08", "w": 0.8}],
        "MENTAL_HEALTH_ALLEGATION": [{"mv": "MV07", "w": 0.9}, {"mv": "MV08", "w": 0.5}],
        "UNFIT_PARENT_LANGUAGE": [{"mv": "MV09", "w": 0.6}, {"mv": "MV08", "w": 0.4}],
        "EX_PARTE_OVERREACH": [{"mv": "MV04", "w": 0.8}, {"mv": "MV02", "w": 0.6}],
        "NOTICE_DEFECT": [{"mv": "MV04", "w": 0.9}],
        "EVIDENCE_BLOCKED": [{"mv": "MV05", "w": 0.9}, {"mv": "MV01", "w": 0.5}],
        "HEARSAY_RELIANCE": [{"mv": "MV05", "w": 0.6}],
        "CONTEMPT_RETALIATION": [{"mv": "MV03", "w": 0.8}, {"mv": "MV06", "w": 0.3}],
        "FEE_BOND_BARRIER": [{"mv": "MV06", "w": 0.8}],
        "PPO_WEAPONIZATION": [{"mv": "MV02", "w": 0.9}],
        "NEGATIVE_EMOTION_LANGUAGE": [{"mv": "MV08", "w": 0.4}],
    },
}


# -----------------------------
# Adversarial config
# -----------------------------

DEFAULT_ADVERSARIAL_CONFIG: Dict[str, Any] = {
    "version": "adversarial_config.v2_2",
    "notes": [
        "Patterns are detectors (text triggers), not findings.",
        "Override defaults by placing ADVERSARIAL_CONFIG.json into the output root.",
    ],
    "actors": {
        "judge": ["McNeill", "Judge McNeill", "Jenny L. McNeill"],
        "opponent": ["Emily Watson", "Emily Ann Watson", "Watson"],
        "child": ["Lincoln"],
        "agency": ["FOC", "Friend of the Court", "HealthWest", "Norton Shores", "Police", "CPS", "DHHS"],
    },
    "patterns": [
        {
            "id": "bias_asymmetric_ruling",
            "category": "BIAS_OR_PARTIALITY",
            "weight": 0.8,
            "regex": r"\b(asymmetric|one[-\s]?sided|favor(?:ed|itism)|double\s+standard|outcome[-\s]?driven)\b",
            "flags": ["I"],
            "mv_ids": ["MV01"],
            "notes": "",
        },
        {
            "id": "bias_credibility_pref",
            "category": "BIAS_OR_PARTIALITY",
            "weight": 0.7,
            "regex": r"\b(credible\s+testimony|not\s+credible|credibility\s+determination|the\s+court\s+finds\s+.*credible)\b",
            "flags": ["I"],
            "mv_ids": ["MV01"],
            "notes": "",
        },
        {
            "id": "exparte_terms",
            "category": "EX_PARTE_OVERREACH",
            "weight": 0.9,
            "regex": r"\b(ex\s+parte|without\s+notice|irreparable\s+injury|immediate\s+danger|emergency\s+order)\b",
            "flags": ["I"],
            "mv_ids": ["MV04", "MV02"],
            "notes": "",
        },
        {
            "id": "notice_defect",
            "category": "NOTICE_DEFECT",
            "weight": 0.9,
            "regex": r"\b(no\s+notice|lack\s+of\s+notice|insufficient\s+notice|was\s+not\s+served|service\s+was\s+defective|not\s+properly\s+noticed)\b",
            "flags": ["I"],
            "mv_ids": ["MV04"],
            "notes": "",
        },
        {
            "id": "evidence_blocked",
            "category": "EVIDENCE_BLOCKED",
            "weight": 0.95,
            "regex": r"\b(not\s+allowed\s+to\s+present|refused\s+to\s+admit|excluded\s+evidence|would\s+not\s+hear|prevented\s+from\s+introducing|proffer\s+denied)\b",
            "flags": ["I"],
            "mv_ids": ["MV05", "MV01"],
            "notes": "",
        },
        {
            "id": "hearsay_reliance",
            "category": "HEARSAY_RELIANCE",
            "weight": 0.6,
            "regex": r"\b(hearsay|unnamed\s+friend|told\s+me\s+that|someone\s+said|alleged\s+that)\b",
            "flags": ["I"],
            "mv_ids": ["MV05"],
            "notes": "",
        },
        {
            "id": "ppo_weaponization_terms",
            "category": "PPO_WEAPONIZATION",
            "weight": 0.9,
            "regex": r"\b(PPO|personal\s+protection\s+order|stalking|harassment|no\s+contact|violation\s+of\s+PPO)\b",
            "flags": ["I"],
            "mv_ids": ["MV02"],
            "notes": "",
        },
        {
            "id": "contempt_retaliation",
            "category": "CONTEMPT_RETALIATION",
            "weight": 0.8,
            "regex": r"\b(retaliat(?:e|ion)|punish(?:ed|ment)|contempt\s+for\s+filing|sanction(?:ed|s)|jail|incarcerat(?:ed|ion))\b",
            "flags": ["I"],
            "mv_ids": ["MV03", "MV06"],
            "notes": "",
        },
        {
            "id": "fee_bond_barrier",
            "category": "FEE_BOND_BARRIER",
            "weight": 0.8,
            "regex": r"\b(filing\s+fee|bond\s+requirement|cash\s+bond|security\s+for\s+costs|paywall|unable\s+to\s+file)\b",
            "flags": ["I"],
            "mv_ids": ["MV06"],
            "notes": "",
        },
        {
            "id": "substance_allegation",
            "category": "SUBSTANCE_ALLEGATION",
            "weight": 0.9,
            "regex": r"\b(meth|amphetamine|cocaine|heroin|opioid|drug\s+use|under\s+the\s+influence|intoxicated|drug\s+screen|urinalysis)\b",
            "flags": ["I"],
            "mv_ids": ["MV08"],
            "notes": "",
        },
        {
            "id": "mental_health_allegation",
            "category": "MENTAL_HEALTH_ALLEGATION",
            "weight": 0.95,
            "regex": r"\b(delusional|psychosis|paranoid|bipolar|schizo|mental\s+health\s+eval|assessment\s+required|diagnos(?:is|ed)|rule\s+out\s+delusional)\b",
            "flags": ["I"],
            "mv_ids": ["MV07", "MV08"],
            "notes": "",
        },
        {
            "id": "unfit_parent_language",
            "category": "UNFIT_PARENT_LANGUAGE",
            "weight": 0.7,
            "regex": r"\b(unfit\s+parent|unsafe\s+for\s+child|danger\s+to\s+child|supervised\s+visitation|parenting\s+time\s+suspend(?:ed|)|withhold(?:ing)?\s+parenting\s+time)\b",
            "flags": ["I"],
            "mv_ids": ["MV09", "MV08"],
            "notes": "",
        },
        {
            "id": "negative_emotion_language",
            "category": "NEGATIVE_EMOTION_LANGUAGE",
            "weight": 0.4,
            "regex": r"\b(crazy|insane|unstable|manipulative|liar|lying|dangerous|abusive|toxic|threat(?:en|ening))\b",
            "flags": ["I"],
            "mv_ids": ["MV08"],
            "notes": "",
        },
        {
            "id": "credibility_attack",
            "category": "CREDIBILITY_ATTACK",
            "weight": 0.6,
            "regex": r"\b(not\s+credible|fabricat(?:ed|ion)|exaggerat(?:ed|ion)|false\s+report|made\s+up|misrepresent(?:ed|ation)|perjur(?:y|ed))\b",
            "flags": ["I"],
            "mv_ids": ["MV08", "MV01"],
            "notes": "",
        },
    ],
    "synonyms": {
        "liar": ["dishonest", "untruthful", "deceptive"],
        "abusive": ["violent", "harassing", "controlling"],
        "dangerous": ["unsafe", "risk", "threat"],
        "withhold": ["deny", "block", "refuse"],
        "excluded": ["barred", "precluded", "stricken"],
        "notice": ["served", "service", "mailing"],
        "bias": ["partial", "prejudice", "unfair"],
    },
}


# -----------------------------
# Data classes
# -----------------------------

@dataclass(frozen=True)
class Segment:
    locator: str
    text: str
    extra: Dict[str, Any]


@dataclass(frozen=True)
class FileParseResult:
    ok: bool
    doctype: str
    segments: List[Segment]
    errors: List[str]
    ocr_needed: bool


# -----------------------------
# Utility: IDs + time
# -----------------------------

def utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def stable_event_id(path: str, locator: str, pattern_id: str, match_span: Tuple[int, int]) -> str:
    return "EVT_" + sha256_hex(f"{path}|{locator}|{pattern_id}|{match_span[0]}|{match_span[1]}")[:32]


def stable_eaid(path: str, locator: str, snippet: str) -> str:
    return "EA_" + sha256_hex(f"{path}|{locator}|{snippet}")[:32]


def build_line_index(text: str) -> List[int]:
    line_starts = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(idx + 1)
    return line_starts


def line_for_offset(line_starts: List[int], offset: int) -> int:
    if not line_starts:
        return 1
    return max(1, bisect.bisect_right(line_starts, offset) - 1 + 1)


def chunk_text_with_lines(text: str, base_locator: Optional[str] = None) -> List[Segment]:
    if not text:
        locator = "line:0-0" if not base_locator else f"{base_locator}|line:0-0|char:0-0"
        return [Segment(locator=locator, text="", extra={"line_start": 0, "line_end": 0, "char_start": 0, "char_end": 0})]
    line_starts = build_line_index(text)
    segs: List[Segment] = []
    step = max(1, SEGMENT_SIZE - SEGMENT_OVERLAP)
    for start in range(0, len(text), step):
        end = min(len(text), start + SEGMENT_SIZE)
        chunk = text[start:end]
        line_start = line_for_offset(line_starts, start)
        line_end = line_for_offset(line_starts, max(start, end - 1))
        locator = f"line:{line_start}-{line_end}|char:{start + 1}-{end}"
        if base_locator:
            locator = f"{base_locator}|{locator}"
        segs.append(
            Segment(
                locator=locator,
                text=chunk,
                extra={"line_start": line_start, "line_end": line_end, "char_start": start + 1, "char_end": end},
            )
        )
        if end >= len(text):
            break
    return segs


def chunk_text_generic(text: str, base_locator: str, extra: Optional[Dict[str, Any]] = None) -> List[Segment]:
    if not text:
        return [Segment(locator=base_locator, text="", extra=extra or {})]
    segs: List[Segment] = []
    step = max(1, SEGMENT_SIZE - SEGMENT_OVERLAP)
    chunk_index = 0
    for start in range(0, len(text), step):
        end = min(len(text), start + SEGMENT_SIZE)
        chunk = text[start:end]
        locator = f"{base_locator}|chunk:{chunk_index}|char:{start + 1}-{end}"
        segs.append(Segment(locator=locator, text=chunk, extra=extra or {}))
        chunk_index += 1
        if end >= len(text):
            break
    return segs


# -----------------------------
# Filesystem: scanning + bucketing
# -----------------------------


def is_probably_system_path(p: Path) -> bool:
    s = str(p).lower()
    bad = [
        "\\windows\\",
        "\\program files",
        "\\programdata\\",
        "\\$recycle.bin\\",
        "/proc/",
        "/sys/",
        "/dev/",
        "/run/",
        "/var/lib/",
        "/var/run/",
        "/mnt/",
    ]
    return any(b in s for b in bad)


def bucket_for_ext(ext: str) -> str:
    ext = ext.lower()
    for bucket in BUCKET_RULES["buckets"]:
        if "*" in bucket["ext"]:
            continue
        if ext in bucket["ext"]:
            return bucket["bucket"]
    return "B15_OTHER"


def iter_files(roots: List[Path], follow_symlinks: bool = False) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            yield root
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
            dp = Path(dirpath)
            if is_probably_system_path(dp):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d.lower() not in ("node_modules", "__pycache__")]
            for fn in filenames:
                if fn.startswith("."):
                    continue
                yield dp / fn


# -----------------------------
# Text extraction (best-effort, no OCR)
# -----------------------------


def read_text_file(p: Path, max_bytes: int = 8_000_000) -> FileParseResult:
    errors: List[str] = []
    try:
        b = p.read_bytes()
        if len(b) > max_bytes:
            b = b[:max_bytes]
            errors.append(f"TRUNCATED:max_bytes={max_bytes}")
        text = b.decode("utf-8", errors="replace")
        segs = chunk_text_with_lines(text)
        return FileParseResult(
            ok=True,
            doctype=DOCTYPE_REGISTRY[p.suffix.lower()]["doctype"],
            segments=segs,
            errors=errors,
            ocr_needed=False,
        )
    except Exception as exc:
        return FileParseResult(
            ok=False,
            doctype=DOCTYPE_REGISTRY.get(p.suffix.lower(), {}).get("doctype", "UNKNOWN"),
            segments=[],
            errors=[f"READ_TEXT_ERROR:{type(exc).__name__}:{exc}"],
            ocr_needed=False,
        )


def read_csv_strings(p: Path, delimiter: str, max_rows: int = 5000, max_cell: int = 4000) -> FileParseResult:
    errors: List[str] = []
    segs: List[Segment] = []
    try:
        with p.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            row_i = 0
            for row in reader:
                row_i += 1
                if row_i > max_rows:
                    errors.append(f"TRUNCATED:max_rows={max_rows}")
                    break
                cells = []
                for cell in row:
                    cell_str = cell.strip()
                    if len(cell_str) > max_cell:
                        cell_str = f"{cell_str[:max_cell]} [TRUNCATED]"
                    if cell_str:
                        cells.append(cell_str)
                if not cells:
                    continue
                row_text = " | ".join(cells)
                segs.extend(chunk_text_generic(row_text, f"row:{row_i}", extra={"row": row_i}))
        return FileParseResult(
            ok=True,
            doctype=DOCTYPE_REGISTRY[p.suffix.lower()]["doctype"],
            segments=segs,
            errors=errors,
            ocr_needed=False,
        )
    except Exception as exc:
        return FileParseResult(
            ok=False,
            doctype=DOCTYPE_REGISTRY.get(p.suffix.lower(), {}).get("doctype", "UNKNOWN"),
            segments=[],
            errors=[f"READ_CSV_ERROR:{type(exc).__name__}:{exc}"],
            ocr_needed=False,
        )


def _extract_json_strings(obj: Any, out: List[str], max_strings: int = 20000, max_len: int = 2000) -> None:
    if len(out) >= max_strings:
        return
    if obj is None:
        return
    if isinstance(obj, str):
        s = obj.strip()
        if s:
            out.append(s[:max_len])
        return
    if isinstance(obj, (int, float, bool)):
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(key, str) and key.strip():
                out.append(key.strip()[:max_len])
            _extract_json_strings(value, out, max_strings=max_strings, max_len=max_len)
            if len(out) >= max_strings:
                break
        return
    if isinstance(obj, list):
        for item in obj:
            _extract_json_strings(item, out, max_strings=max_strings, max_len=max_len)
            if len(out) >= max_strings:
                break
        return


def read_json_strings(p: Path) -> FileParseResult:
    errors: List[str] = []
    try:
        obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        strings: List[str] = []
        _extract_json_strings(obj, strings)
        segs: List[Segment] = []
        block: List[str] = []
        start_i = 0
        for i, s in enumerate(strings):
            block.append(s)
            if len(block) >= 50:
                block_text = "\n".join(block)
                segs.extend(chunk_text_generic(block_text, f"jsonstr:{start_i}-{i}", extra={"i0": start_i, "i1": i}))
                block = []
                start_i = i + 1
        if block:
            block_text = "\n".join(block)
            segs.extend(
                chunk_text_generic(
                    block_text,
                    f"jsonstr:{start_i}-{start_i + len(block) - 1}",
                    extra={"i0": start_i, "i1": start_i + len(block) - 1},
                )
            )
        return FileParseResult(
            ok=True,
            doctype=DOCTYPE_REGISTRY[p.suffix.lower()]["doctype"],
            segments=segs,
            errors=errors,
            ocr_needed=False,
        )
    except Exception as exc:
        return FileParseResult(
            ok=False,
            doctype=DOCTYPE_REGISTRY.get(p.suffix.lower(), {}).get("doctype", "UNKNOWN"),
            segments=[],
            errors=[f"READ_JSON_ERROR:{type(exc).__name__}:{exc}"],
            ocr_needed=False,
        )


def read_jsonl_strings(p: Path, max_lines: int = 20000) -> FileParseResult:
    errors: List[str] = []
    segs: List[Segment] = []
    try:
        with p.open("r", encoding="utf-8", errors="replace") as handle:
            for i, line in enumerate(handle, start=1):
                if i > max_lines:
                    errors.append(f"TRUNCATED:max_lines={max_lines}")
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    strings: List[str] = []
                    _extract_json_strings(obj, strings, max_strings=2000)
                    if strings:
                        segs.extend(
                            chunk_text_generic(
                                "\n".join(strings[:200]),
                                f"jsonl:{i}",
                                extra={"line": i},
                            )
                        )
                except Exception:
                    segs.extend(chunk_text_generic(line[:4000], f"line:{i}", extra={"line": i}))
        return FileParseResult(
            ok=True,
            doctype=DOCTYPE_REGISTRY[p.suffix.lower()]["doctype"],
            segments=segs,
            errors=errors,
            ocr_needed=False,
        )
    except Exception as exc:
        return FileParseResult(
            ok=False,
            doctype=DOCTYPE_REGISTRY.get(p.suffix.lower(), {}).get("doctype", "UNKNOWN"),
            segments=[],
            errors=[f"READ_JSONL_ERROR:{type(exc).__name__}:{exc}"],
            ocr_needed=False,
        )


def read_rtf_text(p: Path, max_bytes: int = 8_000_000) -> FileParseResult:
    errors: List[str] = []
    try:
        raw = p.read_bytes()
        if len(raw) > max_bytes:
            raw = raw[:max_bytes]
            errors.append(f"TRUNCATED:max_bytes={max_bytes}")
        s = raw.decode("utf-8", errors="replace")
        s = re.sub(r"{\\\*?\\[^{}]+;}", " ", s)
        s = re.sub(r"\\'[0-9a-fA-F]{2}", " ", s)
        s = re.sub(r"\\[a-zA-Z]+\d*\s?", " ", s)
        s = s.replace("{", " ").replace("}", " ")
        s = re.sub(r"\s+", " ", s).strip()
        segs = chunk_text_with_lines(s)
        return FileParseResult(
            ok=True,
            doctype=DOCTYPE_REGISTRY[p.suffix.lower()]["doctype"],
            segments=segs,
            errors=errors,
            ocr_needed=False,
        )
    except Exception as exc:
        return FileParseResult(
            ok=False,
            doctype=DOCTYPE_REGISTRY.get(p.suffix.lower(), {}).get("doctype", "UNKNOWN"),
            segments=[],
            errors=[f"READ_RTF_ERROR:{type(exc).__name__}:{exc}"],
            ocr_needed=False,
        )


def read_html_text(p: Path, max_bytes: int = 8_000_000) -> FileParseResult:
    errors: List[str] = []
    try:
        b = p.read_bytes()
        if len(b) > max_bytes:
            b = b[:max_bytes]
            errors.append(f"TRUNCATED:max_bytes={max_bytes}")
        html = b.decode("utf-8", errors="replace")
        if _HAS_BS4:
            soup = BeautifulSoup(html, "lxml" if "lxml" in sys.modules else "html.parser")
            txt = soup.get_text("\n")
        else:
            txt = re.sub(r"<[^>]+>", " ", html)
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        segs = chunk_text_with_lines(txt)
        return FileParseResult(
            ok=True,
            doctype=DOCTYPE_REGISTRY[p.suffix.lower()]["doctype"],
            segments=segs,
            errors=errors,
            ocr_needed=False,
        )
    except Exception as exc:
        return FileParseResult(
            ok=False,
            doctype=DOCTYPE_REGISTRY.get(p.suffix.lower(), {}).get("doctype", "UNKNOWN"),
            segments=[],
            errors=[f"READ_HTML_ERROR:{type(exc).__name__}:{exc}"],
            ocr_needed=False,
        )


def read_docx_text(p: Path) -> FileParseResult:
    errors: List[str] = []
    segs: List[Segment] = []
    if not _HAS_DOCX:
        return FileParseResult(ok=False, doctype="DOCX", segments=[], errors=["MISSING_DEP:python-docx"], ocr_needed=False)
    try:
        doc = docx.Document(str(p))
        for i, para in enumerate(doc.paragraphs, start=1):
            t = (para.text or "").strip()
            if t:
                segs.extend(chunk_text_generic(t, f"para:{i}", extra={"para": i}))
        if not segs:
            segs.append(Segment(locator="para:0", text="", extra={}))
        return FileParseResult(ok=True, doctype="DOCX", segments=segs, errors=errors, ocr_needed=False)
    except Exception as exc:
        return FileParseResult(
            ok=False,
            doctype="DOCX",
            segments=[],
            errors=[f"READ_DOCX_ERROR:{type(exc).__name__}:{exc}"],
            ocr_needed=False,
        )


def read_pdf_text(p: Path, max_pages: int = 400) -> FileParseResult:
    errors: List[str] = []
    segs: List[Segment] = []
    ocr_needed = False
    if _HAS_FITZ:
        try:
            doc = fitz.open(str(p))
            n = min(doc.page_count, max_pages)
            if doc.page_count > max_pages:
                errors.append(f"TRUNCATED:max_pages={max_pages}")
            total_chars = 0
            for i in range(n):
                page = doc.load_page(i)
                txt = page.get_text("text") or ""
                total_chars += len(txt)
                if txt.strip():
                    segs.extend(chunk_text_with_lines(txt, base_locator=f"page:{i + 1}"))
            doc.close()
            if total_chars < 50:
                ocr_needed = True
                errors.append("LOW_TEXT:OCR_UNKNOWN")
            return FileParseResult(ok=True, doctype="PDF", segments=segs or [Segment(locator="page:0", text="", extra={})], errors=errors, ocr_needed=ocr_needed)
        except Exception as exc:
            errors.append(f"FITZ_ERROR:{type(exc).__name__}:{exc}")
    if _HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(str(p)) as pdf:
                n = min(len(pdf.pages), max_pages)
                if len(pdf.pages) > max_pages:
                    errors.append(f"TRUNCATED:max_pages={max_pages}")
                total_chars = 0
                for i in range(n):
                    txt = pdf.pages[i].extract_text() or ""
                    total_chars += len(txt)
                    if txt.strip():
                        segs.extend(chunk_text_with_lines(txt, base_locator=f"page:{i + 1}"))
                if total_chars < 50:
                    ocr_needed = True
                    errors.append("LOW_TEXT:OCR_UNKNOWN")
            return FileParseResult(ok=True, doctype="PDF", segments=segs or [Segment(locator="page:0", text="", extra={})], errors=errors, ocr_needed=ocr_needed)
        except Exception as exc:
            return FileParseResult(
                ok=False,
                doctype="PDF",
                segments=[],
                errors=errors + [f"PDFPLUMBER_ERROR:{type(exc).__name__}:{exc}"],
                ocr_needed=True,
            )
    return FileParseResult(ok=False, doctype="PDF", segments=[], errors=errors + ["MISSING_DEP:PyMuPDF_or_pdfplumber"], ocr_needed=True)


def inventory_zip(p: Path, max_entries: int = 50000) -> FileParseResult:
    errors: List[str] = []
    segs: List[Segment] = []
    try:
        import zipfile

        with zipfile.ZipFile(str(p), "r") as zip_handle:
            names = zip_handle.namelist()
            if len(names) > max_entries:
                errors.append(f"TRUNCATED:max_entries={max_entries}")
                names = names[:max_entries]
            block: List[str] = []
            start = 0
            for i, name in enumerate(names):
                block.append(name)
                if len(block) >= 200:
                    segs.extend(chunk_text_generic("\n".join(block), f"zip:{start}-{i}", extra={"i0": start, "i1": i}))
                    block = []
                    start = i + 1
            if block:
                segs.extend(
                    chunk_text_generic(
                        "\n".join(block),
                        f"zip:{start}-{start + len(block) - 1}",
                        extra={"i0": start, "i1": start + len(block) - 1},
                    )
                )
        return FileParseResult(ok=True, doctype="ZIP", segments=segs, errors=errors, ocr_needed=False)
    except Exception as exc:
        return FileParseResult(ok=False, doctype="ZIP", segments=[], errors=[f"ZIP_ERROR:{type(exc).__name__}:{exc}"], ocr_needed=False)


def deferred_archive(p: Path) -> FileParseResult:
    return FileParseResult(
        ok=True,
        doctype=DOCTYPE_REGISTRY.get(p.suffix.lower(), {}).get("doctype", "ARCHIVE"),
        segments=[Segment(locator="archive:deferred", text=str(p), extra={})],
        errors=["DEFERRED_ARCHIVE_EXTRACTION"],
        ocr_needed=False,
    )


def parse_file(p: Path) -> FileParseResult:
    ext = p.suffix.lower()
    if ext not in DOCTYPE_REGISTRY:
        return FileParseResult(ok=False, doctype="UNSUPPORTED", segments=[], errors=["UNSUPPORTED_EXT"], ocr_needed=False)

    parser = DOCTYPE_REGISTRY[ext]["parser"]
    if parser == "read_text":
        return read_text_file(p)
    if parser == "read_pdf_text":
        return read_pdf_text(p)
    if parser == "read_docx_text":
        return read_docx_text(p)
    if parser == "read_rtf_text":
        return read_rtf_text(p)
    if parser == "read_html_text":
        return read_html_text(p)
    if parser == "read_json_strings":
        return read_json_strings(p)
    if parser == "read_jsonl_strings":
        return read_jsonl_strings(p)
    if parser == "read_csv_strings":
        delim = "\t" if ext == ".tsv" else ","
        return read_csv_strings(p, delimiter=delim)
    if parser == "inventory_zip":
        return inventory_zip(p)
    if parser == "deferred_archive":
        return deferred_archive(p)

    return FileParseResult(ok=False, doctype="UNKNOWN", segments=[], errors=["UNKNOWN_PARSER"], ocr_needed=False)


# -----------------------------
# Pattern compilation + enrichment
# -----------------------------


def load_override_config(out_root: Path) -> Optional[Dict[str, Any]]:
    cfg_path = out_root / "ADVERSARIAL_CONFIG.json"
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return None
    return None


def normalize_flags(flag_list: List[str]) -> int:
    flags = 0
    for flag in flag_list:
        if flag.upper() == "I":
            flags |= re.IGNORECASE
        if flag.upper() == "M":
            flags |= re.MULTILINE
        if flag.upper() == "S":
            flags |= re.DOTALL
    return flags


def expand_patterns_deterministic(cfg: Dict[str, Any], cycle: int) -> Dict[str, Any]:
    if cycle <= 0:
        return cfg

    syn = cfg.get("synonyms", {})
    patterns = cfg.get("patterns", [])
    expanded = []
    for pattern in patterns:
        expanded.append(pattern)
        if cycle >= 1:
            rx = pattern.get("regex", "")
            for base, syns in syn.items():
                if re.search(rf"\b{re.escape(base)}\b", rx, flags=re.IGNORECASE):
                    alt = [base] + syns
                    alt_rx = rf"(?:{'|'.join(re.escape(x) for x in alt)})"
                    rx2 = re.sub(rf"\b{re.escape(base)}\b", alt_rx, rx, flags=re.IGNORECASE)
                    if rx2 != rx:
                        pattern2 = dict(pattern)
                        pattern2["id"] = f"{pattern['id']}__syn__{base}"
                        pattern2["weight"] = min(0.99, float(pattern.get("weight", 0.5)) + 0.05)
                        pattern2["regex"] = rx2
                        expanded.append(pattern2)
        if cycle >= 2:
            rx = pattern.get("regex", "")
            rx2 = rx.replace("one[-\\s]?sided", "(?:one[-\\s]?sided|one\\s+sided|one-sided)")
            if rx2 != rx:
                pattern3 = dict(pattern)
                pattern3["id"] = f"{pattern['id']}__var__hyphen"
                pattern3["weight"] = min(0.99, float(pattern.get("weight", 0.5)) + 0.03)
                pattern3["regex"] = rx2
                expanded.append(pattern3)

    seen = set()
    outp = []
    for pattern in expanded:
        pid = pattern.get("id")
        if pid in seen:
            continue
        seen.add(pid)
        outp.append(pattern)
    cfg2 = dict(cfg)
    cfg2["patterns"] = outp
    cfg2["enrichment_cycle"] = cycle
    return cfg2


def compile_patterns(cfg: Dict[str, Any]) -> List[Tuple[Dict[str, Any], re.Pattern]]:
    compiled = []
    for pattern in cfg.get("patterns", []):
        flags = normalize_flags(pattern.get("flags", ["I"]))
        try:
            compiled.append((pattern, re.compile(pattern["regex"], flags)))
        except Exception:
            continue
    return compiled


# -----------------------------
# Signal detection
# -----------------------------


def find_actor_tags(text: str, actors: Dict[str, List[str]]) -> List[str]:
    tags = []
    low = text.lower()
    for role, names in actors.items():
        for name in names:
            if name and name.lower() in low:
                tags.append(role)
                break
    return sorted(set(tags))


def map_category_to_mv(category: str) -> List[Dict[str, Any]]:
    mapping = EVENT_TO_MV_MAP.get("map", {}).get(category, [])
    return mapping[:] if isinstance(mapping, list) else []


def redact_snippet(s: str, max_len: int = 280) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        return f"{s[:max_len]} [TRUNCATED]"
    return s


def scan_segments_for_signals(
    file_path: Path,
    parse: FileParseResult,
    compiled: List[Tuple[Dict[str, Any], re.Pattern]],
    actors: Dict[str, List[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    stats = {
        "file": str(file_path),
        "doctype": parse.doctype,
        "ocr_needed": parse.ocr_needed,
        "errors": parse.errors[:],
        "events": 0,
        "categories": {},
    }

    for seg in parse.segments:
        seg_text = seg.text or ""
        if not seg_text.strip():
            continue
        actor_tags = find_actor_tags(seg_text, actors)
        for pattern, rx in compiled:
            for match in rx.finditer(seg_text):
                snippet = redact_snippet(seg_text[max(0, match.start() - 120) : min(len(seg_text), match.end() + 160)])
                locator = seg.locator
                evt_id = stable_event_id(str(file_path), locator, pattern["id"], (match.start(), match.end()))
                eaid = stable_eaid(str(file_path), locator, snippet)

                category = pattern.get("category", "UNSPECIFIED")
                mv = map_category_to_mv(category)

                weight = float(pattern.get("weight", 0.5))
                if weight >= 0.85:
                    severity = "HIGH"
                elif weight >= 0.65:
                    severity = "WARN"
                else:
                    severity = "INFO"

                evt = {
                    "event_id": evt_id,
                    "eaid": eaid,
                    "ts_utc": utc_now_iso(),
                    "path": str(file_path),
                    "bucket": bucket_for_ext(file_path.suffix.lower()),
                    "doctype": parse.doctype,
                    "locator": locator,
                    "pattern_id": pattern.get("id"),
                    "category": category,
                    "severity": severity,
                    "weight": weight,
                    "mv": mv,
                    "actor_tags": actor_tags,
                    "match_text": redact_snippet(match.group(0), max_len=180),
                    "snippet": snippet,
                    "parse_errors": parse.errors[:],
                    "ocr_needed": bool(parse.ocr_needed),
                }
                events.append(evt)
                stats["events"] += 1
                stats["categories"][category] = stats["categories"].get(category, 0) + 1

    return events, stats


# -----------------------------
# Outputs: append-only writers
# -----------------------------


def ensure_dirs(out_root: Path) -> Dict[str, Path]:
    out = {
        "ROOT": out_root,
        "RUN": out_root / "RUN",
        "OUT": out_root / "OUT",
        "SCHEMA": out_root / "SCHEMA",
        "QUERIES": out_root / "QUERIES",
        "NEO4J": out_root / "NEO4J_IMPORT",
        "DEFER": out_root / "DEFERRED",
    }
    for path in out.values():
        path.mkdir(parents=True, exist_ok=True)
    return out


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, s: str) -> None:
    path.write_text(s, encoding="utf-8")


def load_seen_event_ids(events_jsonl: Path, max_lines: int = 2_000_000) -> set:
    if not events_jsonl.exists():
        return set()
    seen = set()
    try:
        with events_jsonl.open("r", encoding="utf-8", errors="replace") as handle:
            for i, line in enumerate(handle):
                if i > max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "event_id" in obj:
                        seen.add(obj["event_id"])
                except Exception:
                    continue
    except Exception:
        return set()
    return seen


def load_file_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}


def save_file_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def config_hash(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg.get("patterns", []), ensure_ascii=False, sort_keys=True)
    return sha256_hex(payload)


# -----------------------------
# Neo4j CSV emission
# -----------------------------


def emit_neo4j_csv(out_neo: Path, events: List[Dict[str, Any]]) -> Dict[str, str]:
    out_neo.mkdir(parents=True, exist_ok=True)

    evidence_rows: Dict[str, Dict[str, Any]] = {}
    event_rows: Dict[str, Dict[str, Any]] = {}
    rel_e2e: List[Dict[str, Any]] = []
    mv_nodes: Dict[str, Dict[str, Any]] = {}
    rel_e2mv: List[Dict[str, Any]] = []

    for event in events:
        eaid = event["eaid"]
        if eaid not in evidence_rows:
            evidence_rows[eaid] = {
                "eaid:ID": eaid,
                "path": event["path"],
                "doctype": event["doctype"],
                "bucket": event["bucket"],
                "locator": event["locator"],
                "snippet": event["snippet"],
                "ocr_needed": str(bool(event.get("ocr_needed", False))).lower(),
            }
        evt_id = event["event_id"]
        if evt_id not in event_rows:
            event_rows[evt_id] = {
                "event_id:ID": evt_id,
                "category": event["category"],
                "pattern_id": event["pattern_id"],
                "severity": event["severity"],
                "weight:float": float(event["weight"]),
                "match_text": event["match_text"],
                "actor_tags": "|".join(event.get("actor_tags", [])),
                "ts_utc": event["ts_utc"],
            }
        rel_e2e.append({"eaid:START_ID": eaid, "event_id:END_ID": evt_id, "type": "EVIDENCE_HAS_EVENT"})

        for mv in event.get("mv", []):
            mv_id = mv.get("mv")
            if not mv_id:
                continue
            if mv_id not in mv_nodes:
                mv_nodes[mv_id] = {"mv_id:ID": mv_id, "name": EVENT_TO_MV_MAP.get("mv", {}).get(mv_id, {}).get("name", mv_id)}
            rel_e2mv.append({"event_id:START_ID": evt_id, "mv_id:END_ID": mv_id, "w:float": float(mv.get("w", 0.5)), "type": "MAPS_TO"})

    def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        headers = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    paths = {
        "evidence_atoms.csv": str(out_neo / "evidence_atoms.csv"),
        "signal_events.csv": str(out_neo / "signal_events.csv"),
        "rel_evidence_has_event.csv": str(out_neo / "rel_evidence_has_event.csv"),
        "mv_nodes.csv": str(out_neo / "mv_nodes.csv"),
        "rel_event_maps_mv.csv": str(out_neo / "rel_event_maps_mv.csv"),
    }

    write_csv(out_neo / "evidence_atoms.csv", list(evidence_rows.values()))
    write_csv(out_neo / "signal_events.csv", list(event_rows.values()))
    write_csv(out_neo / "rel_evidence_has_event.csv", rel_e2e)
    write_csv(out_neo / "mv_nodes.csv", list(mv_nodes.values()))
    write_csv(out_neo / "rel_event_maps_mv.csv", rel_e2mv)

    return paths


def merge_graphs(graph_dir: Path, out_path: Path, run_ledger: Optional[Path] = None) -> Dict[str, Any]:
    graph_files = sorted([p for p in graph_dir.glob("*.json") if p.is_file()])
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[str, Dict[str, Any]] = {}

    def node_key(node: Dict[str, Any]) -> str:
        for key in ("id", "node_id", "uid"):
            if key in node:
                return str(node[key])
        return sha256_hex(json.dumps(node, ensure_ascii=False, sort_keys=True))

    def edge_key(edge: Dict[str, Any]) -> str:
        src = edge.get("source") or edge.get("from") or edge.get("src")
        dst = edge.get("target") or edge.get("to") or edge.get("dst")
        rel = edge.get("type") or edge.get("label") or edge.get("rel")
        return f"{src}|{dst}|{rel}"

    for path in graph_files:
        try:
            obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        for node in obj.get("nodes", []) or []:
            key = node_key(node)
            if key not in nodes:
                nodes[key] = node
        for edge in obj.get("edges", []) or []:
            key = edge_key(edge)
            if key not in edges:
                edges[key] = edge

    merged = {
        "nodes": [nodes[k] for k in sorted(nodes.keys())],
        "edges": [edges[k] for k in sorted(edges.keys())],
        "sources": [str(p) for p in graph_files],
        "ts_utc": utc_now_iso(),
    }
    write_json(out_path, merged)
    report = {
        "ts_utc": merged["ts_utc"],
        "phase": "merge_graphs",
        "graph_dir": str(graph_dir),
        "out_path": str(out_path),
        "files": len(graph_files),
        "nodes": len(nodes),
        "edges": len(edges),
    }
    if run_ledger:
        append_jsonl(run_ledger, report)
    return report


# -----------------------------
# Bootstrap bundle emission
# -----------------------------

BUMPERS_CYPHER = r"""
// BUMPER_QUERY_PACK v1 (Signal Events / Evidence Atoms / MV Mapping)
// Assumes you import NEO4J_IMPORT CSVs as:
//  (:EvidenceAtom {eaid, path, doctype, bucket, locator, snippet, ocr_needed})
//  (:SignalEvent {event_id, category, pattern_id, severity, weight, match_text, actor_tags, ts_utc})
//  (:MisconductVector {mv_id, name})
//  (ea)-[:EVIDENCE_HAS_EVENT]->(ev)
//  (ev)-[:MAPS_TO {w}]->(mv)

// Q1: Top categories by count
MATCH (ev:SignalEvent)
RETURN ev.category AS category, count(*) AS n
ORDER BY n DESC;

// Q2: Top HIGH severity events by weight
MATCH (ev:SignalEvent)
WHERE ev.severity = 'HIGH'
RETURN ev.category, ev.pattern_id, ev.weight, ev.match_text, ev.ts_utc
ORDER BY ev.weight DESC, ev.ts_utc DESC
LIMIT 50;

// Q3: Find evidence that mentions ex parte or no notice
MATCH (ea:EvidenceAtom)-[:EVIDENCE_HAS_EVENT]->(ev:SignalEvent)
WHERE ev.category IN ['EX_PARTE_OVERREACH','NOTICE_DEFECT']
RETURN ea.path, ea.locator, ea.snippet, ev.category, ev.pattern_id, ev.match_text
ORDER BY ea.path;

// Q4: MV heatmap (counts by MV)
MATCH (ev:SignalEvent)-[r:MAPS_TO]->(mv:MisconductVector)
RETURN mv.mv_id, mv.name, count(*) AS n, avg(r.w) AS avg_w
ORDER BY n DESC;

// Q5: Actor tag cross-tab (what roles show up in segments)
MATCH (ev:SignalEvent)
WITH split(coalesce(ev.actor_tags,''),'|') AS roles
UNWIND roles AS role
WITH role WHERE role <> ''
RETURN role, count(*) AS n
ORDER BY n DESC;
""".strip() + "\n"


def emit_bootstrap_bundle(out_root: Path) -> Dict[str, str]:
    paths = ensure_dirs(out_root)
    write_json(paths["SCHEMA"] / "doctype_registry.json", DOCTYPE_REGISTRY)
    write_json(paths["SCHEMA"] / "event_to_mv_map.json", EVENT_TO_MV_MAP)
    write_json(paths["SCHEMA"] / "bucket_rules.json", BUCKET_RULES)
    write_json(paths["SCHEMA"] / "ADVERSARIAL_CONFIG_DEFAULT.json", DEFAULT_ADVERSARIAL_CONFIG)

    write_text(paths["QUERIES"] / "bumper_query_pack.cypher", BUMPERS_CYPHER)
    write_json(
        paths["QUERIES"] / "bumper_query_pack.meta.json",
        {
            "name": "BUMPER_QUERY_PACK",
            "version": "v1",
            "requires": ["EvidenceAtom", "SignalEvent", "MisconductVector", "EVIDENCE_HAS_EVENT", "MAPS_TO"],
            "notes": ["These are analysis queries; they do not modify the graph."],
        },
    )

    readme = (
        "# BOOTSTRAP_BUNDLE (v2_2)\n\n"
        "This folder is designed to be dropped into your LitigationOS schema and queries areas.\n\n"
        "## Contents\n"
        "- SCHEMA/doctype_registry.json\n"
        "- SCHEMA/bucket_rules.json\n"
        "- SCHEMA/event_to_mv_map.json\n"
        "- SCHEMA/ADVERSARIAL_CONFIG_DEFAULT.json\n"
        "- QUERIES/bumper_query_pack.cypher\n"
        "- QUERIES/bumper_query_pack.meta.json\n\n"
        "## Override patterns\n"
        "1) Copy SCHEMA/ADVERSARIAL_CONFIG_DEFAULT.json to OUT_ROOT/ADVERSARIAL_CONFIG.json\n"
        "2) Edit patterns/actors/synonyms\n"
        "3) Re-run scan or watch\n"
    )
    write_text(out_root / "README_BOOTSTRAP.md", readme)

    return {k: str(v) for k, v in paths.items()}


# -----------------------------
# Scan orchestration + convergence
# -----------------------------


def top_k_categories(events: List[Dict[str, Any]], k: int = 20) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for event in events:
        cat = event.get("category", "UNSPECIFIED")
        counts[cat] = counts.get(cat, 0) + 1
    items = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [{"category": cat, "n": count} for cat, count in items[:k]]


def scan_once(
    roots: List[Path],
    out_root: Path,
    compiled: List[Tuple[Dict[str, Any], re.Pattern]],
    cfg: Dict[str, Any],
    allow_exts: Optional[set],
    neo4j_csv: bool,
) -> Dict[str, Any]:
    paths = ensure_dirs(out_root)
    events_path = paths["OUT"] / "adversarial_events.jsonl"
    file_status_path = paths["OUT"] / "file_status.jsonl"
    seen = load_seen_event_ids(events_path)
    cache_path = paths["RUN"] / "file_cache.json"
    file_cache = load_file_cache(cache_path)
    cfg_hash = config_hash(cfg)
    cache_dirty = False

    all_events: List[Dict[str, Any]] = []
    file_stats: List[Dict[str, Any]] = []

    scanned_files = 0
    matched_files = 0
    skipped_files = 0
    ocr_bucket = []
    deferred_archives = []

    for fp in iter_files(roots):
        ext = fp.suffix.lower()
        if allow_exts is not None and ext not in allow_exts:
            continue
        if ext not in DOCTYPE_REGISTRY:
            continue

        try:
            stat_info = fp.stat()
        except Exception as exc:
            skipped_files += 1
            status = {
                "file": str(fp),
                "doctype": DOCTYPE_REGISTRY.get(ext, {}).get("doctype", "UNKNOWN"),
                "ocr_needed": False,
                "errors": [f"STAT_ERROR:{type(exc).__name__}:{exc}"],
                "events": 0,
                "categories": {},
                "skipped": True,
                "skip_reason": "STAT_ERROR",
            }
            append_jsonl(file_status_path, status)
            file_stats.append(status)
            continue

        cache_entry = file_cache.get(str(fp))
        if cache_entry and cache_entry.get("mtime") == stat_info.st_mtime and cache_entry.get("size") == stat_info.st_size and cache_entry.get("cfg_hash") == cfg_hash:
            skipped_files += 1
            status = {
                "file": str(fp),
                "doctype": DOCTYPE_REGISTRY.get(ext, {}).get("doctype", "UNKNOWN"),
                "ocr_needed": False,
                "errors": [],
                "events": 0,
                "categories": {},
                "skipped": True,
                "skip_reason": "CACHE_HIT",
            }
            append_jsonl(file_status_path, status)
            file_stats.append(status)
            continue

        parse = parse_file(fp)
        scanned_files += 1

        if parse.ocr_needed:
            ocr_bucket.append(str(fp))
        if fp.suffix.lower() in (".7z", ".rar"):
            deferred_archives.append(str(fp))

        evts, st = scan_segments_for_signals(fp, parse, compiled, cfg.get("actors", {}))
        if evts:
            matched_files += 1
        new_count = 0
        for event in evts:
            if event["event_id"] in seen:
                continue
            append_jsonl(events_path, event)
            seen.add(event["event_id"])
            all_events.append(event)
            new_count += 1
        st["new_events_appended"] = new_count
        st["skipped"] = False
        file_stats.append(st)
        append_jsonl(file_status_path, st)

        file_cache[str(fp)] = {"mtime": stat_info.st_mtime, "size": stat_info.st_size, "cfg_hash": cfg_hash}
        cache_dirty = True

        if scanned_files % 500 == 0:
            append_jsonl(
                paths["RUN"] / "run_ledger.jsonl",
                {
                    "ts_utc": utc_now_iso(),
                    "phase": "scan_progress",
                    "scanned_files": scanned_files,
                    "matched_files": matched_files,
                    "new_events_appended": len(all_events),
                },
            )

    summary = {
        "ts_utc": utc_now_iso(),
        "scanned_files": scanned_files,
        "matched_files": matched_files,
        "skipped_files": skipped_files,
        "new_events_appended": len(all_events),
        "ocr_needed_count": len(ocr_bucket),
        "deferred_archives_count": len(deferred_archives),
        "top_categories_new": top_k_categories(all_events, k=25),
        "out_root": str(out_root),
    }

    if ocr_bucket:
        append_jsonl(paths["DEFER"] / "ocr_bucket.jsonl", {"ts_utc": utc_now_iso(), "files": ocr_bucket})
    if deferred_archives:
        append_jsonl(paths["DEFER"] / "archive_bucket.jsonl", {"ts_utc": utc_now_iso(), "files": deferred_archives})

    if cache_dirty:
        save_file_cache(cache_path, file_cache)

    provenance = {
        "ts_utc": utc_now_iso(),
        "roots": [str(r) for r in roots],
        "doctype_registry_version": "v1",
        "config_version": cfg.get("version"),
        "enrichment_cycle": cfg.get("enrichment_cycle", 0),
        "deps": {
            "PyMuPDF(fitz)": _HAS_FITZ,
            "pdfplumber": _HAS_PDFPLUMBER,
            "python-docx": _HAS_DOCX,
            "bs4": _HAS_BS4,
        },
        "counts": summary,
        "errors_sample": [err for st in file_stats for err in st.get("errors", [])][:50],
    }
    write_json(paths["RUN"] / "provenance_index.json", provenance)

    write_json(
        paths["OUT"] / "adversarial_summary.json",
        {"summary": summary, "files": file_stats[:5000]},
    )

    neo_paths = None
    if neo4j_csv and all_events:
        neo_paths = emit_neo4j_csv(paths["NEO4J"], all_events)

    write_json(
        paths["RUN"] / "manifest.json",
        {
            "ts_utc": utc_now_iso(),
            "outputs": {
                "events_jsonl": str(events_path),
                "summary_json": str(paths["OUT"] / "adversarial_summary.json"),
                "run_ledger_jsonl": str(paths["RUN"] / "run_ledger.jsonl"),
                "provenance_index_json": str(paths["RUN"] / "provenance_index.json"),
                "schema_dir": str(paths["SCHEMA"]),
                "queries_dir": str(paths["QUERIES"]),
                "neo4j_import_dir": str(paths["NEO4J"]) if neo4j_csv else None,
            },
            "neo4j_csv": neo_paths,
        },
    )

    return {"summary": summary, "new_events": len(all_events), "neo4j_csv_paths": neo_paths}


def run_convergent_scan(
    roots: List[Path],
    out_root: Path,
    max_cycles: int,
    eps: float,
    stable_n: int,
    allow_exts: Optional[set],
    neo4j_csv: bool,
) -> Dict[str, Any]:
    paths = ensure_dirs(out_root)
    append_jsonl(
        paths["RUN"] / "run_ledger.jsonl",
        {"ts_utc": utc_now_iso(), "phase": "convergent_scan_start", "max_cycles": max_cycles, "eps": eps, "stable_n": stable_n},
    )

    base_cfg = load_override_config(out_root) or DEFAULT_ADVERSARIAL_CONFIG
    stable_streak = 0
    total_new = 0
    history = []

    for cycle in range(max_cycles):
        cfg = expand_patterns_deterministic(base_cfg, cycle=cycle)
        compiled = compile_patterns(cfg)
        result = scan_once(roots, out_root, compiled, cfg, allow_exts=allow_exts, neo4j_csv=neo4j_csv)
        new_events = int(result["new_events"])
        total_new += new_events

        history.append(
            {
                "cycle": cycle,
                "enrichment_cycle": cfg.get("enrichment_cycle", cycle),
                "patterns": len(cfg.get("patterns", [])),
                "new_events": new_events,
                "scanned_files": result["summary"]["scanned_files"],
            }
        )

        append_jsonl(
            paths["RUN"] / "run_ledger.jsonl",
            {
                "ts_utc": utc_now_iso(),
                "phase": "cycle_done",
                "cycle": cycle,
                "patterns": len(cfg.get("patterns", [])),
                "new_events": new_events,
            },
        )

        if new_events == 0:
            stable_streak += 1
        else:
            stable_streak = 0

        if stable_streak >= stable_n:
            break

        if eps is not None and cycle >= 1:
            prev = history[-2]["new_events"]
            denom = max(1, prev)
            ratio = new_events / denom
            if ratio <= eps:
                stable_streak += 1
                if stable_streak >= stable_n:
                    break
            else:
                stable_streak = 0

    final = {"ts_utc": utc_now_iso(), "total_new_events_appended": total_new, "cycles_ran": len(history), "history": history}
    write_json(paths["RUN"] / "convergence_report.json", final)
    append_jsonl(paths["RUN"] / "run_ledger.jsonl", {"ts_utc": utc_now_iso(), "phase": "convergent_scan_end", **final})
    return final


# -----------------------------
# Watch mode (polling)
# -----------------------------


def watch_polling(
    roots: List[Path],
    out_root: Path,
    poll_seconds: float,
    allow_exts: Optional[set],
    neo4j_csv: bool,
) -> None:
    paths = ensure_dirs(out_root)
    append_jsonl(paths["RUN"] / "run_ledger.jsonl", {"ts_utc": utc_now_iso(), "phase": "watch_start", "poll_seconds": poll_seconds})

    cfg = load_override_config(out_root) or DEFAULT_ADVERSARIAL_CONFIG
    cfg = expand_patterns_deterministic(cfg, cycle=3)
    compiled = compile_patterns(cfg)

    last: Dict[str, float] = {}
    for fp in iter_files(roots):
        ext = fp.suffix.lower()
        if allow_exts is not None and ext not in allow_exts:
            continue
        if ext not in DOCTYPE_REGISTRY:
            continue
        try:
            last[str(fp)] = fp.stat().st_mtime
        except Exception:
            continue

    while True:
        changed: List[Path] = []
        for fp in iter_files(roots):
            ext = fp.suffix.lower()
            if allow_exts is not None and ext not in allow_exts:
                continue
            if ext not in DOCTYPE_REGISTRY:
                continue
            key = str(fp)
            try:
                mt = fp.stat().st_mtime
            except Exception:
                continue
            if key not in last or mt > last[key]:
                last[key] = mt
                changed.append(fp)

        if changed:
            append_jsonl(paths["RUN"] / "run_ledger.jsonl", {"ts_utc": utc_now_iso(), "phase": "watch_changed", "n": len(changed)})
            scan_once([Path(str(x)) for x in changed], out_root, compiled, cfg, allow_exts=allow_exts, neo4j_csv=neo4j_csv)

        time.sleep(poll_seconds)


# -----------------------------
# CLI
# -----------------------------


def parse_roots(args_roots: List[str]) -> List[Path]:
    return [Path(r) for r in args_roots]


def parse_allow_exts(allow: Optional[str]) -> Optional[set]:
    if not allow:
        return None
    extset = set()
    for ext in allow.split(","):
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        extset.add(ext)
    return extset


def cmd_bootstrap(a: argparse.Namespace) -> int:
    out_root = Path(a.out).resolve()
    emit_bootstrap_bundle(out_root)
    print(f"[OK] bootstrap written to: {out_root}")
    return 0


def cmd_scan(a: argparse.Namespace) -> int:
    out_root = Path(a.out).resolve()
    emit_bootstrap_bundle(out_root)
    roots = parse_roots(a.roots)
    allow_exts = parse_allow_exts(a.allow_exts)
    report = run_convergent_scan(
        roots=roots,
        out_root=out_root,
        max_cycles=int(a.max_cycles),
        eps=float(a.eps),
        stable_n=int(a.stable_n),
        allow_exts=allow_exts,
        neo4j_csv=bool(a.neo4j_csv),
    )
    if a.merge_graphs:
        merge_out = Path(a.merge_graphs_out).resolve() if a.merge_graphs_out else (out_root / "OUT" / "merged_graph.json")
        merge_report = merge_graphs(Path(a.merge_graphs), merge_out, out_root / "RUN" / "run_ledger.jsonl")
        report["merge_graphs"] = merge_report
    print(json.dumps(report, indent=2))
    return 0


def cmd_watch(a: argparse.Namespace) -> int:
    out_root = Path(a.out).resolve()
    emit_bootstrap_bundle(out_root)
    roots = parse_roots(a.roots)
    allow_exts = parse_allow_exts(a.allow_exts)
    watch_polling(
        roots=roots,
        out_root=out_root,
        poll_seconds=float(a.poll_seconds),
        allow_exts=allow_exts,
        neo4j_csv=bool(a.neo4j_csv),
    )
    return 0


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="LITIGATIONOS_ADVERSARIAL_SIGNAL_SUITE", add_help=True)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("bootstrap", help="Emit BOOTSTRAP_BUNDLE (schema + queries + default config) into --out")
    p0.add_argument("--out", required=True, help="Output folder for BOOTSTRAP_BUNDLE structure")
    p0.set_defaults(func=cmd_bootstrap)

    p1 = sub.add_parser("scan", help="Scan roots, detect signals, converge, and write outputs")
    p1.add_argument("--roots", nargs="+", required=True, help="Root paths (folders or files)")
    p1.add_argument("--out", required=True, help="Output folder (append-only; safe)")
    p1.add_argument(
        "--allow-exts",
        default=".txt,.md,.log,.pdf,.docx,.rtf,.html,.htm,.json,.jsonl,.csv,.tsv,.zip,.7z,.rar",
        help="Comma list of extensions to include",
    )
    p1.add_argument("--max-cycles", type=int, default=10, help="Maximum enrichment cycles to run")
    p1.add_argument("--eps", type=float, default=0.0, help="Convergence EPS threshold")
    p1.add_argument("--stable-n", type=int, default=2, help="Stop after this many stable cycles")
    p1.add_argument("--neo4j-csv", action="store_true", help="Emit NEO4J_IMPORT CSVs for new events appended in this run")
    p1.add_argument("--merge-graphs", help="Directory of JSON graph files to merge after scan")
    p1.add_argument("--merge-graphs-out", help="Output path for merged graph JSON")
    p1.set_defaults(func=cmd_scan)

    p2 = sub.add_parser("watch", help="Polling watcher. On file changes, re-scan changed files and append new events")
    p2.add_argument("--roots", nargs="+", required=True, help="Root paths (folders or files)")
    p2.add_argument("--out", required=True, help="Output folder (append-only; safe)")
    p2.add_argument(
        "--allow-exts",
        default=".txt,.md,.log,.pdf,.docx,.rtf,.html,.htm,.json,.jsonl,.csv,.tsv",
        help="Comma list of extensions to include",
    )
    p2.add_argument("--poll-seconds", type=float, default=5.0, help="Polling interval seconds")
    p2.add_argument("--neo4j-csv", action="store_true", help="Emit NEO4J_IMPORT CSVs for new events appended on each change batch")
    p2.set_defaults(func=cmd_watch)

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
