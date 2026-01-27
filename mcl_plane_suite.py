#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCL_PLANE_SUITE v3.0  (2026-01-26)

Executive-grade, non-destructive suite:
- Web harvest (Michigan Legislature MCL seeds + discovery)
- Drive scan (supported file types) + <=15 doctype buckets
- Adversarial / rights-violation / negative-statement detector (content based)
- Chained cycles until convergence (delta-based, stable-N)

Core invariants:
- No destructive actions: never deletes, renames, or moves user originals.
- "Bumpers not blockers": missing deps or unreadable formats do not stop a run.
- Append-only outputs: each run writes to OUT/RUNS/<run_id>/; state snapshots are versioned.
- OCR is never performed; unknown/scan-blocked PDFs/images are quarantined for later OCR.

Outputs per run:
- FINAL_DELIVERABLE.md
- RUN/cycle_ledger.jsonl
- RUN/provenance_index.json
- RUN/blockers_and_acquisition_plan.json
- INVENTORY/doctype_bucket_inventory.csv
- ADVERSARIAL/findings.jsonl + findings.csv + stats.json
- NEO4J_IMPORT/*.csv (Files, Findings, MV taxonomy, edges)
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import hashlib
import html
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple


# ----------------------------
# Versioning
# ----------------------------

SUITE_NAME = "MCL_PLANE_SUITE"
SUITE_VERSION = "v3.0"
UTCNOW = lambda: _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

# ----------------------------
# Optional deps (bumpers)
# ----------------------------

HAS_DOCX = False
try:
    import docx  # python-docx
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False

HAS_STRIPRTF = False
try:
    from striprtf.striprtf import rtf_to_text
    HAS_STRIPRTF = True
except Exception:
    HAS_STRIPRTF = False

HAS_PYMUPDF = False
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

HAS_PY7ZR = False
try:
    import py7zr
    HAS_PY7ZR = True
except Exception:
    HAS_PY7ZR = False

HAS_WATCHDOG = False
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    HAS_WATCHDOG = True
except Exception:
    HAS_WATCHDOG = False


# ----------------------------
# Small utilities
# ----------------------------


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
        h.update(b"\x1f")
    return h.hexdigest()[:24]


def safe_relpath(p: Path, root: Path) -> str:
    try:
        return str(p.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def append_jsonl(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def now_compact() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def mk_run_id() -> str:
    rnd = sha256_hex(os.urandom(32))[:6]
    return f"{now_compact()}_{rnd}"


def is_probably_binary(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(2048)
        if b"\x00" in chunk:
            return True
        text_chars = b"".join(bytes([i]) for i in range(32, 127)) + b"\n\r\t\b"
        non = sum(1 for b in chunk if b not in text_chars)
        return non / max(1, len(chunk)) > 0.30
    except Exception:
        return True


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def strip_html_to_text(s: str) -> str:
    s = re.sub(r"(?is)<(script|style)\b.*?>.*?</\1>", " ", s)
    s = re.sub(r"(?is)<br\s*/?>", "\n", s)
    s = re.sub(r"(?is)</p\s*>", "\n", s)
    s = re.sub(r"(?is)<[^>]+>", " ", s)
    s = html.unescape(s)
    return normalize_ws(s)


def sentence_split(text: str) -> List[str]:
    text = text.replace("\r", "\n")
    parts = re.split(r"(?<=[\.\!\?])\s+|\n{2,}", text)
    out = []
    for p in parts:
        p = normalize_ws(p)
        if len(p) >= 8:
            out.append(p)
    return out


def token_split(text: str) -> List[str]:
    return [t for t in re.split(r"[^A-Za-z0-9_'\-]+", text) if t]


def safe_read_text(path: Path, bumper: "BumperLog", encoding: str = "utf-8") -> str:
    try:
        return path.read_text(encoding=encoding, errors="replace")
    except Exception as e:
        bumper.add("READ_TEXT_FAIL", str(path), {"error": repr(e)})
        try:
            return path.read_text(encoding="latin-1", errors="replace")
        except Exception as e2:
            bumper.add("READ_TEXT_FAIL_2", str(path), {"error": repr(e2)})
            return ""


# ----------------------------
# Bumper log (non-blocking issues)
# ----------------------------


@dataclasses.dataclass
class BumperItem:
    code: str
    target: str
    detail: Dict


class BumperLog:
    def __init__(self) -> None:
        self.items: List[BumperItem] = []

    def add(self, code: str, target: str, detail: Optional[Dict] = None) -> None:
        self.items.append(BumperItem(code=code, target=target, detail=detail or {}))

    def to_json(self) -> Dict:
        return {
            "generated_utc": UTCNOW(),
            "count": len(self.items),
            "items": [dataclasses.asdict(x) for x in self.items],
            "acquisition_plan": self._acq_plan(),
        }

    def _acq_plan(self) -> List[Dict]:
        need = set(x.code for x in self.items)
        plan = []
        if any(c.startswith("DOCX_") for c in need) or ("MISSING_DEP_DOCX" in need):
            plan.append({"install": "python-docx", "command": "pip install python-docx", "why": "DOCX text extraction"})
        if any(c.startswith("PDF_") for c in need) or ("MISSING_DEP_PDF" in need):
            plan.append({"install": "PyMuPDF", "command": "pip install PyMuPDF", "why": "PDF text extraction"})
            plan.append({"install": "poppler-utils", "command": "Install pdftotext (Poppler) and ensure it is on PATH", "why": "PDF text extraction fallback"})
        if any(c.startswith("RTF_") for c in need) or ("MISSING_DEP_RTF" in need):
            plan.append({"install": "striprtf", "command": "pip install striprtf", "why": "RTF text extraction"})
        if any(c.startswith("7Z_") for c in need) or ("MISSING_DEP_7Z" in need):
            plan.append({"install": "py7zr", "command": "pip install py7zr", "why": "7z archive extraction"})
        if any(c.startswith("WATCH_") for c in need) or ("MISSING_DEP_WATCHDOG" in need):
            plan.append({"install": "watchdog", "command": "pip install watchdog", "why": "filesystem watcher"})
        plan.append({
            "note": "OCR is intentionally disabled in this suite.",
            "next": "Run an OCR batch later on files under DEFERRED_OCR/ with your preferred OCR pipeline."
        })
        return plan


# ----------------------------
# Default schema assets
# ----------------------------

DEFAULT_DOCTYPE_REGISTRY = {
    "version": "v1",
    "generated_utc": UTCNOW(),
    "buckets_max": 15,
    "buckets": [
        {"bucket": "B01_TEXT", "ext": [".txt", ".md", ".log"], "extractor": "text"},
        {"bucket": "B02_STRUCTURED", "ext": [".json", ".jsonl", ".csv", ".tsv", ".xml", ".yaml", ".yml"], "extractor": "structured"},
        {"bucket": "B03_HTML", "ext": [".html", ".htm"], "extractor": "html_text"},
        {"bucket": "B04_DOCX", "ext": [".docx"], "extractor": "docx_optional"},
        {"bucket": "B05_RTF", "ext": [".rtf"], "extractor": "rtf_optional"},
        {"bucket": "B06_PDF_TEXT", "ext": [".pdf"], "extractor": "pdf_optional"},
        {"bucket": "B07_IMAGES", "ext": [".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"], "extractor": "none"},
        {"bucket": "B08_AUDIO", "ext": [".mp3", ".wav", ".m4a", ".aac", ".flac"], "extractor": "none"},
        {"bucket": "B09_VIDEO", "ext": [".mp4", ".mov", ".mkv", ".avi"], "extractor": "none"},
        {"bucket": "B10_ARCHIVE", "ext": [".zip", ".7z", ".rar"], "extractor": "archive_optional"},
        {"bucket": "B11_CODE", "ext": [".py", ".ps1", ".bat", ".cmd", ".js", ".ts", ".tsx", ".jsx", ".java", ".cs", ".cpp", ".c", ".rs", ".go"], "extractor": "code_text"},
        {"bucket": "B12_OFFICE_OTHER", "ext": [".doc", ".xls", ".xlsx", ".ppt", ".pptx"], "extractor": "optional"},
        {"bucket": "B13_EMAIL", "ext": [".eml", ".msg"], "extractor": "optional"},
        {"bucket": "B14_DATABASE", "ext": [".db", ".sqlite", ".edb"], "extractor": "none"},
        {"bucket": "B15_OTHER", "ext": ["*"], "extractor": "none"},
    ]
}

DEFAULT_EVENT_TO_MV = {
    "version": "v1",
    "mv_taxonomy": {
        "MV01": "Bias/Partiality",
        "MV02": "Weaponized_PPO",
        "MV03": "Retaliatory_Contempt",
        "MV04": "DueProcess_Denial",
        "MV05": "Evidentiary_Asymmetry",
        "MV06": "ParentingTime_Interference",
        "MV07": "Procedural_Barrier",
        "MV08": "Record_Tampering_Omission",
        "MV09": "False_Report_Pattern",
        "MV10": "Coercive_Settlement_Threats",
    },
    "category_to_mv": {
        "JUDICIAL_BIAS": "MV01",
        "CREDIBILITY_ATTACK": "MV01",
        "EX_PARTE_ABUSE": "MV04",
        "DUE_PROCESS": "MV04",
        "DENIAL_OF_EVIDENCE": "MV05",
        "DENIAL_OF_WITNESS": "MV05",
        "PPO_WEAPONIZATION": "MV02",
        "CONTEMPT_ABUSE": "MV03",
        "PARENTING_TIME_WITHHELD": "MV06",
        "BOND_BARRIER": "MV07",
        "NOTICE_DEFECT": "MV04",
        "HEARSAY_RELIANCE": "MV05",
        "FALSE_REPORT": "MV09",
        "THREAT_OR_EXTORTION": "MV10",
        "MENTAL_HEALTH_SMEAR": "MV01",
        "SUBSTANCE_ABUSE_SMEAR": "MV01"
    }
}

DEFAULT_ADVERSARIAL_CONFIG = {
    "version": "v3.0",
    "generated_utc": UTCNOW(),
    "notes": "Regex patterns are applied to extracted text (sentences). No OCR. Bumpers logged.",
    "max_findings_per_file": 500,
    "min_sentence_len": 8,
    "max_sentence_len": 1200,
    "stopwords": [
        "the", "and", "or", "to", "of", "in", "a", "an", "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "with",
        "for", "on", "at", "as", "by", "from", "but", "not", "no", "yes", "do", "does", "did", "done", "so", "if", "then", "than", "too",
        "i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "they", "them", "their", "theirs",
        "we", "us", "our", "ours", "court", "judge", "hearing", "order", "case", "motion", "trial", "record", "transcript", "evidence"
    ],
    "domain_anchors": [
        "court", "judge", "hearing", "order", "motion", "trial", "evidence", "exhibit", "record", "transcript",
        "parenting time", "custody", "ppo", "contempt", "fo c", "foc", "due process", "recusal", "disqualify"
    ],
    "sentiment_negative": [
        "liar", "lying", "crazy", "insane", "delusional", "unhinged", "unstable", "dangerous", "threat", "abusive",
        "harass", "stalk", "manipulative", "vindictive", "malicious", "violent", "meth", "drug", "addict", "narcissist"
    ],
    "categories": [
        {
            "category": "NEGATIVE_STATEMENT",
            "weight": 1.0,
            "patterns": [
                r"\b(liar|lying|dishonest|crazy|insane|delusional|unhinged|unstable|dangerous)\b",
                r"\b(harass(?:ment)?|stalk(?:ing)?|abuse(?:d|ive)?|violent|threat(?:en|s|ening)?)\b",
                r"\b(manipulative|vindictive|malicious|fabricat(?:e|ed|ion)|false|bogus)\b"
            ]
        },
        {
            "category": "CREDIBILITY_ATTACK",
            "weight": 1.2,
            "patterns": [
                r"\b(not credible|lacks credibility|no credibility)\b",
                r"\b(evasive|contradict(?:ion|ory|s)|inconsistent)\b",
                r"\b(made it up|made up)\b"
            ]
        },
        {
            "category": "MENTAL_HEALTH_SMEAR",
            "weight": 1.4,
            "patterns": [
                r"\b(delusional disorder|psychosis|paranoid|schizophren(?:ia|ic))\b",
                r"\b(mental health evaluation|psych eval|psychiatric)\b",
                r"\b(rule out delusional|rule-out delusional)\b"
            ]
        },
        {
            "category": "SUBSTANCE_ABUSE_SMEAR",
            "weight": 1.4,
            "patterns": [
                r"\b(meth(?:amphetamine)?|cocaine|heroin|fentanyl|opioid)\b",
                r"\b(drug use|substance abuse|addict(?:ion)?|under the influence)\b",
                r"\b(drug screen|urinalysis|ua test)\b"
            ]
        },
        {
            "category": "PPO_WEAPONIZATION",
            "weight": 1.6,
            "patterns": [
                r"\b(ex parte)\b.*\b(ppo|protective order)\b",
                r"\b(weaponiz(?:e|ed|ing)|abuse)\b.*\b(ppo|protective order)\b",
                r"\b(false)\b.*\b(ppo|protective order)\b"
            ]
        },
        {
            "category": "EX_PARTE_ABUSE",
            "weight": 1.6,
            "patterns": [
                r"\b(ex parte)\b.*\b(suspend|suspension|terminate|restrict)\b.*\b(parenting time|visitation)\b",
                r"\b(ex parte)\b.*\b(without notice|no notice)\b",
                r"\b(irrep(?:arable)? injury)\b"
            ]
        },
        {
            "category": "DUE_PROCESS",
            "weight": 1.7,
            "patterns": [
                r"\b(due process)\b",
                r"\b(denied)\b.*\b(opportunity to be heard|a hearing|to present evidence|to testify|cross[- ]examin)\b",
                r"\b(no notice|insufficient notice|lack of notice|without notice)\b",
                r"\b(only.*testimony|testimony only)\b.*\b(no exhibits|no evidence)\b"
            ]
        },
        {
            "category": "DENIAL_OF_EVIDENCE",
            "weight": 1.8,
            "patterns": [
                r"\b(not allowed|refused|prohibited)\b.*\b(show|present|introduce)\b.*\b(evidence|exhibits?)\b",
                r"\b(excluded|struck)\b.*\b(evidence|exhibit)\b",
                r"\b(ignored)\b.*\b(evidence|exhibit|record)\b"
            ]
        },
        {
            "category": "DENIAL_OF_WITNESS",
            "weight": 1.8,
            "patterns": [
                r"\b(denied|refused)\b.*\b(witness|testimony|subpoena)\b",
                r"\b(no witness list|witness list denied)\b"
            ]
        },
        {
            "category": "HEARSAY_RELIANCE",
            "weight": 1.3,
            "patterns": [
                r"\b(hearsay)\b",
                r"\b(anonymous|unnamed)\b.*\b(friend|source)\b",
                r"\b(the child said|child said)\b"
            ]
        },
        {
            "category": "JUDICIAL_BIAS",
            "weight": 2.0,
            "patterns": [
                r"\b(judge)\b.*\b(called me|called him|called her)\b.*\b(liar|crazy|delusional)\b",
                r"\b(asymmetric)\b.*\b(ruling|treatment)\b",
                r"\b(bias|partiality)\b"
            ]
        },
        {
            "category": "CONTEMPT_ABUSE",
            "weight": 1.9,
            "patterns": [
                r"\b(contempt)\b.*\b(jail|incarcerat|14 days|sentenced)\b",
                r"\b(show cause)\b.*\b(2 hours|short notice|no notice)\b",
                r"\b(criminal contempt)\b"
            ]
        },
        {
            "category": "PARENTING_TIME_WITHHELD",
            "weight": 1.9,
            "patterns": [
                r"\b(denied|withheld|refused)\b.*\b(parenting time|visitation)\b",
                r"\b(no contact)\b.*\b(child|son|daughter)\b",
                r"\b(alienat(?:e|ion)|interfer(?:e|ence))\b.*\b(parenting time|relationship)\b"
            ]
        },
        {
            "category": "BOND_BARRIER",
            "weight": 1.6,
            "patterns": [
                r"\b(bond requirement|motion bond)\b",
                r"\b(must post)\b.*\b(bond)\b.*\b(future motions?)\b"
            ]
        },
        {
            "category": "FALSE_REPORT",
            "weight": 1.4,
            "patterns": [
                r"\b(false report|false police report)\b",
                r"\b(welfare check)\b.*\b(false)\b",
                r"\b(filed)\b.*\b(police reports?)\b.*\b(false|bogus)\b"
            ]
        },
        {
            "category": "THREAT_OR_EXTORTION",
            "weight": 1.5,
            "patterns": [
                r"\b(threaten(?:ed|ing)?)\b.*\b(evict|jail|take the child|terminate)\b",
                r"\b(accept)\b.*\b(rent increase)\b.*\b(or)\b.*\b(evict|sell|lose)\b"
            ]
        }
    ],
    "mv_mapping": DEFAULT_EVENT_TO_MV
}


# ----------------------------
# File inventory + bucketization
# ----------------------------


def load_doctype_registry(path: Optional[Path], bumper: BumperLog) -> Dict:
    if path and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            bumper.add("LOAD_DOCTYPE_REG_FAIL", str(path), {"error": repr(e)})
    return DEFAULT_DOCTYPE_REGISTRY


def bucket_for_ext(ext: str, registry: Dict) -> str:
    ext = (ext or "").lower()
    for b in registry.get("buckets", []):
        exts = b.get("ext", [])
        if "*" in exts:
            continue
        if ext in [x.lower() for x in exts]:
            return b.get("bucket", "B15_OTHER")
    return "B15_OTHER"


def iter_files(roots: List[Path], bumper: BumperLog, excludes: List[str], max_files: int) -> Iterator[Path]:
    seen = 0
    ex_re = [re.compile(x, re.I) for x in excludes if x]
    for root in roots:
        root = root.expanduser()
        if not root.exists():
            bumper.add("SCAN_ROOT_MISSING", str(root), {})
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            d = dirpath.replace("\\", "/")
            if any(r.search(d) for r in ex_re):
                dirnames[:] = []
                continue
            dirnames[:] = [dn for dn in dirnames if not any(r.search((d + "/" + dn)) for r in ex_re)]
            for fn in filenames:
                p = Path(dirpath) / fn
                fp = str(p).replace("\\", "/")
                if any(r.search(fp) for r in ex_re):
                    continue
                yield p
                seen += 1
                if max_files > 0 and seen >= max_files:
                    bumper.add("MAX_FILES_REACHED", str(max_files), {"note": "Increase --max-files to scan more."})
                    return


def write_bucket_inventory(files: List[Path], registry: Dict, roots: List[Path], out_csv: Path) -> None:
    ensure_dir(out_csv.parent)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "ext", "size_bytes", "mtime_utc", "path"])
        for p in files:
            ext = p.suffix.lower()
            bucket = bucket_for_ext(ext, registry)
            try:
                st = p.stat()
                mtime = _dt.datetime.utcfromtimestamp(st.st_mtime).replace(microsecond=0).isoformat() + "Z"
                size = st.st_size
            except Exception:
                mtime, size = "", ""
            w.writerow([bucket, ext, size, mtime, str(p)])


# ----------------------------
# Text extraction (no OCR)
# ----------------------------


def extract_text_from_docx(path: Path, bumper: BumperLog) -> str:
    if not HAS_DOCX:
        bumper.add("MISSING_DEP_DOCX", str(path), {"install": "python-docx"})
        return ""
    try:
        doc = docx.Document(str(path))
        paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(paras)
    except Exception as e:
        bumper.add("DOCX_EXTRACT_FAIL", str(path), {"error": repr(e)})
        return ""


def extract_text_from_rtf(path: Path, bumper: BumperLog) -> str:
    if not HAS_STRIPRTF:
        bumper.add("MISSING_DEP_RTF", str(path), {"install": "striprtf"})
        return ""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        return rtf_to_text(raw)
    except Exception as e:
        bumper.add("RTF_EXTRACT_FAIL", str(path), {"error": repr(e)})
        return ""


def extract_text_from_pdf(path: Path, bumper: BumperLog, page_limit: int = 200) -> Tuple[str, List[Dict]]:
    page_spans: List[Dict] = []
    if HAS_PYMUPDF:
        try:
            doc = fitz.open(str(path))
            chunks = []
            pos = 0
            for i, page in enumerate(doc):
                if i >= page_limit:
                    bumper.add("PDF_PAGE_LIMIT", str(path), {"limit": page_limit})
                    break
                t = page.get_text("text") or ""
                start = pos
                chunks.append(t)
                pos += len(t)
                page_spans.append({"page": i + 1, "start": start, "end": pos})
            doc.close()
            return "\n".join(chunks), page_spans
        except Exception as e:
            bumper.add("PDF_PYMUPDF_FAIL", str(path), {"error": repr(e)})
    try:
        proc = subprocess.run(["pdftotext", str(path), "-"],
                              capture_output=True, text=True, timeout=120)
        if proc.returncode == 0:
            txt = proc.stdout or ""
            return txt, page_spans
        bumper.add("PDF_PDFTOTEXT_FAIL", str(path), {"stderr": (proc.stderr or "")[:500]})
    except FileNotFoundError:
        bumper.add("MISSING_DEP_PDF", str(path), {"install": "PyMuPDF or pdftotext"})
    except Exception as e:
        bumper.add("PDF_PDFTOTEXT_ERR", str(path), {"error": repr(e)})
    return "", page_spans


def extract_text_from_structured(path: Path, bumper: BumperLog, max_chars: int = 4_000_000) -> str:
    ext = path.suffix.lower()
    try:
        if ext in [".json", ".jsonl"]:
            txt = safe_read_text(path, bumper)
            if ext == ".jsonl":
                vals = []
                for line in txt.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        vals.append(json.dumps(obj, ensure_ascii=False))
                    except Exception:
                        vals.append(line)
                out = "\n".join(vals)
            else:
                try:
                    obj = json.loads(txt)
                    out = json.dumps(obj, ensure_ascii=False)
                except Exception:
                    out = txt
            return out[:max_chars]
        if ext in [".csv", ".tsv"]:
            delim = "\t" if ext == ".tsv" else ","
            rows = []
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f, delimiter=delim)
                for i, row in enumerate(reader):
                    rows.append(" | ".join(row))
                    if i >= 200_000:
                        bumper.add("CSV_ROW_LIMIT", str(path), {"limit": 200000})
                        break
            return "\n".join(rows)[:max_chars]
        return safe_read_text(path, bumper)[:max_chars]
    except Exception as e:
        bumper.add("STRUCT_EXTRACT_FAIL", str(path), {"error": repr(e)})
        return ""


def extract_text_from_code(path: Path, bumper: BumperLog, max_chars: int = 2_000_000) -> str:
    if is_probably_binary(path):
        bumper.add("CODE_BINARY_SKIP", str(path), {})
        return ""
    return safe_read_text(path, bumper)[:max_chars]


def extract_text_any(path: Path, bumper: BumperLog) -> Tuple[str, Dict]:
    ext = path.suffix.lower()
    meta = {"extractor": None, "deferred": False, "page_spans": []}
    if ext in [".txt", ".md", ".log"]:
        meta["extractor"] = "text"
        return safe_read_text(path, bumper), meta
    if ext in [".html", ".htm"]:
        meta["extractor"] = "html"
        return strip_html_to_text(safe_read_text(path, bumper)), meta
    if ext in [".json", ".jsonl", ".csv", ".tsv", ".xml", ".yaml", ".yml"]:
        meta["extractor"] = "structured"
        return extract_text_from_structured(path, bumper), meta
    if ext in [".py", ".ps1", ".bat", ".cmd", ".js", ".ts", ".tsx", ".jsx", ".java", ".cs", ".cpp", ".c", ".rs", ".go"]:
        meta["extractor"] = "code"
        return extract_text_from_code(path, bumper), meta
    if ext == ".docx":
        meta["extractor"] = "docx"
        text = extract_text_from_docx(path, bumper)
        if not text:
            meta["deferred"] = True
        return text, meta
    if ext == ".rtf":
        meta["extractor"] = "rtf"
        text = extract_text_from_rtf(path, bumper)
        if not text:
            meta["deferred"] = True
        return text, meta
    if ext == ".pdf":
        meta["extractor"] = "pdf"
        text, spans = extract_text_from_pdf(path, bumper)
        meta["page_spans"] = spans
        if not text:
            meta["deferred"] = True
        return text, meta
    meta["extractor"] = "none"
    return "", meta


# ----------------------------
# Adversarial scan
# ----------------------------


@dataclasses.dataclass
class Finding:
    finding_id: str
    file_path: str
    file_rel: str
    location: str
    category: str
    mv_code: str
    weight: float
    pattern: str
    snippet: str
    created_utc: str


def load_adversarial_config(path: Optional[Path], bumper: BumperLog) -> Dict:
    if path and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            bumper.add("LOAD_ADVERSARIAL_CONFIG_FAIL", str(path), {"error": repr(e)})
    return DEFAULT_ADVERSARIAL_CONFIG


def compile_patterns(cfg: Dict) -> List[Tuple[str, float, re.Pattern]]:
    out = []
    for c in cfg.get("categories", []):
        cat = c.get("category")
        w = float(c.get("weight", 1.0))
        for pat in c.get("patterns", []):
            try:
                out.append((cat, w, re.compile(pat, re.I)))
            except re.error:
                continue
    return out


def mv_for_category(category: str, cfg: Dict) -> str:
    mapping = ((cfg.get("mv_mapping") or {}).get("category_to_mv") or {})
    return mapping.get(category, "")


def map_offset_to_page(offset: int, page_spans: List[Dict]) -> Optional[int]:
    for sp in page_spans:
        if sp["start"] <= offset < sp["end"]:
            return int(sp["page"])
    return None


def adversarial_scan_files(files: List[Path], scan_roots: List[Path], out_dir: Path, cfg: Dict, bumper: BumperLog) -> Dict:
    ensure_dir(out_dir)
    findings_jsonl = out_dir / "findings.jsonl"
    findings_csv = out_dir / "findings.csv"
    stats_json = out_dir / "stats.json"
    ensure_dir(out_dir.parent / "DEFERRED_OCR")

    compiled = compile_patterns(cfg)
    max_per_file = int(cfg.get("max_findings_per_file", 500))
    min_len = int(cfg.get("min_sentence_len", 8))
    max_len = int(cfg.get("max_sentence_len", 1200))

    findings: List[Finding] = []
    deferred: List[Dict] = []
    file_count = 0
    scanned_files = 0

    with findings_csv.open("w", encoding="utf-8", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["finding_id", "mv_code", "category", "weight", "pattern", "location", "path", "snippet"])
        for fp in files:
            file_count += 1
            txt, meta = extract_text_any(fp, bumper)

            file_rel = ""
            for r in scan_roots:
                if fp.is_absolute() and str(fp).lower().startswith(str(r).lower()):
                    file_rel = safe_relpath(fp, r)
                    break
            if not file_rel:
                file_rel = fp.name

            if meta.get("deferred"):
                deferred.append({"path": str(fp), "reason": "text_extraction_failed_or_missing_dep", "extractor": meta.get("extractor")})
                continue

            if not txt:
                continue

            scanned_files += 1
            sentences = sentence_split(txt)
            cum = 0
            per_file = 0
            for s in sentences:
                if len(s) < min_len:
                    cum += len(s) + 1
                    continue
                if len(s) > max_len:
                    s = s[:max_len]

                found_in_sentence = 0
                for cat, weight, rx in compiled:
                    m = rx.search(s)
                    if not m:
                        continue
                    mv = mv_for_category(cat, cfg)
                    loc = ""
                    if meta.get("extractor") == "pdf" and meta.get("page_spans"):
                        page = map_offset_to_page(cum, meta.get("page_spans", []))
                        if page:
                            loc = f"page:{page}"
                    if not loc:
                        loc = "sentence"

                    snip = normalize_ws(s)
                    fid = stable_id(str(fp), cat, rx.pattern, snip[:200])
                    finding = Finding(
                        finding_id=fid,
                        file_path=str(fp),
                        file_rel=file_rel,
                        location=loc,
                        category=cat,
                        mv_code=mv,
                        weight=float(weight),
                        pattern=rx.pattern,
                        snippet=snip[:1000],
                        created_utc=UTCNOW(),
                    )
                    findings.append(finding)
                    append_jsonl(findings_jsonl, dataclasses.asdict(finding))
                    w.writerow([fid, mv, cat, weight, rx.pattern, loc, str(fp), snip[:500]])

                    found_in_sentence += 1
                    per_file += 1
                    if found_in_sentence >= 5:
                        break
                    if per_file >= max_per_file:
                        bumper.add("MAX_FINDINGS_PER_FILE", str(fp), {"limit": max_per_file})
                        break
                if per_file >= max_per_file:
                    break
                cum += len(s) + 1

    if deferred:
        write_json(out_dir / "deferred_extraction.json", {"count": len(deferred), "items": deferred})

    stats = {
        "generated_utc": UTCNOW(),
        "files_seen": file_count,
        "files_scanned_text": scanned_files,
        "findings_count": len(findings),
        "unique_files_with_findings": len(set(f.file_path for f in findings)),
        "categories": {},
        "mv_codes": {},
    }
    for fnd in findings:
        stats["categories"][fnd.category] = stats["categories"].get(fnd.category, 0) + 1
        mv = fnd.mv_code or ""
        if mv:
            stats["mv_codes"][mv] = stats["mv_codes"].get(mv, 0) + 1

    write_json(stats_json, stats)
    return stats


def derive_auto_terms(findings: List[Finding], cfg: Dict, top_k: int = 50) -> List[str]:
    stop = set((cfg.get("stopwords") or []))
    counts: Dict[str, int] = {}
    for fnd in findings:
        toks = [t.lower() for t in token_split(fnd.snippet)]
        for t in toks:
            if len(t) < 4:
                continue
            if t in stop:
                continue
            if t.isdigit():
                continue
            counts[t] = counts.get(t, 0) + 1
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, v in items if v >= 2][:top_k]


def augment_config_with_auto_terms(cfg: Dict, auto_terms: List[str]) -> Dict:
    cfg2 = json.loads(json.dumps(cfg))
    if not auto_terms:
        return cfg2
    pats = [rf"\b{re.escape(t)}\b" for t in auto_terms]
    cfg2.setdefault("categories", []).append({
        "category": "AUTO_DISCOVERED_TERM",
        "weight": 0.6,
        "patterns": pats
    })
    return cfg2


def adversarial_converge(files: List[Path], scan_roots: List[Path], run_root: Path, base_cfg: Dict,
                         bumper: BumperLog, cycles: int, eps: int, stable_n: int) -> Dict:
    adv_root = run_root / "ADVERSARIAL"
    ensure_dir(adv_root)
    state_dir = run_root / "STATE"
    ensure_dir(state_dir)

    cycle_ledger = run_root / "RUN" / "cycle_ledger.jsonl"

    last_total = 0
    stable = 0
    cfg = base_cfg
    all_findings: Dict[str, Finding] = {}

    for c in range(1, cycles + 1):
        cycle_dir = adv_root / f"CYCLE_{c:02d}"
        ensure_dir(cycle_dir)

        stats = adversarial_scan_files(files, scan_roots, cycle_dir, cfg, bumper)

        findings_jsonl = cycle_dir / "findings.jsonl"
        new_count = 0
        cycle_findings: List[Finding] = []
        if findings_jsonl.exists():
            for line in findings_jsonl.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    fnd = Finding(**d)
                    cycle_findings.append(fnd)
                    if fnd.finding_id not in all_findings:
                        all_findings[fnd.finding_id] = fnd
                        new_count += 1
                except Exception:
                    continue

        total = len(all_findings)
        delta = total - last_total
        last_total = total

        append_jsonl(cycle_ledger, {
            "ts_utc": UTCNOW(),
            "phase": "adversarial_cycle",
            "cycle": c,
            "cycle_new_findings": new_count,
            "total_findings": total,
            "delta": delta,
            "eps": eps,
            "stable_n": stable_n,
            "stats": stats
        })

        auto_terms = derive_auto_terms(cycle_findings, cfg, top_k=60)
        write_json(state_dir / f"auto_terms_cycle_{c:02d}.json", {"cycle": c, "terms": auto_terms})

        cfg_next = augment_config_with_auto_terms(base_cfg, auto_terms)

        if delta <= eps:
            stable += 1
        else:
            stable = 0

        if stable >= stable_n:
            append_jsonl(cycle_ledger, {
                "ts_utc": UTCNOW(),
                "phase": "adversarial_converged",
                "cycle": c,
                "stable": stable,
                "total_findings": total
            })
            cfg = cfg_next
            break

        cfg = cfg_next

    consolidated_dir = adv_root / "CONSOLIDATED"
    ensure_dir(consolidated_dir)
    con_jsonl = consolidated_dir / "findings.jsonl"
    con_csv = consolidated_dir / "findings.csv"
    with con_jsonl.open("w", encoding="utf-8", newline="\n") as f:
        for fid in sorted(all_findings.keys()):
            f.write(json.dumps(dataclasses.asdict(all_findings[fid]), ensure_ascii=False) + "\n")
    with con_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["finding_id", "mv_code", "category", "weight", "pattern", "location", "path", "snippet"])
        for fid in sorted(all_findings.keys()):
            fnd = all_findings[fid]
            w.writerow([fnd.finding_id, fnd.mv_code, fnd.category, fnd.weight, fnd.pattern, fnd.location, fnd.file_path, fnd.snippet])

    summ = {"generated_utc": UTCNOW(), "total_findings": len(all_findings),
            "categories": {}, "mv_codes": {}}
    for fnd in all_findings.values():
        summ["categories"][fnd.category] = summ["categories"].get(fnd.category, 0) + 1
        if fnd.mv_code:
            summ["mv_codes"][fnd.mv_code] = summ["mv_codes"].get(fnd.mv_code, 0) + 1
    write_json(consolidated_dir / "summary.json", summ)
    return summ


# ----------------------------
# Neo4j export (CSV)
# ----------------------------


def neo4j_export(run_root: Path, files: List[Path], scan_roots: List[Path], adv_summary: Dict, cfg: Dict) -> None:
    neo = run_root / "NEO4J_IMPORT"
    ensure_dir(neo)
    files_csv = neo / "nodes_files.csv"
    findings_csv = neo / "nodes_findings.csv"
    mv_csv = neo / "nodes_mv.csv"
    rel_file_has_finding = neo / "rels_file_has_finding.csv"
    rel_finding_maps_mv = neo / "rels_finding_maps_mv.csv"

    with files_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_id:ID", "path", "relpath", "ext", "bucket", "size_bytes:long", "mtime_utc"])
        reg = DEFAULT_DOCTYPE_REGISTRY
        for p in files:
            ext = p.suffix.lower()
            bucket = bucket_for_ext(ext, reg)
            try:
                st = p.stat()
                size = st.st_size
                mtime = _dt.datetime.utcfromtimestamp(st.st_mtime).replace(microsecond=0).isoformat() + "Z"
            except Exception:
                size, mtime = "", ""
            rel = ""
            for r in scan_roots:
                if str(p).lower().startswith(str(r).lower()):
                    rel = safe_relpath(p, r)
                    break
            fid = stable_id(str(p))
            w.writerow([fid, str(p), rel, ext, bucket, size, mtime])

    mv_tax = (cfg.get("mv_mapping") or {}).get("mv_taxonomy") or {}
    with mv_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mv_code:ID", "label"])
        for code, label in sorted(mv_tax.items()):
            w.writerow([code, label])

    consolidated = run_root / "ADVERSARIAL" / "CONSOLIDATED" / "findings.jsonl"
    if not consolidated.exists():
        return

    with findings_csv.open("w", encoding="utf-8", newline="") as f_nodes, \
         rel_file_has_finding.open("w", encoding="utf-8", newline="") as f_rel1, \
         rel_finding_maps_mv.open("w", encoding="utf-8", newline="") as f_rel2:
        wn = csv.writer(f_nodes)
        wr1 = csv.writer(f_rel1)
        wr2 = csv.writer(f_rel2)
        wn.writerow(["finding_id:ID", "category", "mv_code", "weight:float", "pattern", "location", "snippet", "created_utc", "file_path"])
        wr1.writerow([":START_ID", ":END_ID", ":TYPE"])
        wr2.writerow([":START_ID", ":END_ID", ":TYPE"])
        for line in consolidated.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            wn.writerow([d.get("finding_id"), d.get("category"), d.get("mv_code"), d.get("weight"),
                         d.get("pattern"), d.get("location"), d.get("snippet"), d.get("created_utc"), d.get("file_path")])
            file_id = stable_id(d.get("file_path", ""))
            wr1.writerow([file_id, d.get("finding_id"), "HAS_FINDING"])
            if d.get("mv_code"):
                wr2.writerow([d.get("finding_id"), d.get("mv_code"), "MAPS_TO_MV"])


# ----------------------------
# Web harvest (minimal)
# ----------------------------


def load_seeds(seeds_dir: Path, bumper: BumperLog) -> List[str]:
    urls: Set[str] = set()
    for fn in ["seed_urls.txt", "seed_urls_user.txt"]:
        p = seeds_dir / fn
        if p.exists():
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                u = line.strip()
                if not u or u.startswith("#"):
                    continue
                urls.add(u)
    if not urls:
        urls.update([
            "https://www.legislature.mi.gov/",
            "https://www.legislature.mi.gov/Laws/MCL",
            "https://www.legislature.mi.gov/Laws/ChapterIndex",
            "https://www.legislature.mi.gov/Bills",
            "https://www.legislature.mi.gov/Committees",
        ])
    return sorted(urls)


def is_http_url(u: str) -> bool:
    try:
        p = urllib.parse.urlparse(u)
        return p.scheme in ("http", "https")
    except Exception:
        return False


def normalize_url(u: str) -> str:
    try:
        u = u.strip()
        p = urllib.parse.urlparse(u)
        if not p.scheme:
            return ""
        p = p._replace(fragment="")
        return p.geturl()
    except Exception:
        return ""


def fetch_url(url: str, out_dir: Path, bumper: BumperLog, timeout: int = 30, max_bytes: int = 20_000_000) -> Optional[Path]:
    ensure_dir(out_dir)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (LitigationOS MCL Plane Suite)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ct = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                bumper.add("FETCH_MAX_BYTES", url, {"max_bytes": max_bytes})
                return None
            ext = ".html"
            if "pdf" in ct or url.lower().endswith(".pdf"):
                ext = ".pdf"
            elif "json" in ct or url.lower().endswith(".json"):
                ext = ".json"
            elif "xml" in ct or url.lower().endswith(".xml"):
                ext = ".xml"
            fn = stable_id(url) + ext
            p = out_dir / fn
            p.write_bytes(data)
            return p
    except Exception as e:
        bumper.add("FETCH_FAIL", url, {"error": repr(e)})
        return None


def discover_links_from_html(text: str, base_url: str) -> List[str]:
    links = set()
    for m in re.finditer(r'href=["\']([^"\']+)["\']', text, flags=re.I):
        href = m.group(1).strip()
        if not href:
            continue
        if href.startswith("mailto:") or href.startswith("javascript:"):
            continue
        absu = urllib.parse.urljoin(base_url, href)
        absu = normalize_url(absu)
        if absu and is_http_url(absu):
            links.add(absu)
    return sorted(links)


def web_harvest_cycle(seed_urls: List[str], run_root: Path, bumper: BumperLog,
                      max_fetch: int = 40, max_discover: int = 200) -> Dict:
    web_dir = run_root / "WEB"
    ensure_dir(web_dir)
    fetched_dir = web_dir / "FETCHED"
    ensure_dir(fetched_dir)

    q: queue.Queue[str] = queue.Queue()
    for u in seed_urls:
        u = normalize_url(u)
        if u:
            q.put(u)

    seen: Set[str] = set()
    fetched: List[Dict] = []
    discovered: Set[str] = set()

    while not q.empty() and len(fetched) < max_fetch:
        u = q.get()
        if u in seen:
            continue
        seen.add(u)
        path = fetch_url(u, fetched_dir, bumper)
        if path is None:
            continue
        fetched.append({"url": u, "path": str(path)})
        if path.suffix.lower() in [".html", ".htm"]:
            txt = safe_read_text(path, bumper)
            for lnk in discover_links_from_html(txt, u):
                if len(discovered) >= max_discover:
                    break
                if "legislature.mi.gov" in lnk.lower():
                    discovered.add(lnk)

    write_json(web_dir / "fetched.json", {"generated_utc": UTCNOW(), "count": len(fetched), "items": fetched})
    write_json(web_dir / "discovered.json", {"generated_utc": UTCNOW(), "count": len(discovered), "items": sorted(discovered)})
    return {"fetched": len(fetched), "discovered": len(discovered)}


# ----------------------------
# Convergence orchestrator
# ----------------------------


def converge(run_root: Path, in_root: Path, out_root: Path, scan_roots: List[Path],
             doctype_registry: Dict, adv_cfg_path: Optional[Path], bumper: BumperLog,
             cycles: int, eps: int, stable_n: int, adversarial: bool,
             adv_cycles: int, adv_eps: int, adv_stable_n: int,
             web_enabled: bool, max_files: int, excludes: List[str], max_fetch: int) -> Dict:
    run_dir = run_root
    ensure_dir(run_dir)

    ensure_dir(run_dir / "RUN")
    ensure_dir(run_dir / "INVENTORY")

    ledger = run_dir / "RUN" / "cycle_ledger.jsonl"

    seeds_dir = out_root / "seeds"
    ensure_dir(seeds_dir)
    user_seed = in_root / "seed_urls_user.txt"
    if user_seed.exists():
        shutil.copy2(user_seed, seeds_dir / "seed_urls_user.txt")

    seed_urls = load_seeds(seeds_dir, bumper)

    files = list(iter_files(scan_roots, bumper, excludes=excludes, max_files=max_files))
    inv_csv = run_dir / "INVENTORY" / "doctype_bucket_inventory.csv"
    write_bucket_inventory(files, doctype_registry, scan_roots, inv_csv)

    last_metric = 0
    stable = 0
    best_summary: Dict = {}

    for c in range(1, cycles + 1):
        wh = web_harvest_cycle(seed_urls, run_dir, bumper, max_fetch=max_fetch) if web_enabled else {"fetched": 0, "discovered": 0}

        adv_summary: Dict = {}
        if adversarial:
            adv_cfg = load_adversarial_config(adv_cfg_path, bumper)
            fetched_paths = []
            fetched_json = run_dir / "WEB" / "fetched.json"
            if fetched_json.exists():
                try:
                    data = json.loads(fetched_json.read_text(encoding="utf-8"))
                    for it in data.get("items", []):
                        p = Path(it.get("path", ""))
                        if p.exists():
                            fetched_paths.append(p)
                except Exception:
                    pass
            scan_files = files + fetched_paths
            adv_summary = adversarial_converge(
                files=scan_files,
                scan_roots=scan_roots,
                run_root=run_dir,
                base_cfg=adv_cfg,
                bumper=bumper,
                cycles=max(1, int(adv_cycles)),
                eps=int(adv_eps),
                stable_n=max(1, int(adv_stable_n))
            )
            best_summary = adv_summary

        metric = int((best_summary.get("total_findings") or 0)) + int(wh.get("fetched") or 0)
        delta = metric - last_metric
        last_metric = metric

        append_jsonl(ledger, {
            "ts_utc": UTCNOW(),
            "phase": "outer_cycle",
            "cycle": c,
            "metric": metric,
            "delta": delta,
            "eps": eps,
            "stable_n": stable_n,
            "web": wh,
            "adversarial": adv_summary
        })

        if delta <= eps:
            stable += 1
        else:
            stable = 0

        if stable >= stable_n:
            append_jsonl(ledger, {
                "ts_utc": UTCNOW(),
                "phase": "outer_converged",
                "cycle": c,
                "stable": stable,
                "metric": metric
            })
            break

    if adversarial:
        adv_cfg = load_adversarial_config(adv_cfg_path, bumper)
        neo4j_export(run_dir, files, scan_roots, best_summary, adv_cfg)

    final_md = run_dir / "FINAL_DELIVERABLE.md"
    final = {
        "suite": {"name": SUITE_NAME, "version": SUITE_VERSION},
        "run_id": run_dir.name,
        "generated_utc": UTCNOW(),
        "scan_roots": [str(p) for p in scan_roots],
        "files_seen": len(files),
        "adversarial_enabled": bool(adversarial),
        "adversarial_summary": best_summary,
        "web_harvest_enabled": bool(web_enabled),
    }
    final_md.write_text("# FINAL_DELIVERABLE\n\n```json\n" + json.dumps(final, indent=2) + "\n```\n", encoding="utf-8", newline="\n")

    prov = {
        "generated_utc": UTCNOW(),
        "suite_version": SUITE_VERSION,
        "python": sys.version,
        "platform": sys.platform,
        "optional_deps": {
            "python-docx": HAS_DOCX,
            "striprtf": HAS_STRIPRTF,
            "PyMuPDF": HAS_PYMUPDF,
            "py7zr": HAS_PY7ZR,
            "watchdog": HAS_WATCHDOG,
        }
    }
    write_json(run_dir / "RUN" / "provenance_index.json", prov)
    write_json(run_dir / "RUN" / "blockers_and_acquisition_plan.json", bumper.to_json())

    return final


# ----------------------------
# Watch mode
# ----------------------------

if HAS_WATCHDOG:
    class _WatchHandler(FileSystemEventHandler):  # type: ignore
        def __init__(self, trigger_fn):
            self.trigger_fn = trigger_fn

        def on_any_event(self, event):
            self.trigger_fn(event)
else:
    class _WatchHandler(object):
        def __init__(self, trigger_fn):
            self.trigger_fn = trigger_fn


def watch_and_incremental(args, scan_roots: List[Path], doctype_registry: Dict, bumper: BumperLog) -> None:
    if not HAS_WATCHDOG:
        bumper.add("MISSING_DEP_WATCHDOG", "watch_mode", {"install": "watchdog"})
        return
    run_id = mk_run_id()
    run_root = Path(args.out_path) / "RUNS" / run_id
    ensure_dir(run_root)
    debounce_s = 2.0
    last = {"t": 0.0}

    def trigger(event):
        t = time.time()
        if (t - last["t"]) < debounce_s:
            return
        last["t"] = t
        try:
            converge(
                run_root=run_root,
                in_root=Path(args.in_path),
                out_root=Path(args.out_path),
                scan_roots=scan_roots,
                doctype_registry=doctype_registry,
                adv_cfg_path=Path(args.adversarial_config) if args.adversarial_config else None,
                bumper=bumper,
                cycles=args.cycles,
                eps=args.eps,
                stable_n=args.stable_n,
                adversarial=args.adversarial,
                adv_cycles=args.adv_cycles,
                adv_eps=args.adv_eps,
                adv_stable_n=args.adv_stable_n,
                web_enabled=boolish(args.web),
                max_files=args.max_files,
                excludes=args.exclude,
                max_fetch=args.max_fetch
            )
        except Exception as e:
            bumper.add("WATCH_RUN_FAIL", "watch_cycle", {"error": repr(e)})

    observer = Observer()
    for r in scan_roots:
        if r.exists():
            observer.schedule(_WatchHandler(trigger), str(r), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


# ----------------------------
# CLI
# ----------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mcl_plane_suite.py")
    p.add_argument("--in", dest="in_path", default="IN", help="Input root (default IN)")
    p.add_argument("--out", dest="out_path", default="OUT", help="Output root (default OUT)")
    p.add_argument("--mode", choices=["once", "converge", "watch"], default="converge")
    p.add_argument("--cycles", type=int, default=25)
    p.add_argument("--eps", type=int, default=0, help="Convergence epsilon (delta <= eps)")
    p.add_argument("--stable-n", type=int, default=3, help="Stable cycles required to stop")
    p.add_argument("--adversarial", type=str, default="true", help="Enable adversarial scan (true/false)")
    p.add_argument("--adversarial-config", default="", help="Optional path to adversarial_config.json")
    p.add_argument("--adv-cycles", type=int, default=10, help="Inner adversarial convergence cycles")
    p.add_argument("--adv-eps", type=int, default=0, help="Inner adversarial convergence eps")
    p.add_argument("--adv-stable-n", type=int, default=3, help="Inner adversarial stable cycles to stop")
    p.add_argument("--web", type=str, default="true", help="Enable web harvest (true/false)")
    p.add_argument("--doctype-registry", default="", help="Optional path to doctype_registry.json")
    p.add_argument("--scan-root", action="append", default=[], help="Root to scan (repeatable). Default: IN only.")
    p.add_argument("--scan-all-drives", action="store_true", help="Windows: scan all fixed drives (C:,D:,E:...).")
    p.add_argument("--max-files", type=int, default=200000, help="Max files to enumerate (0 = unlimited)")
    p.add_argument("--exclude", action="append", default=[], help="Regex exclude (repeatable)")
    p.add_argument("--max-fetch", type=int, default=40, help="Max URLs fetched per outer cycle")
    return p.parse_args(argv)


def boolish(s: str) -> bool:
    return str(s).strip().lower() in ("1", "true", "yes", "y", "on")


def windows_drive_roots(bumper: BumperLog) -> List[Path]:
    roots = []
    for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
        p = Path(f"{letter}:/")
        if p.exists():
            roots.append(p)
    if not roots:
        bumper.add("NO_WINDOWS_DRIVES_FOUND", "scan-all-drives", {})
    return roots


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    bumper = BumperLog()

    in_root = Path(getattr(args, "in_path")).resolve()
    out_root = Path(getattr(args, "out_path")).resolve()

    ensure_dir(in_root)
    ensure_dir(out_root)
    ensure_dir(out_root / "RUNS")
    ensure_dir(out_root / "schema")
    ensure_dir(out_root / "queries")
    ensure_dir(out_root / "seeds")

    dr_path = out_root / "schema" / "doctype_registry.json"
    if not dr_path.exists():
        write_json(dr_path, DEFAULT_DOCTYPE_REGISTRY)
    mv_path = out_root / "schema" / "event_to_mv_mapping.json"
    if not mv_path.exists():
        write_json(mv_path, DEFAULT_EVENT_TO_MV)
    cy_path = out_root / "queries" / "bumper_query_pack.cypher"
    if not cy_path.exists():
        cy_path.write_text("// See suite README for query pack.\n", encoding="utf-8", newline="\n")

    doctype_registry = load_doctype_registry(Path(args.doctype_registry) if args.doctype_registry else None, bumper)

    if args.scan_root:
        scan_roots = [Path(x) for x in args.scan_root]
    elif args.scan_all_drives and sys.platform.startswith("win"):
        scan_roots = windows_drive_roots(bumper)
    else:
        scan_roots = [in_root]

    default_ex = [
        r"/windows/", r"/program files", r"/programdata/", r"/appdata/",
        r"/\$recycle\.bin", r"/system volume information", r"/node_modules/",
        r"/\.git/", r"/\.venv/", r"/venv/", r"/__pycache__/", r"/dist/", r"/build/",
    ]
    excludes = default_ex + (args.exclude or [])

    adversarial = boolish(args.adversarial)

    if args.mode == "watch":
        watch_and_incremental(args, scan_roots, doctype_registry, bumper)
        return 0

    run_id = mk_run_id()
    run_root = out_root / "RUNS" / run_id
    ensure_dir(run_root)

    converge(
        run_root=run_root,
        in_root=in_root,
        out_root=out_root,
        scan_roots=scan_roots,
        doctype_registry=doctype_registry,
        adv_cfg_path=Path(args.adversarial_config) if args.adversarial_config else None,
        bumper=bumper,
        cycles=max(1, int(args.cycles)),
        eps=int(args.eps),
        stable_n=max(1, int(args.stable_n)),
        adversarial=adversarial,
        adv_cycles=max(1, int(args.adv_cycles)),
        adv_eps=int(args.adv_eps),
        adv_stable_n=max(1, int(args.adv_stable_n)),
        web_enabled=boolish(args.web),
        max_files=int(args.max_files),
        excludes=excludes,
        max_fetch=int(args.max_fetch),
    )
    print(f"[OK] Run complete: {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
