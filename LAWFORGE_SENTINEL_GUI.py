#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LAWFORGE_SENTINEL_GUI.py — Windows-first, single-file GUI + CLI
All-in-one litigation sentinel: scan folders, extract text (PDF/DOCX/TXT, OCR fallback),
detect MCR/MCL/Canon cites, judges, entities, deadlines, build knowledge graph,
produce affidavit DOCX, XLSX, ICS calendar, HTML dashboard, Parquet, ZIP bundle,
text dumps, page-level JSON, and more. Includes Watch mode, ZIP expansion,
Windows Task Scheduler hook, plugin system, and dependency auto-installer.

Copy/paste-ready. One file. Designed for Windows; also runs on macOS/Linux.

Quickstart (Windows, PowerShell):
  python -V               # ensure 3.10+
  python LAWFORGE_SENTINEL_GUI.py --ensure-deps
  python LAWFORGE_SENTINEL_GUI.py --gui

Headless scan:
  python LAWFORGE_SENTINEL_GUI.py --roots "F:\\MEEK1" "Z:\\LAWFORGE_SERVER" \\
    --out "F:\\LegalResults\\SENTINEL" --ocr --bundle

Watch mode:
  python LAWFORGE_SENTINEL_GUI.py --roots "F:\\ALL" --out "F:\\LegalResults\\SENTINEL" --ocr --bundle --watch

Create Windows Scheduler task:
  python LAWFORGE_SENTINEL_GUI.py --schedule onlogon --roots "F:\\ALL" \\
    --out "F:\\LegalResults\\SENTINEL" --ocr --bundle --watch

PowerShell bootstrap (prints a here-string you can paste):
  python LAWFORGE_SENTINEL_GUI.py --print-ps-bootstrap

Plugin scaffold (adds ./SENTINEL_OUT/plugins/my_ai_plugin.py template):
  python LAWFORGE_SENTINEL_GUI.py --write-plugin my_ai_plugin --out "F:\\LegalResults\\SENTINEL"

LEGAL: Flags potential issues. No legal advice. Verify citations & rules yourself.
"""

from __future__ import annotations

# -----------------------------
# Standard lib
# -----------------------------
import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import sys
import time
import zipfile
import threading
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

# -----------------------------
# Dependency manager (auto-install)
# -----------------------------
REQUIRED_PKGS = [
    "PySide6",
    "watchdog",
    "pdfplumber",
    "pypdfium2",
    "pytesseract",
    "pillow",
    "python-docx",
    "pandas",
    "regex",
    "Unidecode",
    "tqdm",
    "networkx",
    "rapidfuzz",
    "openpyxl",
    "pyarrow",
]


def _pip_install(pkgs: List[str]):
    """Install packages quietly. Best-effort."""
    if not pkgs:
        return
    cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + pkgs
    try:
        subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        pass


def ensure_deps():
    missing = []
    for name in REQUIRED_PKGS:
        try:
            __import__(name if name != "Unidecode" else "unidecode")
        except Exception:
            missing.append(name)
    if missing:
        print("Installing missing packages:", ", ".join(missing))
        _pip_install(missing)


def _imp(name, alt=None):
    return __import__(name if not alt else alt)


# Attempt imports; if missing and user didn't run --ensure-deps yet, we'll retry later
try:
    pdfplumber = _imp("pdfplumber")
    PIL = _imp("PIL")
    Image = PIL.Image
    pd = _imp("pandas")
    docx = _imp("docx")
    regex = _imp("regex")
    unidecode = _imp("unidecode").unidecode
    tqdm = _imp("tqdm").tqdm
    nx = _imp("networkx")
    rapidfuzz = _imp("rapidfuzz")
    from PySide6 import QtCore, QtGui, QtWidgets
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    pytesseract = None
    pypdfium2 = None
    with contextlib.suppress(Exception):
        pytesseract = __import__("pytesseract")
    with contextlib.suppress(Exception):
        pypdfium2 = __import__("pypdfium2")
except Exception:
    pdfplumber = PIL = pd = docx = regex = None

    def _identity(s: str) -> str:
        return s

    unidecode = _identity
    tqdm = nx = rapidfuzz = None
    QtCore = QtGui = QtWidgets = None
    Observer = FileSystemEventHandler = None
    pytesseract = pypdfium2 = None

# -----------------------------
# Constants & patterns
# -----------------------------
SUPPORTED_EXT = {".pdf", ".docx", ".txt"}
DB_NAME = "sentinel_cache.sqlite3"
LOG_NAME = "sentinel.log"
DETECTIONS_JSONL = "detections.jsonl"
DETECTIONS_CSV = "detections.csv"
ENTITIES_CSV = "entities.csv"
DEADLINES_CSV = "deadlines.csv"
CHECKLIST_TXT = "checklist.txt"
AFFIDAVIT_DOCX = "affidavit_table.docx"
GRAPHML = "knowledge.graphml"
GRAPH_NODES_CSV = "graph_nodes.csv"
GRAPH_EDGES_CSV = "graph_edges.csv"
BUNDLE_ZIP = "SENTINEL_bundle.zip"
TEXT_DUMPS_DIR = "text_dumps"
PER_PAGE_JSON_DIR = "pages_json"
ZIP_INVENTORY_CSV = "zip_inventory.csv"
XLSX_BOOK = "sentinel.xlsx"
ICS_FILE = "deadlines.ics"
HTML_INDEX = "index.html"
SLA_JSON = "sla_report.json"
BURST_JSON = "burst_report.json"
PLUGINS_JSONL = "plugins.jsonl"

re2 = regex if regex else re

PATTERNS = {
    "MCR": re2.compile(r"\bMCR\s*\d\.\d+(?:\([a-z0-9]+\))*", re2.I),
    "MCL": re2.compile(r"\bMCL\s*\d{3}\.\d+[a-z0-9]*?(?:\([a-z0-9]+\))*", re2.I),
    "CANON": re2.compile(r"\bCanon\s*\d+(?:\([a-z0-9]+\))*", re2.I),
}
PAT_JUDGE = re2.compile(r"\b(?:Hon\.?|Judge)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", re2.I)
PAT_ENTITY = re2.compile(r"\b([A-Z][A-Z0-9&\-\.,]{3,})\b")
NUM_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
}
PAT_WITHIN_DAYS = re2.compile(
    r"\bwithin\s+(?:(\d{1,3})|(" + "|".join(NUM_WORDS) + r"))\s+day[s]?\b", re2.I
)
PAT_NO_LATER_THAN = re2.compile(
    r"\bno\s+later\s+than\s+(?:(\d{1,3})|(" + "|".join(NUM_WORDS) + r"))\s+day[s]?\b",
    re2.I,
)
PAT_CAPTION = re2.compile(
    r"\b(?:STATE OF|IN THE|CIRCUIT COURT|DISTRICT COURT|PROBATE COURT)\b", re2.I
)
PAT_RELIEF = re2.compile(r"\b(?:RELIEF REQUESTED|PRAYER FOR RELIEF|WHEREFORE)\b", re2.I)
PAT_SERVICE = re2.compile(r"\b(?:PROOF OF SERVICE|CERTIFICATE OF SERVICE)\b", re2.I)
PAT_CITATION = re2.compile(r"\b(?:MCR|MCL|USC|Fed\. R\.)\b", re2.I)
PAT_PARTY_PAIR = re2.compile(
    r"\b(?:Plaintiff|Defendant|Petitioner|Respondent)\s*[:\-–]\s*([A-Z][A-Za-z ,.'-]+)"
)


# -----------------------------
# Utils
# -----------------------------
def setup_logging(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(out_dir / LOG_NAME, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def normalize_text(s: str) -> str:
    try:
        s = unidecode(s or "")
    except Exception:
        s = s or ""
    s = s.replace("\x00", " ")
    return re.sub(r"[ \t]+", " ", s).strip()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def ensure_sqlite(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS files(file_sha TEXT PRIMARY KEY, path TEXT, size INTEGER, mtime REAL, ext TEXT);
    CREATE TABLE IF NOT EXISTS pages(file_sha TEXT, page_num INTEGER, page_sha TEXT, text TEXT,
                                     PRIMARY KEY (file_sha,page_num));
    CREATE TABLE IF NOT EXISTS detections(file_sha TEXT, page_num INTEGER, rule_type TEXT, rule TEXT,
                                          context TEXT, start INTEGER, end INTEGER);
    CREATE TABLE IF NOT EXISTS entities(file_sha TEXT, page_num INTEGER, entity_type TEXT, value TEXT, context TEXT);
    CREATE TABLE IF NOT EXISTS deadlines(file_sha TEXT, page_num INTEGER, phrase TEXT, days INTEGER,
                                         ref_date TEXT, due_date TEXT);
    CREATE TABLE IF NOT EXISTS cadence(entity TEXT, dates TEXT);
    """
    )
    conn.commit()
    return conn


def list_files(roots: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        p = Path(root)
        if not p.exists():
            continue
        if p.is_file():
            if p.suffix.lower() in SUPPORTED_EXT:
                files.append(p)
            continue
        for path in p.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXT:
                files.append(path)
    return files


def expand_zips(roots: Sequence[str], out_dir: Path) -> List[Tuple[str, str]]:
    mapping = []
    target_base = out_dir / "_unzipped"
    target_base.mkdir(parents=True, exist_ok=True)
    for root in roots:
        for z in Path(root).rglob("*.zip"):
            try:
                t = target_base / f"{z.stem}"
                if not t.exists():
                    t.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(z, "r") as zf:
                        zf.extractall(t)
                mapping.append((str(z), str(t)))
            except Exception as e:
                logging.warning(f"Unzip failed {z}: {e}")
    if mapping:
        try:
            pd.DataFrame(mapping, columns=["zip_path", "extracted_dir"]).to_csv(
                out_dir / ZIP_INVENTORY_CSV, index=False
            )
        except Exception:
            pass
    return mapping


def extract_pdf_text(
    path: Path, use_ocr: bool, dpi: int = 200, lang: str = "eng"
) -> List[str]:
    texts: List[str] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                texts.append(normalize_text(t))
    except Exception as e:
        logging.warning(f"pdf fail {path}: {e}")
    if use_ocr and (not any(texts) or all(len(t.strip()) < 10 for t in texts)):
        if not (pypdfium2 and pytesseract):
            logging.warning("OCR requested but pypdfium2/pytesseract missing")
            return texts or [""]
        try:
            pdf = pypdfium2.PdfDocument(str(path))
            for i in range(len(pdf)):
                page = pdf.get_page(i)
                bmp = page.render(scale=dpi / 72.0).to_pil()
                t = pytesseract.image_to_string(bmp, lang=lang)
                texts.append(normalize_text(t))
        except Exception as e:
            logging.warning(f"OCR fail {path}: {e}")
    return texts or [""]


def extract_docx_text(path: Path) -> List[str]:
    try:
        d = docx.Document(str(path))
        return [normalize_text("\n".join(p.text for p in d.paragraphs))]
    except Exception as e:
        logging.warning(f"docx fail {path}: {e}")
        return [""]


def extract_txt_text(path: Path) -> List[str]:
    try:
        return [normalize_text(Path(path).read_text(encoding="utf-8", errors="ignore"))]
    except Exception as e:
        logging.warning(f"txt fail {path}: {e}")
        return [""]


def extract_text(path: Path, use_ocr: bool) -> List[str]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf_text(path, use_ocr)
    if ext == ".docx":
        return extract_docx_text(path)
    if ext == ".txt":
        return extract_txt_text(path)
    return [""]


# -----------------------------
# Detections
# -----------------------------
@dataclasses.dataclass
class FileRecord:
    file_sha: str
    path: str
    size: int
    mtime: float
    ext: str


def upsert_file(conn: sqlite3.Connection, fr: FileRecord):
    conn.execute(
        "INSERT INTO files(file_sha,path,size,mtime,ext) VALUES(?,?,?,?,?) "
        "ON CONFLICT(file_sha) DO UPDATE SET "
        "path=excluded.path,size=excluded.size,"
        "mtime=excluded.mtime,ext=excluded.ext",
        (fr.file_sha, fr.path, fr.size, fr.mtime, fr.ext),
    )
    conn.commit()


def cache_pages(conn: sqlite3.Connection, file_sha: str, pages: List[str]):
    for i, t in enumerate(pages, 1):
        psha = hashlib.sha256((t or "").encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT OR REPLACE INTO pages(file_sha,page_num,page_sha,text) VALUES(?,?,?,?)",
            (file_sha, i, psha, t or ""),
        )
    conn.commit()


def already_processed(
    conn: sqlite3.Connection, file_sha: str, size: int, mtime: float
) -> bool:
    row = conn.execute(
        "SELECT size,mtime FROM files WHERE file_sha=?", (file_sha,)
    ).fetchone()
    return bool(row and row[0] == size and abs(row[1] - mtime) < 1e-6)


def human_snippet(text: str, start: int, end: int, radius: int = 80) -> str:
    a = max(0, start - radius)
    b = min(len(text), end + radius)
    return text[a:b]


def parse_days(m: Any) -> Optional[int]:
    try:
        num = m.group(1) or m.group(2)
    except Exception:
        return None
    if not num:
        return None
    return int(num) if str(num).isdigit() else NUM_WORDS.get(str(num).lower())


def detect_rules(text: str) -> List[Tuple[str, str, int, int]]:
    out = []
    for rtype, pat in PATTERNS.items():
        for m in pat.finditer(text):
            out.append((rtype, m.group(0), m.start(), m.end()))
    return out


def detect_judges(text: str) -> List[str]:
    return [m.group(1).strip() for m in PAT_JUDGE.finditer(text)]


def detect_entities(text: str) -> List[str]:
    vals = []
    for m in PAT_ENTITY.finditer(text):
        w = m.group(1)
        if len(w) < 4 or w in {"THE", "AND", "FOR", "WITH", "THIS"}:
            continue
        vals.append(w)
    seen: set[str] = set()
    out: List[str] = []
    for v in vals:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def detect_deadlines(text: str, ref_date: dt.date) -> List[Tuple[str, int, dt.date]]:
    out = []
    for pat in (PAT_WITHIN_DAYS, PAT_NO_LATER_THAN):
        for m in pat.finditer(text):
            d = parse_days(m)
            if d:
                out.append((m.group(0), d, ref_date + dt.timedelta(days=d)))
    return out


def procedural_audit(text: str) -> Dict[str, bool]:
    return {
        "caption_present": bool(PAT_CAPTION.search(text)),
        "relief_present": bool(PAT_RELIEF.search(text)),
        "citations_present": bool(PAT_CITATION.search(text)),
        "service_present": bool(PAT_SERVICE.search(text)),
    }


def record_detections(
    conn: sqlite3.Connection, file_sha: str, page_num: int, text: str
):
    for rtype, rule, s, e in detect_rules(text):
        conn.execute(
            "INSERT INTO detections(file_sha,page_num,rule_type,rule,context,start,end) VALUES(?,?,?,?,?,?,?)",
            (file_sha, page_num, rtype, rule, human_snippet(text, s, e), s, e),
        )
    for j in detect_judges(text):
        conn.execute(
            "INSERT INTO entities(file_sha,page_num,entity_type,value,context) VALUES(?,?,?,?,?)",
            (file_sha, page_num, "judge", j, ""),
        )
    for ent in detect_entities(text):
        conn.execute(
            "INSERT INTO entities(file_sha,page_num,entity_type,value,context) VALUES(?,?,?,?,?)",
            (file_sha, page_num, "entity", ent, ""),
        )
    ref_date = dt.date.fromtimestamp(time.time())
    for phrase, days, due in detect_deadlines(text, ref_date):
        conn.execute(
            "INSERT INTO deadlines(file_sha,page_num,phrase,days,ref_date,due_date) VALUES(?,?,?,?,?,?)",
            (file_sha, page_num, phrase, days, ref_date.isoformat(), due.isoformat()),
        )
    conn.commit()


def adversary_names(text: str) -> List[str]:
    return [m.group(1).strip() for m in PAT_PARTY_PAIR.finditer(text)]


def update_cadence(conn: sqlite3.Connection, names: List[str], file_mtime: float):
    if not names:
        return
    date_iso = dt.date.fromtimestamp(file_mtime).isoformat()
    for nm in set(names):
        row = conn.execute("SELECT dates FROM cadence WHERE entity=?", (nm,)).fetchone()
        if row:
            dates = json.loads(row[0])
            if date_iso not in dates:
                dates.append(date_iso)
        else:
            dates = [date_iso]
        dates.sort()
        conn.execute(
            "INSERT INTO cadence(entity,dates) VALUES(?,?) ON CONFLICT(entity) DO UPDATE SET dates=excluded.dates",
            (nm, json.dumps(dates)),
        )
    conn.commit()


# -----------------------------
# Plugin system
# -----------------------------
def load_plugins(plugins_dir: Path) -> List[Any]:
    plugins: List[Any] = []
    plugins_dir.mkdir(exist_ok=True)
    for p in sorted(plugins_dir.glob("*.py")):
        if p.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(p.stem, str(p))
            mod = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(mod)  # type: ignore
            plugins.append(mod)
            logging.info(f"Loaded plugin: {p.name}")
        except Exception as e:
            logging.warning(f"Plugin load failed {p.name}: {e}")
    return plugins


def run_plugins_page(
    plugins: List[Any],
    out_path: Path,
    engine: "SentinelEngine",
    file_sha: str,
    page_num: int,
    text: str,
):
    if not plugins:
        return
    for mod in plugins:
        try:
            if hasattr(mod, "analyze_page"):
                res = mod.analyze_page(engine, file_sha, page_num, text)
                if res:
                    with open(out_path, "a", encoding="utf-8") as f:
                        for r in res if isinstance(res, list) else [res]:
                            f.write(
                                json.dumps(
                                    {"plugin": getattr(mod, "NAME", mod.__name__), **r},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
        except Exception as e:
            logging.warning(f"Plugin {mod.__name__} analyze_page error: {e}")


def run_plugins_doc(
    plugins: List[Any],
    out_path: Path,
    engine: "SentinelEngine",
    file_sha: str,
    full_text: str,
):
    if not plugins:
        return
    for mod in plugins:
        try:
            if hasattr(mod, "analyze_document"):
                res = mod.analyze_document(engine, file_sha, full_text)
                if res:
                    with open(out_path, "a", encoding="utf-8") as f:
                        for r in res if isinstance(res, list) else [res]:
                            f.write(
                                json.dumps(
                                    {"plugin": getattr(mod, "NAME", mod.__name__), **r},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
        except Exception as e:
            logging.warning(f"Plugin {mod.__name__} analyze_document error: {e}")


# -----------------------------
# Exports
# -----------------------------


def export_tables(conn: sqlite3.Connection, out_dir: Path):
    rows = conn.execute(
        "SELECT file_sha,page_num,rule_type,rule,context,start,end FROM detections"
    ).fetchall()
    with open(out_dir / DETECTIONS_JSONL, "w", encoding="utf-8") as jf:
        for r in rows:
            jf.write(
                json.dumps(
                    {
                        "file_sha": r[0],
                        "page_num": r[1],
                        "rule_type": r[2],
                        "rule": r[3],
                        "context": r[4],
                        "start": r[5],
                        "end": r[6],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    pd.DataFrame(
        rows,
        columns=[
            "file_sha",
            "page_num",
            "rule_type",
            "rule",
            "context",
            "start",
            "end",
        ],
    ).to_csv(out_dir / DETECTIONS_CSV, index=False)
    pd.read_sql_query(
        "SELECT file_sha,page_num,entity_type,value,context FROM entities", conn
    ).to_csv(out_dir / ENTITIES_CSV, index=False)
    pd.read_sql_query(
        "SELECT file_sha,page_num,phrase,days,ref_date,due_date FROM deadlines", conn
    ).to_csv(out_dir / DEADLINES_CSV, index=False)


def export_checklist(conn: sqlite3.Connection, out_dir: Path):
    cur = conn.execute(
        "SELECT file_sha, GROUP_CONCAT(text, '\n') FROM pages GROUP BY file_sha"
    )
    lines = []
    for file_sha, text in cur.fetchall():
        text = text or ""
        audit = procedural_audit(text)
        score = sum(audit.values()) / 4.0
        lines.append(
            f"{file_sha} :: caption={audit['caption_present']} relief={audit['relief_present']} "
            f"citations={audit['citations_present']} service={audit['service_present']} completeness={score:.2f}"
        )
    (out_dir / CHECKLIST_TXT).write_text("\n".join(lines), encoding="utf-8")


def export_affidavit_docx(conn: sqlite3.Connection, out_dir: Path):
    d = docx.Document()
    d.add_heading("Sentinel Evidence Table", level=1)
    cur = conn.execute(
        "SELECT file_sha,page_num,rule_type,rule,context FROM detections ORDER BY file_sha,page_num"
    )
    rows = cur.fetchall()
    if not rows:
        d.add_paragraph("No rule detections.")
    else:
        table = d.add_table(rows=1, cols=5)
        hdr = table.rows[0].cells
        hdr[0].text, hdr[1].text, hdr[2].text, hdr[3].text, hdr[4].text = (
            "File SHA",
            "Page",
            "Type",
            "Rule",
            "Snippet",
        )
        for r in rows:
            c = table.add_row().cells
            c[0].text, c[1].text, c[2].text, c[3].text, c[4].text = (
                r[0],
                str(r[1]),
                r[2],
                r[3],
                r[4],
            )
    d.save(out_dir / AFFIDAVIT_DOCX)


def build_graph(conn: sqlite3.Connection):
    G = nx.Graph()
    for file_sha, path, ext in conn.execute("SELECT file_sha,path,ext FROM files"):
        G.add_node(f"file:{file_sha}", label="file", path=path, ext=ext)
    for file_sha, rule_type, rule in conn.execute(
        "SELECT file_sha,rule_type,rule FROM detections"
    ):
        rid = f"rule:{rule}"
        if not G.has_node(rid):
            G.add_node(rid, label="rule", rule_type=rule, text=rule)
        G.add_edge(f"file:{file_sha}", rid, relation="cites")
    for file_sha, etype, value in conn.execute(
        "SELECT file_sha,entity_type,value FROM entities"
    ):
        nid = f"{etype}:{value}"
        if not G.has_node(nid):
            G.add_node(nid, label=etype, text=value)
        G.add_edge(f"file:{file_sha}", nid, relation="mentions")
    return G


def export_graph(conn: sqlite3.Connection, out_dir: Path):
    G = build_graph(conn)
    nx.write_graphml(G, out_dir / GRAPHML)
    nodes = [{"id": nid, **data} for nid, data in G.nodes(data=True)]
    edges = [{"source": u, "target": v, **data} for u, v, data in G.edges(data=True)]
    pd.DataFrame(nodes).to_csv(out_dir / GRAPH_NODES_CSV, index=False)
    pd.DataFrame(edges).to_csv(out_dir / GRAPH_EDGES_CSV, index=False)


def export_text_dumps(conn: sqlite3.Connection, out_dir: Path):
    tdir = out_dir / TEXT_DUMPS_DIR
    tdir.mkdir(exist_ok=True)
    for (file_sha,) in conn.execute("SELECT DISTINCT file_sha FROM pages"):
        txt = "\n".join(
            [
                r[0] or ""
                for r in conn.execute(
                    "SELECT text FROM pages WHERE file_sha=? ORDER BY page_num",
                    (file_sha,),
                )
            ]
        )
        (tdir / f"{file_sha}.txt").write_text(txt, encoding="utf-8")


def export_page_json(conn: sqlite3.Connection, out_dir: Path):
    pdir = out_dir / PER_PAGE_JSON_DIR
    pdir.mkdir(exist_ok=True)
    for (file_sha,) in conn.execute("SELECT DISTINCT file_sha FROM pages"):
        rows = conn.execute(
            "SELECT page_num,text FROM pages WHERE file_sha=? ORDER BY page_num",
            (file_sha,),
        ).fetchall()
        pages = [{"page": r[0], "text": r[1] or ""} for r in rows]
        (pdir / f"{file_sha}.json").write_text(
            json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def export_xlsx(conn: sqlite3.Connection, out_dir: Path):
    with pd.ExcelWriter(out_dir / XLSX_BOOK, engine="openpyxl") as xw:
        pd.read_sql_query("SELECT * FROM detections", conn).to_excel(
            xw, sheet_name="detections", index=False
        )
        pd.read_sql_query("SELECT * FROM entities", conn).to_excel(
            xw, sheet_name="entities", index=False
        )
        pd.read_sql_query("SELECT * FROM deadlines", conn).to_excel(
            xw, sheet_name="deadlines", index=False
        )
        pd.read_sql_query("SELECT * FROM files", conn).to_excel(
            xw, sheet_name="files", index=False
        )


def export_parquet(conn: sqlite3.Connection, out_dir: Path):
    try:
        pd.read_sql_query("SELECT * FROM detections", conn).to_parquet(
            out_dir / "detections.parquet"
        )
        pd.read_sql_query("SELECT * FROM entities", conn).to_parquet(
            out_dir / "entities.parquet"
        )
        pd.read_sql_query("SELECT * FROM deadlines", conn).to_parquet(
            out_dir / "deadlines.parquet"
        )
    except Exception as e:
        logging.info(f"Parquet skipped: {e}")


def export_ics(out_dir: Path):
    path = out_dir / DEADLINES_CSV
    if not path.exists():
        return
    df = pd.read_csv(path)
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Lawforge Sentinel//EN"]
    now = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    for _, r in df.iterrows():
        uid = f"{r['file_sha']}-{r['page_num']}-{abs(hash(str(r['phrase'])))}@lawforge"
        due = str(r["due_date"]).replace("-", "")
        lines += [
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{now}",
            f"DTSTART;VALUE=DATE:{due}",
            f"SUMMARY:Deadline {r['days']} days — {r['phrase']}",
            "END:VEVENT",
        ]
    lines.append("END:VCALENDAR")
    (out_dir / ICS_FILE).write_text("\n".join(lines), encoding="utf-8")


def export_html_dashboard(out_dir: Path):
    html = f"""<!doctype html><html><head><meta charset="utf-8">
    <title>Lawforge Sentinel Dashboard</title>
    <style>
    body{{font-family:Segoe UI,Arial,sans-serif;background:#0f1115;color:#e8eaf0;margin:0}}
    header{{padding:16px 24px;border-bottom:1px solid #222;background:#11131a}}
    .wrap{{padding:20px}}
    a{{color:#61dafb}}
    table{{border-collapse:collapse;width:100%;margin:12px 0}}
    th,td{{border-bottom:1px solid #222;padding:8px 10px;text-align:left}}
    th{{background:#171a21;position:sticky;top:0}}
    </style></head><body>
    <header><h2>Lawforge Sentinel</h2></header>
    <div class="wrap">
    <p>Outputs generated at {dt.datetime.now().isoformat()}</p>
    <ul>
      <li><a href="{DETECTIONS_CSV}">detections.csv</a></li>
      <li><a href="{ENTITIES_CSV}">entities.csv</a></li>
      <li><a href="{DEADLINES_CSV}">deadlines.csv</a>  |  <a href="{ICS_FILE}">deadlines.ics</a></li>
      <li><a href="{GRAPHML}">knowledge.graphml</a></li>
      <li><a href="{XLSX_BOOK}">sentinel.xlsx</a></li>
      <li><a href="{ZIP_INVENTORY_CSV}">zip_inventory.csv</a></li>
      <li><a href="{PLUGINS_JSONL}">plugins.jsonl</a></li>
    </ul>
    </div></body></html>"""
    (out_dir / HTML_INDEX).write_text(html, encoding="utf-8")


def make_bundle(out_dir: Path):
    with zipfile.ZipFile(
        out_dir / BUNDLE_ZIP, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for name in [
            DETECTIONS_JSONL,
            DETECTIONS_CSV,
            ENTITIES_CSV,
            DEADLINES_CSV,
            CHECKLIST_TXT,
            AFFIDAVIT_DOCX,
            GRAPHML,
            GRAPH_NODES_CSV,
            GRAPH_EDGES_CSV,
            DB_NAME,
            LOG_NAME,
            XLSX_BOOK,
            ICS_FILE,
            HTML_INDEX,
            ZIP_INVENTORY_CSV,
            PLUGINS_JSONL,
        ]:
            p = out_dir / name
            if p.exists():
                zf.write(p, arcname=p.name)


def sla_and_burst_reports(conn: sqlite3.Connection, out_dir: Path):
    sla = {}
    for entity, dates_json in conn.execute("SELECT entity,dates FROM cadence"):
        dates = [dt.date.fromisoformat(d) for d in json.loads(dates_json)]
        dates.sort()
        if len(dates) < 2:
            continue
        intervals = [(b - a).days for a, b in zip(dates, dates[1:])]
        sla[entity] = {
            "count": len(intervals),
            "avg_days": sum(intervals) / len(intervals),
            "min_days": min(intervals),
            "max_days": max(intervals),
        }
    (out_dir / SLA_JSON).write_text(json.dumps(sla, indent=2), encoding="utf-8")
    all_dates = []
    for _, dates_json in conn.execute("SELECT entity,dates FROM cadence"):
        all_dates += [dt.date.fromisoformat(d) for d in json.loads(dates_json)]
    all_dates.sort()
    buckets = {}
    for d in all_dates:
        buckets[d] = buckets.get(d, 0) + 1
    burst = [{"date": d.isoformat(), "count": buckets[d]} for d in sorted(buckets)]
    (out_dir / BURST_JSON).write_text(json.dumps(burst, indent=2), encoding="utf-8")


# -----------------------------
# Engine
# -----------------------------
class SentinelEngine:
    def __init__(
        self,
        roots: Sequence[str],
        out_dir: Path,
        use_ocr: bool,
        bundle_on_run: bool = False,
        plugins_dir: Optional[Path] = None,
    ):
        self.roots = list(roots)
        self.out_dir = Path(out_dir)
        self.use_ocr = bool(use_ocr)
        self.bundle_on_run = bool(bundle_on_run)
        self._stop = threading.Event()
        self.conn: Optional[sqlite3.Connection] = None
        self.plugins_dir = plugins_dir or (self.out_dir / "plugins")
        self.plugins = load_plugins(self.plugins_dir)

    def stop(self):
        self._stop.set()

    def _conn(self) -> sqlite3.Connection:
        if self.conn is None:
            self.conn = ensure_sqlite(self.out_dir / DB_NAME)
        return self.conn

    def process_once(self, progress_cb=None):
        setup_logging(self.out_dir)
        conn = self._conn()
        expand_zips(self.roots, self.out_dir)
        files = list_files(self.roots)
        out_plugins = self.out_dir / PLUGINS_JSONL
        it = files if progress_cb is None else tqdm(files, desc="Processing files")
        for path in it:
            if self._stop.is_set():
                break
            try:
                stat = path.stat()
            except Exception:
                continue
            file_sha = sha256_file(path)
            if already_processed(conn, file_sha, stat.st_size, stat.st_mtime):
                continue
            fr = FileRecord(
                file_sha, str(path), stat.st_size, stat.st_mtime, path.suffix.lower()
            )
            upsert_file(conn, fr)
            pages = extract_text(path, self.use_ocr)
            cache_pages(conn, file_sha, pages)
            full_text = "\n".join(pages)
            for i, text in enumerate(pages, 1):
                record_detections(conn, file_sha, i, text)
                run_plugins_page(self.plugins, out_plugins, self, file_sha, i, text)
            run_plugins_doc(self.plugins, out_plugins, self, file_sha, full_text)
            update_cadence(conn, adversary_names(full_text), stat.st_mtime)
            if progress_cb:
                progress_cb(str(path))
        export_tables(conn, self.out_dir)
        export_checklist(conn, self.out_dir)
        export_affidavit_docx(conn, self.out_dir)
        export_graph(conn, self.out_dir)
        export_text_dumps(conn, self.out_dir)
        export_page_json(conn, self.out_dir)
        export_xlsx(conn, self.out_dir)
        export_parquet(conn, self.out_dir)
        export_ics(self.out_dir)
        export_html_dashboard(self.out_dir)
        sla_and_burst_reports(conn, self.out_dir)
        if self.bundle_on_run:
            make_bundle(self.out_dir)

    def watch(self):
        if Observer is None or FileSystemEventHandler is None:
            raise RuntimeError("watchdog is required for watch mode")
        engine = self

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                if event.is_directory:
                    return
                if Path(event.src_path).suffix.lower() in SUPPORTED_EXT:
                    engine.process_once()

        observer = Observer()
        handler = Handler()
        for root in self.roots:
            observer.schedule(handler, root, recursive=True)
        observer.start()
        try:
            while not self._stop.is_set():
                time.sleep(1)
        finally:
            observer.stop()
            observer.join()


# -----------------------------
# CLI / GUI helpers
# -----------------------------


def print_ps_bootstrap():
    script = r"""
$ErrorActionPreference = "Stop"
python -V
python LAWFORGE_SENTINEL_GUI.py --ensure-deps
python LAWFORGE_SENTINEL_GUI.py --roots "C:\\DATA" --out "C:\\SENTINEL_OUT" --ocr --bundle --watch
"""
    print(script.strip())


def write_plugin_template(name: str, out_dir: Path):
    plugins_dir = Path(out_dir) / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    path = plugins_dir / f"{name}.py"
    if path.exists():
        print(f"Plugin {name} already exists at {path}")
        return
    tpl = f"""NAME = "{name}"

def analyze_page(engine, file_sha, page_num, text):
    return None

def analyze_document(engine, file_sha, full_text):
    return None
"""
    path.write_text(tpl, encoding="utf-8")
    print(f"Wrote plugin template: {path}")


def schedule_task(args):
    if os.name != "nt":
        print("Scheduling is available only on Windows")
        return
    task_name = "LawforgeSentinel"
    root_args = " ".join(args.roots)
    options = ""
    if args.ocr:
        options += " --ocr"
    if args.bundle:
        options += " --bundle"
    if args.watch:
        options += " --watch"
    cmd = (
        f"schtasks /Create /SC {args.schedule} /TN {task_name} "
        f'/TR "{sys.executable} {Path(__file__).resolve()} --roots {root_args} --out {args.out}{options}" /F'
    )
    subprocess.run(cmd, shell=True, check=False)
    print("Scheduler task created")


class SentinelGUI(QtWidgets.QWidget):
    def __init__(self, engine: SentinelEngine):
        super().__init__()
        self.engine = engine
        self.setWindowTitle("Lawforge Sentinel")
        self.log = QtWidgets.QTextEdit(self)
        self.log.setReadOnly(True)
        self.btn_run = QtWidgets.QPushButton("Run Once", self)
        self.btn_stop = QtWidgets.QPushButton("Stop", self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.log)
        layout.addWidget(self.btn_run)
        layout.addWidget(self.btn_stop)
        self.btn_run.clicked.connect(self.run_scan)
        self.btn_stop.clicked.connect(self.engine.stop)

    def run_scan(self):
        self.log.append("Scanning...")

        def work():
            self.engine.process_once()
            self.log.append("Scan complete")

        threading.Thread(target=work, daemon=True).start()


def run_gui(args):
    if QtWidgets is None:
        raise RuntimeError("PySide6 is required for GUI mode")
    engine = SentinelEngine(args.roots, Path(args.out), args.ocr, args.bundle)
    app = QtWidgets.QApplication(sys.argv)
    gui = SentinelGUI(engine)
    gui.resize(600, 400)
    gui.show()
    app.exec()


# -----------------------------
# Main
# -----------------------------


def main(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Lawforge Sentinel")
    p.add_argument("--roots", nargs="*", default=[os.getcwd()])
    p.add_argument("--out", default="SENTINEL_OUT")
    p.add_argument("--ocr", action="store_true")
    p.add_argument("--bundle", action="store_true")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--ensure-deps", action="store_true")
    p.add_argument("--print-ps-bootstrap", action="store_true")
    p.add_argument("--schedule", choices=["once", "onlogon", "daily"], default=None)
    p.add_argument("--write-plugin", default=None)
    args = p.parse_args(argv)

    if args.ensure_deps:
        ensure_deps()
    if args.print_ps_bootstrap:
        print_ps_bootstrap()
        return
    if args.write_plugin:
        write_plugin_template(args.write_plugin, Path(args.out))
        return
    if args.schedule and args.schedule != "once":
        schedule_task(args)
        return
    if args.gui:
        run_gui(args)
        return
    engine = SentinelEngine(args.roots, Path(args.out), args.ocr, args.bundle)
    engine.process_once()
    if args.watch:
        engine.watch()


if __name__ == "__main__":
    main()
