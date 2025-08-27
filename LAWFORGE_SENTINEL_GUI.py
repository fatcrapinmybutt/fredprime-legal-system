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
  python LAWFORGE_SENTINEL_GUI.py --roots "F:\\MEEK1" "Z:\\LAWFORGE_SERVER" --out "F:\\LegalResults\\SENTINEL" --ocr --bundle

Watch mode:
  python LAWFORGE_SENTINEL_GUI.py --roots "F:\\ALL" --out "F:\\LegalResults\\SENTINEL" --ocr --bundle --watch

Create Windows Scheduler task:
  python LAWFORGE_SENTINEL_GUI.py --schedule onlogon --roots "F:\\ALL" --out "F:\\LegalResults\\SENTINEL" --ocr --bundle --watch

PowerShell bootstrap (prints a here-string you can paste):
  python LAWFORGE_SENTINEL_GUI.py --print-ps-bootstrap

Plugin scaffold (adds ./SENTINEL_OUT/plugins/my_ai_plugin.py template):
  python LAWFORGE_SENTINEL_GUI.py --write-plugin my_ai_plugin

LEGAL: Flags potential issues. No legal advice. Verify citations & rules yourself.
"""

from __future__ import annotations

# -----------------------------
# Standard lib
# -----------------------------
import argparse, contextlib, dataclasses, datetime as dt, functools, hashlib, io, json, logging
import os, re, shutil, sqlite3, subprocess, sys, tempfile, textwrap, time, zipfile, threading, importlib.util, glob
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

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


# Ensure deps on import for CLI; GUI path can also call --ensure-deps explicitly.
try:
    # Lazy import in case user calls --ensure-deps first
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
    PySide6 = _imp("PySide6")
    from PySide6 import QtCore, QtGui, QtWidgets

    watchdog = _imp("watchdog")
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    pytesseract = None
    pypdfium2 = None
    with contextlib.suppress(Exception):
        pytesseract = __import__("pytesseract")
    with contextlib.suppress(Exception):
        pypdfium2 = __import__("pypdfium2")
except Exception:
    # In case user runs with --ensure-deps only; subsequent run loads all
    pdfplumber = PIL = pd = docx = regex = unidecode = tqdm = nx = rapidfuzz = None
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
        "ON CONFLICT(file_sha) DO UPDATE SET path=excluded.path,size=excluded.size,mtime=excluded.mtime,ext=excluded.ext",
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


def parse_days(m) -> Optional[int]:
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
    seen = set()
    out = []
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
    # Rules
    for rtype, rule, s, e in detect_rules(text):
        conn.execute(
            "INSERT INTO detections(file_sha,page_num,rule_type,rule,context,start,end) VALUES(?,?,?,?,?,?,?)",
            (file_sha, page_num, rtype, rule, human_snippet(text, s, e), s, e),
        )
    # Judges & entities
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
    # Deadlines
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
    plugins = []
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
                                    {"plugin": mod.__name__, **r}, ensure_ascii=False
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
                                    {"plugin": mod.__name__, **r}, ensure_ascii=False
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
            fr = FileRecord(
                file_sha=file_sha,
                path=str(path.resolve()),
                size=stat.st_size,
                mtime=stat.st_mtime,
                ext=path.suffix.lower(),
            )
            if already_processed(conn, fr.file_sha, fr.size, fr.mtime):
                continue
            upsert_file(conn, fr)
            pages = extract_text(path, self.use_ocr)
            cache_pages(conn, fr.file_sha, pages)
            for i, text in enumerate(pages, 1):
                t = text or ""
                if t:
                    record_detections(conn, fr.file_sha, i, t)
                    update_cadence(conn, adversary_names(t), fr.mtime)
                    run_plugins_page(self.plugins, out_plugins, self, fr.file_sha, i, t)
            # doc-level plugins
            full_text = "\n".join([p or "" for p in pages])
            if full_text.strip():
                run_plugins_doc(self.plugins, out_plugins, self, fr.file_sha, full_text)

        # Exports
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
        setup_logging(self.out_dir)

        class Handler(FileSystemEventHandler):  # type: ignore
            def __init__(self, engine: "SentinelEngine"):
                self.engine = engine
                self.lock = threading.Lock()

            def on_any_event(self, event):
                if self.engine._stop.is_set():
                    return
                if getattr(event, "is_directory", False):
                    return
                with self.lock:
                    logging.info(f"Change: {event.src_path}")
                    self.engine.process_once()

        self.process_once()
        obs = Observer()  # type: ignore
        h = Handler(self)
        for r in self.roots:
            if Path(r).exists():
                obs.schedule(h, r, recursive=True)
        obs.start()
        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        finally:
            obs.stop()
            obs.join()


# -----------------------------
# Windows scheduler helper
# -----------------------------
def create_windows_task(
    task_name: str,
    roots: Sequence[str],
    out_dir: str,
    ocr: bool = True,
    bundle: bool = True,
    watch: bool = True,
    schedule: str = "ONLOGON",
    interval_minutes: int = 60,
) -> Tuple[bool, str]:
    py_exe = sys.executable
    script_path = str(Path(__file__).resolve())
    args = [py_exe, script_path, "--roots"] + list(roots) + ["--out", out_dir]
    if ocr:
        args.append("--ocr")
    if bundle:
        args.append("--bundle")
    if watch:
        args.append("--watch")
    cmd = " ".join(f'"{a}"' if " " in a else a for a in args)
    if schedule.upper() in ("HOURLY", "MINUTE"):
        sch_cmd = [
            "schtasks",
            "/Create",
            "/TN",
            task_name,
            "/TR",
            cmd,
            "/SC",
            "MINUTE",
            "/MO",
            str(interval_minutes),
            "/F",
            "/RL",
            "HIGHEST",
            "/RU",
            "SYSTEM",
        ]
    else:
        sch_cmd = [
            "schtasks",
            "/Create",
            "/TN",
            task_name,
            "/TR",
            cmd,
            "/SC",
            "ONLOGON",
            "/F",
        ]
    try:
        res = subprocess.run(sch_cmd, capture_output=True, text=True)
        ok = res.returncode == 0
        msg = res.stdout if ok else res.stderr
        return ok, msg
    except Exception as e:
        return False, str(e)


# -----------------------------
# PowerShell bootstrap printer
# -----------------------------
def print_ps_bootstrap():
    script_name = Path(__file__).name
    ps = f"""
# Paste in elevated PowerShell to set up Lawforge Sentinel
$ErrorActionPreference = "Stop"
$script = "{script_name}"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here
python -V
if ($LASTEXITCODE -ne 0) {{ Write-Host "Python not found. Install Python 3.10+ first." -ForegroundColor Red; exit 1 }}
python "$script" --ensure-deps
if ($LASTEXITCODE -ne 0) {{ Write-Host "Dependency install failed." -ForegroundColor Red; exit 1 }}
python "$script" --gui
"""
    print(ps.strip())


# -----------------------------
# Plugin scaffold
# -----------------------------
PLUGIN_TEMPLATE = '''"""
Example plugin: analyze_page/analyze_document hooks.
Optional: use external libs (declare your own imports).
"""

NAME = "MyAIPlugin"

def analyze_page(engine, file_sha: str, page_num: int, text: str):
    # Return list of dicts; they will be written to plugins.jsonl
    hits = []
    if "ex parte" in text.lower():
        hits.append({"file_sha": file_sha, "page": page_num, "flag": "ex_parte_phrase"})
    return hits

def analyze_document(engine, file_sha: str, full_text: str):
    # Example doc-level analysis: count words
    wc = len(full_text.split())
    return [{"file_sha": file_sha, "doc_metric": "word_count", "value": wc}]
'''


def scaffold_plugin(name: str, out_dir: Path):
    plugins_dir = out_dir / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    fname = plugins_dir / f"{name}.py"
    if fname.exists():
        print(f"Plugin already exists: {fname}")
        return
    fname.write_text(PLUGIN_TEMPLATE, encoding="utf-8")
    print(f"Plugin created: {fname}")


# -----------------------------
# GUI
# -----------------------------
APP_NAME = "Lawforge Sentinel"
APP_VER = "0.9.1-beta"


class Worker(QtCore.QThread):  # type: ignore
    finished = QtCore.Signal(bool, str)  # type: ignore

    def __init__(self, engine: SentinelEngine, mode: str):
        super().__init__()
        self.engine = engine
        self.mode = mode

    def run(self):
        try:
            if self.mode == "once":
                self.engine.process_once()
            else:
                self.engine.watch()
            self.finished.emit(True, "done")
        except Exception as e:
            self.finished.emit(False, str(e))


class MainWin(QtWidgets.QMainWindow):  # type: ignore
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {APP_VER}")
        self.setMinimumSize(1120, 720)
        self._engine: Optional[SentinelEngine] = None
        self._worker: Optional[Worker] = None
        self._build_ui()
        self._apply_dark_theme()

    def _build_ui(self):
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        v = QtWidgets.QVBoxLayout(w)

        cfg = QtWidgets.QGroupBox("Configuration")
        v.addWidget(cfg)
        gl = QtWidgets.QGridLayout(cfg)
        self.roots_list = QtWidgets.QListWidget()
        btn_add_root = QtWidgets.QPushButton("Add Root")
        btn_add_root.clicked.connect(self.add_root)
        btn_del_root = QtWidgets.QPushButton("Remove")
        btn_del_root.clicked.connect(self.del_root)
        gl.addWidget(QtWidgets.QLabel("Roots"), 0, 0)
        gl.addWidget(self.roots_list, 1, 0, 3, 1)
        root_btns = QtWidgets.QVBoxLayout()
        root_btns.addWidget(btn_add_root)
        root_btns.addWidget(btn_del_root)
        root_btns.addStretch()
        gl.addLayout(root_btns, 1, 1, 3, 1)

        self.out_edit = QtWidgets.QLineEdit(str(Path.cwd() / "SENTINEL_OUT"))
        btn_out = QtWidgets.QPushButton("Browse")
        btn_out.clicked.connect(self.choose_out)
        gl.addWidget(QtWidgets.QLabel("Output"), 0, 2)
        gl.addWidget(self.out_edit, 1, 2)
        gl.addWidget(btn_out, 1, 3)

        self.chk_ocr = QtWidgets.QCheckBox("Enable OCR")
        self.chk_ocr.setChecked(True)
        self.chk_bundle = QtWidgets.QCheckBox("Create bundle ZIP")
        self.chk_bundle.setChecked(True)
        gl.addWidget(self.chk_ocr, 2, 2)
        gl.addWidget(self.chk_bundle, 2, 3)

        act = QtWidgets.QHBoxLayout()
        v.addLayout(act)
        self.btn_selfcheck = QtWidgets.QPushButton("Self-check")
        self.btn_selfcheck.clicked.connect(self.self_check)
        self.btn_run_once = QtWidgets.QPushButton("Run once")
        self.btn_run_once.clicked.connect(self.run_once)
        self.btn_watch = QtWidgets.QPushButton("Start watch")
        self.btn_watch.clicked.connect(self.toggle_watch)
        self.btn_scheduler = QtWidgets.QPushButton("Create Scheduler Task")
        self.btn_scheduler.clicked.connect(self.create_scheduler)
        self.btn_open_out = QtWidgets.QPushButton("Open Output")
        self.btn_open_out.clicked.connect(self.open_out)
        act.addWidget(self.btn_selfcheck)
        act.addWidget(self.btn_run_once)
        act.addWidget(self.btn_watch)
        act.addWidget(self.btn_scheduler)
        act.addStretch()
        act.addWidget(self.btn_open_out)

        self.tabs = QtWidgets.QTabWidget()
        v.addWidget(self.tabs, 1)
        self.tab_detect = self._make_table_tab(
            "Detections",
            ["file_sha", "page_num", "rule_type", "rule", "context", "start", "end"],
            DETECTIONS_CSV,
        )
        self.tab_entities = self._make_table_tab(
            "Entities",
            ["file_sha", "page_num", "entity_type", "value", "context"],
            ENTITIES_CSV,
        )
        self.tab_deadlines = self._make_table_tab(
            "Deadlines",
            ["file_sha", "page_num", "phrase", "days", "ref_date", "due_date"],
            DEADLINES_CSV,
        )
        self.tabs.addTab(self.tab_detect, "Detections")
        self.tabs.addTab(self.tab_entities, "Entities")
        self.tabs.addTab(self.tab_deadlines, "Deadlines")

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        v.addWidget(self.log, 1)
        self.status = self.statusBar()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tail_log)

    def _apply_dark_theme(self):
        app = QtWidgets.QApplication.instance()
        app.setStyle("Fusion")
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.Window, QtGui.QColor(18, 20, 28))
        pal.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Base, QtGui.QColor(15, 17, 23))
        pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(25, 28, 36))
        pal.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Button, QtGui.QColor(25, 28, 36))
        pal.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
        pal.setColor(QtGui.QPalette.Highlight, QtGui.QColor(97, 218, 251))
        pal.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
        app.setPalette(pal)
        self.setStyleSheet(
            """
        QGroupBox{border:1px solid #2a2f3a;border-radius:6px;margin-top:6px}
        QGroupBox::title{subcontrol-origin:margin;left:8px;padding:0 4px}
        QPushButton{padding:6px 12px;border:1px solid #3a3f4a;border-radius:4px}
        QPushButton:hover{border-color:#61dafb;color:#61dafb}
        QLineEdit,QListWidget{border:1px solid #2a2f3a;border-radius:4px;padding:4px}
        QTabWidget::pane{border:1px solid #2a2f3a;border-radius:6px}
        QHeaderView::section{background:#171a21;color:#e8eaf0;padding:4px;border:0}
        """
        )

    def _make_table_tab(self, title: str, headers: List[str], csv_name: str):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        h = QtWidgets.QHBoxLayout()
        v.addLayout(h)
        btn_reload = QtWidgets.QPushButton("Reload")
        btn_export = QtWidgets.QPushButton("Open CSV")
        h.addWidget(btn_reload)
        h.addWidget(btn_export)
        h.addStretch()
        table = QtWidgets.QTableWidget()
        table.setSortingEnabled(True)
        v.addWidget(table, 1)

        def reload():
            out_dir = Path(self.out_edit.text())
            path = out_dir / csv_name
            if not path.exists():
                table.setRowCount(0)
                table.setColumnCount(len(headers))
                table.setHorizontalHeaderLabels(headers)
                return
            import csv

            with open(path, newline="", encoding="utf-8") as f:
                r = list(csv.reader(f))
            if not r:
                return
            table.setRowCount(len(r) - 1)
            table.setColumnCount(len(r[0]))
            table.setHorizontalHeaderLabels(r[0])
            for i, row in enumerate(r[1:]):
                for j, val in enumerate(row):
                    table.setItem(i, j, QtWidgets.QTableWidgetItem(val))
            table.resizeColumnsToContents()

        btn_reload.clicked.connect(reload)

        def open_csv():
            path = str(Path(self.out_edit.text()) / csv_name)
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

        btn_export.clicked.connect(open_csv)
        reload()
        return w

    # Actions
    def add_root(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose root")
        if d:
            self.roots_list.addItem(d)

    def del_root(self):
        for item in self.roots_list.selectedItems():
            self.roots_list.takeItem(self.roots_list.row(item))

    def choose_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Choose output")
        if d:
            self.out_edit.setText(d)

    def _engine_from_ui(self) -> SentinelEngine:
        roots = [self.roots_list.item(i).text() for i in range(self.roots_list.count())]
        if not roots:
            QtWidgets.QMessageBox.warning(
                self, "Missing roots", "Add at least one root"
            )
            raise RuntimeError("no roots")
        out_dir = Path(self.out_edit.text())
        eng = SentinelEngine(
            roots,
            out_dir,
            use_ocr=self.chk_ocr.isChecked(),
            bundle_on_run=self.chk_bundle.isChecked(),
        )
        return eng

    def self_check(self):
        msgs = []
        import platform

        msgs.append(f"Python {platform.python_version()}")
        for mod in [
            "pdfplumber",
            "PIL",
            "pandas",
            "docx",
            "regex",
            "unidecode",
            "tqdm",
            "networkx",
            "rapidfuzz",
            "watchdog",
            "openpyxl",
            "pyarrow",
            "PySide6",
        ]:
            try:
                __import__(mod)
                msgs.append(f"{mod}: OK")
            except Exception:
                msgs.append(f"{mod}: MISSING")
        tesseract = shutil.which("tesseract") is not None
        msgs.append(f"tesseract in PATH: {'OK' if tesseract else 'MISSING'}")
        QtWidgets.QMessageBox.information(self, "Self-check", "\n".join(msgs))

    def run_once(self):
        try:
            self._engine = self._engine_from_ui()
        except Exception:
            return
        self._worker = Worker(self._engine, "once")
        self._worker.finished.connect(self._work_done)
        self._lock_ui(False)
        self._worker.start()
        self.timer.start(1000)

    def toggle_watch(self):
        if self._worker and self._worker.isRunning():
            if self._engine:
                self._engine.stop()
            return
        try:
            self._engine = self._engine_from_ui()
        except Exception:
            return
        self._worker = Worker(self._engine, "watch")
        self._worker.finished.connect(self._work_done)
        self._lock_ui(False)
        self._worker.start()
        self.timer.start(1000)
        self.btn_watch.setText("Stop watch")

    def _work_done(self, ok: bool, msg: str):
        self._lock_ui(True)
        self.btn_watch.setText("Start watch")
        self.timer.stop()
        self._tail_log()
        if not ok:
            QtWidgets.QMessageBox.critical(self, "Error", msg)

    def _lock_ui(self, unlocked: bool):
        for b in [self.btn_run_once, self.btn_scheduler, self.btn_selfcheck]:
            b.setEnabled(unlocked)

    def _tail_log(self):
        try:
            p = Path(self.out_edit.text()) / LOG_NAME
            if not p.exists():
                return
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()[-100000:]
            self.log.setPlainText(txt)
            self.log.verticalScrollBar().setValue(
                self.log.verticalScrollBar().maximum()
            )
        except Exception:
            pass

    def create_scheduler(self):
        roots = [self.roots_list.item(i).text() for i in range(self.roots_list.count())]
        out_dir = self.out_edit.text()
        ok, msg = create_windows_task(
            "LAWFORGE_SENTINEL_Watch",
            roots,
            out_dir,
            ocr=self.chk_ocr.isChecked(),
            bundle=self.chk_bundle.isChecked(),
            watch=True,
            schedule="ONLOGON",
        )
        QtWidgets.QMessageBox.information(
            self, "Scheduler", ("Created\n" if ok else "Failed\n") + msg
        )

    def open_out(self):
        p = Path(self.out_edit.text())
        p.mkdir(exist_ok=True, parents=True)
        if sys.platform.startswith("win"):
            os.startfile(str(p))  # type: ignore
        else:
            subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(p)])


# -----------------------------
# CLI
# -----------------------------
def build_args():
    p = argparse.ArgumentParser(
        "LAWFORGE_SENTINEL", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--roots", nargs="+", default=[], help="Root directories/files to scan"
    )
    p.add_argument(
        "--out", default=str(Path.cwd() / "SENTINEL_OUT"), help="Output directory"
    )
    p.add_argument("--ocr", action="store_true", help="Enable OCR fallback")
    p.add_argument("--bundle", action="store_true", help="Create bundle ZIP after run")
    p.add_argument("--watch", action="store_true", help="Watch mode")
    p.add_argument("--gui", action="store_true", help="Launch GUI")
    p.add_argument(
        "--ensure-deps", action="store_true", help="Install required Python packages"
    )
    p.add_argument(
        "--schedule", choices=["onlogon", "hourly"], help="Create Windows task (watch)"
    )
    p.add_argument(
        "--print-ps-bootstrap",
        action="store_true",
        help="Print PowerShell setup snippet",
    )
    p.add_argument("--write-plugin", help="Scaffold plugins/<name>.py")
    return p.parse_args()


def main():
    args = build_args()

    if args.ensure_deps:
        ensure_deps()
        print("Dependencies ensured.")
        if len(sys.argv) == 2:
            return

    if args.print_ps_bootstrap:
        print_ps_bootstrap()
        return

    if args.write_plugin:
        scaffold_plugin(args.write_plugin, Path(args.out))
        return

    # GUI
    if args.gui or (not args.roots and not args.schedule and not args.watch):
        ensure_deps()
        from PySide6 import QtWidgets  # re-import after ensure

        app = QtWidgets.QApplication(sys.argv)
        win = MainWin()
        win.show()
        sys.exit(app.exec())

    # Schedule Windows task
    if args.schedule:
        ok, msg = create_windows_task(
            "LAWFORGE_SENTINEL_Watch",
            roots=args.roots,
            out_dir=args.out,
            ocr=args.ocr,
            bundle=args.bundle,
            watch=True,
            schedule="HOURLY" if args.schedule == "hourly" else "ONLOGON",
        )
        print(("Created: " if ok else "Failed: ") + msg.strip())
        return

    # Headless engine
    engine = SentinelEngine(
        args.roots, Path(args.out), use_ocr=args.ocr, bundle_on_run=args.bundle
    )
    if args.watch:
        engine.watch()
    else:
        engine.process_once()


if __name__ == "__main__":
    main()
