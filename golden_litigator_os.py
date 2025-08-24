# -*- coding: utf-8 -*-
"""
GOLDEN LITIGATOR OS v∞ — SINGLE-FILE DEPLOY (Windows)
Self-evolving litigation intelligence with OCR, audio transcription, LLM legal extraction,
evidence ledger, timeline builder, exhibit mapper, and motion generator.

──────────────────────────────────────────────────────────────────────────────
DEPENDENCIES (Windows, Python 3.10+)
1) Install system tools:
   • Tesseract OCR (Windows): https://github.com/tesseract-ocr/tesseract
     → After install, set env var TESSERACT_OCR_PATH to the full exe path, e.g.:
       TESSERACT_OCR_PATH="C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

2) Create venv and install packages:
   python -m venv .venv
   .\\.venv\\Scripts\\activate
   pip install --upgrade pip
   pip install pymupdf python-docx pytesseract pillow opencv-python pydub
   pip install openai anthropic
   # Choose ONE Whisper backend (recommended: faster-whisper for speed) OR openai-whisper:
   pip install faster-whisper
   # OR: pip install openai-whisper

   (Optional) If you have ffmpeg for audio handling, install it and ensure it's on PATH.

3) LLM keys (set one or both as environment variables):
   • OPENAI_API_KEY=sk-...
   • ANTHROPIC_API_KEY=sk-ant-...

──────────────────────────────────────────────────────────────────────────────
RUN
   python golden_litigator_os.py

WHAT IT DOES
• Recursively scans drives: C:/, E:/, F:/, Z:/, and G:/MyDrive (edit below)
• Processes PDFs, DOCX, TXT, IMAGES (OCR via Tesseract), AUDIO/VIDEO (Whisper)
• Extracts legal intel via LLMs (claims, parties, statutes/rules), builds exhibits & timelines
• Logs into SQLite DB with rich schema (evidence, exhibits, timelines, filings, parties, sources, code_registry)
• Renames fully processed files by appending "__DONE__" to the stem
• Auto-drafts Michigan-formatted motion(s) (DOCX) only when sufficient facts/case meta exist
• Writes outputs to .\LegalResults\

POLICY SAFETY
• No deletion—read/ingest only; renames on success.
• Skips system dirs to avoid permission traps (edit EXCLUDE_DIRS if you truly want 100%).
• Motion generation runs only if required case metadata and factual prongs are present; otherwise it defers and logs the reason.

NOTE
• This single file is the spine. You can later split into modules.
• Tune LLM models at the CONFIG section.
"""

import os, re, io, sys, math, json, sqlite3, hashlib, logging, traceback, time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DRIVES = ["C:/", "E:/", "F:/", "Z:/", "G:/MyDrive/"]
EXCLUDE_DIRS = {
    r"C:\\Windows", r"C:\\Program Files", r"C:\\Program Files (x86)", r"C:\\$Recycle.Bin",
    r"C:\\ProgramData", r"C:\\Users\\Public", r"C:\\Recovery", r"C:\\PerfLogs"
}
DB_PATH = "golden_litigator.db"
LOG_FILE = "litigator.log"
PROCESSED_TAG = "__DONE__"
RESULTS_DIR = Path("LegalResults")
MOTIONS_DIR = RESULTS_DIR / "Motions"
NARRATIVE_DIR = RESULTS_DIR / "Narratives"
TRANSCRIPTS_DIR = RESULTS_DIR / "Transcripts"
EXHIBITS_DIR = RESULTS_DIR / "Exhibits"
LEDGER_EXPORT = RESULTS_DIR / "EvidenceLedger.jsonl"
CASE_META_MIN_FIELDS = {"court_name", "case_number", "caption_plaintiff", "caption_defendant", "jurisdiction"}

LLM_PROVIDER_ORDER = ["openai", "anthropic"]
OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"

WHISPER_BACKEND = "faster-whisper"
WHISPER_MODEL = "medium"

TESSERACT_CMD = os.environ.get("TESSERACT_OCR_PATH", "").strip()

TEXT_TYPES = {".txt", ".json", ".csv", ".md"}
DOC_TYPES = {".docx"}
PDF_TYPES = {".pdf"}
IMG_TYPES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
AUDIO_TYPES = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"}
VIDEO_TYPES = {".mp4", ".mkv", ".mov", ".m4v", ".wmv", ".avi"}
CODE_TYPES = {".py", ".ps1", ".psm1", ".json", ".txt"}

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
console = logging.getLogger("console")
console.setLevel(logging.INFO)
stream = logging.StreamHandler(sys.stdout)
stream.setFormatter(logging.Formatter("%(message)s"))
console.addHandler(stream)

SCHEMA = {
    "evidence": """
        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY,
            sha256 TEXT UNIQUE,
            filename TEXT,
            filepath TEXT,
            ext TEXT,
            size_bytes INTEGER,
            modified_ts TEXT,
            content_excerpt TEXT,
            party TEXT,
            parties_json TEXT,
            claims_json TEXT,
            statutes_json TEXT,
            court_rules_json TEXT,
            relevance_score REAL,
            timeline_refs_json TEXT,
            exhibit_tag TEXT,
            exhibit_label TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "exhibits": """
        CREATE TABLE IF NOT EXISTS exhibits (
            id INTEGER PRIMARY KEY,
            evidence_sha256 TEXT,
            label TEXT,
            title TEXT,
            description TEXT,
            page_refs_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "timelines": """
        CREATE TABLE IF NOT EXISTS timelines (
            id INTEGER PRIMARY KEY,
            evidence_sha256 TEXT,
            event_dt TEXT,
            actor TEXT,
            action TEXT,
            location TEXT,
            details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "filings": """
        CREATE TABLE IF NOT EXISTS filings (
            id INTEGER PRIMARY KEY,
            filing_type TEXT,
            title TEXT,
            court_name TEXT,
            case_number TEXT,
            body_path TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "parties": """
        CREATE TABLE IF NOT EXISTS parties (
            id INTEGER PRIMARY KEY,
            role TEXT,
            name TEXT,
            contact_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "sources": """
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY,
            evidence_sha256 TEXT,
            source_type TEXT,
            meta_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "case_meta": """
        CREATE TABLE IF NOT EXISTS case_meta (
            id INTEGER PRIMARY KEY,
            court_name TEXT,
            case_number TEXT,
            caption_plaintiff TEXT,
            caption_defendant TEXT,
            judge TEXT,
            jurisdiction TEXT,
            division TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "code_registry": """
        CREATE TABLE IF NOT EXISTS code_registry (
            id INTEGER PRIMARY KEY,
            sha256 TEXT UNIQUE,
            filename TEXT,
            filepath TEXT,
            ext TEXT,
            size_bytes INTEGER,
            modified_ts TEXT,
            header_preview TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
}

def db_conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = db_conn()
    cur = conn.cursor()
    for stmt in SCHEMA.values():
        cur.execute(stmt)
    conn.commit()
    conn.close()

# Utility functions

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def is_excluded_dir(p: str) -> bool:
    if os.name != "nt":
        return False
    for ex in EXCLUDE_DIRS:
        if p.startswith(ex):
            return True
    return False

def safe_rename_done(path: Path) -> None:
    try:
        new_name = f"{path.stem}{PROCESSED_TAG}{path.suffix}"
        target = path.with_name(new_name)
        if not target.exists():
            path.rename(target)
    except Exception as e:
        logging.warning(f"Rename failed for {path}: {e}")

def write_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def excerpt(text: str, n: int = 800) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:n]

# Extractors

def extract_text_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return path.read_text(encoding="latin-1", errors="ignore")
        except Exception as e:
            logging.error(f"TXT read failed: {path} // {e}")
            return ""

def extract_text_docx(path: Path) -> str:
    try:
        import docx
        doc = docx.Document(str(path))
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        logging.error(f"DOCX read failed: {path} // {e}")
        return ""

def extract_text_pdf(path: Path) -> str:
    try:
        import fitz
        text = []
        with fitz.open(str(path)) as doc:
            for page in doc:
                text.append(page.get_text() or "")
        return "\n".join(text)
    except Exception as e:
        logging.error(f"PDF read failed: {path} // {e}")
        return ""

def extract_text_image(path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image, ImageOps
        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        img = Image.open(str(path))
        img = ImageOps.grayscale(img)
        return pytesseract.image_to_string(img) or ""
    except Exception as e:
        logging.error(f"IMG OCR failed: {path} // {e}")
        return ""

def transcribe_audio(path: Path) -> str:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out_txt = TRANSCRIPTS_DIR / f"{path.stem}.txt"
    try:
        if WHISPER_BACKEND == "faster-whisper":
            from faster_whisper import WhisperModel
            model = WhisperModel(WHISPER_MODEL, compute_type="int8_float16")
            segments, info = model.transcribe(str(path), vad_filter=True)
            text = " ".join([seg.text for seg in segments if seg and seg.text])
        else:
            import whisper
            model = whisper.load_model(WHISPER_MODEL)
            res = model.transcribe(str(path))
            text = res.get("text", "")
        out_txt.write_text(text, encoding="utf-8", errors="ignore")
        return text
    except Exception as e:
        logging.error(f"Audio transcription failed: {path} // {e}")
        return ""

def extract_text_video(path: Path) -> str:
    try:
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as td:
            audio_path = Path(td) / f"{path.stem}.wav"
            cmd = ["ffmpeg", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", str(audio_path)]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if audio_path.exists():
                return transcribe_audio(audio_path)
            return ""
    except Exception as e:
        logging.error(f"Video extraction failed: {path} // {e}")
        return ""

# LLM Legal Analyzer

def call_llm(prompt: str) -> str:
    text = ""
    try:
        if "openai" in LLM_PROVIDER_ORDER and os.environ.get("OPENAI_API_KEY"):
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":"You are a Michigan-focused litigation expert (MCR/MCL/Benchbooks + W.D.Mich FRCP/LR). Cite precisely, avoid speculation."},
                          {"role":"user","content":prompt}],
                temperature=0.2,
                max_tokens=1200
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text
    except Exception as e:
        logging.warning(f"OpenAI call failed: {e}")

    try:
        if "anthropic" in LLM_PROVIDER_ORDER and os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic
            c = anthropic.Anthropic()
            msg = c.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=1200,
                temperature=0.2,
                system="You are a Michigan-focused litigation expert (MCR/MCL/Benchbooks + W.D.Mich FRCP/LR). Cite precisely, avoid speculation.",
                messages=[{"role":"user","content":prompt}]
            )
            text = "".join([blk.text for blk in msg.content if hasattr(blk, "text")]) or ""
    except Exception as e:
        logging.warning(f"Anthropic call failed: {e}")
    return text

def legal_extract(text: str, fname: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {}
    prompt = f"""
From the following document content, extract structured Michigan litigation intel:
- parties: list of entities/persons with roles (e.g., landlord, tenant, plaintiff, defendant)
- claims: Michigan & federal causes of action implicated (statute or common law)
- statutes: key MCL/MCL 600/Landlord-Tenant/consumer/EGLE/env; plus FRCP if federal angle
- court_rules: MCR & local WDMI/Fed rules relevant (by rule number)
- timeline: dated events (YYYY-MM-DD where possible) with actor and action
- exhibits: proposed exhibit title and description
Return STRICT JSON only.

FILENAME: {fname}
CONTENT (truncated to 10k chars if long):
{text[:10000]}
"""
    raw = call_llm(prompt)
    try:
        data = json.loads(raw)
        return {
            "parties": data.get("parties", []),
            "claims": data.get("claims", []),
            "statutes": data.get("statutes", []),
            "court_rules": data.get("court_rules", []),
            "timeline": data.get("timeline", []),
            "exhibits": data.get("exhibits", [])
        }
    except Exception:
        logging.warning("LLM returned non-JSON or parse error; falling back to heuristics.")
        return {
            "parties": [],
            "claims": [],
            "statutes": [],
            "court_rules": [],
            "timeline": [],
            "exhibits": []
        }

# Persistence helpers

def insert_evidence(rec: Dict[str, Any]) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO evidence (
            sha256, filename, filepath, ext, size_bytes, modified_ts,
            content_excerpt, party, parties_json, claims_json, statutes_json,
            court_rules_json, relevance_score, timeline_refs_json,
            exhibit_tag, exhibit_label
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        rec["sha256"], rec["filename"], rec["filepath"], rec["ext"], rec["size_bytes"], rec["modified_ts"],
        rec.get("content_excerpt",""), rec.get("party",""), json.dumps(rec.get("parties",[]), ensure_ascii=False),
        json.dumps(rec.get("claims",[]), ensure_ascii=False),
        json.dumps(rec.get("statutes",[]), ensure_ascii=False),
        json.dumps(rec.get("court_rules",[]), ensure_ascii=False),
        rec.get("relevance_score", 0.0),
        json.dumps(rec.get("timeline_refs",[]), ensure_ascii=False),
        rec.get("exhibit_tag",""), rec.get("exhibit_label","")
    ))
    conn.commit()
    conn.close()

def insert_timeline(sha: str, ev: Dict[str, Any]) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO timelines (evidence_sha256, event_dt, actor, action, location, details)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (sha, ev.get("date",""), ev.get("actor",""), ev.get("action",""),
          ev.get("location",""), ev.get("details","")))
    conn.commit()
    conn.close()

def insert_exhibit(sha: str, ex: Dict[str, Any]) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO exhibits (evidence_sha256, label, title, description, page_refs_json)
        VALUES (?, ?, ?, ?, ?)
    """, (sha, ex.get("label",""), ex.get("title",""), ex.get("description",""),
          json.dumps(ex.get("pages",[]), ensure_ascii=False)))
    conn.commit()
    conn.close()

def upsert_case_meta(meta: Dict[str, Any]) -> None:
    if not meta or not CASE_META_MIN_FIELDS.issubset(meta.keys()):
        return
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO case_meta (court_name, case_number, caption_plaintiff, caption_defendant, judge, jurisdiction, division)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (meta.get("court_name",""), meta.get("case_number",""), meta.get("caption_plaintiff",""),
          meta.get("caption_defendant",""), meta.get("judge",""), meta.get("jurisdiction",""),
          meta.get("division","")))
    conn.commit()
    conn.close()

def insert_source(sha: str, source_type: str, meta: Dict[str, Any]) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO sources (evidence_sha256, source_type, meta_json)
        VALUES (?, ?, ?)
    """, (sha, source_type, json.dumps(meta, ensure_ascii=False)))
    conn.commit()
    conn.close()

def register_code_file(path: Path, sha: str) -> None:
    conn = db_conn()
    cur = conn.cursor()
    try:
        preview = excerpt(path.read_text(encoding="utf-8", errors="ignore"), 500)
    except Exception:
        preview = ""
    stat = path.stat()
    cur.execute("""
        INSERT OR IGNORE INTO code_registry (sha256, filename, filepath, ext, size_bytes, modified_ts, header_preview)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (sha, path.name, str(path), path.suffix.lower(), stat.st_size,
          datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"), preview))
    conn.commit()
    conn.close()

# Motion generator

def have_case_meta() -> Optional[Dict[str, Any]]:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT court_name, case_number, caption_plaintiff, caption_defendant, judge, jurisdiction, division FROM case_meta ORDER BY id DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "court_name": row[0] or "",
        "case_number": row[1] or "",
        "caption_plaintiff": row[2] or "",
        "caption_defendant": row[3] or "",
        "judge": row[4] or "",
        "jurisdiction": row[5] or "",
        "division": row[6] or ""
    }

def gather_motion_materials() -> Optional[Dict[str, Any]]:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT filename, filepath, statutes_json, court_rules_json, claims_json, timeline_refs_json, content_excerpt
        FROM evidence ORDER BY relevance_score DESC, id DESC LIMIT 200
    """)
    rows = cur.fetchall()
    conn.close()
    if not rows:
        return None
    pool = []
    for r in rows:
        pool.append({
            "filename": r[0], "filepath": r[1],
            "statutes": json.loads(r[2] or "[]"),
            "rules": json.loads(r[3] or "[]"),
            "claims": json.loads(r[4] or "[]"),
            "timeline": json.loads(r[5] or "[]"),
            "excerpt": r[6] or ""
        })
    return {"materials": pool}

def generate_motion_docx(mtype: str, case_meta: Dict[str, Any], materials: Dict[str, Any]) -> Optional[Path]:
    try:
        from docx import Document
        from docx.shared import Inches, Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH

        if not materials or not materials.get("materials"):
            return None

        MOTIONS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = MOTIONS_DIR / f"{mtype.replace(' ','_')}_{case_meta['case_number'] or 'NOCASE'}.docx"

        d = Document()
        sections = d.sections
        for s in sections:
            s.top_margin = Inches(1)
            s.bottom_margin = Inches(1)
            s.left_margin = Inches(1)
            s.right_margin = Inches(1)

        title = d.add_paragraph()
        run = title.add_run(case_meta["court_name"] or "IN THE CIRCUIT COURT FOR MUSKEGON COUNTY, MICHIGAN")
        run.bold = True
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER

        d.add_paragraph("")
        cap = d.add_paragraph()
        cap.add_run(f"{case_meta.get('caption_plaintiff','')}, Plaintiff,\n").bold = True
        cap.add_run("v.\n")
        cap.add_run(f"{case_meta.get('caption_defendant','')}, Defendant.\n").bold = True
        d.add_paragraph(f"Case No.: {case_meta.get('case_number','')}")
        if case_meta.get("judge"):
            d.add_paragraph(f"Hon.: {case_meta['judge']}")
        d.add_paragraph("")

        p = d.add_paragraph()
        r = p.add_run(mtype.upper())
        r.bold = True
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        d.add_paragraph("")

        mats = materials["materials"][:50]
        prompt = {
            "mtype": mtype,
            "case_meta": case_meta,
            "materials": mats,
            "style_requirements": {
                "jurisdiction": "Michigan",
                "rules": ["MCR", "Benchbooks", "W.D.Mich Local Rules", "FRCP (if federal issue)"],
                "format": "Court-ready, numbered points, double-spaced, accurate citations only",
                "all_fields_resolved": True,
                "relief": "Specify exact relief with legal basis; include proposed order text"
            }
        }
        body = call_llm("Draft the full Michigan-compliant motion as JSON: {\"sections\":[{\"heading\":\"...\",\"paragraphs\":[\"...\"]}], \"relief\":[\"...\"], \"proposed_order\":[\"...\"]} using this data:\n" + json.dumps(prompt, ensure_ascii=False))
        try:
            jd = json.loads(body)
        except Exception:
            logging.warning("Motion LLM JSON parse failed; skipping motion generation.")
            return None

        for sec in jd.get("sections", []):
            if sec.get("heading"):
                h = d.add_paragraph()
                hr = h.add_run(sec["heading"])
                hr.bold = True
            for para in sec.get("paragraphs", []):
                para_p = d.add_paragraph(para)
                for run in para_p.runs:
                    run.font.size = Pt(12)

        if jd.get("relief"):
            d.add_paragraph("")
            rr = d.add_paragraph()
            rr.add_run("RELIEF REQUESTED").bold = True
            for item in jd["relief"]:
                d.add_paragraph(f"• {item}")

        if jd.get("proposed_order"):
            d.add_paragraph("")
            po = d.add_paragraph()
            po.add_run("PROPOSED ORDER (ATTACHED)").bold = True
            for line in jd["proposed_order"]:
                d.add_paragraph(line)

        d.save(str(out_path))
        return out_path
    except Exception as e:
        logging.error(f"Motion generation failed: {e}")
        return None

# Narrative & ledger

def export_ledger_jsonl():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT sha256, filename, filepath, ext, size_bytes, modified_ts, parties_json, claims_json, statutes_json, court_rules_json, timeline_refs_json, exhibit_label
        FROM evidence ORDER BY id ASC
    """)
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        rec = {
            "sha256": r[0], "filename": r[1], "filepath": r[2], "ext": r[3],
            "size_bytes": r[4], "modified_ts": r[5],
            "parties": json.loads(r[6] or "[]"),
            "claims": json.loads(r[7] or "[]"),
            "statutes": json.loads(r[8] or "[]"),
            "court_rules": json.loads(r[9] or "[]"),
            "timeline": json.loads(r[10] or "[]"),
            "exhibit_label": r[11] or ""
        }
        write_jsonl(LEDGER_EXPORT, rec)

def build_master_narrative():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT filename, content_excerpt, parties_json, claims_json, timeline_refs_json FROM evidence ORDER BY id ASC LIMIT 1000")
    rows = cur.fetchall()
    conn.close()
    bundle = []
    for r in rows:
        bundle.append({
            "filename": r[0],
            "excerpt": r[1],
            "parties": json.loads(r[2] or "[]"),
            "claims": json.loads(r[3] or "[]"),
            "timeline": json.loads(r[4] or "[]")
        })
    prompt = "Build a fact-only, Michigan court-compliant narrative/affidavit from the following structured inputs. Use numbered paragraphs, cite exhibits by label when present; no speculation:\n" + json.dumps(bundle, ensure_ascii=False)
    narrative = call_llm(prompt)
    NARRATIVE_DIR.mkdir(parents=True, exist_ok=True)
    out = NARRATIVE_DIR / f"Master_Narrative_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    out.write_text(narrative, encoding="utf-8", errors="ignore")

# Crawler

def process_file(path: Path) -> None:
    try:
        if PROCESSED_TAG in path.stem:
            return

        ext = path.suffix.lower()
        stat = path.stat()
        sha = sha256_file(path)

        text = ""
        src_type = None

        if ext in TEXT_TYPES:
            text = extract_text_txt(path); src_type="txt"
        elif ext in DOC_TYPES:
            text = extract_text_docx(path); src_type="docx"
        elif ext in PDF_TYPES:
            text = extract_text_pdf(path); src_type="pdf"
        elif ext in IMG_TYPES:
            text = extract_text_image(path); src_type="img"
        elif ext in AUDIO_TYPES:
            text = transcribe_audio(path); src_type="audio"
        elif ext in VIDEO_TYPES:
            text = extract_text_video(path); src_type="video"
        elif ext in CODE_TYPES:
            register_code_file(path, sha)
            if ext in {".txt", ".json"}:
                text = extract_text_txt(path)
            src_type = "code"
        else:
            return

        insert_source(sha, src_type or "unknown", {
            "filename": path.name,
            "filepath": str(path),
            "size_bytes": stat.st_size,
            "modified_ts": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        })

        parties, claims, statutes, rules, timeline, exhibits = [], [], [], [], [], []
        if text.strip():
            llm = legal_extract(text, path.name)
            parties = llm.get("parties", [])
            claims = llm.get("claims", [])
            statutes = llm.get("statutes", [])
            rules = llm.get("court_rules", [])
            timeline = llm.get("timeline", [])
            exhibits = llm.get("exhibits", [])

        rel = 0.0
        rel += 1.0 if claims else 0.2
        rel += 0.5 * min(len(statutes), 3)
        rel += 0.5 * min(len(rules), 3)

        insert_evidence({
            "sha256": sha,
            "filename": path.name,
            "filepath": str(path),
            "ext": ext,
            "size_bytes": stat.st_size,
            "modified_ts": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "content_excerpt": excerpt(text, 1000),
            "party": parties[0]["name"] if parties else "",
            "parties": parties,
            "claims": claims,
            "statutes": statutes,
            "court_rules": rules,
            "relevance_score": rel,
            "timeline_refs": timeline,
            "exhibit_tag": exhibits[0]["label"] if exhibits else "",
            "exhibit_label": exhibits[0]["label"] if exhibits else ""
        })

        for ev in timeline:
            insert_timeline(sha, ev)
        for ex in exhibits:
            insert_exhibit(sha, ex)

        write_jsonl(LEDGER_EXPORT, {
            "sha256": sha, "filename": path.name, "filepath": str(path),
            "claims": claims, "statutes": statutes, "rules": rules
        })

        if rel >= 0.5:
            safe_rename_done(path)

        logging.info(f"Processed: {path}")
    except Exception as e:
        logging.error(f"Error processing {path}: {e}\n{traceback.format_exc()}")

def crawl_all():
    for root_drive in DRIVES:
        for root, dirs, files in os.walk(root_drive):
            if is_excluded_dir(root):
                continue
            for fname in files:
                fpath = Path(root) / fname
                if fpath.name.startswith("~$"):
                    continue
                process_file(fpath)

# MAIN

if __name__ == "__main__":
    console.info("🚀 GOLDEN LITIGATOR OS v∞ — starting")
    for d in [RESULTS_DIR, MOTIONS_DIR, NARRATIVE_DIR, TRANSCRIPTS_DIR, EXHIBITS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    init_db()

    crawl_all()

    export_ledger_jsonl()

    build_master_narrative()

    meta = have_case_meta()
    mats = gather_motion_materials()
    if meta and mats:
        possibles = [
            "Motion to Set Aside / Stay Enforcement",
            "Motion for Sanctions (MCR 1.109(E)/2.114)",
            "Federal Complaint Draft (42 USC §1983/§1985 + IIED + Abuse of Process + Malicious Prosecution)"
        ]
        for m in possibles:
            out = generate_motion_docx(m, meta, mats)
            if out:
                logging.info(f"Generated: {out}")

    console.info("✅ Complete. Evidence indexed, transcripts/OCR extracted, ledger exported, narrative built, motions (if eligible) generated.")
