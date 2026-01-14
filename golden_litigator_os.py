# -*- coding: utf-8 -*-
"""
GOLDEN LITIGATOR OS v∞ — SINGLE-FILE DEPLOY (Windows)
Self-evolving litigation intelligence with OCR, audio transcription, LLM legal extraction,
evidence ledger, timeline builder, exhibit mapper, and motion generator.

RUN
   python golden_litigator_os.py
"""

import json
import logging
import os
import re
import sqlite3
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Configuration
DRIVES = ["C:/", "E:/", "F:/", "Z:/", "G:/MyDrive/"]
EXCLUDE_DIRS = {
    r"C:\\Windows",
    r"C:\\Program Files",
    r"C:\\Program Files (x86)",
    r"C:\\$Recycle.Bin",
    r"C:\\ProgramData",
    r"C:\\Users\Public",
    r"C:\\Recovery",
    r"C:\\PerfLogs",
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

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
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
}


def db_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    conn = db_conn()
    cur = conn.cursor()
    for stmt in SCHEMA.values():
        cur.execute(stmt)
    conn.commit()
    conn.close()


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def is_excluded_dir(p: str) -> bool:
    if os.name != "nt":
        return False
    return any(p.startswith(ex) for ex in EXCLUDE_DIRS)


def safe_rename_done(path: Path) -> None:
    new_name = f"{path.stem}{PROCESSED_TAG}{path.suffix}"
    target = path.with_name(new_name)
    if not target.exists():
        try:
            path.rename(target)
        except OSError:
            logging.warning("Failed to rename %s", path)


def write_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def excerpt(text: str, n: int = 800) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:n]


def extract_text_txt(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def extract_text_docx(path: Path) -> str:
    try:
        import docx

        doc = docx.Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""


def extract_text_pdf(path: Path) -> str:
    try:
        import fitz  # type: ignore[import-not-found]

        text = []
        with fitz.open(str(path)) as doc:
            for page in doc:
                text.append(page.get_text() or "")
        return "\n".join(text)
    except Exception:
        return ""


def extract_text_image(path: Path) -> str:
    try:
        import pytesseract  # type: ignore[import-untyped]
        from PIL import Image, ImageOps

        if TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        img: Image.Image = Image.open(str(path))
        img = ImageOps.grayscale(img)
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""


def transcribe_audio(path: Path) -> str:
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    out_txt = TRANSCRIPTS_DIR / f"{path.stem}.txt"
    try:
        if WHISPER_BACKEND == "faster-whisper":
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]

            model = WhisperModel(WHISPER_MODEL, compute_type="int8_float16")
            segments, _ = model.transcribe(str(path), vad_filter=True)
            text = " ".join(seg.text for seg in segments if getattr(seg, "text", None))
        else:
            import whisper  # type: ignore[import-not-found]

            model = whisper.load_model(WHISPER_MODEL)
            res = model.transcribe(str(path))
            text = res.get("text", "")
        out_txt.write_text(text, encoding="utf-8", errors="ignore")
        return text
    except Exception:
        return ""


def extract_text_video(path: Path) -> str:
    try:
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            audio_path = Path(td) / f"{path.stem}.wav"
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(audio_path),
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if audio_path.exists():
                return transcribe_audio(audio_path)
            return ""
    except Exception:
        return ""


def call_llm(prompt: str) -> str:
    text = ""
    try:
        if "openai" in LLM_PROVIDER_ORDER and os.environ.get("OPENAI_API_KEY"):
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Michigan-focused litigation expert.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1200,
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text
    except Exception:
        pass
    try:
        if "anthropic" in LLM_PROVIDER_ORDER and os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic  # type: ignore[import-not-found]

            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=1200,
                temperature=0.2,
                system="You are a Michigan-focused litigation expert.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(block.text for block in msg.content if hasattr(block, "text"))
    except Exception:
        pass
    return text


def legal_extract(text: str, fname: str) -> Dict[str, Any]:
    if not text.strip():
        return {}
    prompt = f"""
From the following document content, extract structured Michigan litigation intel:
- parties
- claims
- statutes
- court_rules
- timeline
- exhibits
Return JSON only.
FILENAME: {fname}
CONTENT:
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
            "exhibits": data.get("exhibits", []),
        }
    except Exception:
        return {
            "parties": [],
            "claims": [],
            "statutes": [],
            "court_rules": [],
            "timeline": [],
            "exhibits": [],
        }


def insert_evidence(rec: Dict[str, Any]) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO evidence (
            sha256, filename, filepath, ext, size_bytes, modified_ts,
            content_excerpt, party, parties_json, claims_json, statutes_json,
            court_rules_json, relevance_score, timeline_refs_json,
            exhibit_tag, exhibit_label
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


def insert_timeline(sha: str, ev: Dict[str, Any]) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO timelines (evidence_sha256, event_dt, actor, action, location, details)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
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


def insert_exhibit(sha: str, ex: Dict[str, Any]) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO exhibits (evidence_sha256, label, title, description, page_refs_json)
        VALUES (?, ?, ?, ?, ?)
        """,
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


def insert_source(sha: str, source_type: str, meta: Dict[str, Any]) -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO sources (evidence_sha256, source_type, meta_json)
        VALUES (?, ?, ?)
        """,
        (sha, source_type, json.dumps(meta, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


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
            text = extract_text_txt(path)
            src_type = "txt"
        elif ext in DOC_TYPES:
            text = extract_text_docx(path)
            src_type = "docx"
        elif ext in PDF_TYPES:
            text = extract_text_pdf(path)
            src_type = "pdf"
        elif ext in IMG_TYPES:
            text = extract_text_image(path)
            src_type = "img"
        elif ext in AUDIO_TYPES:
            text = transcribe_audio(path)
            src_type = "audio"
        elif ext in VIDEO_TYPES:
            text = extract_text_video(path)
            src_type = "video"
        elif ext in CODE_TYPES:
            src_type = "code"
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                text = ""
        else:
            return
        insert_source(
            sha,
            src_type or "unknown",
            {
                "filename": path.name,
                "filepath": str(path),
                "size_bytes": stat.st_size,
                "modified_ts": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            },
        )
        parties: List[Dict[str, Any]] = []
        claims: List[Dict[str, Any]] = []
        statutes: List[str] = []
        rules: List[str] = []
        timeline: List[Dict[str, Any]] = []
        exhibits: List[Dict[str, Any]] = []
        if text.strip():
            llm = legal_extract(text, path.name)
            parties = llm.get("parties", [])
            claims = llm.get("claims", [])
            statutes = llm.get("statutes", [])
            rules = llm.get("court_rules", [])
            timeline = llm.get("timeline", [])
            exhibits = llm.get("exhibits", [])
        relevance = 0.0
        if claims:
            relevance += 1.0
        relevance += 0.5 * min(len(statutes), 3)
        relevance += 0.5 * min(len(rules), 3)
        insert_evidence(
            {
                "sha256": sha,
                "filename": path.name,
                "filepath": str(path),
                "ext": ext,
                "size_bytes": stat.st_size,
                "modified_ts": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "content_excerpt": excerpt(text, 1000),
                "party": parties[0]["name"] if parties else "",
                "parties": parties,
                "claims": claims,
                "statutes": statutes,
                "court_rules": rules,
                "relevance_score": relevance,
                "timeline_refs": timeline,
                "exhibit_tag": exhibits[0]["label"] if exhibits else "",
                "exhibit_label": exhibits[0]["label"] if exhibits else "",
            }
        )
        for ev in timeline:
            insert_timeline(sha, ev)
        for ex in exhibits:
            insert_exhibit(sha, ex)
        write_jsonl(
            LEDGER_EXPORT,
            {
                "sha256": sha,
                "filename": path.name,
                "filepath": str(path),
                "claims": claims,
                "statutes": statutes,
                "rules": rules,
            },
        )
        if relevance >= 0.5:
            safe_rename_done(path)
        logging.info("Processed %s", path)
    except Exception as e:  # noqa: BLE001
        logging.error("Error processing %s: %s\n%s", path, e, traceback.format_exc())


def crawl_all() -> None:
    for root_drive in DRIVES:
        for root, dirs, files in os.walk(root_drive):
            if is_excluded_dir(root):
                continue
            for fname in files:
                fpath = Path(root) / fname
                if fpath.name.startswith("~$"):
                    continue
                process_file(fpath)


def export_ledger_jsonl() -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT sha256, filename, filepath, ext, size_bytes, modified_ts, parties_json,
               claims_json, statutes_json, court_rules_json, timeline_refs_json, exhibit_label
        FROM evidence ORDER BY id ASC
        """
    )
    rows = cur.fetchall()
    conn.close()
    for r in rows:
        rec = {
            "sha256": r[0],
            "filename": r[1],
            "filepath": r[2],
            "ext": r[3],
            "size_bytes": r[4],
            "modified_ts": r[5],
            "parties": json.loads(r[6] or "[]"),
            "claims": json.loads(r[7] or "[]"),
            "statutes": json.loads(r[8] or "[]"),
            "court_rules": json.loads(r[9] or "[]"),
            "timeline": json.loads(r[10] or "[]"),
            "exhibit_label": r[11] or "",
        }
        write_jsonl(LEDGER_EXPORT, rec)


def build_master_narrative() -> None:
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT filename, content_excerpt, parties_json, claims_json, timeline_refs_json
        FROM evidence ORDER BY id ASC LIMIT 1000
        """
    )
    rows = cur.fetchall()
    conn.close()
    bundle = []
    for r in rows:
        bundle.append(
            {
                "filename": r[0],
                "excerpt": r[1],
                "parties": json.loads(r[2] or "[]"),
                "claims": json.loads(r[3] or "[]"),
                "timeline": json.loads(r[4] or "[]"),
            }
        )
    prompt = (
        "Build a fact-only, Michigan court-compliant narrative/affidavit from the following "
        "structured inputs. Use numbered paragraphs, cite exhibits by label when present; "
        "no speculation:\n" + json.dumps(bundle, ensure_ascii=False)
    )
    narrative = call_llm(prompt)
    NARRATIVE_DIR.mkdir(parents=True, exist_ok=True)
    out = NARRATIVE_DIR / f"Master_Narrative_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    out.write_text(narrative, encoding="utf-8", errors="ignore")


if __name__ == "__main__":
    console.info("Launching GOLDEN LITIGATOR OS v∞")
    for d in [RESULTS_DIR, MOTIONS_DIR, NARRATIVE_DIR, TRANSCRIPTS_DIR, EXHIBITS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    init_db()
    crawl_all()
    export_ledger_jsonl()
    build_master_narrative()
    console.info("Complete. Evidence indexed and narrative built.")
