#!/usr/bin/env python3
"""Golden Litigator OS v∞

A self-contained script that scans mounted drives, extracts text from
supported files, analyses the content for legal insight and records the
results in a persistent SQLite ledger.  Each processed file is renamed with
a ``__DONE__`` suffix so repeated runs only analyse new evidence.

The script intentionally keeps the analysis layer simple so it can run in
resource limited environments.  If an OpenAI API key is available the
``analyze_content`` function will attempt to use it for richer extraction,
otherwise a very small heuristic parser is used.

This file is designed as a starting point for a fully autonomous litigation
engine.  Additional modules may extend the analysis, narrative generation and
filing generation features.
"""
from __future__ import annotations

import os
import re
import importlib
import sqlite3
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence
import zipfile
from datetime import date


def _optional_import(name: str) -> Any:
    try:  # pragma: no cover - best effort
        return importlib.import_module(name)
    except Exception:  # pragma: no cover - optional dependency
        return None


fitz = _optional_import("fitz")
docx = _optional_import("docx")
Document = getattr(docx, "Document", None) if docx else None
Image = _optional_import("PIL.Image")
pytesseract = _optional_import("pytesseract")
try:  # pragma: no cover - optional dependency
    OpenAIClient = importlib.import_module("openai").OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAIClient = None

# configuration -------------------------------------------------------------
TARGET_DRIVES: List[str] = ["F:/", "Z:/", "E:/", "C:/"]
MOUNTED_GDRIVE: str = "G:/MyDrive/"
DATABASE_PATH = "golden_litigator_ledger.db"
PROCESSED_TAG = "__DONE__"


# database ------------------------------------------------------------------
def init_db(db_path: str = DATABASE_PATH) -> None:
    """Create the evidence ledger if it does not already exist."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS evidence (
            id INTEGER PRIMARY KEY,
            filename TEXT,
            filepath TEXT,
            content TEXT,
            parties TEXT,
            claims TEXT,
            statutes TEXT,
            quotes TEXT,
            tags TEXT,
            court_relevance TEXT,
            exhibit_id TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


# parsers -------------------------------------------------------------------
def parse_pdf(path: str) -> str:
    if not fitz:  # pragma: no cover
        raise RuntimeError("PyMuPDF not available")
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)


def parse_docx(path: str) -> str:
    if not docx:  # pragma: no cover
        raise RuntimeError("python-docx not available")
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)


def parse_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def parse_image(path: str) -> str:
    if not (Image and pytesseract):  # pragma: no cover
        raise RuntimeError("pytesseract not available")
    img = Image.open(path)
    return str(pytesseract.image_to_string(img))


PARSERS = {
    ".pdf": parse_pdf,
    ".docx": parse_docx,
    ".txt": parse_txt,
    ".jpg": parse_image,
    ".jpeg": parse_image,
    ".png": parse_image,
}

# analysis ------------------------------------------------------------------
TOKEN_RE = re.compile(r"[A-Z][a-z]+ [A-Z][a-z]+")


def heuristic_analysis(text: str) -> Dict[str, str]:
    parties = ", ".join(sorted(set(TOKEN_RE.findall(text))))
    quotes = "\n".join(re.findall(r'"(.*?)"', text))
    return {
        "parties": parties,
        "claims": "",
        "statutes": "",
        "quotes": quotes,
        "tags": "",
        "court_relevance": "",
        "exhibit_id": "",
    }


def analyze_content(text: str) -> Dict[str, str]:
    if OpenAIClient is not None and os.getenv("OPENAI_API_KEY"):
        try:  # pragma: no cover - network call
            client = OpenAIClient()
            prompt = (
                "Extract parties, claims, statutes and notable quotes from "
                "the following text. Return a JSON object with keys: parties, "
                "claims, statutes, quotes, tags, court_relevance, exhibit_id.\n" + text
            )
            resp = client.responses.create(model="gpt-4o-mini", input=prompt)
            content = resp.output_text
            import json

            data = json.loads(content)
            return {k: str(v) for k, v in data.items()}
        except Exception:
            pass
    return heuristic_analysis(text)


# ledger --------------------------------------------------------------------
def index_to_db(
    path: str,
    file: str,
    content: str,
    analysis: Dict[str, str],
    db_path: str = DATABASE_PATH,
) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO evidence (
            filename, filepath, content, parties, claims, statutes,
            quotes, tags, court_relevance, exhibit_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            file,
            path,
            content[:1000],
            analysis.get("parties", ""),
            analysis.get("claims", ""),
            analysis.get("statutes", ""),
            analysis.get("quotes", ""),
            analysis.get("tags", ""),
            analysis.get("court_relevance", ""),
            analysis.get("exhibit_id", ""),
        ),
    )
    conn.commit()
    conn.close()


# helper utilities ----------------------------------------------------------
def rename_done(path: str) -> None:
    p = Path(path)
    new_path = p.with_name(p.stem + PROCESSED_TAG + p.suffix)
    os.rename(path, new_path)


# main crawl ----------------------------------------------------------------
def crawl(paths: Iterable[str]) -> None:
    for drive in paths:
        if not os.path.exists(drive):
            continue
        for root, _, files in os.walk(drive):
            for file in files:
                if PROCESSED_TAG in file:
                    continue
                full_path = os.path.join(root, file)
                ext = Path(file).suffix.lower()
                parser = PARSERS.get(ext)
                if not parser:
                    continue
                try:
                    text = parser(full_path)
                    data = analyze_content(text)
                    index_to_db(full_path, file, text, data)
                    rename_done(full_path)
                except Exception as exc:  # pragma: no cover
                    print(f"Error processing {file}: {exc}")


# narrative and filings -----------------------------------------------------
from collections import defaultdict


def generate_narrative(db_path: str = DATABASE_PATH) -> str:
    """Create a simple chronological narrative from the ledger."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT filename, quotes FROM evidence ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    narrative_lines = []
    for fname, quote in rows:
        if quote:
            narrative_lines.append(f'{fname}: "{quote}"')
    narrative = "\n".join(narrative_lines)
    with open("narrative.txt", "w", encoding="utf-8") as fh:
        fh.write(narrative)
    return narrative


def generate_filings(
    db_path: str = DATABASE_PATH, output_dir: str = "."
) -> Optional[Path]:
    """Generate a minimal complaint document using stored evidence."""
    if Document is None:  # pragma: no cover
        print("python-docx not available, cannot generate filings")
        return None
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT parties, claims, quotes FROM evidence")
    parties_data = cur.fetchall()
    conn.close()

    parties: DefaultDict[str, int] = defaultdict(int)
    claims: DefaultDict[str, int] = defaultdict(int)
    for party, claim, _quote in parties_data:
        for p in party.split(","):
            p = p.strip()
            if p:
                parties[p] += 1
        for c in claim.split(","):
            c = c.strip()
            if c:
                claims[c] += 1

    doc = Document()
    doc.add_heading("Complaint", 0)
    if parties:
        doc.add_paragraph("Parties involved:")
        for p in sorted(parties, key=lambda k: parties[k], reverse=True):
            doc.add_paragraph(p, style="List Bullet")
    if claims:
        doc.add_paragraph("Claims:")
        for c in sorted(claims, key=lambda k: claims[k], reverse=True):
            doc.add_paragraph(c, style="List Bullet")
    doc.add_paragraph(
        "This complaint is automatically generated from the evidence ledger."
    )
    out_path = Path(output_dir) / "Complaint_Auto.docx"
    doc.save(str(out_path))
    return out_path


def generate_certificate_of_service(defendants: Sequence[str], output_dir: str) -> Path:
    """Create a simple certificate of service document."""
    if Document is None:  # pragma: no cover
        raise RuntimeError("python-docx not available")
    os.makedirs(output_dir, exist_ok=True)
    doc = Document()
    doc.add_heading("CERTIFICATE OF SERVICE", level=1)
    doc.add_paragraph(
        "I hereby certify that on this date, I served a copy of the Complaint, "
        "Summons, and Civil Cover Sheet on the following Defendants by U.S. Mail "
        "and/or personal service, pursuant to the Federal Rules of Civil Procedure:"
    )
    for name in defendants:
        doc.add_paragraph(f"- {name}")
    doc.add_paragraph(
        f"\nExecuted on {date.today()} by:\n[Your Name], Pro Se Plaintiff"
    )
    path = Path(output_dir) / "Certificate_of_Service.docx"
    doc.save(str(path))
    return path


def zip_filing_package(output_dir: str) -> Path:
    """Zip all files in the output directory."""
    zip_path = Path(output_dir).with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(output_dir):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, output_dir)
                zf.write(full_path, arcname)
    return zip_path


def build_filing_package(
    defendants: Sequence[str], output_dir: str, db_path: str = DATABASE_PATH
) -> Path:
    """Generate key filing documents and zip them for submission."""
    os.makedirs(output_dir, exist_ok=True)
    generate_filings(db_path, output_dir)
    generate_certificate_of_service(defendants, output_dir)
    return zip_filing_package(output_dir)


# entry point ---------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    sources = TARGET_DRIVES + [MOUNTED_GDRIVE]
    crawl(sources)
    generate_narrative()
    generate_filings()
    print("Golden Litigator run complete")
