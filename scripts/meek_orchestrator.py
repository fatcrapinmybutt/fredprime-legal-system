#!/usr/bin/env python3
"""Streamlined evidence ingestion and normalization pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import subprocess
from pathlib import Path
from typing import Any, List, Optional, cast

import pandas as pd  # type: ignore[import-untyped]
from chardet.universaldetector import UniversalDetector  # type: ignore[import-not-found]
from tqdm import tqdm  # type: ignore[import-untyped]

# ------------------ basic helpers ------------------

SAFE_EXPORT_SIZE = 300 * 1024 * 1024
TEXT_EXTS = {".txt", ".md", ".csv", ".tsv"}
OFFICE_EXTS = {".docx"}
PDF_EXT = ".pdf"
PARSE_EXTS = TEXT_EXTS | OFFICE_EXTS | {PDF_EXT}

MEEK1_PAT = [r"\bShady\s*Oaks\b", r"\bhabitability\b", r"\beviction\b"]
MEEK2_PAT = [r"\bparenting\s*time\b", r"\bcustody\b", r"\bPPO\b"]

# ---------------------------------------------------


def sh(cmd: List[str]) -> tuple[int, str, str]:
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    out, err = proc.communicate()
    return proc.returncode, out, err


def detect_encoding_text(path: Path) -> str:
    det = UniversalDetector()
    with open(path, "rb") as fh:
        for line in fh:
            det.feed(line)
            if det.done:
                break
    det.close()
    return det.result.get("encoding") or "utf-8"


def read_text_robust(path: Path) -> str:
    enc = detect_encoding_text(path)
    with open(path, "r", encoding=enc, errors="replace") as fh:
        return fh.read()


def parse_pdf_text(path: Path) -> str:
    from pdfminer.high_level import extract_text

    try:
        return extract_text(str(path)) or ""
    except Exception:
        return ""


def parse_docx_text(path: Path) -> str:
    from docx import Document

    try:
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_pdf(raw_path: Path, norm_path: Path) -> bool:
    norm_path.parent.mkdir(parents=True, exist_ok=True)
    code, _, _ = sh(
        ["ocrmypdf", "--skip-text", "--pdfa", str(raw_path), str(norm_path)]
    )
    return code == 0


def meek_bucket(text: str) -> str:
    import re

    m1 = sum(1 for pat in MEEK1_PAT if re.search(pat, text, re.I))
    m2 = sum(1 for pat in MEEK2_PAT if re.search(pat, text, re.I))
    if m1 and not m2:
        return "MEEK1"
    if m2 and not m1:
        return "MEEK2"
    if m1 and m2:
        return "BOTH"
    return "NONE"


def evidence_score(text: str, has_pdfa: bool, hash_ok: bool) -> int:
    base = 2 if has_pdfa else 0
    base += 2 if hash_ok else 0
    base += 3 if len(text) > 400 else 0
    return base


# ------------------- core pipeline -----------------


def init_db(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS evidence("
        "id INTEGER PRIMARY KEY,"
        "rel_path TEXT, raw_path TEXT, norm_path TEXT,"
        "sha256 TEXT, ext TEXT, size INTEGER, mtime TEXT,"
        "meek_bucket TEXT, score INTEGER, preview TEXT, pdfa INTEGER)"
    )
    return conn


def rclone_lsjson(remote: str) -> list[dict[str, Any]]:
    code, out, err = sh(["rclone", "lsjson", "--recursive", "--files-only", remote])
    if code != 0:
        raise RuntimeError(err.strip())
    return cast(list[dict[str, Any]], json.loads(out))


def rclone_copyto(remote_file: str, local_file: Path) -> bool:
    local_file.parent.mkdir(parents=True, exist_ok=True)
    code, _, _ = sh(["rclone", "copyto", remote_file, str(local_file)])
    return code == 0


def process_item(
    item: dict[str, Any],
    cfg: argparse.Namespace,
    conn: sqlite3.Connection,
    logger: logging.Logger,
) -> None:
    name = item.get("Name") or item.get("name") or ""
    path_rel = item.get("Path") or name
    size = item.get("Size", 0)
    mtime = item.get("ModTime", "")
    ext = Path(name).suffix.lower()
    if ext not in PARSE_EXTS or (size and size > SAFE_EXPORT_SIZE):
        return
    remote_file = (
        f"{cfg.remote}/{path_rel}" if item.get("Path") else f"{cfg.remote}/{name}"
    )
    raw_path = cfg.out / "PARSED" / path_rel
    if not rclone_copyto(remote_file, raw_path):
        logger.info("skip copy %s", path_rel)
        return
    sha = sha256_file(raw_path)
    norm_path = cfg.out / "NORMALIZED" / Path(path_rel).with_suffix(".pdf")
    has_pdfa = False
    text = ""
    if ext == PDF_EXT:
        has_pdfa = normalize_pdf(raw_path, norm_path)
        text = parse_pdf_text(norm_path if has_pdfa else raw_path)
    elif ext in TEXT_EXTS:
        text = read_text_robust(raw_path)
        has_pdfa = normalize_pdf(raw_path, norm_path)
    elif ext == ".docx":
        text = parse_docx_text(raw_path)
        has_pdfa = normalize_pdf(raw_path, norm_path)
    bucket = meek_bucket(text)
    score = evidence_score(text, has_pdfa, True)
    preview = text[:2000]
    conn.execute(
        "INSERT INTO evidence("
        "rel_path, raw_path, norm_path, sha256, ext, size, mtime, "
        "meek_bucket, score, preview, pdfa)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (
            path_rel,
            str(raw_path),
            str(norm_path if has_pdfa else ""),
            sha,
            ext,
            int(size or 0),
            mtime,
            bucket,
            score,
            preview,
            int(has_pdfa),
        ),
    )
    conn.commit()


# ----------------------- CLI -----------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MEEK orchestrator")
    parser.add_argument(
        "--remote", default=os.environ.get("MEEK_REMOTE", "gdrive:/LITIGATION_INTAKE")
    )
    parser.add_argument("--out", default=os.environ.get("MEEK_OUT", "./OUTPUT"))
    parser.add_argument(
        "--limit", type=int, default=int(os.environ.get("MEEK_LIMIT", "0"))
    )
    args = parser.parse_args(argv)
    args.out = Path(args.out).resolve()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("meek_orch")
    for d in ["PARSED", "NORMALIZED"]:
        (args.out / d).mkdir(parents=True, exist_ok=True)
    conn = init_db(args.out / "evidence.db")
    items = rclone_lsjson(args.remote)
    for item in tqdm(items, desc="ingest"):
        if (
            args.limit
            and conn.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
            >= args.limit
        ):
            break
        process_item(item, args, conn, logger)
    df = pd.read_sql_query("SELECT * FROM evidence", conn)
    df.to_csv(args.out / "evidence.csv", index=False)
    logger.info("processed %d items", len(df))


if __name__ == "__main__":
    main()
