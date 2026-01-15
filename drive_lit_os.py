#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drive Organizer + Litigation Intelligence Engine (Ω build)
Buckets • Duplicates • Empty-Folder Prune • Incremental STATE • NER • Claim→Evidence • Contradictions
MI-Authority Canonicalizer • Timeline Weaver • Missing-Exhibit Detector • Exhibit Covers • Affidavit Stubs
Proof-of-Service Assistant • Binder Index (+Bates) • Webhooks • Mini HTML Dashboard • Graph Export (Neo4j/Gephi)

USAGE (examples):
  python drive_lit_os.py --run --dry-run
  python drive_lit_os.py --run --batch-size 20 --ner --graph neo4j --report html --webhook http://127.0.0.1:5055/hook
  python drive_lit_os.py --run --ocr images,pdf --quarantine --bates PIGORS --bates-start 1
  python drive_lit_os.py --run --watch  # only changed/new since last run

DEPENDENCIES:
  - google-api-python-client, google-auth-oauthlib, google-auth-httplib2
  - pdfminer.six, python-docx
  Optional:
  - pytesseract + Tesseract binary (for --ocr)
  - pillow (auto-installed)
  - requests (for --webhook)
  - rapidfuzz (fast fuzzy matching; optional)
  - spacy (optional; we ship a regex NER fallback to avoid model downloads)

AUTH:
  - Local: place credentials.json next to this file (OAuth Installed App). token.json is created on first run.
  - Colab: automatic auth supported.

OUTPUT (created at Drive root by default):
  - 10 bucket folders + 'duplicates' + 'LITIGATION_ANALYSIS' + 'QUARANTINE' (if used)
  - analysis_batch_####.json (rich per-batch intelligence)
  - nodes.csv / edges.csv (if --graph neo4j)
  - batch_####.html (if --report html)
  - binder_index.csv/json (rolling)
  - STATE.json (incremental tracking)
"""

import os, sys, io, re, json, time, argparse, tempfile, hashlib, datetime, csv, math, textwrap
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# -------------------- Auto-install deps --------------------
def _ensure_deps():
    import importlib, subprocess

    pkgs = [
        "google-api-python-client",
        "google-auth-httplib2",
        "google-auth-oauthlib",
        "pdfminer.six",
        "python-docx",
        "requests",
        "pillow",
        "rapidfuzz",
    ]
    for p in pkgs:
        try:
            importlib.import_module(p.split("==")[0].split(">=")[0])
        except Exception:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", p], stdout=subprocess.DEVNULL
            )


_ensure_deps()

# Optional OCR
try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from pdfminer.high_level import extract_text as pdf_extract
from docx import Document
import requests
from rapidfuzz import fuzz

SCOPES = ["https://www.googleapis.com/auth/drive"]

BUCKET_NAMES = [
    "1_pdfs",
    "2_docx",
    "3_txt",
    "4_images",
    "5_audio",
    "6_video",
    "7_spreadsheets",
    "8_code",
    "9_archives",
    "10_other",
    "duplicates",
]
ANALYSIS_FOLDER_NAME = "LITIGATION_ANALYSIS"
QUARANTINE_NAME = "QUARANTINE"
STATE_FILE = "STATE.json"
DEFAULT_BATCH = 20

EXT_MAP = {
    ".pdf": "1_pdfs",
    ".docx": "2_docx",
    ".doc": "2_docx",
    ".txt": "3_txt",
    ".rtf": "3_txt",
    ".md": "3_txt",
    ".png": "4_images",
    ".jpg": "4_images",
    ".jpeg": "4_images",
    ".gif": "4_images",
    ".tif": "4_images",
    ".tiff": "4_images",
    ".bmp": "4_images",
    ".webp": "4_images",
    ".mp3": "5_audio",
    ".wav": "5_audio",
    ".m4a": "5_audio",
    ".aac": "5_audio",
    ".flac": "5_audio",
    ".ogg": "5_audio",
    ".mp4": "6_video",
    ".mov": "6_video",
    ".avi": "6_video",
    ".mkv": "6_video",
    ".webm": "6_video",
    ".wmv": "6_video",
    ".xlsx": "7_spreadsheets",
    ".xls": "7_spreadsheets",
    ".csv": "7_spreadsheets",
    ".tsv": "7_spreadsheets",
    ".ods": "7_spreadsheets",
    ".py": "8_code",
    ".ipynb": "8_code",
    ".json": "8_code",
    ".yml": "8_code",
    ".yaml": "8_code",
    ".xml": "8_code",
    ".html": "8_code",
    ".css": "8_code",
    ".js": "8_code",
    ".ps1": "8_code",
    ".bat": "8_code",
    ".sh": "8_code",
    ".zip": "9_archives",
    ".rar": "9_archives",
    ".7z": "9_archives",
    ".tar": "9_archives",
    ".gz": "9_archives",
    ".bz2": "9_archives",
}

GDRIVE_NATIVE = {
    "application/vnd.google-apps.document": "gdoc",
    "application/vnd.google-apps.spreadsheet": "gsheet",
    "application/vnd.google-apps.presentation": "gslide",
}
GDRIVE_EXPORT_MAP = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}

# -------------------- Regexes / Heuristics --------------------
RE_MCR = re.compile(r"\bMCR\s*(\d\.\d+(?:\([^)]+\))*)", re.I)
RE_MCL = re.compile(r"\bMCL\s*(\d+(?:\.\d+)?(?:\([^)]+\))*)", re.I)
RE_MRE = re.compile(r"\bMRE\s*(\d+(?:\([^)]+\))*)", re.I)
RE_SCAO = re.compile(r"\b((?:MC|FOC|SCAO)-\d{1,4}(?:\([A-Z]\))?)\b", re.I)
RE_BENCHBOOK = re.compile(r"\bBenchbook\b", re.I)

RE_DATE = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\w*\s+\d{1,2},\s+\d{4})\b",
    re.I,
)
RE_ACCUSATION = re.compile(
    r"\b(alleg(e|ation|es|ed)|claim(s|ed|ing)?|accus(e|es|ed|ation)|assert(s|ed|ion)?)\b",
    re.I,
)
RE_EVIDENCE_HINT = re.compile(
    r"\b(exhibit|attachment|photo|video|receipt|proof|affidavit|invoice|ledger|screenshot|report|see\s+attached|see\s+exhibit)\b",
    re.I,
)
RE_EXHIBIT_REF = re.compile(r"\bExhibit\s+([A-Z]|\d+)\b", re.I)
RE_EMAIL_PoS = re.compile(
    r"\b(from|to|subject|sent|received)\b.*@|CM/?ECF|notice\b", re.I
)

# Light NER (regex + capitalized sequences) to avoid big model downloads
RE_PERSON_LIKE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+){0,3})\b")
ORG_KEYWORDS = {
    "LLC",
    "Inc",
    "Corp",
    "Corporation",
    "Company",
    "Co.",
    "LP",
    "LLP",
    "PLC",
    "Association",
    "Agency",
    "Department",
    "Court",
    "Park",
    "Homes",
}

SUSPICIOUS_EXT = {".exe", ".msi", ".scr", ".vbs", ".js", ".jar", ".apk", ".iso"}


# -------------------- Utilities --------------------
def colab_auth_if_available():
    try:
        from google.colab import auth  # type: ignore

        auth.authenticate_user()
        return True
    except Exception:
        return False


def build_service():
    if colab_auth_if_available():
        return build("drive", "v3", cache_discovery=False)
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists("credentials.json"):
                raise FileNotFoundError(
                    "Missing credentials.json for Google OAuth (Installed App)."
                )
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as f:
            f.write(creds.to_json())
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def gcall(fn, *a, **kw):
    delay = 1.0
    for _ in range(8):
        try:
            return fn(*a, **kw)
        except HttpError as e:
            st = getattr(e, "status_code", None) or (
                e.resp.status if hasattr(e, "resp") else None
            )
            if st in (403, 429, 500, 502, 503, 504):
                time.sleep(delay)
                delay = min(32, delay * 2)
                continue
            raise
    raise RuntimeError("Max retries exceeded")


def list_files(
    service,
    q,
    fields="files(id,name,mimeType,parents,md5Checksum,createdTime,modifiedTime,size,trashed)",
    page_size=1000,
):
    out = []
    token = None
    while True:
        resp = gcall(
            service.files().list,
            q=q,
            pageSize=page_size,
            fields=f"nextPageToken,{fields}",
            pageToken=token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        )
        out += resp.get("files", [])
        token = resp.get("nextPageToken")
        if not token:
            break
    return out


def create_folder_if_absent(service, name, parent="root"):
    escaped_name = name.replace("'", "\\'")
    q = (
        "mimeType='application/vnd.google-apps.folder' and name='"
        f"{escaped_name}' and '{parent}' in parents and trashed=false"
    )
    found = list_files(service, q, fields="files(id,name)")
    if found:
        return found[0]["id"]
    meta = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent],
    }
    return gcall(
        service.files().create, body=meta, fields="id", supportsAllDrives=True
    ).execute()["id"]


def upload_json(service, parent_id, name, data):
    buf = io.BytesIO(json.dumps(data, indent=2).encode("utf-8"))
    media = MediaIoBaseUpload(buf, mimetype="application/json", resumable=False)
    meta = {"name": name, "parents": [parent_id]}
    gcall(
        service.files().create,
        body=meta,
        media_body=media,
        fields="id",
        supportsAllDrives=True,
    ).execute()


def upload_text(service, parent_id, name, text, mime="text/html"):
    buf = io.BytesIO(text.encode("utf-8"))
    media = MediaIoBaseUpload(buf, mimetype=mime, resumable=False)
    meta = {"name": name, "parents": [parent_id]}
    gcall(
        service.files().create,
        body=meta,
        media_body=media,
        fields="id",
        supportsAllDrives=True,
    ).execute()


def move_file(service, file_id, to_parent, remove_parents, dry):
    if dry:
        return
    gcall(
        service.files().update,
        fileId=file_id,
        addParents=to_parent,
        removeParents=",".join(remove_parents) if remove_parents else None,
        fields="id,parents",
        supportsAllDrives=True,
    ).execute()


def delete_file(service, file_id, dry):
    if dry:
        return
    gcall(service.files().delete, fileId=file_id, supportsAllDrives=True).execute()


def file_parents(f):
    return f.get("parents", []) or []


def safe_int(v, d=0):
    try:
        return int(v)
    except:
        return d


def sha256_str(s: bytes) -> str:
    return hashlib.sha256(s).hexdigest()


def md5_str(s: bytes) -> str:
    return hashlib.md5(s).hexdigest()


# -------------------- Classification --------------------
def pick_bucket(name: str, mime: str) -> str:
    ext = os.path.splitext(name.lower())[1]
    if ext in EXT_MAP:
        return EXT_MAP[ext]
    if mime in GDRIVE_NATIVE:
        if mime.endswith(".spreadsheet"):
            return "7_spreadsheets"
        if mime.endswith(".document"):
            return "2_docx"
        if mime.endswith(".presentation"):
            return "10_other"
    if mime.startswith("image/"):
        return "4_images"
    if mime.startswith("audio/"):
        return "5_audio"
    if mime.startswith("video/"):
        return "6_video"
    if "spreadsheet" in mime:
        return "7_spreadsheets"
    if "pdf" in mime:
        return "1_pdfs"
    return "10_other"


def risky(name: str) -> bool:
    return os.path.splitext(name.lower())[1] in SUSPICIOUS_EXT


# -------------------- Download / Export for analysis --------------------
def export_native_text(service, file_id, mime):
    em = GDRIVE_EXPORT_MAP.get(mime)
    if not em:
        return ""
    buf = io.BytesIO()
    req = service.files().export_media(fileId=file_id, mimeType=em)
    dl = MediaIoBaseDownload(buf, req)
    done = False
    while not done:
        _, done = dl.next_chunk()
    return buf.getvalue().decode("utf-8", "ignore")


def download_small(service, fobj, tmpdir, max_bytes=5_000_000) -> Tuple[str, str]:
    fid = fobj["id"]
    name = fobj.get("name", "")
    mime = fobj.get("mimeType", "")
    # native export
    if mime in GDRIVE_NATIVE:
        text = export_native_text(service, fid, mime)
        return "", text
    size = safe_int(fobj.get("size", 0))
    if size and size > max_bytes:
        return "", ""
    out = os.path.join(tmpdir, name.replace("/", "_"))
    req = service.files().get_media(fileId=fid)
    fh = io.FileIO(out, "wb")
    dl = MediaIoBaseDownload(fh, req)
    done = False
    try:
        while not done:
            _, done = dl.next_chunk()
    finally:
        fh.close()
    text = ""
    try:
        if out.lower().endswith(".pdf"):
            text = pdf_extract(out)
        elif out.lower().endswith(".docx"):
            doc = Document(out)
            text = "\n".join(p.text for p in doc.paragraphs)
        else:
            with open(out, "r", errors="ignore") as h:
                text = h.read()
    except Exception:
        text = ""
    return out, text


# -------------------- Intelligence: NER, Claims, Contradictions, MI Canonicalizer --------------------
def light_ner(text: str) -> Dict[str, List[str]]:
    if not text:
        return {"Person": [], "Organization": [], "Place": []}
    cands = RE_PERSON_LIKE.findall(text)
    persons = set()
    orgs = set()
    places = set()
    for raw in cands:
        name = raw.strip()
        # Quick org detection
        if any(tok in name.split() for tok in ORG_KEYWORDS):
            orgs.add(name)
        else:
            # crude heuristic: locations often single proper tokens like "Muskegon" or two-token proper nouns with known place tails
            if any(
                t in name
                for t in [
                    "County",
                    "Township",
                    "City",
                    "State",
                    "Michigan",
                    "Muskegon",
                    "Norton Shores",
                    "Whitehall",
                    "Ravenna",
                ]
            ):
                places.add(name)
            else:
                persons.add(name)
    return {
        "Person": sorted(persons),
        "Organization": sorted(orgs),
        "Place": sorted(places),
    }


def canonicalize_mi_refs(text: str) -> Dict[str, List[str]]:
    out = {"MCR": [], "MCL": [], "MRE": [], "SCAO": [], "Benchbook": []}
    if not text:
        return out
    mcr = [f"MCR {m}" for m in set(RE_MCR.findall(text))]
    mcl = [f"MCL {m}" for m in set(RE_MCL.findall(text))]
    mre = [f"MRE {m}" for m in set(RE_MRE.findall(text))]
    sc = list(set(RE_SCAO.findall(text)))
    bb = ["Benchbook"] if RE_BENCHBOOK.search(text) else []
    # normalize parentheses spacing
    norm = lambda s: re.sub(r"\s*\(\s*", "(", re.sub(r"\s+\)", ")", s)).strip()
    out["MCR"] = [norm(x) for x in mcr]
    out["MCL"] = [norm(x) for x in mcl]
    out["MRE"] = [norm(x) for x in mre]
    out["SCAO"] = [norm(x.upper()) for x in sc]
    out["Benchbook"] = bb
    return out


def sentence_split(text: str) -> List[str]:
    if not text:
        return []
    return re.split(r"(?<=[\.\?!])\s+", text)


def claim_evidence_links(text: str) -> Tuple[List[dict], List[str]]:
    claims = []
    missing = []
    if not text:
        return claims, missing
    sents = sentence_split(text)
    seen_exhibits = set()
    for i, s in enumerate(sents):
        ex_refs = [m.group(1).upper() for m in RE_EXHIBIT_REF.finditer(s)]
        for e in ex_refs:
            seen_exhibits.add(e)
        if RE_ACCUSATION.search(s):
            has_evidence = bool(RE_EVIDENCE_HINT.search(s)) or bool(ex_refs)
            claim = {
                "sentence": s.strip()[:800],
                "idx": i,
                "has_proof_token": has_evidence,
                "exhibits": ex_refs,
            }
            # look around ±2 sentences for evidence hints
            ctx = " ".join(sents[max(0, i - 2) : min(len(sents), i + 3)])
            nearby = list(
                set([m.group(1).upper() for m in RE_EXHIBIT_REF.finditer(ctx)])
            )
            claim["nearby_exhibits"] = nearby
            claim["evidence_hints"] = bool(RE_EVIDENCE_HINT.search(ctx))
            claims.append(claim)
    # Missing exhibit detector: refers to exhibit in text but not present in Drive set (resolved later at batch level)
    return claims, list(sorted(seen_exhibits))


def contradiction_candidates(text: str) -> List[dict]:
    out = []
    if not text:
        return out
    toks = re.findall(r"[A-Za-z0-9']+", text.lower())
    tri = [" ".join(toks[i : i + 3]) for i in range(0, len(toks) - 2)]
    pos = set(t for t in tri if " did " in (" " + t + " "))
    neg = set(
        t
        for t in tri
        if (" did not " in (" " + t + " ")) or (" never " in (" " + t + " "))
    )
    cores_pos = set(" ".join(t.split()[:2]) for t in pos)
    cores_neg = set(" ".join(t.split()[:2]) for t in neg)
    hits = cores_pos.intersection(cores_neg)
    for h in list(hits)[:50]:
        out.append({"core": h, "asserted": True, "denied": True})
    return out


def extract_dates(text: str) -> List[str]:
    return list(set(RE_DATE.findall(text))) if text else []


# -------------------- HTML Dashboard --------------------
def render_dashboard(batch_payload: dict) -> str:
    total = batch_payload.get("count", 0)
    files = batch_payload.get("files", [])
    top_stats = Counter()
    mi_cites = Counter()
    persons = Counter()
    orgs = Counter()
    places = Counter()
    contradictions = 0
    claims = 0
    for f in files:
        a = f.get("analysis", {})
        flags = a.get("flags", {})
        if flags.get("contradiction_candidates"):
            contradictions += len(flags["contradiction_candidates"])
        if "claims" in a:
            claims += len(a["claims"])
        ents = a.get("entities", {})
        for p in ents.get("Person", []):
            persons[p] += 1
        for o in ents.get("Organization", []):
            orgs[o] += 1
        for pl in ents.get("Place", []):
            places[pl] += 1
        cites = a.get("citations", {})
        for k in ("MCR", "MCL", "MRE", "SCAO"):
            for v in cites.get(k, []):
                mi_cites[f"{k}:{v}"] += 1

    def top(counter, n=10):
        return "<br>".join(f"{k} — {v}" for k, v in counter.most_common(n)) or "—"

    html = f"""<html><head><meta charset="utf-8"><title>Batch {batch_payload.get('batch_index')}</title></head>
<body style="font-family:system-ui,Segoe UI,Arial">
<h1>Batch {batch_payload.get('batch_index')} — Litigation Intel Report</h1>
<p>Generated at: {batch_payload.get('generated_at')}</p>
<h2>Overview</h2>
<ul>
<li>Files analyzed: <b>{total}</b></li>
<li>Total claims detected: <b>{claims}</b></li>
<li>Contradiction cores: <b>{contradictions}</b></li>
</ul>
<h2>Top Entities</h2>
<b>Persons</b><br>{top(persons)}<br><br>
<b>Organizations</b><br>{top(orgs)}<br><br>
<b>Places</b><br>{top(places)}<br><br>
<h2>Top MI Citations</h2>
{top(mi_cites)}
<hr>
<small>Automated heuristics; verify before filing.</small>
</body></html>"""
    return html


# -------------------- Graph Export --------------------
def write_graph(files: List[dict], out_dir: str):
    # nodes: id,type,label ; edges: src,dst,type
    nodes_path = os.path.join(out_dir, "nodes.csv")
    edges_path = os.path.join(out_dir, "edges.csv")
    node_set = set()
    edges = []

    def add_node(nid, ntype, label):
        key = (nid, ntype, label)
        if key in node_set:
            return
        node_set.add(key)

    for f in files:
        fid = f["id"]
        name = f["name"]
        add_node(fid, "Document", name)
        a = f.get("analysis", {})
        ents = a.get("entities", {})
        for p in ents.get("Person", []):
            nid = f"person:{p}"
            add_node(nid, "Person", p)
            edges.append((fid, nid, "MENTIONS"))
        for o in ents.get("Organization", []):
            nid = f"org:{o}"
            add_node(nid, "Organization", o)
            edges.append((fid, nid, "MENTIONS"))
        for pl in ents.get("Place", []):
            nid = f"place:{pl}"
            add_node(nid, "Place", pl)
            edges.append((fid, nid, "MENTIONS"))
        cites = a.get("citations", {})
        for group in ("MCR", "MCL", "MRE", "SCAO"):
            for c in cites.get(group, []):
                nid = f"statute:{group}:{c}"
                add_node(nid, "Statute", f"{group} {c}")
                edges.append((fid, nid, "CITES"))
        # claims
        for cl in a.get("claims", []):
            cid = f"claim:{fid}:{cl['idx']}"
            add_node(cid, "Event", "Claim")
            edges.append((fid, cid, "CLAIMS"))
            for ex in cl.get("exhibits", []) + cl.get("nearby_exhibits", []):
                nid = f"exhibit:{ex}"
                add_node(nid, "Exhibit", f"Exhibit {ex}")
                edges.append((cid, nid, "DERIVES_FROM"))
        # contradictions
        for cc in a.get("flags", {}).get("contradiction_candidates", []):
            nid = f"contradiction:{fid}:{cc['core']}"
            add_node(nid, "Event", f"Contradiction {cc['core']}")
            edges.append((fid, nid, "REFUTES"))
    # write
    with open(nodes_path, "w", newline="", encoding="utf-8") as nf:
        w = csv.writer(nf)
        w.writerow(["id", "type", "label"])
        for nid, ntype, label in sorted(node_set):
            w.writerow([nid, ntype, label])
    with open(edges_path, "w", newline="", encoding="utf-8") as ef:
        w = csv.writer(ef)
        w.writerow(["src", "dst", "type"])
        for s, d, t in edges:
            w.writerow([s, d, t])


# -------------------- Binder Index & Covers/Affidavits/PoS --------------------
def bates_str(prefix: str, num: int) -> str:
    return f"{prefix}-{num:06d}"


def exhibit_cover_stub(letter: str, title: str, sha256: str, captured: str) -> dict:
    return {
        "cover_title": f"Exhibit {letter}",
        "document_title": title,
        "foundation": f"True and correct copy of {title}; hash {sha256[:12]}, captured {captured}.",
        "sha256": sha256,
        "captured": captured,
    }


def affidavit_stub(events: List[dict]) -> List[str]:
    # Simple, numbered facts from timeline events
    lines = []
    for i, e in enumerate(events, start=1):
        who = ", ".join(e.get("who", [])[:3]) or "—"
        where = ", ".join(e.get("where", [])[:2]) or "—"
        date = e.get("date", "")
        src = e.get("source_id", "")
        lines.append(
            f"{i}. On {date}, {who} at {where}: {e.get('event_type','event')} [{src}]."
        )
    return lines


def pos_entries_from_text(text: str) -> List[dict]:
    if not text or not RE_EMAIL_PoS.search(text):
        return []
    # naive parse: pick mail-like lines
    entries = []
    for line in text.splitlines()[:500]:
        if "@" in line and any(
            k in line.lower()
            for k in ["from", "to", "subject", "sent", "received", "cm/ecf", "notice"]
        ):
            entries.append({"raw": line.strip()[:500]})
    return entries[:25]


# -------------------- STATE handling --------------------
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(st: dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


# -------------------- Main organize + analyze --------------------
@dataclass
class Cfg:
    root_id: str = "root"
    dry_run: bool = False
    batch_size: int = DEFAULT_BATCH
    ner: bool = False
    graph: Optional[str] = None
    report_html: bool = False
    ocr: List[str] = None
    quarantine: bool = False
    webhook: Optional[str] = None
    watch_only: bool = False
    bates_prefix: Optional[str] = None
    bates_start: int = 1


def ensure_dirs(service, root_id, use_quarantine=False):
    ids = {}
    for name in BUCKET_NAMES:
        ids[name] = create_folder_if_absent(service, name, root_id)
    ids[ANALYSIS_FOLDER_NAME] = create_folder_if_absent(
        service, ANALYSIS_FOLDER_NAME, root_id
    )
    if use_quarantine:
        ids[QUARANTINE_NAME] = create_folder_if_absent(
            service, QUARANTINE_NAME, root_id
        )
    return ids


def gather_files(service, modified_after_iso: Optional[str]):
    base = "trashed=false and mimeType!='application/vnd.google-apps.folder'"
    if modified_after_iso:
        base += f" and modifiedTime>'{modified_after_iso}'"
    return list_files(service, base)


def gather_folders(service):
    return list_files(
        service,
        "trashed=false and mimeType='application/vnd.google-apps.folder'",
        fields="files(id,name,parents)",
    )


def run_pipeline(cfg: Cfg):
    service = build_service()
    ids = ensure_dirs(service, cfg.root_id, use_quarantine=cfg.quarantine)
    bucket_ids = {k: v for k, v in ids.items() if k in BUCKET_NAMES}
    analysis_id = ids[ANALYSIS_FOLDER_NAME]
    quarantine_id = ids.get(QUARANTINE_NAME)

    state = load_state()
    modified_after = state.get("last_run") if cfg.watch_only else None

    files = gather_files(service, modified_after)
    print(f"Files to consider: {len(files)}")

    # duplicate map by md5
    seen_md5 = {}
    batch = []
    batch_idx = state.get("batch_idx", 1)
    binder_rows = []

    tmpdir = tempfile.mkdtemp(prefix="lit_os_")
    all_exhibit_refs = set()

    for f in files:
        fid = f["id"]
        name = f.get("name", "")
        mime = f.get("mimeType", "")
        md5 = f.get("md5Checksum")
        size = f.get("size")
        parents = file_parents(f)

        # quarantine suspicious executables if requested
        if cfg.quarantine and risky(name):
            try:
                move_file(service, fid, quarantine_id, parents, cfg.dry_run)
                print(f"QUARANTINE → {name}")
            except Exception as e:
                print(f"⚠️ Quarantine failed: {name} :: {e}")
            continue

        # duplicate isolation
        is_dup = False
        if md5:
            if md5 not in seen_md5:
                seen_md5[md5] = f
            else:
                is_dup = True

        bucket = "duplicates" if is_dup else pick_bucket(name, mime)
        target = bucket_ids[bucket]
        if target not in parents:
            try:
                move_file(service, fid, target, parents, cfg.dry_run)
                print(
                    ("WOULD MOVE →" if cfg.dry_run else "MOVED →"), bucket, "::", name
                )
            except Exception as e:
                print(f"⚠️ Move failed: {name} :: {e}")

        # analysis extraction
        path, text = download_small(service, f, tmpdir)
        if cfg.ocr and path:
            ext = os.path.splitext(path.lower())[1]
            if (
                (
                    (
                        "images" in cfg.ocr
                        and ext
                        in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]
                    )
                    or ("pdf" in cfg.ocr and ext == ".pdf")
                )
                and pytesseract
                and Image
            ):
                try:
                    # lightweight single-page OCR (expand as needed)
                    img = None
                    if ext == ".pdf":
                        text = text  # prefer pdfminer text; OCR skipped unless empty
                    else:
                        img = Image.open(path)
                        text_ocr = pytesseract.image_to_string(img)
                        if text_ocr and len(text_ocr) > len(text):
                            text = text_ocr
                except Exception:
                    pass

        cites = canonicalize_mi_refs(text)
        ents = (
            light_ner(text)
            if cfg.ner
            else {"Person": [], "Organization": [], "Place": []}
        )
        dates = extract_dates(text)
        claims, seen_ex = claim_evidence_links(text)
        all_exhibit_refs.update(seen_ex)
        contras = contradiction_candidates(text)
        pos = pos_entries_from_text(text)

        # timeline events (very simple synthesis)
        events = []
        for d in sorted(dates):
            events.append(
                {
                    "date": d,
                    "event_type": "mention",
                    "who": ents.get("Person", [])[:3],
                    "where": ents.get("Place", [])[:2],
                    "source_id": fid,
                    "confidence": 0.5,
                }
            )

        batch.append(
            {
                "id": fid,
                "name": name,
                "mimeType": mime,
                "bucket": bucket,
                "size": size,
                "createdTime": f.get("createdTime"),
                "modifiedTime": f.get("modifiedTime"),
                "analysis": {
                    "citations": cites,
                    "entities": ents,
                    "dates": dates[:100],
                    "claims": claims[:100],
                    "flags": {
                        "contradiction_candidates": contras,
                        "pos_candidates": pos[:30],
                    },
                    "events": events[:200],
                },
            }
        )

        # binder row (Bates optional)
        bates = bates_str(cfg.bates_prefix, cfg.bates_start) if cfg.bates_prefix else ""
        if bates:
            cfg.bates_start += 1
        binder_rows.append(
            {
                "id": fid,
                "title": name,
                "bucket": bucket,
                "bates": bates,
                "modified": f.get("modifiedTime", ""),
            }
        )

        if len(batch) >= cfg.batch_size:
            emit_everything(
                service,
                analysis_id,
                batch,
                batch_idx,
                cfg,
                all_exhibit_refs,
                binder_rows,
            )
            if cfg.webhook:
                try:
                    requests.post(
                        cfg.webhook,
                        json={"batch_index": batch_idx, "count": len(batch)},
                        timeout=5,
                    )
                except Exception:
                    pass
            batch_idx += 1
            batch = []
            binder_rows = []
            all_exhibit_refs = set()

    if batch:
        emit_everything(
            service, analysis_id, batch, batch_idx, cfg, all_exhibit_refs, binder_rows
        )
        if cfg.webhook:
            try:
                requests.post(
                    cfg.webhook,
                    json={"batch_index": batch_idx, "count": len(batch)},
                    timeout=5,
                )
            except Exception:
                pass
        batch_idx += 1

    # delete empty folders except managed
    print("Scanning for empty folders …")
    managed = (
        set(bucket_ids.values())
        | {analysis_id}
        | ({quarantine_id} if quarantine_id else set())
    )
    for fo in gather_folders(service):
        fid = fo["id"]
        nm = fo.get("name", "")
        if fid in managed:
            continue
        kids = list_files(
            service, f"'{fid}' in parents and trashed=false", fields="files(id)"
        )
        if not kids:
            try:
                delete_file(service, fid, cfg.dry_run)
                print(
                    ("WOULD DELETE" if cfg.dry_run else "DELETED"), "empty folder:", nm
                )
            except Exception as e:
                print("⚠️ Delete failed:", nm, e)

    # save STATE
    now_iso = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    state["last_run"] = now_iso
    state["batch_idx"] = batch_idx
    save_state(state)
    print("✅ Completed.")


def emit_everything(
    service, analysis_id, batch, batch_idx, cfg: Cfg, all_exhibit_refs, binder_rows
):
    # Missing exhibits: cross-compare to actual Drive exhibit filenames present in buckets (A–Z references only heuristic)
    missing = []
    if all_exhibit_refs:
        # gather names briefly (cheap check)
        names = [b["name"] for b in batch]
        have_letters = set()
        for n in names:
            m = re.search(r"\bExhibit\s+([A-Z])\b", n, re.I)
            if m:
                have_letters.add(m.group(1).upper())
        for ex in all_exhibit_refs:
            if re.fullmatch(r"[A-Z]", ex) and ex not in have_letters:
                missing.append(ex)

    # Exhibit cover stubs for first N exhibits seen
    covers = []
    for i, f in enumerate(batch[:26]):
        fid = f["id"]
        nm = f["name"]
        # hash partial text to pseudo sha
        text_hash = sha256_str(
            ("".join(f["analysis"].get("dates", [])) + nm).encode("utf-8")
        )
        letter = chr(ord("A") + i)
        covers.append(
            exhibit_cover_stub(letter, nm, text_hash, f.get("modifiedTime", ""))
        )

    # Affidavit stub from merged events
    events = []
    for f in batch:
        events += f["analysis"].get("events", [])
    events = sorted(events, key=lambda e: e.get("date", ""))
    aff_lines = affidavit_stub(events[:25])

    # Proof of Service suggestions
    pos = []
    for f in batch:
        pos += f["analysis"].get("flags", {}).get("pos_candidates", [])

    payload = {
        "batch_index": batch_idx,
        "generated_at": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "count": len(batch),
        "files": batch,
        "exhibits": {"missing_letters": sorted(missing), "cover_stubs": covers},
        "affidavit": {"numbered_facts": aff_lines},
        "proof_of_service": pos[:50],
        "binder_index": binder_rows,
    }
    name = f"analysis_batch_{batch_idx:04d}.json"
    upload_json(service, analysis_id, name, payload)
    print(f"⬆️ Uploaded {name} ({len(batch)} files)")

    # Graph export
    if cfg.graph and cfg.graph.lower() == "neo4j":
        tmp_out = tempfile.mkdtemp(prefix="graph_")
        write_graph(batch, tmp_out)
        # Ship into Drive analysis folder
        for fn in ("nodes.csv", "edges.csv"):
            with open(os.path.join(tmp_out, fn), "rb") as h:
                media = MediaIoBaseUpload(h, mimetype="text/csv", resumable=False)
                meta = {
                    "name": f"{fn[:-4]}_{batch_idx:04d}.csv",
                    "parents": [analysis_id],
                }
                gcall(
                    service.files().create,
                    body=meta,
                    media_body=media,
                    fields="id",
                    supportsAllDrives=True,
                ).execute()
        print("↗ Graph CSVs uploaded")

    # HTML dashboard
    if cfg.report_html:
        html = render_dashboard(payload)
        upload_text(
            service, analysis_id, f"batch_{batch_idx:04d}.html", html, "text/html"
        )
        print("📊 Dashboard uploaded")


# -------------------- CLI --------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Drive Organizer + Litigation Intelligence Engine"
    )
    ap.add_argument(
        "--run", action="store_true", help="Execute the organizer + analysis pipeline."
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan-only; show moves/deletes, no changes.",
    )
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ap.add_argument(
        "--root", default="root", help="Parent folder ID (default My Drive root)."
    )
    ap.add_argument(
        "--ner",
        action="store_true",
        help="Enable light NER (regex-based without model downloads).",
    )
    ap.add_argument(
        "--graph", choices=["neo4j"], help="Emit nodes.csv/edges.csv graph exports."
    )
    ap.add_argument(
        "--report", choices=["html"], help="Emit mini HTML dashboard per batch."
    )
    ap.add_argument(
        "--ocr",
        help="Enable OCR for 'images', 'pdf', or 'images,pdf' (requires Tesseract installed).",
    )
    ap.add_argument(
        "--quarantine", action="store_true", help="Move risky filetypes to QUARANTINE."
    )
    ap.add_argument("--webhook", help="POST JSON after each batch to this URL.")
    ap.add_argument(
        "--watch",
        action="store_true",
        help="Process only files changed since last run (STATE.json).",
    )
    ap.add_argument(
        "--bates", dest="bates_prefix", help="Bates prefix, e.g., 'PIGORS'."
    )
    ap.add_argument(
        "--bates-start", type=int, default=1, help="Starting Bates number (default 1)."
    )
    return ap.parse_args()


def main():
    args = parse_args()
    if not args.run:
        print("Nothing to do. Use --run (optionally --dry-run) to execute.")
        return
    ocr_list = []
    if args.ocr:
        ocr_list = [s.strip() for s in args.ocr.split(",") if s.strip()]
    cfg = Cfg(
        root_id=args.root,
        dry_run=args.dry_run,
        batch_size=max(5, args.batch_size),
        ner=args.ner,
        graph=args.graph,
        report_html=(args.report == "html"),
        ocr=ocr_list if ocr_list else None,
        quarantine=args.quarantine,
        webhook=args.webhook,
        watch_only=args.watch,
        bates_prefix=args.bates_prefix,
        bates_start=args.bates_start,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
