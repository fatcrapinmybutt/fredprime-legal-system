# -*- coding: utf-8 -*-
r"""
GOLDEN GOD MODE — Offline AI + LLM + Litigation OS Bootstrap (Windows)

What you get
------------
A full, local-first litigation workspace with:
  • FastAPI backend (incremental evidence catalog, ask LLM, vector search, OCR, binder build)
  • Minimal static UI (index.html + api.js) for query and control
  • LLM engine that prefers llama.cpp server (HTTP), falls back to local bindings if present
  • Embeddings/search: hnswlib + SBERT if available, TF-IDF fallback
  • OCR/Text: ocrmypdf, PyMuPDF and pdfminer for PDFs; pytesseract if Tesseract is installed
  • Canon/MCR scanner: regex rules + YAML config hooks
  • MiFile bundle: DOCX/PDF packing with labeled exhibits + manifest
  • Logs, config, and Windows .bat runners

Assumptions
-----------
- Windows 10/11. Python 3.10+ installed. Run with admin OFF. Internet optional (offline after model/wheels present).
- Place your local LLM here later:   models\llm\model.gguf
- Optional SBERT embedding model here: models\emb\sentence-transformers\all-MiniLM-L6-v2\
- Install Tesseract (optional) if you want OCR from images (add to PATH).

Quick start
-----------
1) Run:
      python golden_god_mode_bootstrap.py --root "F:\\LAWFORGE_SUPREMACY"
   If offline, add pre-downloaded wheels under {root}\.wheels and re-run with:
      python golden_god_mode_bootstrap.py --root "F:\\LAWFORGE_SUPREMACY" --offline
2) Activate venv:
      F:\\LAWFORGE_SUPREMACY\.venv\Scripts\activate
3) Start backend:
      run_backend.bat
   Open UI:
      double-click frontend\run_frontend.bat  (serves static UI at http://127.0.0.1:7777)
   Backend API at http://127.0.0.1:8000/docs
4) Point scanners to evidence roots:
      Edit config.yaml (paths.MEEK1, paths.MEEK2, paths.EVIDENCE_GLOB)

Switch LLM
----------
- llama.cpp server (preferred): set llama_cpp.server_url in config.yaml
- llama-cpp-python (fallback): set llama_cpp.model_path in config.yaml
- No LLM available: system still scans, OCRs, and builds binders. Q&A degraded.

One-command full reset
----------------------
Delete {root}, re-run this script. Idempotent file writes are guarded.

Author: Strictly synthetic. No opinions. Pure utility.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REQ_TXT = r"""# Core
fastapi==0.115.0
uvicorn==0.30.6
pydantic==2.9.2
python-multipart==0.0.9
pyyaml==6.0.2

# PDF/Text
pymupdf==1.24.10
pdfminer.six==20231228
ocrmypdf==16.2.0

# Search
scikit-learn==1.5.1
hnswlib==0.8.0
watchdog==4.0.0
requests==2.31.0

# Optional
# sentence-transformers==3.0.1
# pillow==10.4.0
# pytesseract==0.3.13
# docx==0.2.4
"""

CONSTRAINTS_TXT = r"""fastapi==0.115.0 --hash=sha256:17ea427674467486e997206a5ab25760f6b09e069f099b96f5b55a32fb6f1631
uvicorn==0.30.6 --hash=sha256:65fd46fe3fda5bdc1b03b94eb634923ff18cd35b2f084813ea79d1f103f711b5
pydantic==2.9.2 --hash=sha256:f048cec7b26778210e28a0459867920654d48e5e62db0958433636cde4254f12
python-multipart==0.0.9 --hash=sha256:97ca7b8ea7b05f977dc3849c3ba99d51689822fab725c3703af7c866a0c2b215
pyyaml==6.0.2 --hash=sha256:3ad2a3decf9aaba3d29c8f537ac4b243e36bef957511b4766cb0057d32b0be85
pymupdf==1.24.10 --hash=sha256:c0d1ccdc062ea9961063790831e838bc43fcf9a8436a8b9f55898addf97c0f86
pdfminer.six==20231228 --hash=sha256:e8d3c3310e6fbc1fe414090123ab01351634b4ecb021232206c4c9a8ca3e3b8f
ocrmypdf==16.2.0 --hash=sha256:d2a68e9040e26a0fe9af06e9eccff68dfbb9a481ee345eb762b66670eabfb25a
scikit-learn==1.5.1 --hash=sha256:689b6f74b2c880276e365fe84fe4f1befd6a774f016339c65655eaff12e10cbf
hnswlib==0.8.0 --hash=sha256:cb6d037eedebb34a7134e7dc78966441dfd04c9cf5ee93911be911ced951c44c
watchdog==4.0.0 --hash=sha256:6a80d5cae8c265842c7419c560b9961561556c4361b297b4c431903f8c33b269
requests==2.31.0 --hash=sha256:58cd2187c01e70e6e26505bca751777aa9f2ee0b7f4300988b709f44e013003f
"""

CONFIG_YAML = r"""# Litigation OS configuration
paths:
  ROOT: ""
  DATA: "data"
  LOGS: "logs"
  MODELS: "models"
  OUTPUT: "output"
  MEEK1: "F:/MEEK1"
  MEEK2: "F:/MEEK2"
  EVIDENCE_GLOB: "**/*.(pdf|PDF|docx|png|jpg|jpeg|tif|tiff|txt)"
llm:
  use_ollama: false
  ollama_model: "llama3:8b-instruct"
  llama_cpp:
    server_url: "http://127.0.0.1:8080/completion"
    model_path: "models/llm/model.gguf"
    n_ctx: 8192
    n_threads: 8
    n_gpu_layers: 0
search:
  use_sbert_if_available: true
  sbert_path: "models/emb/sentence-transformers/all-MiniLM-L6-v2"
ocr:
  use_tesseract_if_available: true
  tesseract_cmd: "tesseract"
scanner:
  max_file_mb: 200
  skip_hidden: true
mcr_canon:
  enable_scanner: true
  ruleset_yaml: "backend/rules/canon_mcr_rules.yaml"
mifile:
  bundle_format: "zip"
  exhibit_label: "Exhibit"
server:
  host: "127.0.0.1"
  port: 8000
signing_key: "signing.key"
"""

CANON_RULES = r"""# Canon + MCR quick rules (extend freely)
# Each rule: id, title, hint, pattern (regex), scope
- id: CANON_2A
  title: "Canon 2A: Promote public confidence"
  hint: "Pattern of bias or appearance undermining confidence"
  pattern: "(?i)\\b(bias|prejudge|appearance of impropriety)\\b"
  scope: "text"
- id: MCR_1_109
  title: "MCR 1.109(E): signatures; MCR 1.109(D): file format"
  hint: "Improper filings, signatures, formats, fraud on the court"
  pattern: "(?i)\\b(1\\.109|signature|sanction|fraud on the court)\\b"
  scope: "text"
- id: MCR_2_114
  title: "MCR 2.114: attorney/self-certification; sanctions"
  hint: "False certifications, frivolous filings"
  pattern: "(?i)\\b(2\\.114|frivolous|sanctions)\\b"
  scope: "text"
- id: MCR_2_116
  title: "MCR 2.116: summary disposition"
  hint: "Rule-driven dispositive motions"
  pattern: "(?i)\\b(2\\.116|summary disposition)\\b"
  scope: "text"
"""

BACKEND_APP = r"""# backend/app.py
import hashlib
import io
import json
import logging
import os
import re
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

def _try_import(name: str):
    try:
        return __import__(name)
    except Exception:
        return None

fitz = _try_import("fitz")
pdfminer = _try_import("pdfminer")
pytesseract = _try_import("pytesseract")
PIL = _try_import("PIL")
st = _try_import("sentence_transformers")
hnswlib = _try_import("hnswlib")
requests = _try_import("requests")

watchdog = _try_import("watchdog")
if watchdog:
    from watchdog.events import FileSystemEventHandler  # type: ignore
    from watchdog.observers import Observer  # type: ignore
else:  # pragma: no cover - watchdog missing
    FileSystemEventHandler = object  # type: ignore
    Observer = None  # type: ignore

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config.yaml", "r", encoding="utf-8"))
LOGS = ROOT / CFG["paths"]["LOGS"]
DATA = ROOT / CFG["paths"]["DATA"]
MODELS = ROOT / CFG["paths"]["MODELS"]
OUTPUT = ROOT / CFG["paths"]["OUTPUT"]
RULES_YAML = ROOT / CFG["mcr_canon"]["ruleset_yaml"]
DB_PATH = ROOT / "catalog.db"

for p in [LOGS, DATA, MODELS, OUTPUT]:
    p.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOGS / "backend.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS files(path TEXT PRIMARY KEY, sha256 TEXT, size INTEGER, mtime INTEGER, text TEXT)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rule_hits(path TEXT, rule_id TEXT, title TEXT, hint TEXT)"
    )
    conn.commit()
    return conn

DB = init_db()

RULES = yaml.safe_load(open(RULES_YAML, "r", encoding="utf-8")) if RULES_YAML.exists() else []

def scan_rules(text: str) -> List[Dict[str, str]]:
    hits: List[Dict[str, str]] = []
    for r in RULES:
        try:
            if re.search(r.get("pattern", ""), text):
                hits.append(
                    {"id": r.get("id", ""), "title": r.get("title", ""), "hint": r.get("hint", "")}
                )
        except Exception:
            continue
    return hits

def extract_text(path: Path) -> str:
    try:
        if path.suffix.lower() == ".pdf":
            if fitz:
                doc = fitz.open(path)
                txt = "\n".join(pg.get_text("text") for pg in doc)
            elif pdfminer:
                from pdfminer.high_level import extract_text as pdf_extract

                txt = pdf_extract(str(path))
            else:
                txt = ""
            if not txt.strip():
                tmp = path.with_suffix(".ocr.pdf")
                try:
                    subprocess.run(["ocrmypdf", "--quiet", str(path), str(tmp)], check=True)
                    txt = extract_text(tmp)
                finally:
                    if tmp.exists():
                        tmp.unlink()
            return txt
        if path.suffix.lower() == ".txt":
            return path.read_text("utf-8", errors="ignore")
        if path.suffix.lower() == ".docx":
            docx = _try_import("docx")
            if docx:
                d = docx.Document(str(path))
                return "\n".join(p.text for p in d.paragraphs)
        if path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]:
            if pytesseract and PIL:
                from PIL import Image

                return pytesseract.image_to_string(Image.open(path))
        return ""
    except Exception as e:  # pragma: no cover
        logging.exception("extract_text error: %s", e)
        return ""


class SearchIndex:
    def __init__(self) -> None:
        self.paths: List[str] = []
        self._sbert = st.SentenceTransformer(str(ROOT / CFG["search"]["sbert_path"])) if st else None
        self._use_hnsw = bool(self._sbert and hnswlib)
        self._tfidf = None
        if not self._use_hnsw:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf = TfidfVectorizer(max_features=200000, ngram_range=(1, 2))
        self._index = None
        self.reload()

    def reload(self) -> None:
        rows = DB.execute("SELECT path, text FROM files").fetchall()
        self.paths = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        if self._use_hnsw and self._sbert:
            import numpy as np

            embs = self._sbert.encode(texts, normalize_embeddings=True)
            dim = embs.shape[1] if len(embs) else self._sbert.get_sentence_embedding_dimension()
            self._index = hnswlib.Index(space="ip", dim=dim)
            self._index.init_index(max_elements=len(embs) + 1000, ef_construction=200, M=16)
            if len(embs):
                self._index.add_items(embs, list(range(len(embs))))
            self._index.save_index(str(MODELS / "ann.index"))
        elif self._tfidf:
            self._tfidf.fit(texts)

    def add_document(self, path: str, text: str) -> None:
        if self._use_hnsw and self._sbert and self._index is not None:
            import numpy as np

            emb = self._sbert.encode([text], normalize_embeddings=True)
            self._index.add_items(emb, [len(self.paths)])
            self._index.save_index(str(MODELS / "ann.index"))
        elif self._tfidf:
            self._tfidf.fit([text])
        self.paths.append(path)

    def search(self, query: str, k: int = 8) -> List[Dict[str, Any]]:
        if not self.paths:
            return []
        if self._use_hnsw and self._sbert and self._index is not None:
            import numpy as np

            qv = self._sbert.encode([query], normalize_embeddings=True)
            D, I = self._index.knn_query(qv, k=min(k, len(self.paths)))
            idxs = I[0].tolist()
            scores = D[0].tolist()
        else:
            from sklearn.metrics.pairwise import cosine_similarity

            texts = [r[0] for r in DB.execute("SELECT text FROM files").fetchall()]
            X = self._tfidf.transform(texts)  # type: ignore[union-attr]
            q = self._tfidf.transform([query])  # type: ignore[union-attr]
            sims = cosine_similarity(q, X).ravel()
            idxs = sims.argsort()[::-1][:k].tolist()
            scores = [float(sims[i]) for i in idxs]
        out: List[Dict[str, Any]] = []
        for i, sc in zip(idxs, scores):
            path = self.paths[i]
            hits = [
                {"id": r[1], "title": r[2], "hint": r[3]}
                for r in DB.execute("SELECT * FROM rule_hits WHERE path=?", (path,))
            ]
            out.append({"path": path, "score": sc, "hits": hits})
        return out

INDEX = SearchIndex()


class EvidenceHandler(FileSystemEventHandler):
    def on_created(self, event):  # type: ignore[override]
        if not getattr(event, "is_directory", False):
            index_file(Path(event.src_path))

    def on_modified(self, event):  # type: ignore[override]
        if not getattr(event, "is_directory", False):
            index_file(Path(event.src_path))

def start_watchers() -> None:
    if Observer is None:
        return
    roots = [Path(CFG["paths"]["MEEK1"]), Path(CFG["paths"]["MEEK2"])]
    for r in roots:
        if r.exists():
            obs = Observer()
            obs.schedule(EvidenceHandler(), str(r), recursive=True)
            obs.daemon = True
            obs.start()

def index_file(path: Path) -> None:
    text = extract_text(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    stat = path.stat()
    DB.execute(
        "REPLACE INTO files(path, sha256, size, mtime, text) VALUES(?,?,?,?,?)",
        (str(path), sha, stat.st_size, int(stat.st_mtime), text),
    )
    DB.execute("DELETE FROM rule_hits WHERE path=?", (str(path),))
    for hit in scan_rules(text):
        DB.execute(
            "INSERT INTO rule_hits(path, rule_id, title, hint) VALUES(?,?,?,?)",
            (str(path), hit["id"], hit["title"], hit["hint"]),
        )
    DB.commit()
    INDEX.add_document(str(path), text)

def initial_scan() -> None:
    roots = [Path(CFG["paths"]["MEEK1"]), Path(CFG["paths"]["MEEK2"])]
    known = {r[0] for r in DB.execute("SELECT path FROM files").fetchall()}
    for r in roots:
        if not r.exists():
            continue
        for p in r.rglob("*"):
            if p.is_file() and str(p) not in known:
                index_file(p)

initial_scan()
start_watchers()


class AskIn(BaseModel):
    question: str
    k: int = 6


class ScanOut(BaseModel):
    ok: bool
    count: int


class LLME:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> str:
        server = self.cfg["llm"]["llama_cpp"].get("server_url")
        if server and requests:
            try:
                payload = {
                    "prompt": (system + "\n" if system else "") + prompt,
                    "n_predict": max_tokens,
                    "temperature": temperature,
                }
                r = requests.post(server, json=payload, timeout=120)
                if r.ok:
                    j = r.json()
                    return j.get("content", "")
            except Exception:
                pass
        model_path = ROOT / self.cfg["llm"]["llama_cpp"]["model_path"]
        llama_cpp = _try_import("llama_cpp")
        if llama_cpp and model_path.exists():
            try:
                from llama_cpp import Llama  # type: ignore

                llm = Llama(
                    model_path=str(model_path),
                    n_ctx=self.cfg["llm"]["llama_cpp"]["n_ctx"],
                    n_threads=self.cfg["llm"]["llama_cpp"]["n_threads"],
                    n_gpu_layers=self.cfg["llm"]["llama_cpp"]["n_gpu_layers"],
                )
                out = llm(
                    prompt=(system + "\n" if system else "") + prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return out["choices"][0]["text"]
            except Exception:
                return ""
        return ""

LLM = LLME(CFG)

app = FastAPI(title="LAWFORGE Litigation OS API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home() -> HTMLResponse:
    return HTMLResponse("<h3>LAWFORGE API online</h3><p>See /docs</p>")

@app.post("/scan")
def scan() -> ScanOut:
    initial_scan()
    INDEX.reload()
    count = DB.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    return ScanOut(ok=True, count=count)

@app.post("/ask")
def ask(inp: AskIn) -> Dict[str, Any]:
    hits = INDEX.search(inp.question, k=inp.k)
    context = []
    for h in hits:
        row = DB.execute("SELECT text FROM files WHERE path=?", (h["path"],)).fetchone()
        snippet = row[0][:4000] if row else ""
        context.append(f"[{Path(h['path']).name}] {snippet}")
    sys_prompt = (
        "You are a Michigan litigation engine. Use only provided context. "
        "Cite by filename in brackets. Flag possible MCR/Canon issues precisely. "
        "Avoid vague language."
    )
    prompt = (
        "Question:\n" + inp.question + "\n\n"
        "Context:\n" + "\n\n".join(context[:8]) + "\n\n"
        "Answer with numbered, court-usable points and minimal prose."
    )
    answer = LLM.generate(prompt, system=sys_prompt, max_tokens=700, temperature=0.1)
    return {"answer": answer, "retrieval": hits}

@app.post("/ingest_files")
def ingest(files: List[UploadFile]) -> Dict[str, Any]:
    saved = []
    for f in files:
        data = f.file.read()
        target = DATA / f.filename
        target.write_bytes(data)
        saved.append(str(target))
        index_file(target)
    return {"ok": True, "saved": saved}

@app.post("/bundle_mifile")
def bundle_mifile() -> Dict[str, Any]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = OUTPUT / f"FILING_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    rows = DB.execute("SELECT path FROM files LIMIT 50").fetchall()
    for i, (path,) in enumerate(rows, start=1):
        src = Path(path)
        label = f"{CFG['mifile']['exhibit_label']} {i:02d} - {src.name}"
        dest = outdir / label
        try:
            import shutil

            shutil.copy2(src, dest)
            manifest.append({"label": label, "source": path})
        except Exception:
            continue
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zip_path = OUTPUT / f"FILING_{stamp}.zip"
    import zipfile

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in outdir.rglob("*"):
            z.write(p, p.relative_to(OUTPUT))
    return {"ok": True, "zip": str(zip_path), "dir": str(outdir)}

@app.get("/metrics")
def metrics() -> Dict[str, Any]:
    files, size = DB.execute("SELECT COUNT(*), SUM(size) FROM files").fetchone()
    return {"files": files, "bytes": size or 0, "llm_mode": "server"}

@app.get("/diag")
def diag() -> Dict[str, Any]:
    return {"config": CFG, "python": os.sys.version}

@app.post("/manifest/sign")
def manifest_sign() -> Dict[str, Any]:
    rows = DB.execute("SELECT path, sha256 FROM files").fetchall()
    manifest = [{"path": p, "sha256": h} for p, h in rows]
    out = OUTPUT / "evidence_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    key = CFG.get("signing_key")
    if key and Path(key).exists():
        sig = out.with_suffix(out.suffix + ".minisig")
        subprocess.run(["minisign", "-S", "-s", key, "-m", out], check=True)
        return {"ok": True, "manifest": str(out), "signature": str(sig)}
    raise HTTPException(status_code=500, detail="signing key not found")
"""

PYI_SPEC = r"""# golden_god_mode_bootstrap.spec
# PyInstaller spec for building a portable EXE
block_cipher = None

a = Analysis(['golden_god_mode_bootstrap.py'], pathex=['.'])
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='golden_god_mode_bootstrap',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=True, upx_exclude=[],
    name='golden_god_mode_bootstrap')
"""

RUN_BACKEND_BAT = r"""@echo off
setlocal
cd /d %~dp0
call .venv\Scripts\activate
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --log-level info
"""

RUN_FRONTEND_BAT = r"""@echo off
setlocal
cd /d %~dp0\frontend
echo Serving static UI at http://127.0.0.1:7777  (Ctrl+C to stop)
python - <<PY
import http.server, socketserver, os
os.chdir(os.path.dirname(__file__))
with socketserver.TCPServer(("127.0.0.1", 7777), http.server.SimpleHTTPRequestHandler) as httpd:
    httpd.serve_forever()
PY
"""

FRONTEND_INDEX = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>LAWFORGE — Litigation OS</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 18px; }
    textarea, input { width: 100%; }
    .row { display: flex; gap: 16px; }
    .col { flex: 1; }
    pre { background: #111; color: #ddd; padding: 12px; overflow:auto; }
    button { padding: 8px 12px; }
  </style>
</head>
<body>
  <h2>LAWFORGE — Offline Litigation OS</h2>
  <div>
    <button onclick="scan()">Scan Evidence</button>
    <small id="scanStatus"></small>
  </div>
  <hr/>
  <div class="row">
    <div class="col">
      <h3>Ask</h3>
      <label for="q">Enter your question:</label>
      <textarea id="q" rows="8"></textarea>
      <button onclick="ask()">Run</button>
      <pre id="ans"></pre>
    </div>
    <div class="col">
      <h3>Bundle (MiFile)</h3>
      <button onclick="bundle()">Build ZIP</button>
      <pre id="bundleOut"></pre>
      <h3>Health</h3>
      <button onclick="health()">Check</button>
      <pre id="healthOut"></pre>
    </div>
  </div>
  <script src="api.js"></script>
</body>
</html>
"""

FRONTEND_API = r"""const API = "http://127.0.0.1:8000";

async function scan() {
  document.getElementById("scanStatus").innerText = " scanning...";
  const r = await fetch(API + "/scan", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({})
});
  const j = await r.json();
  document.getElementById("scanStatus").innerText = ` indexed: ${j.count}`;
}

async function ask() {
  const q = document.getElementById("q").value;
  const r = await fetch(API + "/ask", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({question:q, k:8})
});
  const j = await r.json();
  document.getElementById("ans").innerText = j.answer || "(no LLM output)\n" + JSON.stringify(j.retrieval, null, 2);
}

async function bundle() {
  const r = await fetch(API + "/bundle_mifile", {method:"POST"});
  const j = await r.json();
  document.getElementById("bundleOut").innerText = JSON.stringify(j, null, 2);
}

async function health() {
  const r = await fetch(API + "/metrics");
  const j = await r.json();
  document.getElementById("healthOut").innerText = JSON.stringify(j, null, 2);
}
"""


def write_if_missing(path: Path, content: str, binary: bool = False) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "wb" if binary else "w"
        with open(path, mode, encoding=None if binary else "utf-8") as f:
            f.write(content)


def run(cmd: str, cwd: Path) -> int:
    print(f"$ {cmd}")
    return subprocess.call(cmd, cwd=str(cwd), shell=True)


def install_deps(root: Path, offline: bool) -> None:
    venv = root / ".venv"
    if not (venv / "Scripts" / "activate").exists():
        run(f'python -m venv "{venv}"', root)
    pip = venv / "Scripts" / "pip.exe"
    req = root / "requirements.txt"
    con = root / "constraints.txt"
    req.write_text(REQ_TXT, encoding="utf-8")
    con.write_text(CONSTRAINTS_TXT, encoding="utf-8")
    if offline:
        wheels = root / ".wheels"
        if wheels.exists():
            run(
                f'"{pip}" install --no-index --find-links "{wheels}" -r "{req}" -c "{con}"',
                root,
            )
        else:
            print("[!] Offline mode selected but .wheels not found. Skipping pip.")
    else:
        run(f'"{pip}" install -r "{req}" -c "{con}"', root)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", required=True, help="Install root, e.g., F:\\LAWFORGE_SUPREMACY"
    )
    ap.add_argument("--offline", action="store_true", help="Use local wheels only")
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    write_if_missing(root / "requirements.txt", REQ_TXT)
    write_if_missing(root / "constraints.txt", CONSTRAINTS_TXT)
    write_if_missing(root / "config.yaml", CONFIG_YAML)
    write_if_missing(root / "backend" / "app.py", BACKEND_APP)
    write_if_missing(root / "backend" / "rules" / "__init__.py", "")
    write_if_missing(root / "backend" / "rules" / "canon_mcr_rules.yaml", CANON_RULES)
    write_if_missing(root / "frontend" / "index.html", FRONTEND_INDEX)
    write_if_missing(root / "frontend" / "api.js", FRONTEND_API)
    write_if_missing(root / "run_backend.bat", RUN_BACKEND_BAT)
    write_if_missing(root / "frontend" / "run_frontend.bat", RUN_FRONTEND_BAT)
    write_if_missing(root / "golden_god_mode_bootstrap.spec", PYI_SPEC)

    for d in [
        "data",
        "logs",
        "models/llm",
        "models/emb/sentence-transformers/all-MiniLM-L6-v2",
        "output",
        ".wheels",
    ]:
        (root / d).mkdir(parents=True, exist_ok=True)

    install_deps(root, args.offline)

    print("\n[READY]")
    print(f"Root: {root}")
    print("1) Activate:    " + str(root / ".venv" / "Scripts" / "activate"))
    print("2) Backend:     run_backend.bat")
    print("3) Frontend:    frontend\\run_frontend.bat")
    print(
        "4) Models:      put GGUF at models\\llm\\model.gguf; set config.yaml -> llm.llama_cpp.model_path"
    )
    print(
        "5) Evidence:    set config.yaml paths.MEEK1, paths.MEEK2; file events auto-index"
    )


if __name__ == "__main__":
    main()
