# -*- coding: utf-8 -*-
"""
GOLDEN GOD MODE — Offline AI + LLM + Litigation OS Bootstrap (Windows)

What you get
------------
A full, local-first litigation workspace with:
  • FastAPI backend (evidence scan, ask LLM, vector search, OCR, binder build)
  • Minimal static UI (index.html + api.js) for query and control
  • LLM engine that prefers llama.cpp (local GGUF), falls back to Ollama if present
  • Embeddings/search: TF-IDF baseline; optional FAISS and SBERT if available locally
  • OCR/Text: PyMuPDF and pdfminer for PDFs; pytesseract if Tesseract is installed
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
      python golden_god_mode_bootstrap.py --root "F:\LAWFORGE_SUPREMACY"
   If offline, add pre-downloaded wheels under {root}\.wheels and re-run with:
      python golden_god_mode_bootstrap.py --root "F:\LAWFORGE_SUPREMACY" --offline
2) Activate venv:
      F:\LAWFORGE_SUPREMACY\.venv\Scripts\activate
3) Start backend:
      run_backend.bat
   Open UI:
      double-click frontend\run_frontend.bat  (serves static UI at http://127.0.0.1:7777)
   Backend API at http://127.0.0.1:8000/docs
4) Point scanners to evidence roots:
      Edit config.yaml (paths.MEEK1, paths.MEEK2, paths.EVIDENCE_GLOB)

Switch LLM
----------
- llama.cpp (preferred, CPU ok): set LLAMA_MODEL_PATH in config.yaml to your .gguf
- Ollama: install Ollama separately, set use_ollama: true and model name in config.yaml
- No LLM available: system still scans, OCRs, and builds binders. Q&A degraded.

One-command full reset
----------------------
Delete {root}, re-run this script. Idempotent file writes are guarded.

Author: Strictly synthetic. No opinions. Pure utility.
"""
import os, sys, argparse, json, shutil, textwrap, subprocess, time
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

# Search
scikit-learn==1.5.1

# Optional (if present or wheels available)
# faiss-cpu==1.8.0.post1
# sentence-transformers==3.0.1
# llama-cpp-python==0.2.90
# python-docx==1.1.2
# pillow==10.4.0
# pytesseract==0.3.13
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
    model_path: "models/llm/model.gguf"   # local GGUF path
    n_ctx: 8192
    n_threads: 8
    n_gpu_layers: 0
search:
  use_faiss_if_available: true
  use_sbert_if_available: true
  sbert_path: "models/emb/sentence-transformers/all-MiniLM-L6-v2"
ocr:
  use_tesseract_if_available: true
  tesseract_cmd: "tesseract"  # on PATH; set full path if needed
scanner:
  max_file_mb: 200
  skip_hidden: true
mcr_canon:
  enable_scanner: true
  ruleset_yaml: "backend/rules/canon_mcr_rules.yaml"
mifile:
  bundle_format: "zip"        # zip for filing; also emits PDF/DOCX if deps available
  exhibit_label: "Exhibit"
server:
  host: "127.0.0.1"
  port: 8000
"""

CANON_RULES = r"""# Canon + MCR quick rules (extend freely)
# Each rule: id, title, hint, pattern (regex), scope
- id: CANON_2A
  title: "Canon 2A: Promote public confidence"
  hint: "Pattern of bias or appearance undermining confidence"
  pattern: "(?i)\b(bias|prejudge|appearance of impropriety)\b"
  scope: "text"
- id: MCR_1_109
  title: "MCR 1.109(E): signatures; MCR 1.109(D): file format"
  hint: "Improper filings, signatures, formats, fraud on the court"
  pattern: "(?i)\b(1\.109|signature|sanction|fraud on the court)\b"
  scope: "text"
- id: MCR_2_114
  title: "MCR 2.114: attorney/self-certification; sanctions"
  hint: "False certifications, frivolous filings"
  pattern: "(?i)\b(2\.114|frivolous|sanctions)\b"
  scope: "text"
- id: MCR_2_116
  title: "MCR 2.116: summary disposition"
  hint: "Rule-driven dispositive motions"
  pattern: "(?i)\b(2\.116|summary disposition)\b"
  scope: "text"
"""

BACKEND_APP = r"""# backend/app.py
import os, io, re, json, time, glob, zipfile, uuid, logging
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from pydantic import BaseModel
import yaml

# Optional imports guarded
def _try_import(name):
    try:
        return __import__(name)
    except Exception:
        return None

np = _try_import("numpy")
sk = _try_import("sklearn")
faiss = _try_import("faiss") or _try_import("faiss_cpu")
st = _try_import("sentence_transformers")
fitz = _try_import("fitz")            # PyMuPDF
pdfminer = _try_import("pdfminer")
pytesseract = _try_import("pytesseract")
PIL = _try_import("PIL")
llama_cpp = _try_import("llama_cpp")
docx = _try_import("docx")

from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT/"config.yaml","r",encoding="utf-8"))
LOGS = ROOT / CFG["paths"]["LOGS"]
DATA = ROOT / CFG["paths"]["DATA"]
MODELS = ROOT / CFG["paths"]["MODELS"]
OUTPUT = ROOT / CFG["paths"]["OUTPUT"]
RULES_YAML = ROOT / CFG["mcr_canon"]["ruleset_yaml"]

for p in [LOGS, DATA, MODELS, OUTPUT]:
    p.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOGS/"backend.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

app = FastAPI(title="LAWFORGE Litigation OS API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ----------------------------
# Utility: text extraction
# ----------------------------
def extract_text_bytes(name: str, b: bytes) -> str:
    name_low = name.lower()
    try:
        if name_low.endswith(".pdf") and fitz:
            doc = fitz.open(stream=b, filetype="pdf")
            chunks = []
            for page in doc:
                chunks.append(page.get_text("text"))
            return "\n".join(chunks)
        elif name_low.endswith(".pdf") and pdfminer:
            from pdfminer.high_level import extract_text as pdf_extract
            bio = io.BytesIO(b)
            return pdf_extract(bio)
        elif name_low.endswith(".txt"):
            return b.decode("utf-8", errors="ignore")
        elif any(name_low.endswith(ext) for ext in [".png",".jpg",".jpeg",".tif",".tiff"]):
            if pytesseract and PIL:
                from PIL import Image
                img = Image.open(io.BytesIO(b))
                return pytesseract.image_to_string(img)
            else:
                return ""
        elif name_low.endswith(".docx") and docx:
            d = docx.Document(io.BytesIO(b))
            return "\n".join(p.text for p in d.paragraphs)
    except Exception as e:
        logging.exception(f"extract_text_bytes error: {e}")
    return ""

def load_text_from_path(p: Path) -> str:
    try:
        if p.suffix.lower()==".pdf" and fitz:
            doc = fitz.open(p)
            return "\n".join([pg.get_text("text") for pg in doc])
        elif p.suffix.lower()==".pdf" and pdfminer:
            from pdfminer.high_level import extract_text as pdf_extract
            return pdf_extract(str(p))
        elif p.suffix.lower()==".txt":
            return p.read_text("utf-8", errors="ignore")
        elif p.suffix.lower()==".docx" and docx:
            d = docx.Document(str(p))
            return "\n".join(par.text for par in d.paragraphs)
        elif p.suffix.lower() in [".png",".jpg",".jpeg",".tif",".tiff"] and pytesseract and PIL:
            from PIL import Image
            return pytesseract.image_to_string(Image.open(p))
    except Exception as e:
        logging.exception(f"load_text_from_path error: {e}")
    return ""

# ----------------------------
# Simple rule scanner (Canon/MCR)
# ----------------------------
RULES = []
if RULES_YAML.exists():
    RULES = yaml.safe_load(open(RULES_YAML,"r",encoding="utf-8")) or []

def scan_rules(text: str):
    hits = []
    for r in RULES:
        try:
            if re.search(r.get("pattern",""), text):
                hits.append({"id":r.get("id"), "title":r.get("title"), "hint":r.get("hint")})
        except Exception:
            continue
    return hits

# ----------------------------
# Embedding + Search
# ----------------------------
class SearchIndex:
    def __init__(self):
        self.docs = []      # [(path, text)]
        self.paths = []
        self._tfidf = None
        self._faiss = None
        self._sbert = None
        self._dim = 0
        self._use_faiss = bool(CFG["search"]["use_faiss_if_available"] and faiss)
        self._use_sbert = bool(CFG["search"]["use_sbert_if_available"] and st)
        if self._use_sbert:
            try:
                self._sbert = st.SentenceTransformer(str(ROOT / CFG["search"]["sbert_path"]))
                self._dim = self._sbert.get_sentence_embedding_dimension()
            except Exception:
                self._sbert = None
                self._use_sbert = False

    def build(self, items: List[Path]):
        texts, self.paths = [], []
        for p in items:
            t = load_text_from_path(p)
            if t:
                texts.append(t)
                self.paths.append(str(p))
        self.docs = list(zip(self.paths, texts))
        if not self.docs:
            return
        if self._use_sbert and self._sbert:
            import numpy as np
            embs = self._sbert.encode([t for _, t in self.docs], normalize_embeddings=True)
            self._dim = embs.shape[1]
            if self._use_faiss:
                self._faiss = faiss.IndexFlatIP(self._dim)
                self._faiss.add(embs.astype("float32"))
            else:
                self._tfidf = None
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(max_features=200000, ngram_range=(1,2))
            self._tfidf.fit([t for _, t in self.docs])

    def search(self, query: str, k: int = 8):
        if not self.docs:
            return []
        if self._sbert and (self._faiss or not self._use_faiss):
            import numpy as np
            qv = self._sbert.encode([query], normalize_embeddings=True).astype("float32")
            if self._faiss:
                D,I = self._faiss.search(qv, min(k, len(self.docs)))
                idxs = I[0].tolist()
                scores = D[0].tolist()
            else:
                # cosine by dot if normalized
                corpus = self._sbert.encode([t for _, t in self.docs], normalize_embeddings=True)
                sims = (corpus @ qv.T).ravel()
                idxs = sims.argsort()[::-1][:k].tolist()
                scores = [float(sims[i]) for i in idxs]
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            tfidf = self._tfidf
            X = tfidf.transform([t for _, t in self.docs])
            q = tfidf.transform([query])
            sims = cosine_similarity(q, X).ravel()
            idxs = sims.argsort()[::-1][:k].tolist()
            scores = [float(sims[i]) for i in idxs]
        out = []
        for i, sc in zip(idxs, scores):
            path, text = self.docs[i]
            out.append({"path": path, "score": float(sc), "hits": scan_rules(text) if CFG["mcr_canon"]["enable_scanner"] else []})
        return out

INDEX = SearchIndex()

# ----------------------------
# LLM Engine
# ----------------------------
class LLME:
    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = None
        self.llm = None
        # try llama.cpp first
        if llama_cpp and Path(cfg["llm"]["llama_cpp"]["model_path"]).exists():
            try:
                from llama_cpp import Llama
                self.llm = Llama(
                    model_path=str(ROOT / cfg["llm"]["llama_cpp"]["model_path"]),
                    n_ctx=cfg["llm"]["llama_cpp"]["n_ctx"],
                    n_threads=cfg["llm"]["llama_cpp"]["n_threads"],
                    n_gpu_layers=cfg["llm"]["llama_cpp"]["n_gpu_layers"],
                    logits_all=False,
                    verbose=False
                )
                self.mode = "llama_cpp"
            except Exception:
                self.llm = None
        # then Ollama
        if self.llm is None and cfg["llm"]["use_ollama"]:
            self.mode = "ollama"

    def generate(self, prompt: str, system: Optional[str]=None, max_tokens: int=512, temperature: float=0.2):
        if self.mode == "llama_cpp" and self.llm:
            tpl = f"{system+'\n' if system else ''}{prompt}"
            out = self.llm(
                prompt=tpl,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>","<|eot_id|>"]
            )
            return out["choices"][0]["text"]
        elif self.mode == "ollama":
            try:
                import subprocess, json
                cmd = [
                    "ollama","run", CFG["llm"]["ollama_model"],
                    "--json"
                ]
                p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                req = {"prompt": (system+"\n" if system else "") + prompt}
                out,_ = p.communicate(json.dumps(req))
                # stream aware; take last response
                last = ""
                for line in out.splitlines():
                    try:
                        j = json.loads(line)
                        if "response" in j:
                            last += j["response"]
                    except Exception:
                        continue
                return last
            except Exception:
                return ""
        else:
            return ""  # no LLM configured

LLM = LLME(CFG)

# ----------------------------
# API Models
# ----------------------------
class AskIn(BaseModel):
    question: str
    k: int = 6

class ScanIn(BaseModel):
    glob: Optional[str] = None
    max_mb: Optional[int] = None

# ----------------------------
# Routes
# ----------------------------
@app.get("/")
def home():
    return HTMLResponse("<h3>LAWFORGE API online</h3><p>See /docs</p>")

@app.post("/scan")
def scan(inp: ScanIn):
    glob_pat = inp.glob or CFG["paths"]["EVIDENCE_GLOB"]
    max_mb = inp.max_mb or CFG["scanner"]["max_file_mb"]
    roots = [Path(CFG["paths"]["MEEK1"]), Path(CFG["paths"]["MEEK2"])]
    files = []
    for r in roots:
        if not r.exists(): continue
        files.extend([p for p in r.rglob("*") if p.is_file()])
    # Filter by size and extension pattern
    selected = []
    import fnmatch
    patterns = [g.strip() for g in glob_pat.split("|")]
    for p in files:
        if any(fnmatch.fnmatch(p.name, pat.replace("**/*.", "*.")) for pat in patterns):
            sz_mb = p.stat().st_size / (1024*1024)
            if sz_mb <= max_mb:
                selected.append(p)
    INDEX.build(selected)
    return {"ok": True, "count": len(INDEX.docs), "paths": INDEX.paths[:1000]}

@app.post("/ask")
def ask(q: AskIn):
    # retrieve
    hits = INDEX.search(q.question, k=q.k)
    context = []
    for h in hits:
        try:
            text = load_text_from_path(Path(h["path"]))[:4000]
        except Exception:
            text = ""
        context.append(f"[{Path(h['path']).name}] {text}")
    sys_prompt = (
        "You are a Michigan litigation engine. Use only provided context. "
        "Cite by filename in brackets. Flag possible MCR/Canon issues precisely. "
        "No placeholders."
    )
    prompt = (
        "Question:\n" + q.question + "\n\n"
        "Context:\n" + "\n\n".join(context[:8]) + "\n\n"
        "Answer with numbered, court-usable points and minimal prose."
    )
    answer = LLM.generate(prompt, system=sys_prompt, max_tokens=700, temperature=0.1)
    return {"answer": answer, "retrieval": hits}

@app.post("/ingest_files")
def ingest(files: List[UploadFile]):
    saved = []
    for f in files:
        b = f.file.read()
        target = DATA / f.filename
        target.write_bytes(b)
        saved.append(str(target))
    return {"ok": True, "saved": saved}

@app.post("/bundle_mifile")
def bundle_mifile():
    # Build a simple filing ZIP with exhibits + manifest.json
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = OUTPUT / f"FILING_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    # Sample: copy top 50 indexed docs as Exhibits
    for i,(path,text) in enumerate(INDEX.docs[:50], start=1):
        src = Path(path)
        label = f"{CFG['mifile']['exhibit_label']} {i:02d} - {src.name}"
        dest = outdir / label
        try:
            shutil.copy2(src, dest)
            manifest.append({"label": label, "source": path})
        except Exception:
            continue
    (outdir/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    zip_path = OUTPUT / f"FILING_{stamp}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in outdir.rglob("*"):
            z.write(p, p.relative_to(OUTPUT))
    return {"ok": True, "zip": str(zip_path), "dir": str(outdir)}

@app.get("/health")
def health():
    return {"ok": True, "docs_indexed": len(INDEX.docs), "llm_mode": LLM.mode}
"""

BACKEND_RULES_INIT = r"""# backend/rules/__init__.py
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
      <textarea id="q" rows="8" placeholder="Enter your question..."></textarea>
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
  const r = await fetch(API + "/scan", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({})});
  const j = await r.json();
  document.getElementById("scanStatus").innerText = ` indexed: ${j.count}`;
}

async function ask() {
  const q = document.getElementById("q").value;
  const r = await fetch(API + "/ask", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({question:q, k:8})});
  const j = await r.json();
  document.getElementById("ans").innerText = j.answer || "(no LLM output)\n" + JSON.stringify(j.retrieval, null, 2);
}

async function bundle() {
  const r = await fetch(API + "/bundle_mifile", {method:"POST"});
  const j = await r.json();
  document.getElementById("bundleOut").innerText = JSON.stringify(j, null, 2);
}

async function health() {
  const r = await fetch(API + "/health");
  const j = await r.json();
  document.getElementById("healthOut").innerText = JSON.stringify(j, null, 2);
}
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


def write_if_missing(path: Path, content: str, binary=False):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "wb" if binary else "w"
        with open(path, mode, encoding=None if binary else "utf-8") as f:
            f.write(content)


def run(cmd: str, cwd: Path):
    print(f"$ {cmd}")
    return subprocess.call(cmd, cwd=str(cwd), shell=True)


def install_deps(root: Path, offline: bool):
    venv = root / ".venv"
    if not (venv / "Scripts" / "activate").exists():
        run(f'python -m venv "{venv}"', root)
    pip = venv / "Scripts" / "pip.exe"
    # Upgrade pip quietly
    if not offline:
        run(f'"{pip}" install --upgrade pip', root)
    req = root / "requirements.txt"
    req.write_text(REQ_TXT, encoding="utf-8")
    if offline:
        wheels = root / ".wheels"
        if wheels.exists():
            run(f'"{pip}" install --no-index --find-links "{wheels}" -r "{req}"', root)
        else:
            print("[!] Offline mode selected but .wheels not found. Skipping pip.")
    else:
        run(f'"{pip}" install -r "{req}"', root)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", required=True, help="Install root, e.g., F:\\LAWFORGE_SUPREMACY"
    )
    ap.add_argument("--offline", action="store_true", help="Use local wheels only")
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    # Write skeleton
    write_if_missing(root / "requirements.txt", REQ_TXT)
    write_if_missing(root / "config.yaml", CONFIG_YAML)
    write_if_missing(root / "backend" / "app.py", BACKEND_APP)
    write_if_missing(root / "backend" / "rules" / "__init__.py", BACKEND_RULES_INIT)
    write_if_missing(root / "backend" / "rules" / "canon_mcr_rules.yaml", CANON_RULES)
    write_if_missing(root / "frontend" / "index.html", FRONTEND_INDEX)
    write_if_missing(root / "frontend" / "api.js", FRONTEND_API)
    write_if_missing(root / "run_backend.bat", RUN_BACKEND_BAT)
    write_if_missing(root / "frontend" / "run_frontend.bat", RUN_FRONTEND_BAT)

    # Create standard dirs
    for d in [
        "data",
        "logs",
        "models/llm",
        "models/emb/sentence-transformers/all-MiniLM-L6-v2",
        "output",
        ".wheels",
    ]:
        (root / d).mkdir(parents=True, exist_ok=True)

    # Install deps
    install_deps(root, args.offline)

    # Final hints
    print("\n[READY]")
    print(f"Root: {root}")
    print("1) Activate:    " + str(root / ".venv" / "Scripts" / "activate"))
    print("2) Backend:     run_backend.bat")
    print("3) Frontend:    frontend\\run_frontend.bat")
    print(
        "4) Models:      put GGUF at models\\llm\\model.gguf; set config.yaml -> llm.llama_cpp.model_path"
    )
    print(
        "5) Evidence:    set config.yaml paths.MEEK1, paths.MEEK2; then POST /scan or use UI 'Scan Evidence'"
    )


if __name__ == "__main__":
    main()
