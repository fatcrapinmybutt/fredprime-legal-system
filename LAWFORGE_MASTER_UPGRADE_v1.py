#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LAWFORGE_MASTER_UPGRADE_v1.py — hardened
Scanner + OCR + embed + search + RAG API with chain-of-custody.
- Fixed pdfminer import
- Added HEIC support (pillow-heif if present)
- Added audio transcription (faster-whisper or openai-whisper if present)
- Optional PDF text-layer repair (ocrmypdf if present) before extraction
- Deterministic JSONL corpus + index manifest (hash, size, times)
- ChromaDB safe upsert (delete-if-exists, then add)
- JSON event log (index/embed/serve)
- /manifest API route

Install extras as available:
  pip install chromadb sentence-transformers pdfminer.six pypdf pdfplumber pillow pytesseract rapidfuzz unstructured[all-docs]
  pip install watchdog tiktoken uvicorn fastapi python-magic-bin
  # Optional add-ons:
  pip install pillow-heif faster-whisper openai-whisper ocrmypdf
"""
import os, re, time, json, hashlib, pathlib, mimetypes, shutil, tempfile, subprocess
from typing import List, Dict, Optional, Tuple

DEFAULT_PATTERN = r".*\.(pdf|txt|rtf|docx?|xlsx?|pptx?|csv|png|jpg|jpeg|tiff|bmp|gif|heic|mp3|wav|m4a|ogg)$"
IGNORE_DIRS = {
    ".git",
    ".cache",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
}
CHUNK_TOKENS = 600
CHUNK_OVERLAP = 80
MAX_FILE_MB = 200


# -------- Lazy imports --------
def _lazy_imports() -> Dict[str, object]:
    L: Dict[str, object] = {}
    try:
        import pdfplumber

        L["pdfplumber"] = pdfplumber
    except Exception:
        L["pdfplumber"] = None
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text  # FIXED

        L["pdfminer_extract_text"] = pdfminer_extract_text
    except Exception:
        L["pdfminer_extract_text"] = None
    try:
        from PIL import Image

        L["PIL_Image"] = Image
    except Exception:
        L["PIL_Image"] = None
    try:
        import pillow_heif  # HEIC support

        pillow_heif.register_heif_opener()
        L["pillow_heif"] = pillow_heif
    except Exception:
        L["pillow_heif"] = None
    try:
        import pytesseract

        L["pytesseract"] = pytesseract
    except Exception:
        L["pytesseract"] = None
    try:
        import easyocr

        L["easyocr"] = easyocr
    except Exception:
        L["easyocr"] = None
    try:
        from unstructured.partition.auto import partition

        L["unstructured_partition"] = partition
    except Exception:
        L["unstructured_partition"] = None
    try:
        from sentence_transformers import SentenceTransformer

        L["SentenceTransformer"] = SentenceTransformer
    except Exception:
        L["SentenceTransformer"] = None
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        L["chromadb"] = chromadb
        L["ChromaSettings"] = ChromaSettings
    except Exception:
        L["chromadb"] = None
        L["ChromaSettings"] = None
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel

        L["FastAPI"] = FastAPI
        L["PydanticBase"] = BaseModel
    except Exception:
        L["FastAPI"] = None
        L["PydanticBase"] = object
    try:
        import uvicorn

        L["uvicorn"] = uvicorn
    except Exception:
        L["uvicorn"] = None
    try:
        from rapidfuzz import fuzz

        L["fuzz"] = fuzz
    except Exception:
        L["fuzz"] = None
    try:
        import tiktoken

        L["tiktoken"] = tiktoken
    except Exception:
        L["tiktoken"] = None
    # Audio transcription
    try:
        from faster_whisper import WhisperModel

        L["faster_whisper"] = WhisperModel
    except Exception:
        L["faster_whisper"] = None
    try:
        import whisper as openai_whisper

        L["openai_whisper"] = openai_whisper
    except Exception:
        L["openai_whisper"] = None
    # PDF text-layer repair
    try:
        import ocrmypdf  # noqa: F401

        L["ocrmypdf"] = True
    except Exception:
        L["ocrmypdf"] = False
    return L


L = _lazy_imports()


# -------- Logging --------
def _log_path(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, "events.log.jsonl")


def jlog(outdir: str, event: str, **kw) -> None:
    rec = {"ts": int(time.time()), "event": event, **kw}
    try:
        with open(_log_path(outdir), "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


# -------- Utils --------
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_relpath(path: str, root: str) -> str:
    try:
        return str(
            pathlib.Path(path).resolve().relative_to(pathlib.Path(root).resolve())
        )
    except Exception:
        return os.path.abspath(path)


def is_binary_by_mime(path: str) -> bool:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        return False
    return mime.startswith(("audio/", "video/", "application/x-executable"))


def file_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def tokenize_len(s: str) -> int:
    if L["tiktoken"]:
        try:
            enc = L["tiktoken"].get_encoding("cl100k_base")
            return len(enc.encode(s))
        except Exception:
            pass
    return max(1, len(s) // 4)


def chunk_text(
    s: str, tokens: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP
) -> List[str]:
    words = s.split()
    stride = max(1, tokens - overlap)
    chunks, i = [], 0
    while i < len(words):
        seg = " ".join(words[i : i + tokens]).strip()
        if seg:
            chunks.append(seg)
        i += stride
    return chunks


# -------- Audio transcription --------
def transcribe_audio(path: str) -> str:
    if L["faster_whisper"]:
        try:
            model = L["faster_whisper"]("base", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(path, beam_size=1)
            return "\n".join(seg.text.strip() for seg in segments if seg.text.strip())
        except Exception:
            pass
    if L["openai_whisper"]:
        try:
            model = L["openai_whisper"].load_model("base")
            result = model.transcribe(path)
            return result.get("text", "").strip()
        except Exception:
            pass
    return ""


# -------- PDF repair (optional) --------
def _repair_pdf_to_temp(path: str) -> Optional[str]:
    if not L["ocrmypdf"]:
        return None
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.close()
        subprocess.check_call(["ocrmypdf", "--skip-text", "--quiet", path, tmp.name])
        return tmp.name
    except Exception:
        return None


# -------- Extractors --------
def extract_text_pdf(path: str) -> str:
    text = ""
    if L["pdfplumber"]:
        try:
            out = []
            with L["pdfplumber"].open(path) as pdf:
                for page in pdf.pages:
                    out.append(page.extract_text() or "")
            text = ("\n".join(out)).strip()
            if text:
                return text
        except Exception:
            pass
    if L["pdfminer_extract_text"]:
        try:
            text = L["pdfminer_extract_text"](path) or ""
            if text.strip():
                return text.strip()
        except Exception:
            pass
    repaired = _repair_pdf_to_temp(path)
    if repaired:
        try:
            if L["pdfminer_extract_text"]:
                text = L["pdfminer_extract_text"](repaired) or ""
                os.unlink(repaired)
                if text.strip():
                    return text.strip()
        except Exception:
            try:
                os.unlink(repaired)
            except Exception:
                pass
    return ""


def extract_text_image(path: str) -> str:
    if L["pytesseract"] and L["PIL_Image"]:
        try:
            img = L["PIL_Image"].open(path)
            txt = L["pytesseract"].image_to_string(img)
            if txt and txt.strip():
                return txt.strip()
        except Exception:
            pass
    if L["easyocr"]:
        try:
            reader = L["easyocr"].Reader(["en"], gpu=False)
            res = reader.readtext(path, detail=0, paragraph=True)
            if res:
                return "\n".join([r.strip() for r in res if r.strip()])
        except Exception:
            pass
    return ""


def extract_text_unstructured(path: str) -> str:
    if L["unstructured_partition"]:
        try:
            elements = L["unstructured_partition"](filename=path)
            return "\n".join([str(e) for e in elements]).strip()
        except Exception:
            pass
    return ""


def extract_text_generic(path: str) -> str:
    p = path.lower()
    if p.endswith(".pdf"):
        return extract_text_pdf(path)
    if p.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".heic")):
        return extract_text_image(path)
    if p.endswith((".mp3", ".wav", ".m4a", ".ogg")):
        return transcribe_audio(path)
    return extract_text_unstructured(path)


# -------- Indexer --------
def _write_manifest(outdir: str, manifest: Dict) -> None:
    mpath = os.path.join(outdir, "index_manifest.json")
    with open(mpath, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False, indent=2))


def index_folder(
    root: str, outdir: str, pattern: str = DEFAULT_PATTERN, ocr: bool = True
) -> None:
    os.makedirs(outdir, exist_ok=True)
    corpus_path = os.path.join(outdir, "corpus.jsonl")
    seen_hashes = set()
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    h = j.get("sha256", "")
                    if h:
                        seen_hashes.add(h)
                except Exception:
                    pass
    rx = re.compile(pattern, re.I)
    added = skipped = 0
    start = int(time.time())
    jlog(outdir, "index_start", root=root, pattern=pattern, ocr=ocr)
    with open(corpus_path, "a", encoding="utf-8") as out:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            for fname in filenames:
                full = os.path.join(dirpath, fname)
                if not rx.match(fname):
                    continue
                try:
                    if file_mb(full) > MAX_FILE_MB:
                        skipped += 1
                        continue
                    if is_binary_by_mime(full) and not full.lower().endswith(
                        (".mp3", ".wav", ".m4a", ".ogg")
                    ):
                        skipped += 1
                        continue
                    h = sha256_file(full)
                    if h in seen_hashes:
                        continue
                    txt = extract_text_generic(full)
                    if (
                        not txt
                        and ocr
                        and full.lower().endswith(
                            (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".heic")
                        )
                    ):
                        txt = extract_text_image(full)
                    record = {
                        "sha256": h,
                        "path": os.path.abspath(full),
                        "relpath": safe_relpath(full, root),
                        "bytes": os.path.getsize(full),
                        "created": int(os.path.getctime(full)),
                        "modified": int(os.path.getmtime(full)),
                        "text": (txt or "").strip(),
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    added += 1
                except Exception:
                    skipped += 1
    end = int(time.time())
    manifest = {
        "root": os.path.abspath(root),
        "outdir": os.path.abspath(outdir),
        "pattern": pattern,
        "added": added,
        "skipped": skipped,
        "started": start,
        "ended": end,
        "duration_sec": end - start,
        "schema_version": "1.0",
    }
    _write_manifest(outdir, manifest)
    jlog(outdir, "index_complete", **manifest)
    print(
        f"[index] added={added} skipped={skipped} secs={manifest['duration_sec']} out={corpus_path}"
    )


# -------- Embeddings + Vector store --------
def ensure_vector_store(
    index_dir: str, model_name: str = "all-MiniLM-L6-v2"
) -> Tuple[object, object]:
    if L["chromadb"] is None or L["SentenceTransformer"] is None:
        raise RuntimeError("chromadb and sentence-transformers required")
    chroma_path = os.path.join(index_dir, "chroma")
    client = L["chromadb"].PersistentClient(
        path=chroma_path, settings=L["ChromaSettings"](anonymized_telemetry=False)
    )
    coll = client.get_or_create_collection("lawforge")
    model = L["SentenceTransformer"](model_name)
    return coll, model


def _chroma_upsert(coll, ids, texts, metas, embs) -> None:
    try:
        if hasattr(coll, "upsert"):
            coll.upsert(ids=ids, embeddings=embs, metadatas=metas, documents=texts)  # type: ignore
            return
    except Exception:
        pass
    try:
        coll.delete(ids=ids)
    except Exception:
        pass
    coll.add(ids=ids, embeddings=embs, metadatas=metas, documents=texts)


def embed_corpus(
    index_dir: str, model_name: str = "all-MiniLM-L6-v2", batch: int = 64
) -> None:
    corpus_path = os.path.join(index_dir, "corpus.jsonl")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(corpus_path)
    coll, model = ensure_vector_store(index_dir, model_name=model_name)
    ids, texts, metas = [], [], []
    start = int(time.time())
    added = 0
    jlog(index_dir, "embed_start", model=model_name)
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            text = j.get("text", "").strip()
            if not text:
                continue
            for i, chunk in enumerate(chunk_text(text)):
                ids.append(f"{j['sha256']}:{i}")
                texts.append(chunk)
                metas.append(
                    {
                        "path": j["path"],
                        "relpath": j["relpath"],
                        "sha256": j["sha256"],
                        "chunk": i,
                    }
                )
                if len(ids) >= batch:
                    embs = model.encode(
                        texts, show_progress_bar=False, convert_to_numpy=True
                    ).tolist()
                    _chroma_upsert(coll, ids, texts, metas, embs)
                    added += len(ids)
                    ids, texts, metas = [], [], []
    if ids:
        embs = model.encode(
            texts, show_progress_bar=False, convert_to_numpy=True
        ).tolist()
        _chroma_upsert(coll, ids, texts, metas, embs)
        added += len(ids)
    end = int(time.time())
    jlog(index_dir, "embed_complete", added=added, duration_sec=end - start)
    print("[embed] complete")


# -------- Search --------
def search_keyword(index_dir: str, q: str, k: int = 20) -> List[Dict]:
    corpus_path = os.path.join(index_dir, "corpus.jsonl")
    results = []
    if not os.path.exists(corpus_path):
        return results
    ql = q.lower()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            text = j.get("text", "")
            if not text:
                continue
            score = text.lower().count(ql)
            if L["fuzz"]:
                try:
                    score = max(score, int(L["fuzz"].partial_ratio(ql, text.lower())))
                except Exception:
                    pass
            if score > 0:
                results.append({"score": score, **j})
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:k]


def ensure_vs(index_dir: str, model_name="all-MiniLM-L6-v2"):
    return ensure_vector_store(index_dir, model_name)


def search_vector(
    index_dir: str, q: str, k: int = 10, model_name="all-MiniLM-L6-v2"
) -> List[Dict]:
    coll, model = ensure_vs(index_dir, model_name=model_name)
    emb = model.encode([q], convert_to_numpy=True).tolist()[0]
    out = coll.query(query_embeddings=[emb], n_results=k)
    docs = out.get("documents", [[]])[0]
    metas = out.get("metadatas", [[]])[0]
    ids = out.get("ids", [[]])[0]
    res = []
    for i in range(len(docs)):
        m = metas[i] if i < len(metas) else {}
        res.append(
            {
                "id": ids[i],
                "path": m.get("path"),
                "relpath": m.get("relpath"),
                "sha256": m.get("sha256"),
                "chunk": m.get("chunk"),
                "text": docs[i],
            }
        )
    return res


# -------- Q&A --------
def answer_with_llm(
    context: str, question: str, provider: str = "openai", max_tokens: int = 512
) -> str:
    provider = provider.lower().strip()
    if provider == "openai":
        try:
            import openai

            client = openai.OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Michigan litigation assistant. Use only the provided context.",
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}",
                    },
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"[openai-error] {e}"
    if provider == "anthropic":
        try:
            import anthropic

            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=max_tokens,
                temperature=0.2,
                system="You are a Michigan litigation assistant. Use only the provided context.",
                messages=[
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}",
                    }
                ],
            )
            return msg.content[0].text.strip()
        except Exception as e:
            return f"[anthropic-error] {e}"
    if provider == "llama":
        try:
            import llama_cpp

            model_path = os.getenv("LLAMA_CPP_MODEL")
            if not model_path:
                return "[llama-error] LLAMA_CPP_MODEL env var not set"
            ctx = int(os.getenv("LLAMA_CTX", "4096"))
            llm = llama_cpp.Llama(
                model_path=model_path, n_ctx=ctx, logits_all=False, n_threads=8
            )
            prompt = (
                "You are a Michigan litigation assistant. Use only the provided context.\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
            out = llm(prompt, max_tokens=max_tokens, temperature=0.2, stop=["\n\n"])
            return out["choices"][0]["text"].strip()
        except Exception as e:
            return f"[llama-error] {e}"
    if provider == "hf":
        try:
            from huggingface_hub import InferenceClient

            token = os.getenv("HF_TOKEN")
            client = InferenceClient(token=token) if token else InferenceClient()
            prompt = (
                "You are a Michigan litigation assistant. Use only the provided context.\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
            gen = client.text_generation(
                prompt,
                model="meta-llama/Meta-Llama-3-8B-Instruct",
                max_new_tokens=max_tokens,
                temperature=0.2,
            )
            return gen.strip()
        except Exception as e:
            return f"[hf-error] {e}"
    return "[provider-error] Unknown provider"


def retrieve_then_answer(
    index_dir: str, q: str, top_k: int = 6, provider: str = "openai"
) -> Dict:
    kw = search_keyword(index_dir, q, k=top_k)
    vs = search_vector(index_dir, q, k=top_k)
    seen, ctx_parts = set(), []
    for r in kw + vs:
        key = (r.get("sha256"), r.get("chunk"))
        if key in seen:
            continue
        seen.add(key)
        text = r.get("text") or ""
        if text:
            ctx_parts.append(text.strip())
        if len(" ".join(ctx_parts)) > 16000:
            break
    context = "\n\n---\n\n".join(ctx_parts)
    answer = answer_with_llm(context, q, provider=provider)
    return {
        "answer": answer,
        "context_len": len(context),
        "chunks_used": len(ctx_parts),
    }


# -------- API --------
def run_api(index_dir: str, host: str = "127.0.0.1", port: int = 8000):
    if L["FastAPI"] is None:
        raise RuntimeError("fastapi and pydantic required to serve API")
    app = L["FastAPI"]()

    class Q(L["PydanticBase"]):  # type: ignore
        q: str
        provider: Optional[str] = "openai"
        top_k: Optional[int] = 6

    @app.get("/health")
    def health() -> Dict[str, object]:
        mf = os.path.join(index_dir, "index_manifest.json")
        meta = {}
        if os.path.exists(mf):
            try:
                meta = json.loads(open(mf, "r", encoding="utf-8").read())
            except Exception:
                meta = {}
        return {"ok": True, "indexed": meta.get("added", 0), "manifest": bool(meta)}

    @app.post("/search")
    def _search(q: Q) -> Dict[str, List[Dict]]:
        r_kw = search_keyword(index_dir, q.q, k=q.top_k or 6)
        r_vs = search_vector(index_dir, q.q, k=q.top_k or 6)
        return {"keyword": r_kw, "vector": r_vs}

    @app.post("/ask")
    def _ask(q: Q) -> Dict[str, object]:
        return retrieve_then_answer(
            index_dir, q.q, top_k=q.top_k or 6, provider=q.provider or "openai"
        )

    @app.get("/manifest")
    def manifest() -> Dict[str, object]:
        mf = os.path.join(index_dir, "index_manifest.json")
        if not os.path.exists(mf):
            return {"ok": False, "error": "manifest missing"}
        return {
            "ok": True,
            "manifest": json.loads(open(mf, "r", encoding="utf-8").read()),
        }

    if L["uvicorn"] is None:
        raise RuntimeError("uvicorn required to run server")
    L["uvicorn"].run(app, host=host, port=port)


# -------- CLI --------
def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="LAWFORGE MASTER UPGRADE (hardened)")
    sub = p.add_subparsers(dest="cmd")

    p_index = sub.add_parser("index", help="Scan and build corpus")
    p_index.add_argument("--root", required=True)
    p_index.add_argument("--out", required=True)
    p_index.add_argument("--pattern", default=DEFAULT_PATTERN)
    p_index.add_argument("--ocr", choices=["yes", "no"], default="yes")

    p_embed = sub.add_parser("embed", help="Embed corpus to vector store")
    p_embed.add_argument("--index", required=True)
    p_embed.add_argument("--model", default="all-MiniLM-L6-v2")

    p_search = sub.add_parser("search", help="Keyword search")
    p_search.add_argument("--index", required=True)
    p_search.add_argument("--q", required=True)
    p_search.add_argument("--k", type=int, default=20)

    p_vsearch = sub.add_parser("vsearch", help="Vector search")
    p_vsearch.add_argument("--index", required=True)
    p_vsearch.add_argument("--q", required=True)
    p_vsearch.add_argument("--k", type=int, default=10)
    p_vsearch.add_argument("--model", default="all-MiniLM-L6-v2")

    p_ask = sub.add_parser("ask", help="RAG Q&A with context")
    p_ask.add_argument("--index", required=True)
    p_ask.add_argument("--q", required=True)
    p_ask.add_argument(
        "--provider", default="openai", choices=["openai", "anthropic", "llama", "hf"]
    )
    p_ask.add_argument("--top_k", type=int, default=6)

    p_srv = sub.add_parser("serve", help="Start local API")
    p_srv.add_argument("--index", required=True)
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8000)

    args = p.parse_args()
    if args.cmd == "index":
        index_folder(args.root, args.out, pattern=args.pattern, ocr=(args.ocr == "yes"))
        print("Next: embed with 'embed' command.")
        return
    if args.cmd == "embed":
        embed_corpus(args.index, model_name=args.model)
        return
    if args.cmd == "search":
        res = search_keyword(args.index, args.q, k=args.k)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    if args.cmd == "vsearch":
        res = search_vector(args.index, args.q, k=args.k, model_name=args.model)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    if args.cmd == "ask":
        res = retrieve_then_answer(
            args.index, args.q, top_k=args.top_k, provider=args.provider
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return
    if args.cmd == "serve":
        run_api(args.index, host=args.host, port=args.port)
        return
    p.print_help()


if __name__ == "__main__":
    main()
