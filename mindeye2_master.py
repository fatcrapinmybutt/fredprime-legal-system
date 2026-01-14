#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MindEye2 MASTER v10 — Unzip-and-Run Local GUI + Graph + Dossier + Correlation + Neo4j Export

Zero-dependency baseline: standard library only.
Optional accelerators (auto-detected): PyMuPDF (fitz) for PDF page text; pdftotext binary fallback.

Run GUI (Windows example):
  python mindeye2_master.py gui --root "F:\\" --out "F:\\LegalResults\\MindEye2" --autorun --watch 60 --port 8787 --page-anchors --correlate --dossier --neo4j

Termux:
  python mindeye2_master.py gui --root /storage/emulated/0/Download --out /storage/emulated/0/Download/MindEye2_Out --autorun --watch 60 --port 8787 --page-anchors --correlate --dossier --neo4j
"""
from __future__ import annotations
import argparse, csv, datetime as _dt, hashlib, html, http.server, json, logging, os, platform, re, shutil, sqlite3, subprocess, sys, threading, time, urllib.parse, webbrowser
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

APP_NAME = "MindEye2 MASTER"
APP_VERSION = "v10.0"
DEFAULT_PORT = 8787
DEFAULT_WATCH = 0
DEFAULT_MAX_MB = 200
DEFAULT_LARGE_MB = 20

TEXT_EXTS = {".txt", ".md", ".csv", ".json", ".log", ".rtf"}
DOC_EXTS = {".docx"}
HTML_EXTS = {".html", ".htm"}
PDF_EXTS = {".pdf"}

RE_CASE = re.compile(r"\b(20\d{2}-\d{6,7}-(?:DC|DS|PP|PO|DD|DT|CZ|CV|NH|FC))\b", re.IGNORECASE)
RE_MCR  = re.compile(r"\bMCR\s+\d+\.\d+(?:\([A-Za-z0-9]+\))?\b")
RE_MCL  = re.compile(r"\bMCL\s+\d+\.\d+[a-z]?(?:\([0-9A-Za-z]+\))?\b", re.IGNORECASE)
RE_MRE  = re.compile(r"\bMRE\s+\d+\b", re.IGNORECASE)
RE_DATE = re.compile(r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4})\b", re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
RE_NAME = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z]\.)?\s+[A-Z][a-z]+)\b")
RE_ORDER_HINT = re.compile(r"\b(ORDER|JUDGMENT|OPINION|DECISION)\b", re.IGNORECASE)
RE_FILING_HINT = re.compile(r"\b(MOTION|OBJECTION|AFFIDAVIT|BRIEF|APPLICATION|PETITION|NOTICE)\b", re.IGNORECASE)
RE_POLICE_HINT = re.compile(r"\b(POLICE\s+REPORT|INCIDENT\s+REPORT|CASE\s+REPORT|COMPLAINT\s+REPORT)\b", re.IGNORECASE)

SEED_PERSONS = ["Andrew J. Pigors","Emily A. Watson","Jenny L. McNeill","Albert Watson","Lori Watson","Cody Watson","Mandi Martini","Jennifer Barnes"]
SEED_ORGS = ["HealthWest","Health West","Friend of the Court","FOC","RUSCOPA","Muskegon County Circuit Court","14th Circuit Court","Michigan Court of Appeals","Michigan Supreme Court"]

STOPWORDS_NAME = set("the a an and or for with to of in on at from by as is are was were be been being court judge plaintiff defendant order motion hearing evidentiary ex parte parenting time custody".split())

def now_iso() -> str:
    return _dt.datetime.now().replace(microsecond=0).isoformat()

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}:{sha1(value)[:16]}"

def safe_relpath(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root)).replace('\\','/')
    except Exception:
        return p.name

def is_large(p: Path, mb: int) -> bool:
    try:
        return p.stat().st_size >= mb*1024*1024
    except Exception:
        return False

def read_bytes_limited(p: Path, max_mb: int) -> bytes:
    max_bytes = max_mb*1024*1024
    data = p.read_bytes()
    return data if len(data) <= max_bytes else data[:max_bytes]

def decode_best_effort(b: bytes) -> str:
    try:
        return b.decode("utf-8", errors="replace")
    except Exception:
        return b.decode("latin-1", errors="replace")

def setup_logger(out_dir: Path, verbose: bool) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("mindeye2_master")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setLevel(logging.DEBUG if verbose else logging.INFO); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(out_dir/"mindeye2_master.log", encoding="utf-8"); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt); logger.addHandler(fh)
    return logger

def has_pymupdf() -> bool:
    try:
        import fitz  # noqa
        return True
    except Exception:
        return False

def extract_pdf_pymupdf(p: Path, max_pages: int = 400) -> List[Tuple[int,str]]:
    import fitz  # type: ignore
    doc = fitz.open(str(p))
    pages=[]
    n=min(len(doc), max_pages)
    for i in range(n):
        try:
            t = doc.load_page(i).get_text("text") or ""
        except Exception:
            t=""
        t=t.strip()
        if t:
            pages.append((i+1,t))
    doc.close()
    return pages

def extract_pdf_pdftotext(p: Path) -> str:
    cmd = shutil.which("pdftotext")
    if not cmd:
        return ""
    try:
        proc = subprocess.run([cmd,"-layout",str(p),"-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return proc.stdout.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""

def extract_docx_xml(p: Path) -> str:
    import zipfile
    try:
        with zipfile.ZipFile(str(p),"r") as z:
            with z.open("word/document.xml") as f:
                xml=f.read().decode("utf-8", errors="replace")
        txt=re.findall(r"<w:t[^>]*>(.*?)</w:t>", xml, flags=re.DOTALL)
        txt=[html.unescape(re.sub(r"\s+"," ",t)) for t in txt]
        return "\n".join([t.strip() for t in txt if t.strip()])
    except Exception:
        return ""

def strip_html(raw: str) -> str:
    raw=re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>"," ",raw)
    raw=re.sub(r"(?is)<br\s*/?>","\n",raw)
    raw=re.sub(r"(?is)</p\s*>","\n",raw)
    text=re.sub(r"(?is)<[^>]+>"," ",raw)
    text=re.sub(r"\s+"," ",text)
    return text.strip()

def sniff_type(text: str) -> Dict[str,bool]:
    return {"is_order": bool(RE_ORDER_HINT.search(text)), "is_filing": bool(RE_FILING_HINT.search(text)), "is_police": bool(RE_POLICE_HINT.search(text))}

def extract_mentions(text: str) -> Dict[str,List[str]]:
    cases=sorted(set(m.group(1).upper() for m in RE_CASE.finditer(text)))
    mcr=sorted(set(m.group(0).upper() for m in RE_MCR.finditer(text)))
    mcl=sorted(set(m.group(0).upper() for m in RE_MCL.finditer(text)))
    mre=sorted(set(m.group(0).upper() for m in RE_MRE.finditer(text)))
    dates=sorted(set(m.group(0) for m in RE_DATE.finditer(text)))
    emails=sorted(set(m.group(0).lower() for m in RE_EMAIL.finditer(text)))
    names=set()
    for nm in SEED_PERSONS:
        if nm.lower() in text.lower(): names.add(nm)
    for m in RE_NAME.finditer(text):
        nm=m.group(1).strip()
        low=nm.lower()
        if any(w in STOPWORDS_NAME for w in low.split()): continue
        if "court" in low or "judge" in low: continue
        names.add(nm)
    names=sorted(names)
    orgs=set()
    for org in SEED_ORGS:
        if org.lower() in text.lower(): orgs.add(org)
    for m in re.finditer(r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,5}\s+(?:County|Department|Services|Clinic|Hospital|Court|School|Company|Inc\.|LLC))\b", text):
        orgs.add(m.group(1).strip())
    orgs=sorted(orgs)
    return {"cases":cases,"mcr":mcr,"mcl":mcl,"mre":mre,"dates":dates,"emails":emails,"names":names,"orgs":orgs}

def classify_track(case_no: str) -> str:
    c=case_no.upper()
    if c.endswith("-PP") or c.endswith("-PO"): return "MEEK3"
    if c.endswith("-DC") or c.endswith("-DS"): return "MEEK2"
    return "UNKNOWN"

def snippet(text: str, term: str, max_len: int=240) -> str:
    if not text: return ""
    low=text.lower(); idx=low.find(term.lower())
    if idx<0: return text[:max_len].replace("\n"," ").strip()
    start=max(0, idx-90); end=min(len(text), idx+120)
    s=text[start:end].replace("\n"," ").strip()
    return s[:max_len]

@dataclass
class Node:
    id: str
    type: str
    label: str
    props: Dict[str,Any]

@dataclass
class Edge:
    source: str
    target: str
    type: str
    props: Dict[str,Any]

class Graph:
    def __init__(self)->None:
        self.nodes: Dict[str,Node]={}
        self.edges: List[Edge]=[]
    def upsert(self, n: Node)->None:
        cur=self.nodes.get(n.id)
        if not cur:
            self.nodes[n.id]=n; return
        merged=dict(cur.props); merged.update({k:v for k,v in (n.props or {}).items() if v not in (None,"",[],{})})
        self.nodes[n.id]=Node(id=cur.id, type=cur.type, label=cur.label or n.label, props=merged)
    def add_edge(self, e: Edge)->None:
        self.edges.append(e)

def seed_graph(g: Graph)->None:
    for p in SEED_PERSONS:
        g.upsert(Node(id=stable_id("person", p.lower()), type="Person", label=p, props={"seed": True}))
    for o in SEED_ORGS:
        g.upsert(Node(id=stable_id("org", o.lower()), type="Org", label=o, props={"seed": True}))

SCHEMA_SQL = """PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, mtime INTEGER, sha1 TEXT, ext TEXT, rel TEXT);
"""

def open_index(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con=sqlite3.connect(str(db_path))
    con.executescript(SCHEMA_SQL)
    return con

def file_fingerprint(p: Path, max_mb: int) -> Tuple[int,int,str]:
    st=p.stat()
    b=read_bytes_limited(p, max_mb=max_mb)
    h=hashlib.sha1(); h.update(b)
    return st.st_size, int(st.st_mtime), h.hexdigest()

def needs_reprocess(con: sqlite3.Connection, p: Path, max_mb: int) -> bool:
    size, mtime, h = file_fingerprint(p, max_mb=max_mb)
    cur = con.execute("SELECT size,mtime,sha1 FROM files WHERE path=?", (str(p),)).fetchone()
    if not cur: return True
    return cur[0]!=size or cur[1]!=mtime or cur[2]!=h

def upsert_file_row(con: sqlite3.Connection, p: Path, rel: str, max_mb: int)->None:
    ext=p.suffix.lower()
    size, mtime, h = file_fingerprint(p, max_mb=max_mb)
    con.execute("INSERT OR REPLACE INTO files(path,size,mtime,sha1,ext,rel) VALUES (?,?,?,?,?,?)", (str(p), size, mtime, h, ext, rel))

def walk_roots(roots: List[Path], large_only: bool, large_mb: int) -> List[Tuple[Path,Path,str]]:
    out=[]
    for root in roots:
        if not root.exists(): continue
        if root.is_file():
            out.append((root.parent, root, root.name)); continue
        for p in root.rglob("*"):
            if not p.is_file(): continue
            if large_only and not is_large(p, large_mb): continue
            out.append((root,p,safe_relpath(p,root)))
    out.sort(key=lambda t: str(t[1]).lower())
    return out

def ingest_file_text(p: Path, max_mb: int, logger: logging.Logger) -> Tuple[str, List[Tuple[int,str]]]:
    ext=p.suffix.lower()
    pages=[]
    text=""
    try:
        if ext in TEXT_EXTS:
            text=decode_best_effort(read_bytes_limited(p, max_mb=max_mb))
        elif ext in DOC_EXTS:
            text=extract_docx_xml(p)
        elif ext in HTML_EXTS:
            raw=decode_best_effort(read_bytes_limited(p, max_mb=max_mb))
            text=strip_html(raw)
        elif ext in PDF_EXTS:
            if has_pymupdf():
                pages=extract_pdf_pymupdf(p)
                text="\n\n".join([t for _,t in pages])
            else:
                text=extract_pdf_pdftotext(p)
        else:
            text=decode_best_effort(read_bytes_limited(p, max_mb=max_mb))
    except Exception as e:
        logger.debug("extract failed %s (%s)", p, e)
    return (text.strip() if text else ""), pages

def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def write_csv_any(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8"); return
    fields=sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows: w.writerow(r)

def build_correlations(out_dir: Path, logger: logging.Logger, min_len: int=5) -> None:
    metas=json.loads((out_dir/"files_meta.json").read_text(encoding="utf-8")) if (out_dir/"files_meta.json").exists() else []
    token_map: Dict[str,set]={}
    def add(tok: str, path: str):
        tok=(tok or "").strip().lower()
        if len(tok) < min_len: return
        token_map.setdefault(tok,set()).add(path)
    for m in metas:
        path=m.get("path","")
        add(m.get("name",""), path)
        mentions=m.get("mentions",{}) or {}
        for k in ("emails","cases","mcr","mcl","mre","names","orgs"):
            for t in mentions.get(k,[]): add(t, path)
    rows=[{"token":tok,"hits":len(paths),"paths":sorted(paths)} for tok,paths in token_map.items() if len(paths)>=2]
    rows.sort(key=lambda r: (-r["hits"], r["token"]))
    (out_dir/"correlations.json").write_text(json.dumps({"generated_at":now_iso(),"correlations":rows[:10000]}, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("correlations.json written")

DOSSIER_HEAD = """<!doctype html><html><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>MindEye2 Living Dossier</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:0;background:#0b0f14;color:#e6edf3}
header{padding:14px 18px;background:#0f1620;border-bottom:1px solid #233;position:sticky;top:0}
h1{margin:0;font-size:18px}
.small{color:#9fb1c3;font-size:12px;margin-top:4px}
main{padding:14px 18px}
section{margin:18px 0;padding:12px;border:1px solid #233;border-radius:14px;background:#0f1620}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{border:1px solid #2b3a4a;padding:6px;vertical-align:top}
th{background:#101b27}
</style></head><body>
<header><h1>MindEye2 Living Dossier</h1><div class="small">Append-only. Each run appends a new section; prior sections remain intact.</div></header>
<main id="log"></main></body></html>"""

def ensure_dossier(out_dir: Path) -> Path:
    p=out_dir/"living_dossier.html"
    if not p.exists(): p.write_text(DOSSIER_HEAD, encoding="utf-8")
    return p

def build_record_spine(graph: dict, max_items: int=350) -> str:
    hits=[n for n in graph.get("nodes",[]) if n.get("type")=="PageHit"]
    def weight(n):
        term=((n.get("props",{}) or {}).get("term","")).lower()
        s=0
        if "mcr " in term: s+=30
        if "mcl " in term: s+=25
        if "mre " in term: s+=20
        if "2024-" in term or "2023-" in term: s+=40
        if "mcneill" in term: s+=15
        if "health" in term or "west" in term: s+=10
        return -s
    hits.sort(key=weight)
    rows=[]
    for h in hits[:max_items]:
        pr=h.get("props",{}) or {}
        rows.append((pr.get("rel") or pr.get("file",""), pr.get("page",""), pr.get("term",""), pr.get("snippet","")))
    if not rows:
        return "<p><em>No PageHit nodes were generated (PDF page anchors require PyMuPDF).</em></p>"
    out=["<table><tr><th>File (relative)</th><th>Page</th><th>Term</th><th>Snippet</th></tr>"]
    for rel,pg,term,sn in rows:
        out.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>" % (html.escape(str(rel)), html.escape(str(pg)), html.escape(str(term)), html.escape(str(sn))))
    out.append("</table>")
    return "".join(out)

def append_dossier(out_dir: Path, logger: logging.Logger, graph: dict) -> None:
    p=ensure_dossier(out_dir)
    raw=p.read_text(encoding="utf-8")
    stamp=now_iso()
    spine=build_record_spine(graph)
    block=f"<section><div class='small'>{html.escape(stamp)}</div><h3>Record Spine (PageHits)</h3>{spine}</section>\n"
    raw2=raw.replace("<main id=\"log\"></main>", "<main id=\"log\"></main>\n"+block)
    p.write_text(raw2, encoding="utf-8")
    logger.info("living_dossier.html appended")

def export_neo4j(out_dir: Path, logger: logging.Logger, graph: dict) -> None:
    nrows=[{"id:ID":n.get("id",""),"type":n.get("type",""),"label":n.get("label",""),"props_json":json.dumps(n.get("props",{}) or {}, ensure_ascii=False)} for n in graph.get("nodes",[])]
    erows=[{":START_ID":e.get("source",""),":END_ID":e.get("target",""),":TYPE":e.get("type",""),"props_json":json.dumps(e.get("props",{}) or {}, ensure_ascii=False)} for e in graph.get("edges",[])]
    def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
            for r in rows: w.writerow(r)
    write_csv(out_dir/"neo4j_nodes.csv", nrows, ["id:ID","type","label","props_json"])
    write_csv(out_dir/"neo4j_edges.csv", erows, [":START_ID",":END_ID",":TYPE","props_json"])
    cy = """// MindEye2 MASTER v10 — Neo4j MERGE (APOC recommended)
// Copy neo4j_nodes.csv and neo4j_edges.csv into Neo4j import/ directory.
// In Neo4j Browser:
// :param nodes => 'file:///neo4j_nodes.csv';
// :param edges => 'file:///neo4j_edges.csv';

CREATE CONSTRAINT mindeye2_id IF NOT EXISTS FOR (n:MindEye2) REQUIRE n.id IS UNIQUE;

LOAD CSV WITH HEADERS FROM $nodes AS row
MERGE (n:MindEye2 {id: row[`id:ID`]})
SET n.type=row.type, n.label=row.label, n.props_json=row.props_json
WITH n, row CALL apoc.create.addLabels(n, [row.type]) YIELD node RETURN count(*) as nodes;

LOAD CSV WITH HEADERS FROM $edges AS row
MATCH (a:MindEye2 {id: row[`:START_ID`]})
MATCH (b:MindEye2 {id: row[`:END_ID`]})
CALL apoc.create.relationship(a, row[`:TYPE`], {props_json: row.props_json}, b) YIELD rel
RETURN count(rel) as rels_created;
"""
    (out_dir/"neo4j_merge.cypher").write_text(cy, encoding="utf-8")
    logger.info("Neo4j export written")

STANDALONE_VIEWER_HTML = """<!doctype html><html><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width,initial-scale=1'/><title>MindEye2 Graph</title></head>
<body style='margin:0;background:#0b0f14;color:#e6edf3;font-family:system-ui'>
<div style='padding:12px 16px;border-bottom:1px solid #233;background:#0f1620'>MindEye2 Graph — loads GUI</div>
<iframe src='./gui/index.html#tab=graph' style='border:0;width:100%;height:calc(100vh - 50px)'></iframe>
</body></html>"""

def render_graph_viewer(out_dir: Path, logger: logging.Logger) -> None:
    (out_dir/"mindeye2_graph_standalone.html").write_text(STANDALONE_VIEWER_HTML, encoding="utf-8")

def ingest_build_graph(roots: List[Path], out_dir: Path, logger: logging.Logger,
                      max_mb: int, large_only: bool, large_mb: int,
                      page_anchors: bool, correlate: bool, dossier: bool, neo4j: bool,
                      incremental: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    con=open_index(out_dir/"index.sqlite")
    con.execute("BEGIN")
    g=Graph(); seed_graph(g)
    all_files=walk_roots(roots, large_only=large_only, large_mb=large_mb)
    changed=processed=pagehits=0
    metas=[]
    for i,(root,p,rel) in enumerate(all_files, start=1):
        try:
            do=True
            if incremental:
                do=needs_reprocess(con,p,max_mb=max_mb)
            upsert_file_row(con,p,rel,max_mb=max_mb)
            if not do: continue
            changed += 1
            text,pages=ingest_file_text(p,max_mb=max_mb,logger=logger)
            processed += 1
            st=p.stat()
            fid=stable_id("file", str(p).lower())
            meta={"path":str(p),"rel":rel,"name":p.name,"ext":p.suffix.lower(),"size":st.st_size,
                  "mtime": _dt.datetime.fromtimestamp(st.st_mtime).isoformat(), "parse_ok": bool(text), "pages": len(pages)}
            if text:
                mentions=extract_mentions(text); meta["mentions"]=mentions; meta["sniff"]=sniff_type(text)
                g.upsert(Node(id=fid,type="File",label=p.name,props={"path":str(p),"rel":rel,"ext":p.suffix.lower(),"size":st.st_size,"mtime":meta["mtime"]}))
                for c in mentions["cases"]:
                    cid=stable_id("case", c.upper())
                    g.upsert(Node(id=cid,type="Case",label=c.upper(),props={"case_no":c.upper(),"track":classify_track(c)}))
                    g.add_edge(Edge(source=fid,target=cid,type="MENTIONS_CASE",props={}))
                for a in mentions["mcr"]:
                    aid=stable_id("auth", a.upper()); g.upsert(Node(id=aid,type="Authority",label=a.upper(),props={"kind":"MCR"}))
                    g.add_edge(Edge(source=fid,target=aid,type="CITES",props={"kind":"MCR"}))
                for a in mentions["mcl"]:
                    aid=stable_id("auth", a.upper()); g.upsert(Node(id=aid,type="Authority",label=a.upper(),props={"kind":"MCL"}))
                    g.add_edge(Edge(source=fid,target=aid,type="CITES",props={"kind":"MCL"}))
                for a in mentions["mre"]:
                    aid=stable_id("auth", a.upper()); g.upsert(Node(id=aid,type="Authority",label=a.upper(),props={"kind":"MRE"}))
                    g.add_edge(Edge(source=fid,target=aid,type="CITES",props={"kind":"MRE"}))
                for nm in mentions["names"]:
                    pid=stable_id("person", nm.lower()); g.upsert(Node(id=pid,type="Person",label=nm,props={"seed": nm in SEED_PERSONS}))
                    g.add_edge(Edge(source=fid,target=pid,type="MENTIONS_PERSON",props={}))
                for org in mentions["orgs"]:
                    oid=stable_id("org", org.lower()); g.upsert(Node(id=oid,type="Org",label=org,props={"seed": org in SEED_ORGS}))
                    g.add_edge(Edge(source=fid,target=oid,type="MENTIONS_ORG",props={}))
                for em in mentions["emails"]:
                    eid=stable_id("email", em.lower()); g.upsert(Node(id=eid,type="EmailAddress",label=em.lower(),props={}))
                    g.add_edge(Edge(source=fid,target=eid,type="MENTIONS_EMAIL",props={}))
                sniff=meta.get("sniff",{})
                if sniff.get("is_order"):
                    oid=stable_id("order", str(p).lower()); g.upsert(Node(id=oid,type="Order",label=f"Order: {p.name}",props={"file":str(p)}))
                    g.add_edge(Edge(source=fid,target=oid,type="IS_ORDER_DOC",props={}))
                if sniff.get("is_filing"):
                    flid=stable_id("filing", str(p).lower()); g.upsert(Node(id=flid,type="Filing",label=f"Filing: {p.name}",props={"file":str(p)}))
                    g.add_edge(Edge(source=fid,target=flid,type="IS_FILING_DOC",props={}))
                if sniff.get("is_police"):
                    prid=stable_id("police", str(p).lower()); g.upsert(Node(id=prid,type="PoliceReport",label=f"Police: {p.name}",props={"file":str(p)}))
                    g.add_edge(Edge(source=fid,target=prid,type="IS_POLICE_REPORT",props={}))
                if page_anchors and pages:
                    anchor_terms = (mentions["cases"] + mentions["mcr"] + mentions["mcl"] + mentions["mre"] + SEED_PERSONS + SEED_ORGS)[:120]
                    seen=set()
                    anchors=[]
                    for t in anchor_terms:
                        t=t.strip()
                        if not t: continue
                        k=t.lower()
                        if k in seen: continue
                        seen.add(k); anchors.append(t)
                    for pgno,pgtext in pages:
                        lowpg=pgtext.lower()
                        for term in anchors:
                            if term.lower() in lowpg:
                                hid=stable_id("hit", f"{p}:{pgno}:{term}".lower())
                                g.upsert(Node(id=hid,type="PageHit",label=f"{p.name} p.{pgno}",props={"file":str(p),"rel":rel,"page":pgno,"term":term,"snippet":snippet(pgtext,term)}))
                                g.add_edge(Edge(source=fid,target=hid,type="HAS_PAGE_HIT",props={"page":pgno,"term":term}))
                                pagehits += 1
            else:
                meta["mentions"]={}; meta["sniff"]={}
            metas.append(meta)
        except Exception as e:
            logger.warning("Ingest failed: %s (%s)", p, e)
        if i % 200 == 0:
            logger.info("Scan progress: %d/%d (changed=%d processed=%d)", i, len(all_files), changed, processed)
    con.execute("COMMIT")
    nodes=list(g.nodes.values()); edges=g.edges
    graph_json={"app":APP_NAME,"version":APP_VERSION,"generated_at":now_iso(),"roots":[str(r) for r in roots],
               "counts":{"nodes":len(nodes),"edges":len(edges),"files_seen":len(all_files),"files_changed":changed,"files_processed":processed,"pagehits":pagehits},
               "nodes":[asdict(n) for n in nodes],"edges":[asdict(e) for e in edges]}
    (out_dir/"graph.json").write_text(json.dumps(graph_json, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(out_dir/"nodes.jsonl", [asdict(n) for n in nodes])
    write_jsonl(out_dir/"edges.jsonl", [asdict(e) for e in edges])
    write_csv_any(out_dir/"nodes.csv", [{"id":n.id,"type":n.type,"label":n.label,"props_json":json.dumps(n.props, ensure_ascii=False)} for n in nodes])
    write_csv_any(out_dir/"edges.csv", [{"source":e.source,"target":e.target,"type":e.type,"props_json":json.dumps(e.props, ensure_ascii=False)} for e in edges])
    (out_dir/"files_meta.json").write_text(json.dumps(metas, ensure_ascii=False, indent=2), encoding="utf-8")
    if correlate: build_correlations(out_dir, logger)
    if dossier: append_dossier(out_dir, logger, graph_json)
    if neo4j: export_neo4j(out_dir, logger, graph_json)
    render_graph_viewer(out_dir, logger)
    summary={"app":APP_NAME,"version":APP_VERSION,"generated_at":now_iso(),"roots":[str(r) for r in roots],"out_dir":str(out_dir),
             "counts":graph_json["counts"],"incremental":bool(incremental),
             "pdf_engine":"pymupdf" if has_pymupdf() else ("pdftotext" if shutil.which("pdftotext") else "none")}
    (out_dir/"run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary

# ----- Server / API -----
def app_static_dir() -> Path:
    return Path(__file__).resolve().parent / "app"

class AppState:
    def __init__(self):
        self.out_dir: Path = Path(".").resolve()
        self.roots: List[Path] = []
        self.max_mb: int = DEFAULT_MAX_MB
        self.large_only: bool = False
        self.large_mb: int = DEFAULT_LARGE_MB
        self.page_anchors: bool = True
        self.correlate: bool = True
        self.dossier: bool = True
        self.neo4j: bool = True
        self.incremental: bool = True
        self.watch: int = 0
        self.last_summary: Optional[dict] = None
        self.lock = threading.Lock()

STATE=AppState()

def api_json(handler, obj, status=200):
    payload=json.dumps(obj, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type","application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(payload)))
    handler.end_headers()
    handler.wfile.write(payload)

def api_text(handler, txt: str, status=200, ctype="text/plain; charset=utf-8"):
    b=txt.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(b)))
    handler.end_headers()
    handler.wfile.write(b)

def load_graph(out_dir: Path) -> dict:
    gp=out_dir/"graph.json"
    return json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else {}

def run_ingest(logger: logging.Logger) -> dict:
    with STATE.lock:
        roots=list(STATE.roots); out_dir=STATE.out_dir
        max_mb=STATE.max_mb; large_only=STATE.large_only; large_mb=STATE.large_mb
        page_anchors=STATE.page_anchors; correlate=STATE.correlate; dossier=STATE.dossier; neo4j=STATE.neo4j; incremental=STATE.incremental
    summary=ingest_build_graph(roots,out_dir,logger,max_mb,large_only,large_mb,page_anchors,correlate,dossier,neo4j,incremental)
    with STATE.lock:
        STATE.last_summary=summary
    return summary

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        parsed=urllib.parse.urlparse(path)
        pth=parsed.path
        if pth.startswith("/api/"): return ""
        if pth.startswith("/out/"):
            rel=pth[len("/out/"):]
            return str((STATE.out_dir/rel).resolve())
        base=app_static_dir()
        rel=pth.lstrip("/") or "index.html"
        return str((base/rel).resolve())

    def do_GET(self):
        parsed=urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/"):
            return self.handle_api_get(parsed)
        return super().do_GET()

    def do_POST(self):
        parsed=urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/"):
            return self.handle_api_post(parsed)
        self.send_error(405,"POST not supported")

    def handle_api_get(self, parsed):
        if parsed.path=="/api/status":
            with STATE.lock:
                st={"app":APP_NAME,"version":APP_VERSION,"out_dir":str(STATE.out_dir),"roots":[str(r) for r in STATE.roots],
                    "max_mb":STATE.max_mb,"large_only":STATE.large_only,"large_mb":STATE.large_mb,"page_anchors":STATE.page_anchors,
                    "correlate":STATE.correlate,"dossier":STATE.dossier,"neo4j":STATE.neo4j,"incremental":STATE.incremental,
                    "watch":STATE.watch,"pdf_engine":"pymupdf" if has_pymupdf() else ("pdftotext" if shutil.which("pdftotext") else "none"),
                    "last_summary":STATE.last_summary}
            return api_json(self, st)
        if parsed.path=="/api/graph":
            return api_json(self, load_graph(STATE.out_dir))
        if parsed.path=="/api/files":
            fp=STATE.out_dir/"files_meta.json"
            return api_json(self, json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else [])
        if parsed.path=="/api/correlations":
            cp=STATE.out_dir/"correlations.json"
            return api_json(self, json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {"generated_at":None,"correlations":[]})
        if parsed.path=="/api/dossier":
            dp=STATE.out_dir/"living_dossier.html"
            return api_text(self, dp.read_text(encoding="utf-8") if dp.exists() else "<!doctype html><html><body>No dossier yet.</body></html>", ctype="text/html; charset=utf-8")
        self.send_error(404,"Unknown API endpoint")

    def handle_api_post(self, parsed):
        length=int(self.headers.get("Content-Length","0") or "0")
        raw=self.rfile.read(length) if length>0 else b"{}"
        try: data=json.loads(raw.decode("utf-8", errors="replace"))
        except Exception: data={}
        if parsed.path=="/api/config":
            with STATE.lock:
                if "out_dir" in data: STATE.out_dir=Path(str(data["out_dir"])).expanduser()
                if "roots" in data and isinstance(data["roots"], list): STATE.roots=[Path(str(x)).expanduser() for x in data["roots"]]
                for k in ("max_mb","large_mb","watch"):
                    if k in data:
                        try: setattr(STATE, k, int(data[k]))
                        except Exception: pass
                for k in ("large_only","page_anchors","correlate","dossier","neo4j","incremental"):
                    if k in data: setattr(STATE, k, bool(data[k]))
            return api_json(self, {"ok":True})
        if parsed.path=="/api/run":
            logger=logging.getLogger("mindeye2_master")
            summary=run_ingest(logger)
            return api_json(self, {"ok":True,"summary":summary})
        self.send_error(404,"Unknown API endpoint")

def serve(port: int, logger: logging.Logger):
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), RequestHandler) as httpd:
        logger.info("Serving GUI at http://127.0.0.1:%d/  (output under /out/)", port)
        try: httpd.serve_forever()
        except KeyboardInterrupt: logger.info("Stopped.")

def watch_loop(logger: logging.Logger, interval: int):
    if interval<=0: return
    last_fp=""
    while True:
        time.sleep(interval)
        fp=compute_roots_fingerprint(STATE.roots)
        if fp and fp!=last_fp:
            last_fp=fp
            logger.info("Change detected. Rebuilding…")
            try: run_ingest(logger)
            except Exception as e: logger.warning("Watch rebuild failed: %s", e)

def compute_roots_fingerprint(roots: List[Path]) -> str:
    parts=[]
    for r in roots:
        if not r.exists(): continue
        if r.is_file():
            st=r.stat(); parts.append(f"{r}|{st.st_size}|{int(st.st_mtime)}"); continue
        for p in r.rglob("*"):
            if not p.is_file(): continue
            try:
                st=p.stat(); parts.append(f"{p}|{st.st_size}|{int(st.st_mtime)}")
            except Exception:
                continue
    parts.sort()
    return sha1("\n".join(parts))

def cmd_run(args) -> int:
    out_dir=Path(args.out).expanduser()
    logger=setup_logger(out_dir, args.verbose)
    with STATE.lock:
        STATE.out_dir=out_dir
        STATE.roots=[Path(r).expanduser() for r in args.root]
        STATE.max_mb=args.max_mb
        STATE.large_only=args.large_only
        STATE.large_mb=args.large_mb
        STATE.page_anchors=args.page_anchors
        STATE.correlate=args.correlate
        STATE.dossier=args.dossier
        STATE.neo4j=args.neo4j
        STATE.incremental=not args.no_incremental
        STATE.watch=0
    run_ingest(logger)
    logger.info("Done. To serve: python mindeye2_master.py serve --out \"%s\" --port %d", out_dir, args.port)
    return 0

def cmd_serve(args) -> int:
    out_dir=Path(args.out).expanduser()
    logger=setup_logger(out_dir, args.verbose)
    with STATE.lock:
        STATE.out_dir=out_dir
        STATE.roots=[]
        STATE.watch=0
    serve(args.port, logger)
    return 0

def cmd_gui(args) -> int:
    out_dir=Path(args.out).expanduser()
    logger=setup_logger(out_dir, args.verbose)
    with STATE.lock:
        STATE.out_dir=out_dir
        STATE.roots=[Path(r).expanduser() for r in args.root]
        STATE.max_mb=args.max_mb
        STATE.large_only=args.large_only
        STATE.large_mb=args.large_mb
        STATE.page_anchors=args.page_anchors
        STATE.correlate=args.correlate
        STATE.dossier=args.dossier
        STATE.neo4j=args.neo4j
        STATE.incremental=not args.no_incremental
        STATE.watch=args.watch
    if args.autorun:
        try: run_ingest(logger)
        except Exception as e: logger.warning("Initial ingest failed: %s", e)
    if args.watch and args.watch>0:
        threading.Thread(target=watch_loop, args=(logger, args.watch), daemon=True).start()
    url=f"http://127.0.0.1:{args.port}/"
    if platform.system().lower().startswith("win"):
        try: webbrowser.open(url)
        except Exception: pass
    serve(args.port, logger)
    return 0

def cmd_status(args) -> int:
    out_dir=Path(args.out).expanduser()
    db=out_dir/"index.sqlite"
    if not db.exists():
        print("No index yet at:", db); return 2
    con=open_index(db)
    n=con.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    print(json.dumps({"out_dir":str(out_dir),"indexed_files":n}, indent=2))
    return 0

def build_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(prog="mindeye2_master.py", description=f"{APP_NAME} {APP_VERSION}")
    p.add_argument("--version", action="version", version=f"{APP_NAME} {APP_VERSION}")
    sub=p.add_subparsers(dest="cmd", required=True)
    def add_common(sp):
        sp.add_argument("--root", action="append", default=[], help="Input root path (repeatable).")
        sp.add_argument("--out", required=True, help="Output folder.")
        sp.add_argument("--max-mb", type=int, default=DEFAULT_MAX_MB, help="Max MB to read per file.")
        sp.add_argument("--large-only", action="store_true", help="Only ingest files >= --large-mb.")
        sp.add_argument("--large-mb", type=int, default=DEFAULT_LARGE_MB, help="Large threshold MB.")
        sp.add_argument("--page-anchors", action="store_true", help="Create PDF PageHit anchors (best with PyMuPDF).")
        sp.add_argument("--correlate", action="store_true", help="Create correlations.json.")
        sp.add_argument("--dossier", action="store_true", help="Append living_dossier.html per run.")
        sp.add_argument("--neo4j", action="store_true", help="Export Neo4j CSV+Cypher.")
        sp.add_argument("--no-incremental", action="store_true", help="Disable incremental reprocess; re-read all files.")
        sp.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for GUI server.")
        sp.add_argument("--verbose", action="store_true", help="Verbose logs.")
    sp=sub.add_parser("run", help="One-shot ingest + exports."); add_common(sp); sp.set_defaults(func=cmd_run)
    sp=sub.add_parser("serve", help="Serve an existing output folder (no ingest).")
    sp.add_argument("--out", required=True); sp.add_argument("--port", type=int, default=DEFAULT_PORT); sp.add_argument("--verbose", action="store_true"); sp.set_defaults(func=cmd_serve)
    sp=sub.add_parser("gui", help="GUI mode: server + config + optional auto-run + watch."); add_common(sp)
    sp.add_argument("--autorun", action="store_true"); sp.add_argument("--watch", type=int, default=DEFAULT_WATCH); sp.set_defaults(func=cmd_gui)
    sp=sub.add_parser("status", help="Print index statistics."); sp.add_argument("--out", required=True); sp.set_defaults(func=cmd_status)
    return p

def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args=build_parser().parse_args(argv)
    if args.cmd in ("run","gui") and not args.root:
        print("[ERROR] Provide at least one --root path."); return 2
    return args.func(args)

if __name__=="__main__":
    raise SystemExit(main())


# ============================
# v11: Graph Enhancement Lane
# ============================

def _lp_communities(nodes, edges, max_iter=25):
    """Label propagation communities (no external deps)."""
    neigh = {n.get("id"): [] for n in nodes if n.get("id") is not None}
    for e in edges:
        a=e.get("source"); b=e.get("target")
        if a in neigh and b in neigh:
            neigh[a].append(b); neigh[b].append(a)
    labels = {nid: nid for nid in neigh.keys()}
    ids = list(labels.keys())
    for _ in range(max_iter):
        random.shuffle(ids)
        changed = 0
        for nid in ids:
            nbs = neigh.get(nid, [])
            if not nbs:
                continue
            counts={}
            for nb in nbs:
                lab = labels.get(nb, nb)
                counts[lab] = counts.get(lab, 0) + 1
            best = max(counts.items(), key=lambda kv: (kv[1], str(kv[0])))
            if labels[nid] != best[0]:
                labels[nid] = best[0]
                changed += 1
        if changed == 0:
            break
    uniq = {lab:i for i,lab in enumerate(sorted(set(labels.values()), key=str))}
    return {nid: uniq[lab] for nid,lab in labels.items()}

def _fr_layout(nodes, edges, seed=1337, iters=200, width=2200.0, height=1400.0):
    """Fruchterman–Reingold layout (deterministic)."""
    rnd = random.Random(seed)
    ids = [n.get("id") for n in nodes if n.get("id") is not None]
    n = len(ids)
    if n == 0:
        return {}
    pos = {nid: [rnd.random()*width, rnd.random()*height] for nid in ids}
    disp = {nid: [0.0, 0.0] for nid in ids}
    area = width * height
    k = math.sqrt(area / max(1, n))
    def _dist(a,b):
        dx=a[0]-b[0]; dy=a[1]-b[1]
        d=math.sqrt(dx*dx+dy*dy)+1e-9
        return dx,dy,d
    adj = [(e.get("source"), e.get("target")) for e in edges if e.get("source") in pos and e.get("target") in pos]
    t = width/10.0
    cool = t / max(1, iters)
    for _ in range(iters):
        for nid in ids:
            disp[nid] = [0.0,0.0]
        # repulsive
        for i in range(n):
            v=ids[i]
            for j in range(i+1, n):
                u=ids[j]
                dx,dy,d=_dist(pos[v],pos[u])
                f=(k*k)/d
                disp[v][0] += (dx/d)*f; disp[v][1] += (dy/d)*f
                disp[u][0] -= (dx/d)*f; disp[u][1] -= (dy/d)*f
        # attractive
        for v,u in adj:
            dx,dy,d=_dist(pos[v],pos[u])
            f=(d*d)/k
            disp[v][0] -= (dx/d)*f; disp[v][1] -= (dy/d)*f
            disp[u][0] += (dx/d)*f; disp[u][1] += (dy/d)*f
        # move
        for v in ids:
            dx,dy=disp[v]
            d=math.sqrt(dx*dx+dy*dy)+1e-9
            pos[v][0] += (dx/d)*min(d,t)
            pos[v][1] += (dy/d)*min(d,t)
            pos[v][0] = min(width, max(0.0, pos[v][0]))
            pos[v][1] = min(height, max(0.0, pos[v][1]))
        t -= cool
        if t <= 0:
            break
    return {nid: {"x": pos[nid][0], "y": pos[nid][1]} for nid in ids}

def enhance_graph(graph_obj: dict, out_dir: Path, seed: int = 1337) -> dict:
    """Compute clusters + deterministic layout and write graph_enhanced.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    nodes = graph_obj.get("nodes", []) or []
    edges = graph_obj.get("edges", []) or []
    deg = {n.get("id"): 0 for n in nodes if n.get("id") is not None}
    for e in edges:
        a=e.get("source"); b=e.get("target")
        if a in deg: deg[a] += 1
        if b in deg: deg[b] += 1
    comm = _lp_communities(nodes, edges, max_iter=25)
    layout = _fr_layout(nodes, edges, seed=seed, iters=200)
    for n in nodes:
        nid=n.get("id")
        if nid is None:
            continue
        n["degree"] = int(deg.get(nid, 0))
        n["community"] = int(comm.get(nid, 0))
        xy = layout.get(nid)
        if xy:
            n["x"] = float(xy["x"]); n["y"] = float(xy["y"])
    enhanced = {"meta": {**(graph_obj.get("meta", {}) or {}), "enhanced": True, "seed": seed}, "nodes": nodes, "edges": edges}
    (out_dir / "graph_enhanced.json").write_text(json.dumps(enhanced, ensure_ascii=False, indent=2), encoding="utf-8")
    # CSVs
    import csv
    node_fields = sorted({k for n in nodes for k in n.keys()})
    edge_fields = sorted({k for e in edges for k in e.keys()})
    with (out_dir / "nodes_enhanced.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=node_fields); w.writeheader()
        for n in nodes: w.writerow(n)
    with (out_dir / "edges_enhanced.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=edge_fields); w.writeheader()
        for e in edges: w.writerow(e)
    return enhanced
