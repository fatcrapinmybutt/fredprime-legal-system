#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LITIGATIONOS_HF_TRANSFORMERS_DOC_HARVESTER_BLOOM_MAX_v2026_01_27.py

Executive-grade, append-only harvester for Hugging Face Transformers docs.

Design goals
- Deterministic IDs (stable hashes) for append-only merging.
- Robust extraction: pages, sections, code blocks, API entities, parameters, issues.
- Dual emission: JSONL/CSV binder + Neo4j import CSV tables + run ledger + provenance + manifest.
- Optional discovery crawl (BFS) restricted by allow/deny regex.

Dependencies (free):
- requests
- beautifulsoup4 (optional but strongly recommended; fallback parser exists but is weaker)

Install:
  python -m pip install requests beautifulsoup4

Example:
  python LITIGATIONOS_HF_TRANSFORMERS_DOC_HARVESTER_BLOOM_MAX_v2026_01_27.py ^
    --out "F:\LitigationOS\HF_DOC_OUT" ^
    --seed-file "SEEDS\hf_transformers_seed_urls.txt" ^
    --discover --max-depth 2 --max-urls 400 --skip-unchanged

Outputs (under --out):
  RAW/                     raw HTML (doc_id.html)
  EXTRACT/                 per-doc extract JSON (doc_id.json)
  BINDER/                  JSONL + CSV (pages/sections/code/entities/mentions/params/issues)
  NEO4J_IMPORT/            Neo4j import CSV tables
  RUN/                     run ledger (jsonl), provenance_index.json, manifest.json

Truth-lock note
- This script does not assert facts about the content; it extracts and indexes what it reads.
- All "issue" signals are heuristics and include evidence pointers back to the source URL and section.

License
- You are responsible for respecting the terms of use of sites you crawl.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

try:
    import requests
except Exception as e:
    raise SystemExit("Missing dependency: requests. Install with: python -m pip install requests") from e

try:
    from bs4 import BeautifulSoup  # type: ignore
    _HAS_BS4 = True
except Exception:
    BeautifulSoup = None  # type: ignore
    _HAS_BS4 = False


# ----------------------------
# Utilities
# ----------------------------

def iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_json(path: Path, obj: Any) -> None:
    safe_mkdir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    safe_mkdir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    safe_mkdir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def normalize_url(u: str) -> str:
    u = u.strip()
    if not u:
        return u
    parsed = urlparse(u)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc
    path = parsed.path
    if not netloc and path.startswith("//"):
        parsed2 = urlparse(scheme + ":" + u)
        netloc = parsed2.netloc
        path = parsed2.path
    # drop fragments/query for canonicality; we store fragments separately as evidence pointers
    cleaned = urlunparse((scheme, netloc, path.rstrip("/"), "", "", ""))
    return cleaned

def to_abs_href(base_url: str, href: str) -> str:
    try:
        return normalize_url(urljoin(base_url, href))
    except Exception:
        return ""

def clean_text(s: str) -> str:
    # Normalize whitespace without destroying meaning
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def compile_patterns(patterns: List[str]) -> List[re.Pattern]:
    out: List[re.Pattern] = []
    for p in patterns:
        try:
            out.append(re.compile(p))
        except re.error:
            # skip broken patterns; do not crash the run
            continue
    return out

def allow_url(url: str, allow_pats: List[re.Pattern], deny_pats: List[re.Pattern]) -> bool:
    if not url:
        return False
    for dp in deny_pats:
        if dp.search(url):
            return False
    for ap in allow_pats:
        if ap.search(url):
            return True
    return False


# ----------------------------
# DocType registry
# ----------------------------

def infer_doctype(url: str) -> str:
    u = normalize_url(url)
    if "/docs/transformers/model_doc/" in u:
        return "HF_TRANSFORMERS_MODEL_DOC"
    if "/docs/transformers/quantization/torchao" in u:
        return "HF_TRANSFORMERS_QUANT_TORCHAO"
    if "/docs/transformers/kv_cache" in u:
        return "HF_TRANSFORMERS_KV_CACHE"
    if "/docs/transformers/main_classes/" in u:
        return "HF_TRANSFORMERS_MAIN_CLASSES"
    if "/docs/transformers/en/" in u or re.search(r"/docs/transformers/[a-z]{2}/en/", u):
        return "HF_TRANSFORMERS_GUIDE_DOC"
    if "/docs/transformers/" in u:
        return "HF_TRANSFORMERS_DOC_OTHER"
    return "UNKNOWN"

def infer_product(url: str) -> str:
    u = normalize_url(url)
    if "/docs/transformers" in u:
        return "transformers"
    return "unknown"


# ----------------------------
# HTTP fetch
# ----------------------------

@dataclass
class FetchResult:
    url: str
    status: str
    http_status: int
    etag: str
    last_modified: str
    bytes_len: int
    sha256: str
    html_text: str
    error: str

def fetch_html(session: requests.Session, url: str, timeout_s: int, throttle_ms: int,
               cache_dir: Path, skip_unchanged: bool) -> FetchResult:
    """
    Fetch HTML with basic conditional caching:
      - stores response headers and body under cache_dir by doc_id (sha1(url))
      - sends If-None-Match and If-Modified-Since if skip_unchanged=True
    """
    u = normalize_url(url)
    doc_id = sha1(u)
    cache_meta = cache_dir / f"{doc_id}.headers.json"
    cache_body = cache_dir / f"{doc_id}.body.html"

    headers: Dict[str, str] = {
        "User-Agent": "LitigationOS-HFDocHarvester/1.0 (+local indexer)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    if skip_unchanged and cache_meta.exists():
        try:
            m = json.loads(cache_meta.read_text(encoding="utf-8"))
            etag = m.get("etag", "")
            lm = m.get("last_modified", "")
            if etag:
                headers["If-None-Match"] = etag
            if lm:
                headers["If-Modified-Since"] = lm
        except Exception:
            pass

    # throttle
    if throttle_ms > 0:
        time.sleep(throttle_ms / 1000.0)

    try:
        resp = session.get(u, headers=headers, timeout=timeout_s)
    except Exception as e:
        return FetchResult(url=u, status="FAIL", http_status=0, etag="", last_modified="",
                           bytes_len=0, sha256="", html_text="", error=str(e))

    http_status = int(getattr(resp, "status_code", 0) or 0)

    if http_status == 304 and skip_unchanged and cache_body.exists():
        # Use cached body
        try:
            body = cache_body.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            body = ""
        meta = {}
        try:
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        return FetchResult(url=u, status="PASS", http_status=304,
                           etag=str(meta.get("etag", "")),
                           last_modified=str(meta.get("last_modified", "")),
                           bytes_len=len(body.encode("utf-8", errors="ignore")),
                           sha256=sha256_bytes(body.encode("utf-8", errors="ignore")) if body else "",
                           html_text=body, error="")

    # Fresh body
    try:
        body_bytes = resp.content or b""
    except Exception:
        body_bytes = b""
    body = body_bytes.decode("utf-8", errors="ignore")
    etag = resp.headers.get("ETag", "")
    last_modified = resp.headers.get("Last-Modified", "")

    # persist cache
    safe_mkdir(cache_dir)
    cache_body.write_text(body, encoding="utf-8", errors="ignore")
    cache_meta.write_text(json.dumps({"etag": etag, "last_modified": last_modified}, indent=2),
                          encoding="utf-8")

    return FetchResult(url=u, status="PASS" if 200 <= http_status < 400 else "FAIL",
                       http_status=http_status, etag=etag, last_modified=last_modified,
                       bytes_len=len(body_bytes), sha256=sha256_bytes(body_bytes),
                       html_text=body, error="" if 200 <= http_status < 400 else f"HTTP {http_status}")


# ----------------------------
# Parsing (BS4)
# ----------------------------

def bs4_main_soup(html_text: str) -> Any:
    if not _HAS_BS4:
        return None
    soup = BeautifulSoup(html_text, "html.parser")  # type: ignore
    # HF docs usually have a <main> tag. fallback to body.
    main = soup.find("main")
    if main is None:
        main = soup.find("body")
    return main if main is not None else soup

def extract_title(main: Any, html_text: str) -> str:
    # Prefer <h1>, then <title>
    if _HAS_BS4 and main is not None:
        h1 = main.find(["h1"])
        if h1 is not None:
            t = clean_text(h1.get_text(" ", strip=True))
            if t:
                return t
    # fallback
    m = re.search(r"<title>(.*?)</title>", html_text, flags=re.I | re.S)
    if m:
        return clean_text(re.sub(r"<.*?>", " ", m.group(1)))
    return ""

def extract_lang_from_url(url: str) -> str:
    # Example: /docs/transformers/en/... or /docs/transformers/fr/...
    parts = urlparse(url).path.strip("/").split("/")
    if len(parts) >= 3 and parts[0] == "docs" and parts[1] == "transformers":
        if parts[2] in ("en", "fr", "de", "es", "it", "ja", "ko", "zh", "ru", "pt"):
            return parts[2]
        if len(parts) >= 4 and re.fullmatch(r"[a-z]{2}", parts[2]) and parts[3] == "en":
            return parts[2] + "_en"
    return ""

@dataclass
class SectionRec:
    section_id: str
    doc_id: str
    path: str
    level: int
    heading: str
    heading_id: str
    ordinal: int
    text: str
    text_sha1: str
    has_parameters_heading: bool

@dataclass
class CodeRec:
    code_id: str
    doc_id: str
    section_id: str
    ordinal: int
    language: str
    code: str
    code_sha1: str

@dataclass
class EntityRec:
    entity_id: str
    doc_id: str
    name: str
    fqname: str
    kind: str
    signature: str
    doc_anchor: str
    source_url: str
    confidence: float
    evidence_ptr: str

@dataclass
class ParamRec:
    param_id: str
    entity_id: str
    name: str
    type: str
    default: str
    description: str

@dataclass
class MentionRec:
    section_id: str
    entity_id: str
    count: int
    evidence_ptr: str

@dataclass
class IssueRec:
    issue_id: str
    doc_id: str
    severity: str
    code: str
    message: str
    evidence_ptr: str


def section_stack_update(stack: List[Tuple[int, str]], level: int, heading: str) -> None:
    # pop while stack level >= new level
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, heading))


def parse_sections_and_code(main: Any, url: str) -> Tuple[List[SectionRec], List[CodeRec], Dict[str, int]]:
    """
    Extract ordered sections and code blocks.
    - Builds a hierarchical section path using heading stack.
    - Assigns code blocks to the current section.
    Returns sections, codeblocks, and stats.
    """
    doc_id = sha1(normalize_url(url))
    sections: List[SectionRec] = []
    codeblocks: List[CodeRec] = []
    stats = {"github_source_links": 0, "deprecation_mentions": 0, "torchao_string_quant_pattern_hits": 0}

    if not _HAS_BS4 or main is None:
        return sections, codeblocks, stats

    # Count GitHub source links if present
    for a in main.find_all("a"):
        href = a.get("href", "") or ""
        if "github.com" in href:
            stats["github_source_links"] += 1

    # Traverse in document order. We consider headings and pre blocks and text blocks.
    heading_stack: List[Tuple[int, str]] = []
    current_section_id = ""
    current_section_path = ""
    current_heading = ""
    current_heading_id = ""
    current_level = 0
    current_text_parts: List[str] = []
    ordinal = 0

    def flush_section() -> None:
        nonlocal current_section_id, current_section_path, current_heading, current_heading_id, current_level, current_text_parts, ordinal
        if not current_heading and not current_text_parts:
            return
        path = current_section_path or current_heading or "ROOT"
        sid = sha1(doc_id + "|" + path)
        text = clean_text("\n".join([t for t in current_text_parts if t]))
        has_params = bool(re.search(r"\bParameters\b", current_heading, flags=re.I)) or bool(re.search(r"\bParameters\b", text, flags=re.I))
        sections.append(SectionRec(
            section_id=sid,
            doc_id=doc_id,
            path=path,
            level=current_level or 1,
            heading=current_heading or "ROOT",
            heading_id=current_heading_id,
            ordinal=len(sections),
            text=text,
            text_sha1=sha1(text) if text else "",
            has_parameters_heading=has_params
        ))
        # reset for next section accumulation
        current_text_parts = []

    # Find candidate elements in main content. We include headings, paragraphs, lists, pre/code, tables, blockquotes.
    candidates = main.find_all(["h1","h2","h3","h4","h5","h6","p","li","pre","table","blockquote","div"])
    # This is intentionally broad; we filter inside.
    for el in candidates:
        tag = el.name.lower() if getattr(el, "name", None) else ""

        # Admonitions/callouts often are divs with specific classes; detect deprecation text.
        if tag == "div":
            cls = " ".join(el.get("class", []) or [])
            txt = clean_text(el.get_text(" ", strip=True))
            if re.search(r"\bdeprecat", txt, flags=re.I):
                stats["deprecation_mentions"] += 1
            # skip divs that are pure containers with no text
            continue

        if tag in ("h1","h2","h3","h4","h5","h6"):
            # flush previous section when new heading appears
            if current_heading or current_text_parts:
                flush_section()

            level = int(tag[1])
            heading = clean_text(el.get_text(" ", strip=True))
            hid = el.get("id", "") or ""
            if not heading:
                continue
            section_stack_update(heading_stack, level, heading)
            current_section_path = "/".join([h for _, h in heading_stack])
            current_heading = heading
            current_heading_id = hid
            current_level = level
            current_section_id = sha1(doc_id + "|" + (current_section_path or heading))
            continue

        if tag == "pre":
            code_text = el.get_text("\n", strip=False)
            code_text = code_text.replace("\r\n", "\n")
            code_text = code_text.strip("\n")
            if not code_text.strip():
                continue

            lang = ""
            # attempt language from nested code tag class
            try:
                code_tag = el.find("code")
                if code_tag is not None:
                    cls = " ".join(code_tag.get("class", []) or [])
                    m = re.search(r"language-([a-zA-Z0-9_+-]+)", cls)
                    if m:
                        lang = m.group(1)
            except Exception:
                lang = ""

            sec_id = current_section_id or sha1(doc_id + "|ROOT")
            code_id = sha1(doc_id + "|" + sec_id + "|" + str(len([c for c in codeblocks if c.section_id == sec_id])))
            codeblocks.append(CodeRec(
                code_id=code_id,
                doc_id=doc_id,
                section_id=sec_id,
                ordinal=len(codeblocks),
                language=lang,
                code=code_text,
                code_sha1=sha1(code_text)
            ))

            # TorchAO stale-pattern heuristic
            if "TorchAoConfig(\"int4_weight_only\"" in code_text or "TorchAoConfig('int4_weight_only'" in code_text:
                stats["torchao_string_quant_pattern_hits"] += 1

            # add code to section text too (for mention detection)
            current_text_parts.append(code_text)
            continue

        # For other text-bearing tags
        if tag in ("p","li","blockquote","table"):
            txt = clean_text(el.get_text(" ", strip=True))
            if not txt:
                continue
            current_text_parts.append(txt)
            if re.search(r"\bdeprecat", txt, flags=re.I):
                stats["deprecation_mentions"] += 1
            continue

    # flush last section
    flush_section()
    return sections, codeblocks, stats


def extract_entities_and_params(sections: List[SectionRec], codeblocks: List[CodeRec], url: str) -> Tuple[List[EntityRec], List[ParamRec]]:
    """
    Extract API entities and their parameters.
    Heuristics:
      - Entities: patterns like "class transformers.X" or "transformers.X" in headings/text.
      - Params: bullets/rows that look like "name (type, optional) — description" inside a Parameters section.
    """
    doc_id = sha1(normalize_url(url))
    entities: Dict[str, EntityRec] = {}
    params: List[ParamRec] = []

    # Entity extraction from text
    entity_pat = re.compile(r"\btransformers\.([A-Za-z][A-Za-z0-9_]+)\b")
    class_pat = re.compile(r"\bclass\s+(transformers\.[A-Za-z][A-Za-z0-9_]+)\b")
    func_pat = re.compile(r"\b([a-z_][a-z0-9_]*)\s*<\s*source\s*>", flags=re.I)

    def add_entity(fq: str, kind: str, evidence_ptr: str, confidence: float, signature: str = "", anchor: str = "", source_url: str = "") -> None:
        name = fq.split(".")[-1]
        entity_id = sha1(doc_id + "|" + fq + "|" + kind)
        if entity_id in entities:
            return
        entities[entity_id] = EntityRec(
            entity_id=entity_id,
            doc_id=doc_id,
            name=name,
            fqname=fq,
            kind=kind,
            signature=signature,
            doc_anchor=anchor,
            source_url=source_url,
            confidence=confidence,
            evidence_ptr=evidence_ptr
        )

    # Scan headings and section text
    for s in sections:
        ep = url + (("#" + s.heading_id) if s.heading_id else "")
        for m in class_pat.finditer(s.text):
            add_entity(m.group(1), "class", ep, 0.95)
        for m in entity_pat.finditer(s.text):
            fq = "transformers." + m.group(1)
            add_entity(fq, "entity", ep, 0.70)

    # Scan code blocks for function-like headings from rendered docs (often show "forward <source>" and signature)
    for c in codeblocks:
        ep = url + f"#code:{c.section_id}:{c.ordinal}"
        if "class transformers." in c.code:
            for m in class_pat.finditer(c.code):
                add_entity(m.group(1), "class", ep, 0.90)
        # extract simple "forward(" signature lines for known heads
        sig_line = ""
        m = re.search(r"^\s*forward\s*\n<\s*source\s*>\s*\n\(\s*(.*?)\)\s*→", c.code, flags=re.S | re.M)
        if m:
            sig_line = clean_text(m.group(1))
        # detect mention of LlamaConfig-like
        for m2 in entity_pat.finditer(c.code):
            fq = "transformers." + m2.group(1)
            add_entity(fq, "entity", ep, 0.80, signature=sig_line)

    # Parameter extraction: very heuristic
    # We look for sections where has_parameters_heading=True and then parse bullet-like lines.
    param_line_pat = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*[—-]\s*(.+)$")
    for s in sections:
        if not s.has_parameters_heading:
            continue
        ep = url + (("#" + s.heading_id) if s.heading_id else "")
        lines = [clean_text(x) for x in s.text.split("\n") if clean_text(x)]
        # assign parameters to the best-matching entity: if heading contains an entity mention
        target_entity_id = ""
        target_fq = ""
        for ent in entities.values():
            if ent.name.lower() in (s.heading or "").lower():
                target_entity_id = ent.entity_id
                target_fq = ent.fqname
                break
        # fallback: if doc is llama model doc, prefer LlamaConfig if present
        if not target_entity_id:
            for ent in entities.values():
                if ent.fqname.endswith(".LlamaConfig"):
                    target_entity_id = ent.entity_id
                    target_fq = ent.fqname
                    break

        for ln in lines:
            m = param_line_pat.match(ln)
            if not m:
                continue
            pname = m.group(1)
            pmeta = m.group(2)
            pdesc = m.group(3)
            ptype = pmeta
            pdefault = ""
            # try parse defaults "defaults to X"
            md = re.search(r"defaults?\s+to\s+([^\s,;]+)", pmeta, flags=re.I)
            if md:
                pdefault = md.group(1)
            if not target_entity_id:
                # create a synthetic entity "DocParameters" to anchor params to the doc
                fq = f"transformers.DocParameters[{doc_id}]"
                synthetic_id = sha1(doc_id + "|" + fq + "|synthetic")
                if synthetic_id not in entities:
                    entities[synthetic_id] = EntityRec(
                        entity_id=synthetic_id, doc_id=doc_id, name="DocParameters",
                        fqname=fq, kind="synthetic", signature="", doc_anchor="", source_url="",
                        confidence=0.10, evidence_ptr=ep
                    )
                target_entity_id = synthetic_id
                target_fq = fq

            pid = sha1(target_entity_id + "|" + pname)
            params.append(ParamRec(
                param_id=pid,
                entity_id=target_entity_id,
                name=pname,
                type=ptype,
                default=pdefault,
                description=pdesc
            ))

    return list(entities.values()), params


def compute_mentions(sections: List[SectionRec], entities: List[EntityRec], url: str) -> List[MentionRec]:
    # Simple mention counts: count occurrences of entity name and fqname tokens
    mentions: List[MentionRec] = []
    ent_tokens = [(e.entity_id, e.name, e.fqname) for e in entities]
    for s in sections:
        txt = (s.text or "")
        if not txt:
            continue
        for (eid, name, fq) in ent_tokens:
            c1 = len(re.findall(r"\b" + re.escape(name) + r"\b", txt))
            c2 = len(re.findall(re.escape(fq), txt))
            count = c1 + c2
            if count <= 0:
                continue
            ep = url + (("#" + s.heading_id) if s.heading_id else "")
            mentions.append(MentionRec(section_id=s.section_id, entity_id=eid, count=count, evidence_ptr=ep))
    return mentions


def compute_issues(url: str, doc_id: str, sections: List[SectionRec], codeblocks: List[CodeRec],
                   entities: List[EntityRec], params: List[ParamRec], stats: Dict[str, int]) -> List[IssueRec]:
    issues: List[IssueRec] = []
    def add(severity: str, code: str, message: str, evidence_ptr: str) -> None:
        iid = sha1(doc_id + "|" + code + "|" + evidence_ptr + "|" + message)
        issues.append(IssueRec(issue_id=iid, doc_id=doc_id, severity=severity, code=code, message=message, evidence_ptr=evidence_ptr))

    if len(sections) < 2:
        add("MED", "PARSE_WEAK", "Less than 2 sections extracted. Parser coverage may be weak for this page.", url)

    if stats.get("github_source_links", 0) == 0:
        add("LOW", "GITHUB_SOURCE_LINKS_MISSING", "No GitHub source links detected in page HTML.", url)

    if stats.get("deprecation_mentions", 0) > 0:
        add("MED", "DEPRECATION_MENTION", f"Detected {stats.get('deprecation_mentions', 0)} deprecation mentions (heuristic).", url)

    if stats.get("torchao_string_quant_pattern_hits", 0) > 0:
        add("MED", "TORCHAO_STRING_PATTERN_RISK",
            "Detected TorchAoConfig(\"int4_weight_only\"...) pattern in code blocks; torchao config APIs may evolve. Verify against latest Transformers docs.",
            url)

    # Parameters heading present but no params
    if any(s.has_parameters_heading for s in sections) and len(params) == 0:
        add("MED", "PARAM_EXTRACT_EMPTY", "Parameters heading detected but no parameters extracted.", url)

    # Codeblocks missing but many sections
    if len(sections) >= 3 and len(codeblocks) == 0:
        add("LOW", "CODEBLOCKS_MISSING", "Multiple sections extracted but no code blocks captured. Page may have no <pre>, or parser missed it.", url)

    # Entities missing
    if "/model_doc/" in normalize_url(url) and len(entities) == 0:
        add("LOW", "ENTITY_EXTRACT_EMPTY", "Model doc page but no API entities extracted.", url)

    return issues


# ----------------------------
# Discovery crawl
# ----------------------------

def discover_urls(session: requests.Session, seed_urls: List[str], allow_pats: List[re.Pattern],
                  deny_pats: List[re.Pattern], timeout_s: int, throttle_ms: int, max_depth: int,
                  max_urls: int, cache_dir: Path, skip_unchanged: bool, run_log: Path, cycle: int) -> List[str]:
    """
    BFS discovery crawl restricted by allow/deny patterns.
    Uses fetched HTML to extract <a href> links and pushes into the queue.
    """
    queue: List[Tuple[str, int]] = [(normalize_url(u), 0) for u in seed_urls if normalize_url(u)]
    seen: Set[str] = set()
    out: List[str] = []

    while queue and len(out) < max_urls:
        url, depth = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        if not allow_url(url, allow_pats, deny_pats):
            continue

        out.append(url)

        if depth >= max_depth:
            continue

        fr = fetch_html(session, url, timeout_s=timeout_s, throttle_ms=throttle_ms, cache_dir=cache_dir,
                        skip_unchanged=skip_unchanged)
        append_jsonl(run_log, {"ts": iso_now(), "cycle": cycle, "action": "fetch", "doc_url": fr.url,
                               "status": fr.status, "http_status": fr.http_status, "etag": fr.etag,
                               "last_modified": fr.last_modified, "bytes": fr.bytes_len, "error": fr.error})

        if fr.status != "PASS":
            continue
        if not _HAS_BS4:
            continue

        main = bs4_main_soup(fr.html_text)
        if main is None:
            continue
        for a in main.find_all("a"):
            href = a.get("href", "") or ""
            if not href:
                continue
            absu = to_abs_href(url, href)
            if not absu:
                continue
            if absu in seen:
                continue
            if allow_url(absu, allow_pats, deny_pats):
                queue.append((absu, depth + 1))

    return out


# ----------------------------
# Main harvest
# ----------------------------

def harvest(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).expanduser().resolve()
    safe_mkdir(out_dir)

    raw_dir = out_dir / "RAW"
    extract_dir = out_dir / "EXTRACT"
    binder_dir = out_dir / "BINDER"
    neo4j_dir = out_dir / "NEO4J_IMPORT"
    run_dir = out_dir / "RUN"

    for d in (raw_dir, extract_dir, binder_dir, neo4j_dir, run_dir):
        safe_mkdir(d)

    run_log = run_dir / "run_ledger.jsonl"
    cycle = int(time.time())

    # patterns
    allow_pats = compile_patterns(args.allow_regex)
    deny_pats = compile_patterns(args.deny_regex)

    # seeds
    seed_urls: List[str] = [normalize_url(u) for u in (args.seed_url or []) if normalize_url(u)]
    if args.seed_file:
        p = Path(args.seed_file).expanduser()
        if p.exists():
            for ln in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                u = normalize_url(ln)
                if u:
                    seed_urls.append(u)
    # dedup seeds preserving order
    seen_seed: Set[str] = set()
    seed_urls = [u for u in seed_urls if not (u in seen_seed or seen_seed.add(u))]

    append_jsonl(run_log, {"ts": iso_now(), "cycle": cycle, "action": "run_start",
                           "seed_urls": seed_urls, "discover": bool(args.discover),
                           "max_depth": args.max_depth, "max_urls": args.max_urls, "status": "PASS"})

    session = requests.Session()

    cache_dir = run_dir / "HTTP_CACHE"
    urls_to_process = seed_urls

    if args.discover:
        urls_to_process = discover_urls(
            session=session,
            seed_urls=seed_urls,
            allow_pats=allow_pats,
            deny_pats=deny_pats,
            timeout_s=args.timeout,
            throttle_ms=args.throttle_ms,
            max_depth=args.max_depth,
            max_urls=args.max_urls,
            cache_dir=cache_dir,
            skip_unchanged=bool(args.skip_unchanged),
            run_log=run_log,
            cycle=cycle
        )

    # outputs accumulated in-memory for this run (we also write per-doc JSON to EXTRACT)
    pages: List[Dict[str, Any]] = []
    sections_out: List[Dict[str, Any]] = []
    code_out: List[Dict[str, Any]] = []
    entities_out: List[Dict[str, Any]] = []
    params_out: List[Dict[str, Any]] = []
    mentions_out: List[Dict[str, Any]] = []
    issues_out: List[Dict[str, Any]] = []

    for url in urls_to_process[: args.max_urls]:
        if not allow_url(url, allow_pats, deny_pats):
            continue

        doc_id = sha1(normalize_url(url))
        fr = fetch_html(session, url, timeout_s=args.timeout, throttle_ms=args.throttle_ms,
                        cache_dir=cache_dir, skip_unchanged=bool(args.skip_unchanged))

        append_jsonl(run_log, {"ts": iso_now(), "cycle": cycle, "action": "fetch", "doc_url": fr.url,
                               "status": fr.status, "http_status": fr.http_status, "etag": fr.etag,
                               "last_modified": fr.last_modified, "bytes": fr.bytes_len, "error": fr.error})

        if fr.status != "PASS":
            # still record page meta (fail)
            pages.append({
                "doc_id": doc_id, "url": url, "doctype": infer_doctype(url), "product": infer_product(url),
                "title": "", "doc_version_hint": "", "lang": extract_lang_from_url(url),
                "fetched_at": iso_now(), "http_status": fr.http_status, "etag": fr.etag,
                "last_modified": fr.last_modified, "bytes": fr.bytes_len, "sha256": fr.sha256,
                "status": "FAIL", "error": fr.error
            })
            continue

        # Save raw html
        raw_path = raw_dir / f"{doc_id}.html"
        raw_path.write_text(fr.html_text, encoding="utf-8", errors="ignore")

        # Parse
        main = bs4_main_soup(fr.html_text) if _HAS_BS4 else None
        title = extract_title(main, fr.html_text)
        lang = extract_lang_from_url(url)
        doctype = infer_doctype(url)
        product = infer_product(url)

        stats = {"github_source_links": 0, "deprecation_mentions": 0, "torchao_string_quant_pattern_hits": 0}

        sections: List[SectionRec] = []
        codeblocks: List[CodeRec] = []
        entities: List[EntityRec] = []
        params: List[ParamRec] = []
        mentions: List[MentionRec] = []
        issues: List[IssueRec] = []

        parse_status = "PASS"
        parse_error = ""
        try:
            sections, codeblocks, stats = parse_sections_and_code(main, url)
            entities, params = extract_entities_and_params(sections, codeblocks, url)
            mentions = compute_mentions(sections, entities, url)
            issues = compute_issues(url, doc_id, sections, codeblocks, entities, params, stats)
        except Exception as e:
            parse_status = "FAIL"
            parse_error = str(e)

        append_jsonl(run_log, {"ts": iso_now(), "cycle": cycle, "action": "parse", "doc_url": url,
                               "doc_id": doc_id, "status": parse_status, "sections": len(sections),
                               "codeblocks": len(codeblocks), "entities": len(entities), "params": len(params),
                               "issues": len(issues), "error": parse_error})

        # Page record
        page_row = {
            "doc_id": doc_id, "url": url, "doctype": doctype, "product": product, "title": title,
            "doc_version_hint": "", "lang": lang, "fetched_at": iso_now(), "http_status": fr.http_status,
            "etag": fr.etag, "last_modified": fr.last_modified, "bytes": fr.bytes_len, "sha256": fr.sha256,
            "status": parse_status, "error": parse_error
        }
        pages.append(page_row)

        # Emit per-doc extract JSON
        extract_obj = {
            "extract_id": f"HF_DOC_EXTRACT.v1:{doc_id}",
            "ts": iso_now(),
            "doc": page_row,
            "stats": stats,
            "sections": [s.__dict__ for s in sections],
            "codeblocks": [c.__dict__ for c in codeblocks],
            "entities": [e.__dict__ for e in entities],
            "params": [p.__dict__ for p in params],
            "mentions": [m.__dict__ for m in mentions],
            "issues": [i.__dict__ for i in issues],
        }
        write_json(extract_dir / f"{doc_id}.json", extract_obj)

        # Accumulate binder/neo4j
        sections_out.extend([s.__dict__ for s in sections])
        code_out.extend([c.__dict__ for c in codeblocks])
        entities_out.extend([e.__dict__ for e in entities])
        params_out.extend([p.__dict__ for p in params])
        mentions_out.extend([m.__dict__ for m in mentions])
        issues_out.extend([i.__dict__ for i in issues])

    # Dedup for this run (stable IDs)
    def dedup_by(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
        seen: Set[str] = set()
        out: List[Dict[str, Any]] = []
        for r in rows:
            k = str(r.get(key, ""))
            if not k:
                continue
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
        return out

    pages = dedup_by(pages, "doc_id")
    sections_out = dedup_by(sections_out, "section_id")
    code_out = dedup_by(code_out, "code_id")
    entities_out = dedup_by(entities_out, "entity_id")
    params_out = dedup_by(params_out, "param_id")
    issues_out = dedup_by(issues_out, "issue_id")
    # mentions: composite key
    mentions_dedup: Set[str] = set()
    mentions2: List[Dict[str, Any]] = []
    for m in mentions_out:
        k = f"{m.get('section_id','')}|{m.get('entity_id','')}"
        if not m.get("section_id") or not m.get("entity_id"):
            continue
        if k in mentions_dedup:
            continue
        mentions_dedup.add(k)
        mentions2.append(m)
    mentions_out = mentions2

    # Write binder JSONL and CSV
    binder_pages_jsonl = binder_dir / "pages.jsonl"
    binder_sections_jsonl = binder_dir / "sections.jsonl"
    binder_code_jsonl = binder_dir / "codeblocks.jsonl"
    binder_entities_jsonl = binder_dir / "entities.jsonl"
    binder_params_jsonl = binder_dir / "params.jsonl"
    binder_mentions_jsonl = binder_dir / "mentions.jsonl"
    binder_issues_jsonl = binder_dir / "issues.jsonl"

    for row in pages:
        append_jsonl(binder_pages_jsonl, row)
    for row in sections_out:
        append_jsonl(binder_sections_jsonl, row)
    for row in code_out:
        append_jsonl(binder_code_jsonl, row)
    for row in entities_out:
        append_jsonl(binder_entities_jsonl, row)
    for row in params_out:
        append_jsonl(binder_params_jsonl, row)
    for row in mentions_out:
        append_jsonl(binder_mentions_jsonl, row)
    for row in issues_out:
        append_jsonl(binder_issues_jsonl, row)

    # CSV snapshots (per-run overwrite)
    write_csv(binder_dir / "pages.csv", pages,
              ["doc_id","url","doctype","product","title","doc_version_hint","lang","fetched_at","http_status","etag","last_modified","bytes","sha256","status","error"])
    write_csv(binder_dir / "sections.csv", sections_out,
              ["section_id","doc_id","path","level","heading","heading_id","ordinal","text","text_sha1","has_parameters_heading"])
    write_csv(binder_dir / "codeblocks.csv", code_out,
              ["code_id","doc_id","section_id","ordinal","language","code","code_sha1"])
    write_csv(binder_dir / "entities.csv", entities_out,
              ["entity_id","doc_id","name","fqname","kind","signature","doc_anchor","source_url","confidence","evidence_ptr"])
    write_csv(binder_dir / "params.csv", params_out,
              ["param_id","entity_id","name","type","default","description"])
    write_csv(binder_dir / "mentions.csv", mentions_out,
              ["section_id","entity_id","count","evidence_ptr"])
    write_csv(binder_dir / "issues.csv", issues_out,
              ["issue_id","doc_id","severity","code","message","evidence_ptr"])

    # Neo4j import CSV tables
    write_csv(neo4j_dir / "nodes_docpage.csv", pages,
              ["doc_id","url","doctype","product","title","doc_version_hint","lang","fetched_at","http_status","etag","last_modified","bytes","sha256"])
    write_csv(neo4j_dir / "nodes_docsection.csv", sections_out,
              ["section_id","doc_id","path","level","heading","ordinal","text","text_sha1","has_parameters_heading"])
    write_csv(neo4j_dir / "nodes_codeblock.csv", code_out,
              ["code_id","doc_id","section_id","ordinal","language","code","code_sha1"])
    write_csv(neo4j_dir / "nodes_apientity.csv", entities_out,
              ["entity_id","doc_id","name","fqname","kind","signature","doc_anchor","source_url","confidence","evidence_ptr"])
    write_csv(neo4j_dir / "nodes_param.csv", params_out,
              ["param_id","entity_id","name","type","default","description"])
    write_csv(neo4j_dir / "edges_mentions.csv", mentions_out,
              ["section_id","entity_id","count","evidence_ptr"])
    write_csv(neo4j_dir / "nodes_issue.csv", issues_out,
              ["issue_id","doc_id","severity","code","message","evidence_ptr"])

    append_jsonl(run_log, {"ts": iso_now(), "cycle": cycle, "action": "emit", "status": "PASS",
                           "out_dir": str(out_dir),
                           "pages": len(pages), "sections": len(sections_out), "codeblocks": len(code_out),
                           "entities": len(entities_out), "mentions": len(mentions_out),
                           "issues": len(issues_out), "neo4j_tables": 7})

    append_jsonl(run_log, {"ts": iso_now(), "cycle": cycle, "action": "run_end",
                           "docs_harvested": len(pages), "urls_processed": len(urls_to_process),
                           "status": "PASS"})

    # Provenance + manifest
    prov = {
        "provenance_id": f"HF_DOC_PROVENANCE.v1:{cycle}",
        "ts": iso_now(),
        "script": Path(__file__).name,
        "args": vars(args),
        "outputs": {
            "raw_dir": str(raw_dir),
            "extract_dir": str(extract_dir),
            "binder_dir": str(binder_dir),
            "neo4j_dir": str(neo4j_dir),
            "run_dir": str(run_dir)
        },
        "notes": [
            "All emitted IDs are stable hashes for append-only merges.",
            "Issue signals are heuristic; validate against the source URL and section."
        ]
    }
    write_json(run_dir / "provenance_index.json", prov)

    # manifest: list top-level files for quick verification
    manifest_files: List[Dict[str, Any]] = []
    for root, _, files in os.walk(out_dir):
        for fn in files:
            fp = Path(root) / fn
            try:
                sz = fp.stat().st_size
            except Exception:
                sz = 0
            rel = str(fp.relative_to(out_dir)).replace("\\", "/")
            manifest_files.append({"path": rel, "bytes": sz})
    manifest = {
        "manifest_id": f"HF_DOC_MANIFEST.v1:{cycle}",
        "ts": iso_now(),
        "file_count": len(manifest_files),
        "files": sorted(manifest_files, key=lambda x: x["path"])
    }
    write_json(run_dir / "manifest.json", manifest)

    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Harvest HF Transformers docs into binder + Neo4j import tables.")
    p.add_argument("--out", required=True, help="Output directory (will be created if missing).")
    p.add_argument("--seed-url", action="append", default=[], help="Seed doc URL (repeatable).")
    p.add_argument("--seed-file", default="", help="Text file with one doc URL per line.")
    p.add_argument("--discover", action="store_true", help="Enable BFS discovery crawl from seed URLs.")
    p.add_argument("--max-depth", type=int, default=2, help="Discovery BFS max depth.")
    p.add_argument("--max-urls", type=int, default=250, help="Max URLs processed (seed + discovered).")
    p.add_argument("--timeout", type=int, default=30, help="HTTP timeout seconds.")
    p.add_argument("--throttle-ms", type=int, default=250, help="Delay between requests in ms.")
    p.add_argument("--allow-regex", action="append", default=[
        r"^https://huggingface\.co/docs/transformers/(?:[a-z]{2}/)?model_doc/[^?#]+$",
        r"^https://huggingface\.co/docs/transformers/(?:[a-z]{2}/)?en/[^?#]+$",
        r"^https://huggingface\.co/docs/transformers/kv_cache/?$",
        r"^https://huggingface\.co/docs/transformers/quantization/torchao/?$",
        r"^https://huggingface\.co/docs/transformers/main_classes/[^?#]+$",
        r"^https://huggingface\.co/docs/transformers/installation/?$",
    ], help="Allowlist regex (repeatable).")
    p.add_argument("--deny-regex", action="append", default=[
        r"/enterprise", r"/pricing", r"/datasets", r"/spaces", r"/community", r"/login", r"/signup",
        r"/AcceptableUse", r"/terms", r"/privacy"
    ], help="Denylist regex (repeatable).")
    p.add_argument("--skip-unchanged", action="store_true", help="Send If-None-Match/If-Modified-Since and reuse cached HTML on 304.")
    args = p.parse_args(argv)

    if not _HAS_BS4:
        # do not fail; warn and continue with limited parsing
        sys.stderr.write("WARNING: beautifulsoup4 not installed. Parsing will be weak. Install: python -m pip install beautifulsoup4\n")

    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return harvest(args)
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        sys.stderr.write(f"FATAL: {e}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
