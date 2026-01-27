#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LITIGATIONOS_ADVERSARIAL_SIGNAL_SUITE_v2_2

Non-destructive, append-only adversarial signal scanner with bootstrap outputs.

Core features:
- Constants for bucket rules and document type registry.
- Pattern objects with id, category, regex, flags, weight, mv_ids, notes.
- Synonym expansion via expand_config for deterministic enrichment.
- Text extraction for .txt/.md/.log/.pdf/.docx (best-effort, no OCR).
- 800-character segments with overlap and line/page/paragraph locators.
- Append-only JSONL outputs for events and file processing statuses.
- Convergence cycles with caching to avoid reprocessing.
- Watch mode for polling-based rescan on changes.
- Bootstrap bundle creation: schema + query pack + README.
- Neo4j CSV export for EvidenceAtom, SignalEvent, and optional MV nodes.
- Graph merge via --merge-graphs with deterministic output and logging.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SEGMENT_SIZE = 800
SEGMENT_OVERLAP = 120

DOCTYPE_REGISTRY: Dict[str, Dict[str, str]] = {
    ".txt": {"doctype": "TEXT", "parser": "read_text"},
    ".md": {"doctype": "MARKDOWN", "parser": "read_text"},
    ".log": {"doctype": "LOG", "parser": "read_text"},
    ".pdf": {"doctype": "PDF", "parser": "read_pdf"},
    ".docx": {"doctype": "DOCX", "parser": "read_docx"},
}

BUCKET_RULES: Dict[str, Any] = {
    "version": "bucket_rules.v1",
    "max_buckets": 15,
    "buckets": [
        {"bucket": "B01_TEXT", "ext": [".txt", ".md", ".log"]},
        {"bucket": "B02_PDF", "ext": [".pdf"]},
        {"bucket": "B03_DOCX", "ext": [".docx"]},
        {"bucket": "B15_OTHER", "ext": ["*"]},
    ],
}

DEFAULT_EVENT_TO_MV_MAP: Dict[str, Any] = {
    "version": "event_to_mv_map.v1",
    "mv": {
        "MV01": {"name": "Bias/Partiality"},
        "MV02": {"name": "Weaponized_PPO"},
        "MV03": {"name": "Retaliatory_Contempt"},
        "MV04": {"name": "Due_Process/Notice_Defect"},
        "MV05": {"name": "Evidentiary_Exclusion/Record_Suppression"},
        "MV06": {"name": "Barrier/Paywall/Access_To_Court"},
        "MV07": {"name": "Mental_Health_Label_Weaponization"},
        "MV08": {"name": "Defamation/Character_Assassination"},
        "MV09": {"name": "Parenting_Time_Interference/Alienation_Signal"},
    },
    "map": {
        "BIAS_OR_PARTIALITY": [{"mv": "MV01", "w": 0.9}],
        "CREDIBILITY_ATTACK": [{"mv": "MV08", "w": 0.6}, {"mv": "MV01", "w": 0.4}],
        "SUBSTANCE_ALLEGATION": [{"mv": "MV08", "w": 0.8}],
        "MENTAL_HEALTH_ALLEGATION": [{"mv": "MV07", "w": 0.9}, {"mv": "MV08", "w": 0.5}],
        "UNFIT_PARENT_LANGUAGE": [{"mv": "MV09", "w": 0.6}, {"mv": "MV08", "w": 0.4}],
        "EX_PARTE_OVERREACH": [{"mv": "MV04", "w": 0.8}, {"mv": "MV02", "w": 0.6}],
        "NOTICE_DEFECT": [{"mv": "MV04", "w": 0.9}],
        "EVIDENCE_BLOCKED": [{"mv": "MV05", "w": 0.9}, {"mv": "MV01", "w": 0.5}],
        "HEARSAY_RELIANCE": [{"mv": "MV05", "w": 0.6}],
        "CONTEMPT_RETALIATION": [{"mv": "MV03", "w": 0.8}, {"mv": "MV06", "w": 0.3}],
        "FEE_BOND_BARRIER": [{"mv": "MV06", "w": 0.8}],
        "PPO_WEAPONIZATION": [{"mv": "MV02", "w": 0.9}],
        "NEGATIVE_EMOTION_LANGUAGE": [{"mv": "MV08", "w": 0.4}],
    },
}

DEFAULT_ADVERSARIAL_CONFIG: Dict[str, Any] = {
    "version": "adversarial_config.v2_2",
    "actors": {
        "judge": ["Judge", "Magistrate"],
        "opponent": ["respondent", "petitioner", "plaintiff", "defendant"],
        "agency": ["FOC", "Friend of the Court", "CPS", "DHHS"],
    },
    "patterns": [
        {
            "id": "bias_asymmetric_ruling",
            "category": "BIAS_OR_PARTIALITY",
            "regex": r"\\b(asymmetric|one[-\\s]?sided|favor(?:ed|itism)|double\\s+standard|outcome[-\\s]?driven)\\b",
            "flags": ["I"],
            "weight": 0.8,
            "mv_ids": ["MV01"],
            "notes": "Asymmetric rulings or outcome-driven language.",
        },
        {
            "id": "notice_defect",
            "category": "NOTICE_DEFECT",
            "regex": r"\\b(no\\s+notice|lack\\s+of\\s+notice|insufficient\\s+notice|was\\s+not\\s+served|service\\s+was\\s+defective|not\\s+properly\\s+noticed)\\b",
            "flags": ["I"],
            "weight": 0.9,
            "mv_ids": ["MV04"],
            "notes": "Notice defects or service failures.",
        },
        {
            "id": "evidence_blocked",
            "category": "EVIDENCE_BLOCKED",
            "regex": r"\\b(not\\s+allowed\\s+to\\s+present|refused\\s+to\\s+admit|excluded\\s+evidence|would\\s+not\\s+hear|prevented\\s+from\\s+introducing|proffer\\s+denied)\\b",
            "flags": ["I"],
            "weight": 0.95,
            "mv_ids": ["MV05"],
            "notes": "Evidence exclusion or record suppression.",
        },
        {
            "id": "mental_health_allegation",
            "category": "MENTAL_HEALTH_ALLEGATION",
            "regex": r"\\b(delusional|psychosis|paranoid|bipolar|schizo|mental\\s+health\\s+eval|assessment\\s+required|diagnos(?:is|ed)|rule\\s+out\\s+delusional)\\b",
            "flags": ["I"],
            "weight": 0.95,
            "mv_ids": ["MV07", "MV08"],
            "notes": "Mental health allegations or labels.",
        },
        {
            "id": "negative_emotion_language",
            "category": "NEGATIVE_EMOTION_LANGUAGE",
            "regex": r"\\b(crazy|insane|unstable|manipulative|liar|lying|dangerous|abusive|toxic|threat(?:en|ening))\\b",
            "flags": ["I"],
            "weight": 0.4,
            "mv_ids": ["MV08"],
            "notes": "Inflammatory language.",
        },
    ],
    "synonyms": {
        "liar": ["dishonest", "untruthful", "deceptive"],
        "abusive": ["violent", "harassing", "controlling"],
        "dangerous": ["unsafe", "risk", "threat"],
        "excluded": ["barred", "precluded", "stricken"],
        "notice": ["served", "service", "mailing"],
        "bias": ["partial", "prejudice", "unfair"],
    },
}


@dataclass(frozen=True)
class Segment:
    locator: str
    text: str
    extra: Dict[str, Any]


@dataclass(frozen=True)
class FileParseResult:
    ok: bool
    doctype: str
    segments: List[Segment]
    errors: List[str]
    ocr_needed: bool


@dataclass(frozen=True)
class CacheEntry:
    size: int
    mtime: float
    pattern_hash: str


def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def stable_event_id(path: str, locator: str, pattern_id: str, span: Tuple[int, int]) -> str:
    return "EVT_" + sha256_hex(f"{path}|{locator}|{pattern_id}|{span[0]}|{span[1]}")[:32]


def stable_evidence_id(path: str, locator: str, snippet: str) -> str:
    return "EA_" + sha256_hex(f"{path}|{locator}|{snippet}")[:32]


def optional_import(module_name: str) -> Optional[Any]:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    return importlib.import_module(module_name)


def bucket_for_ext(ext: str) -> str:
    ext = ext.lower()
    for bucket in BUCKET_RULES["buckets"]:
        if "*" in bucket["ext"]:
            continue
        if ext in bucket["ext"]:
            return bucket["bucket"]
    return "B15_OTHER"


def iter_files(roots: List[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            yield root
            continue
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.startswith("."):
                    continue
                yield Path(dirpath) / name


def split_text_segments(text: str, segment_size: int, overlap: int) -> List[Tuple[int, int]]:
    if not text:
        return [(0, 0)]
    step = max(1, segment_size - overlap)
    segments = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + segment_size)
        segments.append((start, end))
        if end >= length:
            break
        start += step
    return segments


def line_index_map(text: str) -> List[int]:
    line_starts = [0]
    for idx, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(idx + 1)
    return line_starts


def locator_for_span(text: str, start: int, end: int) -> str:
    line_starts = line_index_map(text)
    start_line = 1
    end_line = 1
    for i, pos in enumerate(line_starts, start=1):
        if pos <= start:
            start_line = i
        if pos <= end:
            end_line = i
    return f"line:{start_line}-{end_line}"


def read_text(path: Path) -> FileParseResult:
    errors: List[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return FileParseResult(ok=False, doctype="TEXT", segments=[], errors=[f"READ_TEXT_ERROR:{type(exc).__name__}:{exc}"], ocr_needed=False)

    segments: List[Segment] = []
    for start, end in split_text_segments(text, SEGMENT_SIZE, SEGMENT_OVERLAP):
        locator = locator_for_span(text, start, end)
        segments.append(Segment(locator=locator, text=text[start:end], extra={"start": start, "end": end}))
    return FileParseResult(ok=True, doctype=DOCTYPE_REGISTRY[path.suffix.lower()]["doctype"], segments=segments, errors=errors, ocr_needed=False)


def read_docx(path: Path) -> FileParseResult:
    errors: List[str] = []
    docx = optional_import("docx")
    if docx is None:
        return FileParseResult(ok=False, doctype="DOCX", segments=[], errors=["MISSING_DEP:python-docx"], ocr_needed=False)

    segments: List[Segment] = []
    try:
        document = docx.Document(str(path))
    except Exception as exc:
        return FileParseResult(ok=False, doctype="DOCX", segments=[], errors=[f"READ_DOCX_ERROR:{type(exc).__name__}:{exc}"], ocr_needed=False)

    for idx, para in enumerate(document.paragraphs, start=1):
        text = (para.text or "").strip()
        if not text:
            continue
        for start, end in split_text_segments(text, SEGMENT_SIZE, SEGMENT_OVERLAP):
            locator = f"para:{idx}"
            segments.append(
                Segment(
                    locator=locator,
                    text=text[start:end],
                    extra={"para": idx, "start": start, "end": end},
                )
            )
    if not segments:
        segments.append(Segment(locator="para:0", text="", extra={"para": 0}))
    return FileParseResult(ok=True, doctype="DOCX", segments=segments, errors=errors, ocr_needed=False)


def read_pdf(path: Path) -> FileParseResult:
    errors: List[str] = []
    segments: List[Segment] = []
    ocr_needed = False

    fitz = optional_import("fitz")
    pdfplumber = optional_import("pdfplumber")

    if fitz is None and pdfplumber is None:
        return FileParseResult(ok=False, doctype="PDF", segments=[], errors=["MISSING_DEP:PyMuPDF_or_pdfplumber"], ocr_needed=True)

    if fitz is not None:
        doc = fitz.open(str(path))
        total_chars = 0
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            total_chars += len(text)
            for start, end in split_text_segments(text, SEGMENT_SIZE, SEGMENT_OVERLAP):
                locator = f"page:{page_index + 1}"
                segments.append(
                    Segment(
                        locator=locator,
                        text=text[start:end],
                        extra={"page": page_index + 1, "start": start, "end": end},
                    )
                )
        doc.close()
        if total_chars < 50:
            ocr_needed = True
            errors.append("LOW_TEXT:OCR_UNKNOWN")
        return FileParseResult(ok=True, doctype="PDF", segments=segments, errors=errors, ocr_needed=ocr_needed)

    with pdfplumber.open(str(path)) as pdf:
        total_chars = 0
        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            total_chars += len(text)
            for start, end in split_text_segments(text, SEGMENT_SIZE, SEGMENT_OVERLAP):
                locator = f"page:{page_index}"
                segments.append(
                    Segment(
                        locator=locator,
                        text=text[start:end],
                        extra={"page": page_index, "start": start, "end": end},
                    )
                )
        if total_chars < 50:
            ocr_needed = True
            errors.append("LOW_TEXT:OCR_UNKNOWN")
    return FileParseResult(ok=True, doctype="PDF", segments=segments, errors=errors, ocr_needed=ocr_needed)


def extract_text_from_file(path: Path) -> FileParseResult:
    ext = path.suffix.lower()
    if ext not in DOCTYPE_REGISTRY:
        return FileParseResult(ok=False, doctype="UNSUPPORTED", segments=[], errors=["UNSUPPORTED_EXT"], ocr_needed=False)

    parser = DOCTYPE_REGISTRY[ext]["parser"]
    if parser == "read_text":
        return read_text(path)
    if parser == "read_docx":
        return read_docx(path)
    if parser == "read_pdf":
        return read_pdf(path)
    return FileParseResult(ok=False, doctype="UNKNOWN", segments=[], errors=["UNKNOWN_PARSER"], ocr_needed=False)


def normalize_flags(flags: List[str]) -> int:
    combined = 0
    for flag in flags:
        if flag.upper() == "I":
            combined |= re.IGNORECASE
        if flag.upper() == "M":
            combined |= re.MULTILINE
        if flag.upper() == "S":
            combined |= re.DOTALL
    return combined


def compile_patterns(config: Dict[str, Any]) -> List[Tuple[Dict[str, Any], re.Pattern]]:
    compiled: List[Tuple[Dict[str, Any], re.Pattern]] = []
    for pattern in config.get("patterns", []):
        flags = normalize_flags(pattern.get("flags", ["I"]))
        compiled.append((pattern, re.compile(pattern["regex"], flags)))
    return compiled


def expand_config(config: Dict[str, Any], cycle: int) -> Dict[str, Any]:
    if cycle <= 0:
        return config
    synonyms = config.get("synonyms", {})
    expanded_patterns: List[Dict[str, Any]] = []
    for pattern in config.get("patterns", []):
        expanded_patterns.append(pattern)
        regex = pattern.get("regex", "")
        for base, syns in synonyms.items():
            if re.search(rf"\\b{re.escape(base)}\\b", regex, flags=re.IGNORECASE):
                alt = [base] + syns
                alt_rx = rf"(?:{'|'.join(re.escape(item) for item in alt)})"
                new_regex = re.sub(rf"\\b{re.escape(base)}\\b", alt_rx, regex, flags=re.IGNORECASE)
                if new_regex != regex:
                    expanded_patterns.append(
                        {
                            **pattern,
                            "id": f"{pattern['id']}__syn__{base}",
                            "regex": new_regex,
                            "weight": min(0.99, float(pattern.get("weight", 0.5)) + 0.05),
                            "notes": f"Synonym expansion for {base}.",
                        }
                    )
    expanded = dict(config)
    expanded["patterns"] = expanded_patterns
    expanded["enrichment_cycle"] = cycle
    return expanded


def find_actor_tags(text: str, actors: Dict[str, List[str]]) -> List[str]:
    tags = []
    lowered = text.lower()
    for role, names in actors.items():
        for name in names:
            if name.lower() in lowered:
                tags.append(role)
                break
    return sorted(set(tags))


def map_category_to_mv(category: str) -> List[Dict[str, Any]]:
    mapping = DEFAULT_EVENT_TO_MV_MAP.get("map", {}).get(category, [])
    return list(mapping)


def redact_snippet(text: str, max_len: int = 280) -> str:
    cleaned = re.sub(r"\\s+", " ", text).strip()
    if len(cleaned) > max_len:
        return cleaned[:max_len] + " [TRUNCATED]"
    return cleaned


def ensure_dirs(out_root: Path) -> Dict[str, Path]:
    paths = {
        "ROOT": out_root,
        "RUN": out_root / "RUN",
        "OUT": out_root / "OUT",
        "SCHEMA": out_root / "SCHEMA",
        "QUERIES": out_root / "QUERIES",
        "NEO4J": out_root / "NEO4J_IMPORT",
        "MERGE": out_root / "MERGE",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_seen_event_ids(path: Path, max_lines: int = 2_000_000) -> set:
    if not path.exists():
        return set()
    seen = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for idx, line in enumerate(handle):
            if idx >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_id = obj.get("event_id")
            if event_id:
                seen.add(event_id)
    return seen


def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    write_json(path, cache)


def file_signature(path: Path) -> Tuple[int, float]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime


def pattern_signature(config: Dict[str, Any]) -> str:
    raw = json.dumps(config.get("patterns", []), sort_keys=True)
    return sha256_hex(raw)[:24]


def scan_segments_for_signals(
    path: Path,
    parse: FileParseResult,
    compiled: List[Tuple[Dict[str, Any], re.Pattern]],
    actors: Dict[str, List[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    stats = {
        "file": str(path),
        "doctype": parse.doctype,
        "ocr_needed": parse.ocr_needed,
        "errors": parse.errors[:],
        "events": 0,
        "categories": {},
    }

    for seg in parse.segments:
        if not seg.text.strip():
            continue
        actor_tags = find_actor_tags(seg.text, actors)
        for pattern, regex in compiled:
            for match in regex.finditer(seg.text):
                locator = seg.locator
                snippet = redact_snippet(seg.text[max(0, match.start() - 120):match.end() + 160])
                event_id = stable_event_id(str(path), locator, pattern["id"], (match.start(), match.end()))
                evidence_id = stable_evidence_id(str(path), locator, snippet)
                category = pattern.get("category", "UNSPECIFIED")
                mv = map_category_to_mv(category)
                weight = float(pattern.get("weight", 0.5))
                severity = "INFO"
                if weight >= 0.85:
                    severity = "HIGH"
                elif weight >= 0.65:
                    severity = "WARN"

                evt = {
                    "event_id": event_id,
                    "evidence_id": evidence_id,
                    "ts_utc": utc_now_iso(),
                    "path": str(path),
                    "bucket": bucket_for_ext(path.suffix.lower()),
                    "doctype": parse.doctype,
                    "locator": locator,
                    "pattern_id": pattern.get("id"),
                    "category": category,
                    "severity": severity,
                    "weight": weight,
                    "mv": mv,
                    "actor_tags": actor_tags,
                    "match_text": redact_snippet(match.group(0), max_len=180),
                    "snippet": snippet,
                    "parse_errors": parse.errors[:],
                    "ocr_needed": bool(parse.ocr_needed),
                }
                events.append(evt)
                stats["events"] += 1
                stats["categories"][category] = stats["categories"].get(category, 0) + 1

    return events, stats


def emit_neo4j_csv(out_dir: Path, events: List[Dict[str, Any]]) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence_rows = {}
    event_rows = {}
    rel_evidence_has_event = []
    mv_nodes = {}
    rel_event_maps_mv = []

    for event in events:
        evidence_id = event["evidence_id"]
        if evidence_id not in evidence_rows:
            evidence_rows[evidence_id] = {
                "evidence_id:ID": evidence_id,
                "path": event["path"],
                "doctype": event["doctype"],
                "bucket": event["bucket"],
                "locator": event["locator"],
                "snippet": event["snippet"],
                "ocr_needed": str(bool(event.get("ocr_needed", False))).lower(),
            }

        event_id = event["event_id"]
        if event_id not in event_rows:
            event_rows[event_id] = {
                "event_id:ID": event_id,
                "category": event["category"],
                "pattern_id": event["pattern_id"],
                "severity": event["severity"],
                "weight:float": float(event["weight"]),
                "match_text": event["match_text"],
                "actor_tags": "|".join(event.get("actor_tags", [])),
                "ts_utc": event["ts_utc"],
            }

        rel_evidence_has_event.append(
            {"evidence_id:START_ID": evidence_id, "event_id:END_ID": event_id, "type": "EVIDENCE_HAS_EVENT"}
        )

        for mv in event.get("mv", []):
            mv_id = mv.get("mv")
            if not mv_id:
                continue
            if mv_id not in mv_nodes:
                mv_nodes[mv_id] = {
                    "mv_id:ID": mv_id,
                    "name": DEFAULT_EVENT_TO_MV_MAP.get("mv", {}).get(mv_id, {}).get("name", mv_id),
                }
            rel_event_maps_mv.append(
                {
                    "event_id:START_ID": event_id,
                    "mv_id:END_ID": mv_id,
                    "w:float": float(mv.get("w", 0.5)),
                    "type": "MAPS_TO",
                }
            )

    def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        headers = list(rows[0].keys())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    write_csv(out_dir / "evidence_atoms.csv", list(evidence_rows.values()))
    write_csv(out_dir / "signal_events.csv", list(event_rows.values()))
    write_csv(out_dir / "rel_evidence_has_event.csv", rel_evidence_has_event)
    write_csv(out_dir / "mv_nodes.csv", list(mv_nodes.values()))
    write_csv(out_dir / "rel_event_maps_mv.csv", rel_event_maps_mv)

    return {
        "evidence_atoms.csv": str(out_dir / "evidence_atoms.csv"),
        "signal_events.csv": str(out_dir / "signal_events.csv"),
        "rel_evidence_has_event.csv": str(out_dir / "rel_evidence_has_event.csv"),
        "mv_nodes.csv": str(out_dir / "mv_nodes.csv"),
        "rel_event_maps_mv.csv": str(out_dir / "rel_event_maps_mv.csv"),
    }


def emit_bootstrap_bundle(out_root: Path) -> None:
    paths = ensure_dirs(out_root)
    write_json(paths["SCHEMA"] / "doctype_registry.json", DOCTYPE_REGISTRY)
    write_json(paths["SCHEMA"] / "bucket_rules.json", BUCKET_RULES)
    write_json(paths["SCHEMA"] / "event_to_mv_map.json", DEFAULT_EVENT_TO_MV_MAP)
    write_json(paths["SCHEMA"] / "adversarial_config_default.json", DEFAULT_ADVERSARIAL_CONFIG)

    bumper_query_pack = (
        "// BUMPER_QUERY_PACK v1\n"
        "MATCH (ev:SignalEvent)\n"
        "RETURN ev.category AS category, count(*) AS n\n"
        "ORDER BY n DESC;\n"
    )
    (paths["QUERIES"] / "bumper_query_pack.cypher").write_text(bumper_query_pack, encoding="utf-8")
    write_json(
        paths["QUERIES"] / "bumper_query_pack.meta.json",
        {
            "name": "BUMPER_QUERY_PACK",
            "version": "v1",
            "requires": ["EvidenceAtom", "SignalEvent", "MisconductVector", "EVIDENCE_HAS_EVENT", "MAPS_TO"],
            "notes": ["Analysis-only queries."],
        },
    )

    readme = (
        "# BOOTSTRAP_BUNDLE (v2_2)\n\n"
        "Schema and query pack for the adversarial signal suite.\n\n"
        "## Contents\n"
        "- SCHEMA/doctype_registry.json\n"
        "- SCHEMA/bucket_rules.json\n"
        "- SCHEMA/event_to_mv_map.json\n"
        "- SCHEMA/adversarial_config_default.json\n"
        "- QUERIES/bumper_query_pack.cypher\n"
        "- QUERIES/bumper_query_pack.meta.json\n"
    )
    (out_root / "README_BOOTSTRAP.md").write_text(readme, encoding="utf-8")


def scan_once(
    roots: List[Path],
    out_root: Path,
    compiled: List[Tuple[Dict[str, Any], re.Pattern]],
    config: Dict[str, Any],
    cache: Dict[str, Dict[str, Any]],
    pattern_hash: str,
    neo4j_csv: bool,
) -> Dict[str, Any]:
    paths = ensure_dirs(out_root)
    events_path = paths["OUT"] / "adversarial_events.jsonl"
    status_path = paths["OUT"] / "file_status.jsonl"
    cache_path = paths["RUN"] / "cache.json"

    seen_event_ids = load_seen_event_ids(events_path)
    new_events: List[Dict[str, Any]] = []
    file_summaries: List[Dict[str, Any]] = []

    for file_path in iter_files(roots):
        ext = file_path.suffix.lower()
        if ext not in DOCTYPE_REGISTRY:
            continue

        size, mtime = file_signature(file_path)
        cache_key = str(file_path)
        cached = cache.get(cache_key)
        if cached and cached.get("size") == size and cached.get("mtime") == mtime and cached.get("pattern_hash") == pattern_hash:
            continue

        parse = extract_text_from_file(file_path)
        events, stats = scan_segments_for_signals(file_path, parse, compiled, config.get("actors", {}))
        stats["ts_utc"] = utc_now_iso()
        stats["pattern_hash"] = pattern_hash
        append_jsonl(status_path, stats)
        file_summaries.append(stats)

        for event in events:
            if event["event_id"] in seen_event_ids:
                continue
            append_jsonl(events_path, event)
            seen_event_ids.add(event["event_id"])
            new_events.append(event)

        cache[cache_key] = {"size": size, "mtime": mtime, "pattern_hash": pattern_hash}

    save_cache(cache_path, cache)

    summary = {
        "ts_utc": utc_now_iso(),
        "scanned_files": len(file_summaries),
        "new_events_appended": len(new_events),
        "top_categories_new": top_k_categories(new_events, k=25),
    }
    write_json(paths["OUT"] / "adversarial_summary.json", {"summary": summary, "files": file_summaries[:5000]})

    neo4j_paths = None
    if neo4j_csv and new_events:
        neo4j_paths = emit_neo4j_csv(paths["NEO4J"], new_events)

    return {"summary": summary, "new_events": new_events, "neo4j_paths": neo4j_paths}


def top_k_categories(events: List[Dict[str, Any]], k: int = 20) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for event in events:
        category = event.get("category", "UNSPECIFIED")
        counts[category] = counts.get(category, 0) + 1
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [{"category": category, "n": count} for category, count in items[:k]]


def run_convergent_scan(
    roots: List[Path],
    out_root: Path,
    max_cycles: int,
    eps: float,
    stable_n: int,
    neo4j_csv: bool,
) -> Dict[str, Any]:
    cache: Dict[str, Dict[str, Any]] = {}
    history = []
    stable_streak = 0
    total_new = 0

    for cycle in range(max_cycles):
        config = expand_config(DEFAULT_ADVERSARIAL_CONFIG, cycle)
        compiled = compile_patterns(config)
        pattern_hash = pattern_signature(config)
        result = scan_once(roots, out_root, compiled, config, cache, pattern_hash, neo4j_csv)
        new_events = len(result["new_events"])
        total_new += new_events

        history.append(
            {
                "cycle": cycle,
                "pattern_hash": pattern_hash,
                "new_events": new_events,
                "scanned_files": result["summary"]["scanned_files"],
            }
        )

        if new_events == 0:
            stable_streak += 1
        else:
            stable_streak = 0

        if stable_streak >= stable_n:
            break

        if eps is not None and cycle >= 1:
            prev = history[-2]["new_events"]
            ratio = new_events / max(1, prev)
            if ratio <= eps:
                stable_streak += 1
                if stable_streak >= stable_n:
                    break

    report = {
        "ts_utc": utc_now_iso(),
        "total_new_events_appended": total_new,
        "cycles_ran": len(history),
        "history": history,
    }
    write_json(ensure_dirs(out_root)["RUN"] / "convergence_report.json", report)
    return report


def watch_polling(roots: List[Path], out_root: Path, poll_seconds: float, neo4j_csv: bool) -> None:
    cache: Dict[str, Dict[str, Any]] = {}
    config = expand_config(DEFAULT_ADVERSARIAL_CONFIG, cycle=3)
    compiled = compile_patterns(config)
    pattern_hash = pattern_signature(config)

    last_mtimes: Dict[str, float] = {}
    for fp in iter_files(roots):
        if fp.suffix.lower() not in DOCTYPE_REGISTRY:
            continue
        try:
            last_mtimes[str(fp)] = fp.stat().st_mtime
        except OSError:
            continue

    while True:
        changed: List[Path] = []
        for fp in iter_files(roots):
            if fp.suffix.lower() not in DOCTYPE_REGISTRY:
                continue
            try:
                current_mtime = fp.stat().st_mtime
            except OSError:
                continue
            key = str(fp)
            if key not in last_mtimes or current_mtime > last_mtimes[key]:
                last_mtimes[key] = current_mtime
                changed.append(fp)

        if changed:
            scan_once(changed, out_root, compiled, config, cache, pattern_hash, neo4j_csv)

        time.sleep(poll_seconds)


def merge_graphs(graph_dir: Path, out_root: Path) -> Path:
    paths = ensure_dirs(out_root)
    merged_nodes: Dict[str, Dict[str, Any]] = {}
    merged_edges: List[Dict[str, Any]] = []
    log_path = paths["MERGE"] / "merge_log.jsonl"

    graph_files = sorted(graph_dir.glob("*.json"))
    for graph_path in graph_files:
        data = json.loads(graph_path.read_text(encoding="utf-8"))
        for node in data.get("nodes", []):
            node_id = str(node.get("id"))
            if node_id not in merged_nodes:
                merged_nodes[node_id] = node
        for edge in data.get("edges", []):
            merged_edges.append(edge)

        append_jsonl(
            log_path,
            {
                "ts_utc": utc_now_iso(),
                "source": str(graph_path),
                "nodes_added": len(data.get("nodes", [])),
                "edges_added": len(data.get("edges", [])),
            },
        )

    merged = {
        "nodes": sorted(merged_nodes.values(), key=lambda n: str(n.get("id"))),
        "edges": sorted(merged_edges, key=lambda e: json.dumps(e, sort_keys=True)),
    }

    out_path = paths["MERGE"] / "merged_graph.json"
    write_json(out_path, merged)
    return out_path


def package_bundle(script_path: Path, bundle_root: Path, output_zip: Path) -> None:
    emit_bootstrap_bundle(bundle_root)
    with output_zip.open("wb") as handle:
        pass
    import zipfile

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(script_path, arcname=script_path.name)
        for path in bundle_root.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(bundle_root.parent)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="LITIGATIONOS_ADVERSARIAL_SIGNAL_SUITE_v2_2")
    sub = parser.add_subparsers(dest="command", required=True)

    bootstrap = sub.add_parser("bootstrap", help="Emit schema + queries bundle into --out")
    bootstrap.add_argument("--out", required=True, help="Output root for bootstrap bundle")

    scan = sub.add_parser("scan", help="Scan roots and append events with convergence cycles")
    scan.add_argument("--roots", nargs="+", required=True, help="Root paths to scan")
    scan.add_argument("--out", required=True, help="Output folder for append-only results")
    scan.add_argument("--max-cycles", type=int, default=10, help="Maximum convergence cycles")
    scan.add_argument("--eps", type=float, default=0.0, help="Convergence EPS ratio")
    scan.add_argument("--stable-n", type=int, default=2, help="Stop after N stable cycles")
    scan.add_argument("--neo4j-csv", action="store_true", help="Emit Neo4j CSVs for new events")
    scan.add_argument("--merge-graphs", type=str, default=None, help="Merge JSON graphs from a directory")

    watch = sub.add_parser("watch", help="Watch for changes and rescan")
    watch.add_argument("--roots", nargs="+", required=True, help="Root paths to scan")
    watch.add_argument("--out", required=True, help="Output folder for append-only results")
    watch.add_argument("--poll-seconds", type=float, default=5.0, help="Polling interval in seconds")
    watch.add_argument("--neo4j-csv", action="store_true", help="Emit Neo4j CSVs for new events")

    merge = sub.add_parser("merge-graphs", help="Merge JSON graphs in a directory")
    merge.add_argument("--graph-dir", required=True, help="Directory with JSON graph files")
    merge.add_argument("--out", required=True, help="Output folder for merged graph")

    package = sub.add_parser("package", help="Create a zip with script + BOOTSTRAP_BUNDLE")
    package.add_argument("--bundle-root", required=True, help="Output root for BOOTSTRAP_BUNDLE")
    package.add_argument("--zip", required=True, help="Zip file output path")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "bootstrap":
        out_root = Path(args.out).resolve()
        emit_bootstrap_bundle(out_root)
        return 0

    if args.command == "scan":
        out_root = Path(args.out).resolve()
        emit_bootstrap_bundle(out_root)
        roots = [Path(r) for r in args.roots]
        report = run_convergent_scan(
            roots=roots,
            out_root=out_root,
            max_cycles=args.max_cycles,
            eps=args.eps,
            stable_n=args.stable_n,
            neo4j_csv=args.neo4j_csv,
        )
        if args.merge_graphs:
            merge_graphs(Path(args.merge_graphs), out_root)
        print(json.dumps(report, indent=2))
        return 0

    if args.command == "watch":
        out_root = Path(args.out).resolve()
        emit_bootstrap_bundle(out_root)
        roots = [Path(r) for r in args.roots]
        watch_polling(roots, out_root, args.poll_seconds, args.neo4j_csv)
        return 0

    if args.command == "merge-graphs":
        out_root = Path(args.out).resolve()
        merge_graphs(Path(args.graph_dir), out_root)
        return 0

    if args.command == "package":
        bundle_root = Path(args.bundle_root).resolve()
        output_zip = Path(args.zip).resolve()
        package_bundle(Path(__file__).resolve(), bundle_root, output_zip)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
