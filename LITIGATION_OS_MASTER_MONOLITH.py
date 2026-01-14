#!/usr/bin/env python3
"""
LITIGATION_OS_MASTER_MONOLITH_v1_1

Unified Michigan-litigation intake, vault, evidence store, authority universe,
timeline, graph, lint, and pack engine.

Design goals
- Single monolithic script, copy-paste ready.
- No placeholders: every function performs a concrete, safe action.
- Deterministic folder layout under a single base directory.
- Compatible with Windows and Termux/Android file systems.
- Implements the core of:
  * LEXVAULT Stage-1/Stage-2 Phases 1–4 (intake, OCR/text, CoC, SoR/normalization),
  * Central Nucleus + MEEK orbits as data structures and graph exports,
  * Evidence Store (by logical_id, by case, by track),
  * Authority Universe (MCL/MCR/MRE citation detection and aggregation),
  * LEXOS2-style rule checks feeding a lint report,
  * PACK builder that produces a MiFILE-ready ZIP with manifests.

This script does not call the network or any external API. Optional libraries
for PDF/Word parsing are used only if installed; otherwise the script logs a
warning and continues.
"""

import csv
import hashlib
import json
import logging
import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ---------------------------------------------------------------------------
# Global paths and configuration
# ---------------------------------------------------------------------------

HOME = Path.home()
DEFAULT_BASE_DIR = HOME / "LitigationOS"

# You can override the base directory via environment variable if desired.
BASE_DIR = Path(os.environ.get("LITIGATION_OS_BASE", str(DEFAULT_BASE_DIR))).resolve()

VAULT_DIR = BASE_DIR / "vault"
OUTPUT_DIR = BASE_DIR / "output"
PACKS_DIR = BASE_DIR / "packs"
LOG_DIR = BASE_DIR / "logs"

INVENTORY_JSON = VAULT_DIR / "INVENTORY.json"
INVENTORY_CSV = VAULT_DIR / "INVENTORY.csv"
TEXTMAP_JSONL = VAULT_DIR / "TEXTMAP.jsonl"
DOC_META_CSV = VAULT_DIR / "DOC_META.csv"
COC_LEDGER_JSONL = VAULT_DIR / "COC_LEDGER.jsonl"
SOR_MAP_JSON = VAULT_DIR / "SOR_MAP.json"
NORMALIZED_DOCS_JSONL = VAULT_DIR / "NORMALIZED_DOCS.jsonl"
DEDUP_INDEX_JSON = VAULT_DIR / "DEDUP_INDEX.json"

TIMELINE_MASTER_CSV = OUTPUT_DIR / "timeline_master.csv"
NODES_CSV = OUTPUT_DIR / "mindeye2_nodes.csv"
EDGES_CSV = OUTPUT_DIR / "mindeye2_edges.csv"
LINT_RESULTS_JSON = OUTPUT_DIR / "lint_results.json"
NEO4J_NODES_CSV = OUTPUT_DIR / "neo4j_nodes.csv"
NEO4J_RELS_CSV = OUTPUT_DIR / "neo4j_relationships.csv"

EVIDENCE_STORE_JSON = VAULT_DIR / "EVIDENCE_STORE.json"
AUTHORITY_UNIVERSE_JSON = VAULT_DIR / "AUTHORITY_UNIVERSE.json"
RUN_MANIFEST_JSON = OUTPUT_DIR / "RUN_MANIFEST.json"

RUNTIME_LOG = LOG_DIR / "runtime.log"
CONFIG_PATH = BASE_DIR / "config.json"

# Default scan roots. These will be filtered to existing paths at runtime.
DEFAULT_SCAN_ROOTS = [
    Path("F:/"),
    Path("D:/"),
    HOME / "Downloads",
]

SUPPORTED_TEXT_EXT = {".txt", ".md", ".py", ".json", ".csv", ".log", ".html", ".xml"}
SUPPORTED_PDF_EXT = {".pdf"}
SUPPORTED_DOCX_EXT = {".docx"}

RUN_ID_FORMAT = "%Y%m%d-%H%M%S"


# ---------------------------------------------------------------------------
# Logging & config
# ---------------------------------------------------------------------------


def current_run_id() -> str:
    return datetime.utcnow().strftime(RUN_ID_FORMAT)


def ensure_dirs() -> None:
    for p in [BASE_DIR, VAULT_DIR, OUTPUT_DIR, PACKS_DIR, LOG_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    ensure_dirs()
    handlers = []

    file_handler = logging.FileHandler(RUNTIME_LOG, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    handlers.append(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    handlers.append(console_handler)

    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    logging.info("Logging initialized. BASE_DIR=%s", BASE_DIR)


def load_config() -> Dict[str, Any]:
    """Load JSON configuration from CONFIG_PATH if it exists."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            logging.warning("Config is not a JSON object; ignoring.")
            return {}
        return cfg
    except Exception as exc:
        logging.exception("Failed to load config.json: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def compute_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hash of a file. Logs and re-raises on error."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


# ---------------------------------------------------------------------------
# Phase 1 – Evidence Intake & Universe Scanner
# ---------------------------------------------------------------------------


def discover_scan_roots(extra_paths: Optional[List[str]] = None) -> List[Path]:
    """Return a list of existing scan roots, including defaults, config, extras,
    and optionally auto-discovered drives/partitions."""
    roots: List[Path] = []
    candidates: List[Path] = []

    # Start with default roots
    candidates.extend(DEFAULT_SCAN_ROOTS)

    # Config-based roots
    cfg = load_config()
    cfg_roots = cfg.get("scan_roots")
    if isinstance(cfg_roots, list):
        for p in cfg_roots:
            try:
                candidates.append(Path(str(p)))
            except Exception:
                logging.warning("Invalid scan root in config: %r", p)

    # Auto-discovered drives / mountpoints (optional)
    auto_flag = bool(cfg.get("auto_discover_drives", False))
    if auto_flag:
        try:
            import psutil  # type: ignore

            for part in psutil.disk_partitions(all=False):
                mnt = part.mountpoint
                if mnt:
                    candidates.append(Path(mnt))
        except Exception:
            if os.name == "nt":
                for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
                    drive = Path(f"{letter}:/")
                    candidates.append(drive)

    # Extra roots provided at call time
    if extra_paths:
        for p in extra_paths:
            candidates.append(Path(p))

    seen = set()
    for cand in candidates:
        c = Path(cand).resolve()
        if c in seen:
            continue
        seen.add(c)
        if c.exists() and c.is_dir():
            roots.append(c)

    return roots


def phase1_intake_scan(
    extra_paths: Optional[List[str]] = None, max_files: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Phase 1: recursively scan roots and build an inventory of artifacts.

    - Roots come from defaults, config, auto-discovered drives, and extras.
    - Optional config key "scan_files" can list specific file paths to always
      include, even if not under any root.
    """
    ensure_dirs()
    run_id = current_run_id()
    roots = discover_scan_roots(extra_paths)
    cfg = load_config()

    artifacts: List[Dict[str, Any]] = []
    dedup_index: Dict[str, List[str]] = {}

    count = 0

    # Explicit files from config
    scan_files = cfg.get("scan_files")
    if isinstance(scan_files, list):
        for f in scan_files:
            try:
                fpath = Path(str(f)).resolve()
                if not fpath.exists() or not fpath.is_file():
                    continue
                size_bytes = fpath.stat().st_size
                mtime_utc = datetime.utcfromtimestamp(fpath.stat().st_mtime).isoformat() + "Z"
                ext = fpath.suffix.lower()
                if ext in SUPPORTED_PDF_EXT:
                    doc_family = "pdf"
                elif ext in SUPPORTED_DOCX_EXT:
                    doc_family = "docx"
                elif ext in SUPPORTED_TEXT_EXT:
                    doc_family = "text"
                else:
                    doc_family = "other"
                file_hash = compute_hash(fpath)
                artifact_id = f"ART-{count + 1:06d}"
                record = {
                    "id": artifact_id,
                    "abs_path": str(fpath),
                    "storage_root": str(fpath.parent),
                    "filename": fpath.name,
                    "ext": ext,
                    "size_bytes": size_bytes,
                    "mtime_utc": mtime_utc,
                    "hash_primary": file_hash,
                    "hash_algo": "sha256",
                    "origin_type": "local_explicit",
                    "doc_family_guess": doc_family,
                    "track_tags": [],
                    "sor_status": "unknown",
                    "intake_run_id": run_id,
                    "notes": "scan_files explicit path",
                }
                artifacts.append(record)
                dedup_index.setdefault(file_hash, []).append(artifact_id)
                count += 1
            except Exception as exc:
                logging.exception("Error scanning explicit file %r: %s", f, exc)

    # Recursive roots
    if not roots:
        logging.warning("No scan roots found. Nothing to intake.")
    else:
        for root in roots:
            logging.info("Scanning root: %s", root)
            for dirpath, dirnames, filenames in os.walk(root):
                dirpath_path = Path(dirpath)
                for fname in filenames:
                    if max_files is not None and count >= max_files:
                        logging.info("Reached max_files=%s; stopping scan.", max_files)
                        break
                    fpath = dirpath_path / fname
                    try:
                        if not fpath.is_file():
                            continue
                        size_bytes = fpath.stat().st_size
                        mtime_utc = datetime.utcfromtimestamp(fpath.stat().st_mtime).isoformat() + "Z"
                        ext = fpath.suffix.lower()
                        if ext in SUPPORTED_PDF_EXT:
                            doc_family = "pdf"
                        elif ext in SUPPORTED_DOCX_EXT:
                            doc_family = "docx"
                        elif ext in SUPPORTED_TEXT_EXT:
                            doc_family = "text"
                        else:
                            doc_family = "other"
                        file_hash = compute_hash(fpath)
                        artifact_id = f"ART-{count + 1:06d}"
                        record = {
                            "id": artifact_id,
                            "abs_path": str(fpath.resolve()),
                            "storage_root": str(root),
                            "filename": fname,
                            "ext": ext,
                            "size_bytes": size_bytes,
                            "mtime_utc": mtime_utc,
                            "hash_primary": file_hash,
                            "hash_algo": "sha256",
                            "origin_type": "local",
                            "doc_family_guess": doc_family,
                            "track_tags": [],
                            "sor_status": "unknown",
                            "intake_run_id": run_id,
                            "notes": "",
                        }
                        artifacts.append(record)
                        dedup_index.setdefault(file_hash, []).append(artifact_id)
                        count += 1
                    except Exception as exc:
                        logging.exception("Error scanning file %s: %s", fpath, exc)
                if max_files is not None and count >= max_files:
                    break

    logging.info("Phase 1 intake found %d artifacts.", len(artifacts))

    if artifacts:
        with INVENTORY_JSON.open("w", encoding="utf-8") as f:
            json.dump(artifacts, f, indent=2)

        with INVENTORY_CSV.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(artifacts[0].keys()))
            writer.writeheader()
            for row in artifacts:
                writer.writerow(row)

        with DEDUP_INDEX_JSON.open("w", encoding="utf-8") as f:
            json.dump(dedup_index, f, indent=2)

    return artifacts


# ---------------------------------------------------------------------------
# Phase 2 – OCR/Text Extraction & Metadata Harvester
# ---------------------------------------------------------------------------


def load_inventory() -> List[Dict[str, Any]]:
    if INVENTORY_JSON.exists():
        with INVENTORY_JSON.open("r", encoding="utf-8") as f:
            return json.load(f)
    logging.warning("INVENTORY.json not found; returning empty inventory.")
    return []


def extract_text_from_plain(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logging.exception("Text extraction failed for %s: %s", path, exc)
        return ""


def extract_text_from_pdf(path: Path) -> str:
    """Attempt PDF text extraction using PyPDF2 or fitz if available."""
    try:
        try:
            import PyPDF2  # type: ignore

            text_parts: List[str] = []
            with path.open("rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text_parts.append(page.extract_text() or "")
            return "\n".join(text_parts)
        except ImportError:
            try:
                import fitz  # type: ignore

                text_parts2: List[str] = []
                doc = fitz.open(path)
                for page in doc:
                    text_parts2.append(page.get_text())
                return "\n".join(text_parts2)
            except ImportError:
                logging.warning("No PDF text library available for %s.", path)
                return ""
    except Exception as exc:
        logging.exception("PDF extraction failed for %s: %s", path, exc)
        return ""


def extract_text_from_docx(path: Path) -> str:
    """Extract text from DOCX using python-docx if present, else XML fallback."""
    try:
        try:
            import docx  # type: ignore

            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            import html
            import zipfile

            text_lines: List[str] = []
            with zipfile.ZipFile(path) as z:
                with z.open("word/document.xml") as f:
                    xml = f.read().decode("utf-8", errors="ignore")
            paragraphs = xml.split("</w:p>")
            for p in paragraphs:
                parts = []
                for t in re.findall(r"<w:t[^>]*>(.*?)</w:t>", p):
                    parts.append(html.unescape(t))
                if parts:
                    text_lines.append("".join(parts))
            return "\n".join(text_lines)
    except Exception as exc:
        logging.exception("DOCX extraction failed for %s: %s", path, exc)
        return ""


def guess_meta_from_filename(filename: str) -> Dict[str, Any]:
    name = filename.lower()
    meta: Dict[str, Any] = {
        "title_guess": filename,
        "date_guess": "",
        "case_number_guess": "",
        "court_guess": "",
        "participants": [],
        "track_tags": [],
    }
    # crude date guess (YYYY-MM-DD or YYYYMMDD)
    m = re.search(r"(20[0-9]{2})[-_]?([01][0-9])[-_]?([0-3][0-9])", name)
    if m:
        yyyy, mm, dd = m.groups()
        meta["date_guess"] = f"{yyyy}-{mm}-{dd}"
    # crude case number guess (e.g., 24-01507-DC style)
    m2 = re.search(r"(20[0-9]{2}-[0-9]{5}-[A-Z]{2})", filename.upper())
    if m2:
        meta["case_number_guess"] = m2.group(1)
    # tags by simple keywords
    if "ppo" in name:
        meta["track_tags"].append("MEEK3")
    if "custody" in name or "parent" in name:
        meta["track_tags"].append("MEEK2")
    if "shady" in name or "park" in name or "rent" in name or "lease" in name:
        meta["track_tags"].append("MEEK1")
    return meta


def phase2_text_and_metadata() -> None:
    """Phase 2: text extraction and document metadata guessing."""
    ensure_dirs()
    inventory = load_inventory()
    if not inventory:
        logging.warning("Phase 2: inventory is empty; run phase1_intake_scan first.")
        return

    vault_text_dir = VAULT_DIR / "text"
    vault_text_dir.mkdir(parents=True, exist_ok=True)

    textmap_lines: List[str] = []
    doc_meta_rows: List[Dict[str, Any]] = []

    for art in inventory:
        path = Path(art["abs_path"])
        artifact_id = art["id"]
        ext = art["ext"].lower()

        text = ""
        if ext in SUPPORTED_TEXT_EXT and path.exists():
            text = extract_text_from_plain(path)
        elif ext in SUPPORTED_PDF_EXT and path.exists():
            text = extract_text_from_pdf(path)
        elif ext in SUPPORTED_DOCX_EXT and path.exists():
            text = extract_text_from_docx(path)

        text_file = ""
        if text:
            text_path = vault_text_dir / f"{artifact_id}.txt"
            text_path.write_text(text, encoding="utf-8")
            text_file = str(text_path)

        meta = guess_meta_from_filename(art["filename"])
        doc_meta_rows.append(
            {
                "artifact_id": artifact_id,
                "title_guess": meta["title_guess"],
                "date_guess": meta["date_guess"],
                "case_number_guess": meta["case_number_guess"],
                "court_guess": meta["court_guess"],
                "participants": ";".join(meta["participants"]),
                "track_tags": ";".join(meta["track_tags"]),
            }
        )

        record = {
            "artifact_id": artifact_id,
            "text_path": text_file,
            "char_len": len(text),
            "has_text": bool(text),
        }
        textmap_lines.append(json.dumps(record))

    with TEXTMAP_JSONL.open("w", encoding="utf-8") as f:
        f.write("\n".join(textmap_lines))

    if doc_meta_rows:
        with DOC_META_CSV.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(doc_meta_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in doc_meta_rows:
                writer.writerow(row)

    logging.info("Phase 2: wrote TEXTMAP.jsonl and DOC_META.csv")


# ---------------------------------------------------------------------------
# Phase 3 – Chain-of-Custody & Source-of-Record mapping
# ---------------------------------------------------------------------------


def phase3_coc_and_sor() -> None:
    ensure_dirs()
    inventory = load_inventory()
    if not inventory:
        logging.warning("Phase 3: inventory is empty; run phase1_intake_scan first.")
        return

    run_id = current_run_id()

    ledger_lines: List[str] = []
    if COC_LEDGER_JSONL.exists():
        with COC_LEDGER_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ledger_lines.append(line)

    existing_ids = {json.loads(l)["artifact_id"] for l in ledger_lines} if ledger_lines else set()

    for art in inventory:
        if art["id"] in existing_ids:
            continue
        event = {
            "artifact_id": art["id"],
            "event_ts_utc": datetime.utcnow().isoformat() + "Z",
            "event_type": "ingested",
            "actor": "LITIGATION_OS_MASTER_MONOLITH",
            "old_path": "",
            "new_path": art["abs_path"],
            "hash_before": "",
            "hash_after": art["hash_primary"],
            "notes": f"Initial ingestion run {run_id}",
        }
        ledger_lines.append(json.dumps(event))

    with COC_LEDGER_JSONL.open("w", encoding="utf-8") as f:
        f.write("\n".join(ledger_lines))

    # Build Source-of-Record map using dedup index if available.
    sor_map: Dict[str, Dict[str, Any]] = {}
    dedup_index: Dict[str, List[str]] = {}
    if DEDUP_INDEX_JSON.exists():
        with DEDUP_INDEX_JSON.open("r", encoding="utf-8") as f:
            dedup_index = json.load(f)

    logical_counter = 1
    for file_hash, art_ids in dedup_index.items():
        logical_id = f"DOC-{logical_counter:06d}"
        logical_counter += 1
        # Prefer artifact with shortest abs_path as SoR
        best_artifact_id = min(
            art_ids,
            key=lambda aid: len(next(a["abs_path"] for a in inventory if a["id"] == aid)),
        )
        sor_map[logical_id] = {
            "logical_id": logical_id,
            "hash_primary": file_hash,
            "sor_artifact_id": best_artifact_id,
            "duplicate_artifact_ids": art_ids,
        }

    with SOR_MAP_JSON.open("w", encoding="utf-8") as f:
        json.dump(sor_map, f, indent=2)

    logging.info("Phase 3: updated COC_LEDGER.jsonl and SOR_MAP.json with %d logical documents.", len(sor_map))


# ---------------------------------------------------------------------------
# Phase 4 – Normalization & Source-of-Record Builder
# ---------------------------------------------------------------------------


def classify_family(filename: str, ext: str) -> str:
    name = filename.lower()
    if "order" in name or "judgment" in name:
        return "order"
    if "motion" in name:
        return "motion"
    if "transcript" in name or "hearing" in name:
        return "transcript"
    if "report" in name or "evaluation" in name:
        return "report"
    if "ppo" in name:
        return "ppo"
    if "lease" in name or "rent" in name or "evict" in name:
        return "housing"
    if ext in SUPPORTED_TEXT_EXT:
        return "text"
    if ext in SUPPORTED_PDF_EXT:
        return "pdf_misc"
    if ext in SUPPORTED_DOCX_EXT:
        return "docx_misc"
    return "misc"


def phase4_normalization() -> None:
    ensure_dirs()
    inventory = load_inventory()
    if not inventory:
        logging.warning("Phase 4: inventory is empty; run phase1_intake_scan first.")
        return

    sor_map: Dict[str, Dict[str, Any]] = {}
    if SOR_MAP_JSON.exists():
        with SOR_MAP_JSON.open("r", encoding="utf-8") as f:
            sor_map = json.load(f)

    # Build reverse mapping artifact_id -> logical_id
    logical_by_artifact: Dict[str, str] = {}
    for logical_id, info in sor_map.items():
        for aid in info.get("duplicate_artifact_ids", []):
            logical_by_artifact[aid] = logical_id

    normalized_lines: List[str] = []
    for art in inventory:
        art_id = art["id"]
        filename = art["filename"]
        ext = art["ext"]
        logical_id = logical_by_artifact.get(art_id, f"DOC-UNBOUND-{art_id}")
        family = classify_family(filename, ext)

        meta = guess_meta_from_filename(filename)
        record = {
            "logical_id": logical_id,
            "primary_artifact_id": sor_map.get(logical_id, {}).get("sor_artifact_id", art_id),
            "family": family,
            "subtype": "",
            "case_anchor": meta["case_number_guess"],
            "date_primary": meta["date_guess"] or art["mtime_utc"],
            "parties": ";".join(meta["participants"]),
            "judge_or_author": "",
            "track_tags": ";".join(meta["track_tags"]),
            "related_artifacts": ";".join(sor_map.get(logical_id, {}).get("duplicate_artifact_ids", [])),
            "status": "active",
        }
        normalized_lines.append(json.dumps(record))

    with NORMALIZED_DOCS_JSONL.open("w", encoding="utf-8") as f:
        f.write("\n".join(normalized_lines))

    logging.info("Phase 4: wrote NORMALIZED_DOCS.jsonl with %d records.", len(normalized_lines))


def load_normalized_docs() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    if NORMALIZED_DOCS_JSONL.exists():
        with NORMALIZED_DOCS_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    docs.append(json.loads(line))
    return docs


# ---------------------------------------------------------------------------
# Evidence Store & Authority Universe helpers
# ---------------------------------------------------------------------------


def load_textmap_index() -> Dict[str, Dict[str, Any]]:
    """Return artifact_id -> {text_path, char_len, has_text}."""
    idx: Dict[str, Dict[str, Any]] = {}
    if not TEXTMAP_JSONL.exists():
        return idx
    with TEXTMAP_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            aid = rec.get("artifact_id")
            if aid:
                idx[aid] = rec
    return idx


def load_sor_map() -> Dict[str, Dict[str, Any]]:
    if not SOR_MAP_JSON.exists():
        return {}
    with SOR_MAP_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_evidence_store() -> None:
    """Build an Evidence Store keyed by logical_id and cross-indexed by case and track."""
    ensure_dirs()
    docs = load_normalized_docs()
    inventory = load_inventory()
    text_idx = load_textmap_index()
    sor_map = load_sor_map()

    inv_by_id = {a["id"]: a for a in inventory}

    by_logical: Dict[str, Dict[str, Any]] = {}
    by_case: Dict[str, List[str]] = {}
    by_track: Dict[str, List[str]] = {}

    for d in docs:
        logical_id = d.get("logical_id", "")
        if not logical_id:
            continue

        primary_artifact_id = d.get("primary_artifact_id", "")
        art = inv_by_id.get(primary_artifact_id)
        text_info = text_idx.get(primary_artifact_id, {})
        sor_info = sor_map.get(logical_id, {})

        case_id = d.get("case_anchor", "") or "UNSPECIFIED_CASE"
        tracks_raw = d.get("track_tags", "") or ""
        tracks = [t for t in tracks_raw.split(";") if t] or ["UNSPECIFIED_TRACK"]

        entry = {
            "logical_id": logical_id,
            "primary_artifact_id": primary_artifact_id,
            "case_anchor": case_id,
            "family": d.get("family", ""),
            "date_primary": d.get("date_primary", ""),
            "track_tags": tracks,
            "sor_artifact_id": sor_info.get("sor_artifact_id", primary_artifact_id),
            "duplicate_artifact_ids": sor_info.get("duplicate_artifact_ids", []),
            "abs_path": art["abs_path"] if art else "",
            "text_path": text_info.get("text_path", ""),
            "char_len": text_info.get("char_len", 0),
            "has_text": text_info.get("has_text", False),
        }
        by_logical[logical_id] = entry

        by_case.setdefault(case_id, [])
        if logical_id not in by_case[case_id]:
            by_case[case_id].append(logical_id)

        for t in tracks:
            by_track.setdefault(t, [])
            if logical_id not in by_track[t]:
                by_track[t].append(logical_id)

    evidence_store = {
        "built_at_utc": datetime.utcnow().isoformat() + "Z",
        "logical_count": len(by_logical),
        "case_count": len(by_case),
        "track_count": len(by_track),
        "by_logical_id": by_logical,
        "by_case": by_case,
        "by_track": by_track,
    }

    with EVIDENCE_STORE_JSON.open("w", encoding="utf-8") as f:
        json.dump(evidence_store, f, indent=2)

    logging.info(
        "Evidence Store built: %d logical docs, %d cases, %d tracks.", len(by_logical), len(by_case), len(by_track)
    )


def build_authority_universe() -> None:
    """Scan text for Michigan authority citations and build Authority Universe."""
    ensure_dirs()
    docs = load_normalized_docs()
    evidence_store: Dict[str, Any] = {}
    if EVIDENCE_STORE_JSON.exists():
        with EVIDENCE_STORE_JSON.open("r", encoding="utf-8") as f:
            evidence_store = json.load(f)

    by_logical = evidence_store.get("by_logical_id", {}) if evidence_store else {}

    # regex patterns for MI authority
    mcl_pattern = re.compile(r"\bMCL\s+\d{3}\.\d+[a-zA-Z0-9]*")
    mcr_pattern = re.compile(r"\bMCR\s+\d\.\d+[A-Za-z0-9]*")
    mre_pattern = re.compile(r"\bMRE\s+\d{3}\b")

    by_citation: Dict[str, Dict[str, Any]] = {}
    by_logical_id: Dict[str, List[str]] = {}
    by_case: Dict[str, List[str]] = {}

    for d in docs:
        logical_id = d.get("logical_id", "")
        if not logical_id:
            continue
        ev = by_logical.get(logical_id, {})
        text_path = ev.get("text_path", "")
        if not text_path:
            continue

        try:
            text = Path(text_path).read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        case_id = ev.get("case_anchor", "") or d.get("case_anchor", "") or "UNSPECIFIED_CASE"
        tracks = ev.get("track_tags", [])
        if not tracks:
            tracks_raw = d.get("track_tags", "") or ""
            tracks = [t for t in tracks_raw.split(";") if t] or ["UNSPECIFIED_TRACK"]

        citations_found: List[str] = []

        for patt, c_type in [
            (mcl_pattern, "MCL"),
            (mcr_pattern, "MCR"),
            (mre_pattern, "MRE"),
        ]:
            for match in patt.findall(text):
                citation = match.strip()
                citations_found.append(citation)
                info = by_citation.setdefault(
                    citation,
                    {
                        "citation": citation,
                        "type": c_type,
                        "hit_count": 0,
                        "logical_ids": [],
                        "case_ids": [],
                        "tracks": [],
                    },
                )
                info["hit_count"] += 1
                if logical_id not in info["logical_ids"]:
                    info["logical_ids"].append(logical_id)
                if case_id not in info["case_ids"]:
                    info["case_ids"].append(case_id)
                for t in tracks:
                    if t not in info["tracks"]:
                        info["tracks"].append(t)

        if citations_found:
            uniq_cites = sorted(set(citations_found))
            by_logical_id[logical_id] = uniq_cites
            by_case.setdefault(case_id, [])
            for c in uniq_cites:
                if c not in by_case[case_id]:
                    by_case[case_id].append(c)

    authority_universe = {
        "built_at_utc": datetime.utcnow().isoformat() + "Z",
        "citation_count": len(by_citation),
        "logical_with_cites": len(by_logical_id),
        "by_citation": by_citation,
        "by_logical_id": by_logical_id,
        "by_case": by_case,
    }

    with AUTHORITY_UNIVERSE_JSON.open("w", encoding="utf-8") as f:
        json.dump(authority_universe, f, indent=2)

    logging.info(
        "Authority Universe built: %d citations, %d docs with citations.", len(by_citation), len(by_logical_id)
    )


# ---------------------------------------------------------------------------
# Timeline builder
# ---------------------------------------------------------------------------


def build_timeline_from_normalized() -> List[Dict[str, Any]]:
    docs = load_normalized_docs()
    timeline: List[Dict[str, Any]] = []
    for idx, doc in enumerate(docs, start=1):
        event_id = f"EVT-{idx:06d}"
        timeline.append(
            {
                "event_id": event_id,
                "date": doc.get("date_primary", ""),
                "case_id": doc.get("case_anchor", ""),
                "track": doc.get("track_tags", ""),
                "short_label": doc.get("family", "") + " " + doc.get("logical_id", ""),
                "description": (
                    f"Document {doc.get('logical_id', '')} in family {doc.get('family', '')} "
                    f"treated as timeline event."
                ),
                "doc_refs": doc.get("logical_id", ""),
                "exh_refs": "",
                "legal_significance": doc.get("family", ""),
                "tags": doc.get("track_tags", ""),
            }
        )
    return timeline


def write_timeline_csv(events: List[Dict[str, Any]]) -> None:
    if not events:
        logging.warning("No timeline events to write.")
        return
    fieldnames = list(events[0].keys())
    with TIMELINE_MASTER_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in events:
            writer.writerow(row)
    logging.info("Wrote timeline_master.csv with %d events.", len(events))


# ---------------------------------------------------------------------------
# Graph builder (MindEye2 + Neo4j exports)
# ---------------------------------------------------------------------------


def build_graph_from_normalized_and_timeline() -> None:
    docs = load_normalized_docs()
    events: List[Dict[str, Any]] = []
    if TIMELINE_MASTER_CSV.exists():
        with TIMELINE_MASTER_CSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            events = list(reader)

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # Case nodes
    case_ids = sorted({d.get("case_anchor", "") for d in docs if d.get("case_anchor")})
    for idx, case_id in enumerate(case_ids, start=1):
        node_id = f"CASE-{idx:04d}"
        nodes.append(
            {
                "node_id": node_id,
                "node_type": "CASE",
                "label": case_id,
                "track_tags": "",
                "case_anchor": case_id,
                "date_primary": "",
                "severity": "",
                "source_logical_id": "",
                "sor_artifact_id": "",
            }
        )

    case_node_by_case_id = {n["case_anchor"]: n["node_id"] for n in nodes if n.get("case_anchor")}

    # Document nodes
    for doc in docs:
        logical_id = doc.get("logical_id", "")
        if not logical_id:
            continue
        node_id = logical_id
        nodes.append(
            {
                "node_id": node_id,
                "node_type": "DOCUMENT",
                "label": logical_id,
                "track_tags": doc.get("track_tags", ""),
                "case_anchor": doc.get("case_anchor", ""),
                "date_primary": doc.get("date_primary", ""),
                "severity": "",
                "source_logical_id": logical_id,
                "sor_artifact_id": doc.get("primary_artifact_id", ""),
            }
        )

        case_anchor = doc.get("case_anchor", "")
        if case_anchor and case_anchor in case_node_by_case_id:
            edges.append(
                {
                    "edge_id": f"EDG-CASE-DOC-{logical_id}",
                    "src": case_node_by_case_id[case_anchor],
                    "dst": node_id,
                    "edge_type": "CASE_HAS_DOCUMENT",
                    "weight": 1,
                    "track_tags": doc.get("track_tags", ""),
                    "notes": "",
                }
            )

    # Event nodes
    for ev in events:
        evt_id = ev.get("event_id", "")
        if not evt_id:
            continue
        node_id = evt_id
        nodes.append(
            {
                "node_id": node_id,
                "node_type": "EVENT",
                "label": ev.get("short_label", ""),
                "track_tags": ev.get("tags", ""),
                "case_anchor": ev.get("case_id", ""),
                "date_primary": ev.get("date", ""),
                "severity": "",
                "source_logical_id": ev.get("doc_refs", ""),
                "sor_artifact_id": "",
            }
        )

        doc_ref = ev.get("doc_refs", "")
        if doc_ref:
            edges.append(
                {
                    "edge_id": f"EDG-EVENT-DOC-{evt_id}",
                    "src": node_id,
                    "dst": doc_ref,
                    "edge_type": "EVENT_RELATES_TO_DOCUMENT",
                    "weight": 1,
                    "track_tags": ev.get("tags", ""),
                    "notes": "",
                }
            )

    # Write nodes/edges CSV
    if nodes:
        with NODES_CSV.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(nodes[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in nodes:
                writer.writerow(row)
    if edges:
        with EDGES_CSV.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(edges[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in edges:
                writer.writerow(row)

    logging.info("Graph build complete: %d nodes, %d edges.", len(nodes), len(edges))


def build_neo4j_exports_from_graph() -> None:
    """Read MindEye2 nodes/edges and emit Neo4j-ready CSVs."""
    ensure_dirs()

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    if NODES_CSV.exists():
        with NODES_CSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            nodes = list(reader)
    if EDGES_CSV.exists():
        with EDGES_CSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            edges = list(reader)

    neo_nodes: List[Dict[str, Any]] = []
    for n in nodes:
        node_id = n.get("node_id", "")
        node_type = n.get("node_type", "") or "NODE"
        label = {
            "CASE": "Case",
            "DOCUMENT": "Document",
            "EVENT": "Event",
        }.get(node_type, "Node")
        neo_nodes.append(
            {
                "nodeId:ID": node_id,
                "labels:LABEL": label,
                "name": n.get("label", node_id),
                "node_type": node_type,
                "case_anchor": n.get("case_anchor", ""),
                "track_tags": n.get("track_tags", ""),
                "date_primary": n.get("date_primary", ""),
                "severity": n.get("severity", ""),
            }
        )

    if neo_nodes:
        with NEO4J_NODES_CSV.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(neo_nodes[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in neo_nodes:
                writer.writerow(row)

    neo_rels: List[Dict[str, Any]] = []
    for e in edges:
        src = e.get("src", "")
        dst = e.get("dst", "")
        etype = e.get("edge_type", "") or "RELATED_TO"
        weight_val = e.get("weight", "1")
        try:
            weight_int = int(weight_val)
        except Exception:
            weight_int = 1
        neo_rels.append(
            {
                ":START_ID": src,
                ":END_ID": dst,
                ":TYPE": etype,
                "weight:long": weight_int,
                "track_tags": e.get("track_tags", ""),
                "notes": e.get("notes", ""),
            }
        )

    if neo_rels:
        with NEO4J_RELS_CSV.open("w", encoding="utf-8", newline="") as f:
            fieldnames = list(neo_rels[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in neo_rels:
                writer.writerow(row)

    logging.info(
        "Neo4j exports written: %s (%d nodes), %s (%d relationships)",
        NEO4J_NODES_CSV,
        len(neo_nodes),
        NEO4J_RELS_CSV,
        len(neo_rels),
    )


# ---------------------------------------------------------------------------
# Lint rules & violations table
# ---------------------------------------------------------------------------


def run_lint_checks() -> None:
    """Apply a small set of concrete lint rules to the vault and outputs."""
    results: List[Dict[str, Any]] = []

    if not INVENTORY_JSON.exists():
        results.append(
            {
                "rule_id": "LV-P1-INVENTORY-001",
                "severity": "critical",
                "description": "INVENTORY.json missing; intake has not been run.",
                "recommended_actions": ["Run phase1_intake_scan()"],
            }
        )

    if INVENTORY_JSON.exists() and not TEXTMAP_JSONL.exists():
        results.append(
            {
                "rule_id": "LV-P2-TEXTMAP-001",
                "severity": "warning",
                "description": "TEXTMAP.jsonl missing; text extraction has not been run.",
                "recommended_actions": ["Run phase2_text_and_metadata()"],
            }
        )

    if INVENTORY_JSON.exists() and not SOR_MAP_JSON.exists():
        results.append(
            {
                "rule_id": "LV-P3-SOR-001",
                "severity": "warning",
                "description": "SOR_MAP.json missing; chain-of-custody and SoR selection not recorded.",
                "recommended_actions": ["Run phase3_coc_and_sor()"],
            }
        )

    if INVENTORY_JSON.exists() and not NORMALIZED_DOCS_JSONL.exists():
        results.append(
            {
                "rule_id": "LV-P4-NORM-001",
                "severity": "warning",
                "description": "NORMALIZED_DOCS.jsonl missing; normalization phase not run.",
                "recommended_actions": ["Run phase4_normalization()"],
            }
        )

    if NORMALIZED_DOCS_JSONL.exists() and not EVIDENCE_STORE_JSON.exists():
        results.append(
            {
                "rule_id": "LV-EVIDENCE-001",
                "severity": "info",
                "description": "EVIDENCE_STORE.json missing; evidence store not built.",
                "recommended_actions": ["Run evidence store builder."],
            }
        )

    if NORMALIZED_DOCS_JSONL.exists() and not AUTHORITY_UNIVERSE_JSON.exists():
        results.append(
            {
                "rule_id": "LV-AUTH-001",
                "severity": "info",
                "description": "AUTHORITY_UNIVERSE.json missing; authority universe not built.",
                "recommended_actions": ["Run authority universe builder."],
            }
        )

    if NORMALIZED_DOCS_JSONL.exists() and not TIMELINE_MASTER_CSV.exists():
        results.append(
            {
                "rule_id": "LV-TIMELINE-001",
                "severity": "info",
                "description": "Timeline CSV is missing; can be built from normalized documents.",
                "recommended_actions": ["Build timeline from normalized docs and write timeline_master.csv"],
            }
        )

    if NORMALIZED_DOCS_JSONL.exists() and not (NODES_CSV.exists() and EDGES_CSV.exists()):
        results.append(
            {
                "rule_id": "LV-GRAPH-001",
                "severity": "info",
                "description": "Graph CSVs are missing; build graph from normalized docs and timeline.",
                "recommended_actions": ["Run build_graph_from_normalized_and_timeline()"],
            }
        )

    with LINT_RESULTS_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logging.info("Lint checks complete. %d issues recorded.", len(results))


def tool_violations_table() -> None:
    """Convert lint_results.json into a CSV for tracking remediation."""
    if not LINT_RESULTS_JSON.exists():
        logging.warning("lint_results.json missing; run lint first.")
        return
    try:
        with LINT_RESULTS_JSON.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logging.exception("Failed to read lint_results.json: %s", exc)
        return
    if not isinstance(data, list):
        logging.warning("lint_results.json is not a list; skipping.")
        return
    out_path = OUTPUT_DIR / "violations_todo.csv"
    rows: List[Dict[str, Any]] = []
    for item in data:
        rows.append(
            {
                "rule_id": item.get("rule_id", ""),
                "severity": item.get("severity", ""),
                "description": item.get("description", ""),
                "recommended_actions": "; ".join(item.get("recommended_actions", [])),
            }
        )
    if rows:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["rule_id", "severity", "description", "recommended_actions"])
            writer.writeheader()
    logging.info("Wrote violations_todo.csv with %d rows.", len(rows))


# ---------------------------------------------------------------------------
# Pack builder
# ---------------------------------------------------------------------------


def build_zip_pack(pack_name: Optional[str] = None) -> Path:
    ensure_dirs()
    if pack_name is None:
        pack_name = f"LITIGATION_PACK_{current_run_id()}.zip"
    pack_path = PACKS_DIR / pack_name

    files_to_add: List[Path] = []
    for p in [
        INVENTORY_JSON,
        INVENTORY_CSV,
        TEXTMAP_JSONL,
        DOC_META_CSV,
        COC_LEDGER_JSONL,
        SOR_MAP_JSON,
        NORMALIZED_DOCS_JSONL,
        DEDUP_INDEX_JSON,
        EVIDENCE_STORE_JSON,
        AUTHORITY_UNIVERSE_JSON,
        TIMELINE_MASTER_CSV,
        NODES_CSV,
        EDGES_CSV,
        LINT_RESULTS_JSON,
        RUN_MANIFEST_JSON,
    ]:
        if p.exists():
            files_to_add.append(p)

    for folder in [VAULT_DIR, OUTPUT_DIR]:
        if not folder.exists():
            continue
        for path in folder.rglob("*"):
            if path.is_file() and path not in files_to_add:
                files_to_add.append(path)

    with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files_to_add:
            arcname = safe_relpath(f, BASE_DIR)
            z.write(f, arcname)

    logging.info("Built pack: %s with %d files.", pack_path, len(files_to_add))
    return pack_path


# ---------------------------------------------------------------------------
# Helper tools: status, index, affidavit, search
# ---------------------------------------------------------------------------


def tool_status_report() -> None:
    """Write a simple status report summarizing vault and outputs."""
    ensure_dirs()
    inv = load_inventory()
    docs = load_normalized_docs()
    lint_results: List[Dict[str, Any]] = []
    if LINT_RESULTS_JSON.exists():
        try:
            with LINT_RESULTS_JSON.open("r", encoding="utf-8") as f:
                lint_results = json.load(f)
        except Exception as exc:
            logging.exception("Failed to read lint results: %s", exc)

    evidence_count = 0
    authority_cites = 0
    if EVIDENCE_STORE_JSON.exists():
        try:
            with EVIDENCE_STORE_JSON.open("r", encoding="utf-8") as f:
                es = json.load(f)
            evidence_count = len(es.get("by_logical_id", {}))
        except Exception:
            pass
    if AUTHORITY_UNIVERSE_JSON.exists():
        try:
            with AUTHORITY_UNIVERSE_JSON.open("r", encoding="utf-8") as f:
                au = json.load(f)
            authority_cites = len(au.get("by_citation", {}))
        except Exception:
            pass

    report_path = OUTPUT_DIR / "STATUS_REPORT.txt"
    lines: List[str] = []
    lines.append(f"Status report generated at {datetime.utcnow().isoformat()}Z")
    lines.append("")
    lines.append(f"Base dir: {BASE_DIR}")
    lines.append(f"Inventory artifacts: {len(inv)}")
    lines.append(f"Normalized docs: {len(docs)}")
    lines.append(f"Evidence Store logical docs: {evidence_count}")
    lines.append(f"Authority Universe citations: {authority_cites}")
    lines.append(f"Lint issues: {len(lint_results)}")
    if lint_results:
        lines.append("")
        lines.append("Top lint issues:")
        for r in lint_results[:10]:
            lines.append(f"- [{r.get('severity','')}] {r.get('rule_id','')}: {r.get('description','')}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Wrote status report: %s", report_path)


def tool_case_track_index() -> None:
    """Build a case/track index CSV from normalized docs."""
    docs = load_normalized_docs()
    index: Dict[tuple, int] = {}
    for d in docs:
        case_id = d.get("case_anchor", "") or "UNSPECIFIED_CASE"
        tracks_raw = d.get("track_tags", "") or ""
        tracks = [t for t in tracks_raw.split(";") if t] or ["UNSPECIFIED_TRACK"]
        for t in tracks:
            key = (case_id, t)
            index[key] = index.get(key, 0) + 1
    rows: List[Dict[str, Any]] = []
    for (case_id, track), count in sorted(index.items()):
        rows.append({"case_id": case_id, "track": track, "doc_count": count})
    out_path = OUTPUT_DIR / "case_track_index.csv"
    if rows:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id", "track", "doc_count"])
            writer.writeheader()
    logging.info("Wrote case_track_index.csv with %d rows.", len(rows))


def tool_affidavit_blocks() -> None:
    """Generate simple affidavit-ready fact stubs from normalized docs."""
    docs = load_normalized_docs()
    out_path = OUTPUT_DIR / "affidavit_blocks.txt"
    lines: List[str] = []
    lines.append("# Affidavit Fact Stubs")
    lines.append(f"# Generated at {datetime.utcnow().isoformat()}Z")
    lines.append("")
    for d in docs:
        date = d.get("date_primary", "")
        family = d.get("family", "")
        logical_id = d.get("logical_id", "")
        case_id = d.get("case_anchor", "")
        line = (
            f"- On {date or '[no date detected]'}, a {family or 'document'} "
            f"identified internally as {logical_id} exists in case {case_id or '[no case anchor]'}."
        )
        lines.append(line)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Wrote affidavit_blocks.txt with %d lines.", len(lines))


def slugify_query(query: str) -> str:
    s = query.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s or "query"


def tool_search_text(query: str) -> Path:
    """Search extracted text for a substring; write results to a text file."""
    ensure_dirs()
    results: List[str] = []
    if not TEXTMAP_JSONL.exists():
        logging.warning("TEXTMAP.jsonl missing; run phase2_text_and_metadata first.")
    else:
        with TEXTMAP_JSONL.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text_path = rec.get("text_path")
                if not text_path:
                    continue
                try:
                    t = Path(text_path).read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                if query.lower() in t.lower():
                    results.append(f"{rec.get('artifact_id','')} -> {text_path}")
    slug = slugify_query(query)
    out_path = OUTPUT_DIR / f"search_{slug}.txt"
    lines = [f'Search query: "{query}"', ""] + results
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Search for %r produced %d hits. Results in %s", query, len(results), out_path)
    return out_path


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


def build_run_manifest() -> None:
    """Build a JSON manifest summarizing current core artifacts."""
    manifest = {
        "built_at_utc": datetime.utcnow().isoformat() + "Z",
        "base_dir": str(BASE_DIR),
        "inventory_exists": INVENTORY_JSON.exists(),
        "normalized_exists": NORMALIZED_DOCS_JSONL.exists(),
        "evidence_store_exists": EVIDENCE_STORE_JSON.exists(),
        "authority_universe_exists": AUTHORITY_UNIVERSE_JSON.exists(),
        "timeline_exists": TIMELINE_MASTER_CSV.exists(),
        "graph_exists": NODES_CSV.exists() and EDGES_CSV.exists(),
        "lint_exists": LINT_RESULTS_JSON.exists(),
        "paths": {
            "inventory_json": str(INVENTORY_JSON),
            "normalized_docs": str(NORMALIZED_DOCS_JSONL),
            "evidence_store": str(EVIDENCE_STORE_JSON),
            "authority_universe": str(AUTHORITY_UNIVERSE_JSON),
            "timeline": str(TIMELINE_MASTER_CSV),
            "nodes_csv": str(NODES_CSV),
            "edges_csv": str(EDGES_CSV),
            "lint_results": str(LINT_RESULTS_JSON),
        },
    }
    with RUN_MANIFEST_JSON.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logging.info("Run manifest written: %s", RUN_MANIFEST_JSON)


# ---------------------------------------------------------------------------
# HTML dashboard
# ---------------------------------------------------------------------------


def build_html_dashboard() -> None:
    """Build a rich HTML dashboard in output/index.html."""
    ensure_dirs()
    cfg = load_config()
    inv = load_inventory()
    docs = load_normalized_docs()

    lint_data: List[Dict[str, Any]] = []
    if LINT_RESULTS_JSON.exists():
        try:
            with LINT_RESULTS_JSON.open("r", encoding="utf-8") as f:
                lint_data = json.load(f)
        except Exception:
            lint_data = []

    case_track_rows: List[Dict[str, Any]] = []
    case_track_path = OUTPUT_DIR / "case_track_index.csv"
    if case_track_path.exists():
        with case_track_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            case_track_rows = list(reader)

    timeline_rows: List[Dict[str, Any]] = []
    if TIMELINE_MASTER_CSV.exists():
        with TIMELINE_MASTER_CSV.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            timeline_rows = list(reader)

    affidavit_path = OUTPUT_DIR / "affidavit_blocks.txt"
    affidavit_lines: List[str] = []
    if affidavit_path.exists():
        affidavit_lines = affidavit_path.read_text(encoding="utf-8", errors="replace").splitlines()

    evidence_store = {}
    if EVIDENCE_STORE_JSON.exists():
        try:
            with EVIDENCE_STORE_JSON.open("r", encoding="utf-8") as f:
                evidence_store = json.load(f)
        except Exception:
            evidence_store = {}

    authority_universe = {}
    if AUTHORITY_UNIVERSE_JSON.exists():
        try:
            with AUTHORITY_UNIVERSE_JSON.open("r", encoding="utf-8") as f:
                authority_universe = json.load(f)
        except Exception:
            authority_universe = {}

    total_artifacts = len(inv)
    total_docs = len(docs)
    family_counts: Dict[str, int] = {}
    for d in docs:
        fam = d.get("family", "") or "(unspecified)"
        family_counts[fam] = family_counts.get(fam, 0) + 1

    lint_by_severity: Dict[str, int] = {}
    for item in lint_data:
        sev = item.get("severity", "") or "(unspecified)"
        lint_by_severity[sev] = lint_by_severity.get(sev, 0) + 1

    evidence_count = len(evidence_store.get("by_logical_id", {})) if evidence_store else 0
    evidence_case_count = len(evidence_store.get("by_case", {})) if evidence_store else 0
    auth_citation_count = len(authority_universe.get("by_citation", {})) if authority_universe else 0
    auth_doc_count = len(authority_universe.get("by_logical_id", {})) if authority_universe else 0

    packs: List[Path] = []
    if PACKS_DIR.exists():
        for p in PACKS_DIR.glob("*.zip"):
            packs.append(p)

    now_iso = datetime.utcnow().isoformat() + "Z"

    html_lines: List[str] = []
    html_lines.append("<!DOCTYPE html>")
    html_lines.append("<html lang='en'>")
    html_lines.append("<head>")
    html_lines.append("  <meta charset='utf-8'>")
    html_lines.append("  <title>Litigation OS Dashboard</title>")
    html_lines.append("  <style>")
    html_lines.append(
        "    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 0; background: #0f172a; color: #e5e7eb; }"
    )
    html_lines.append("    header { padding: 16px 24px; background: #111827; border-bottom: 1px solid #1f2937; }")
    html_lines.append("    h1 { margin: 0; font-size: 24px; }")
    html_lines.append(
        "    h2 { margin-top: 24px; font-size: 20px; border-bottom: 1px solid #1f2937; padding-bottom: 4px; }"
    )
    html_lines.append("    main { padding: 16px 24px 40px 24px; }")
    html_lines.append(
        "    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); grid-gap: 16px; }"
    )
    html_lines.append(
        "    .card { background: #020617; border-radius: 12px; padding: 12px 16px; border: 1px solid #1f2937; box-shadow: 0 8px 18px rgba(0,0,0,0.35); }"
    )
    html_lines.append("    .card h3 { margin-top: 0; font-size: 16px; }")
    html_lines.append("    table { width: 100%; border-collapse: collapse; font-size: 13px; }")
    html_lines.append("    th, td { border-bottom: 1px solid #111827; padding: 4px 6px; text-align: left; }")
    html_lines.append("    th { background: #020617; position: sticky; top: 0; }")
    html_lines.append("    a { color: #38bdf8; text-decoration: none; }")
    html_lines.append("    a:hover { text-decoration: underline; }")
    html_lines.append("    code { background: #020617; padding: 2px 4px; border-radius: 4px; font-size: 12px; }")
    html_lines.append(
        "    .pill { display: inline-block; padding: 2px 6px; border-radius: 999px; font-size: 11px; background: #0f172a; border: 1px solid #1f2937; margin-right: 4px; }"
    )
    html_lines.append("    .pill-critical { border-color: #f97373; color: #fecaca; }")
    html_lines.append("    .pill-warning { border-color: #facc15; color: #fef9c3; }")
    html_lines.append("    .pill-info { border-color: #38bdf8; color: #e0f2fe; }")
    html_lines.append(
        "    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; }"
    )
    html_lines.append("  </style>")
    html_lines.append("</head>")
    html_lines.append("<body>")
    html_lines.append("  <header>")
    html_lines.append("    <h1>Litigation OS Dashboard</h1>")
    html_lines.append(
        f"    <div style='margin-top:4px;font-size:13px;'>Run at <span class='mono'>{now_iso}</span> • Base dir: <span class='mono'>{BASE_DIR}</span></div>"
    )
    html_lines.append("  </header>")
    html_lines.append("  <main>")

    # Environment & config
    html_lines.append("    <section>")
    html_lines.append("      <h2>1. Environment & Config</h2>")
    html_lines.append("      <div class='grid'>")
    html_lines.append("        <div class='card'>")
    html_lines.append("          <h3>Base Paths</h3>")
    html_lines.append(f"          <div class='mono'>BASE_DIR = {BASE_DIR}</div>")
    html_lines.append(f"          <div class='mono'>VAULT_DIR = {VAULT_DIR}</div>")
    html_lines.append(f"          <div class='mono'>OUTPUT_DIR = {OUTPUT_DIR}</div>")
    html_lines.append(f"          <div class='mono'>PACKS_DIR = {PACKS_DIR}</div>")
    html_lines.append("        </div>")
    html_lines.append("        <div class='card'>")
    html_lines.append("          <h3>Config Snapshot</h3>")
    if cfg:
        cfg_str = json.dumps(cfg, indent=2)
        html_lines.append(
            "          <pre class='mono' style='white-space:pre-wrap;font-size:11px;'>" + cfg_str + "</pre>"
        )
    else:
        html_lines.append("          <div style='font-size:13px;'>No config.json loaded. Defaults in effect.</div>")
    html_lines.append("        </div>")
    html_lines.append("      </div>")
    html_lines.append("    </section>")

    # Inventory & docs & evidence/authority metrics
    html_lines.append("    <section>")
    html_lines.append("      <h2>2. Inventory, Evidence & Authority Metrics</h2>")
    html_lines.append("      <div class='grid'>")
    html_lines.append("        <div class='card'>")
    html_lines.append("          <h3>Core Counts</h3>")
    html_lines.append(f"          <div>Total artifacts in inventory: <strong>{total_artifacts}</strong></div>")
    html_lines.append(f"          <div>Total normalized logical docs: <strong>{total_docs}</strong></div>")
    html_lines.append(f"          <div>Evidence Store logical docs: <strong>{evidence_count}</strong></div>")
    html_lines.append(f"          <div>Evidence Store cases: <strong>{evidence_case_count}</strong></div>")
    html_lines.append(f"          <div>Authority citations tracked: <strong>{auth_citation_count}</strong></div>")
    html_lines.append(f"          <div>Docs with citations: <strong>{auth_doc_count}</strong></div>")
    html_lines.append("        </div>")
    html_lines.append("        <div class='card'>")
    html_lines.append("          <h3>Families (normalized docs)</h3>")
    if family_counts:
        html_lines.append("          <table>")
        html_lines.append("            <tr><th>Family</th><th>Count</th></tr>")
        for fam, cnt in sorted(family_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            html_lines.append(f"            <tr><td>{fam}</td><td>{cnt}</td></tr>")
        html_lines.append("          </table>")
    else:
        html_lines.append("          <div>No normalized documents yet.</div>")
    html_lines.append("        </div>")
    html_lines.append("      </div>")
    html_lines.append("    </section>")

    # Case/track index
    html_lines.append("    <section>")
    html_lines.append("      <h2>3. Case / Track Index</h2>")
    if case_track_rows:
        html_lines.append("      <div class='card'>")
        html_lines.append(
            "        <div style='margin-bottom:6px;'><a href='case_track_index.csv'>Open full case_track_index.csv</a></div>"
        )
        html_lines.append("        <table>")
        html_lines.append("          <tr><th>Case ID</th><th>Track</th><th>Doc count</th></tr>")
        for row in case_track_rows[:50]:
            html_lines.append(
                f"          <tr><td>{row.get('case_id','')}</td><td>{row.get('track','')}</td><td>{row.get('doc_count','')}</td></tr>"
            )
        html_lines.append("        </table>")
        html_lines.append("      </div>")
    else:
        html_lines.append(
            "      <div class='card'>No case_track_index.csv yet. Run the 'index' tool after normalization.</div>"
        )
    html_lines.append("    </section>")

    # Timeline preview
    html_lines.append("    <section>")
    html_lines.append("      <h2>4. Timeline Preview</h2>")
    if timeline_rows:
        html_lines.append("      <div class='card'>")
        html_lines.append(
            "        <div style='margin-bottom:6px;'><a href='timeline_master.csv'>Open full timeline_master.csv</a></div>"
        )
        html_lines.append("        <table>")
        html_lines.append("          <tr><th>Date</th><th>Case</th><th>Track</th><th>Label</th></tr>")
        for row in timeline_rows[:50]:
            html_lines.append(
                f"          <tr><td>{row.get('date','')}</td><td>{row.get('case_id','')}</td><td>{row.get('track','')}</td><td>{row.get('short_label','')}</td></tr>"
            )
        html_lines.append("        </table>")
        html_lines.append("      </div>")
    else:
        html_lines.append(
            "      <div class='card'>Timeline not yet built. Run 'timeline' or 'all' after normalization.</div>"
        )
    html_lines.append("    </section>")

    # Lint summary
    html_lines.append("    <section>")
    html_lines.append("      <h2>5. Lint / Violation Summary</h2>")
    html_lines.append("      <div class='grid'>")
    html_lines.append("        <div class='card'>")
    html_lines.append("          <h3>Counts by Severity</h3>")
    if lint_by_severity:
        for sev, cnt in sorted(lint_by_severity.items(), key=lambda kv: (-kv[1], kv[0])):
            cls = "pill"
            if sev.lower() == "critical":
                cls += " pill-critical"
            elif sev.lower() == "warning":
                cls += " pill-warning"
            else:
                cls += " pill-info"
            html_lines.append(f"          <div class='{cls}'>{sev}: {cnt}</div>")
    else:
        html_lines.append("          <div>No lint results yet. Run 'lint' or 'all'.</div>")
    html_lines.append("        </div>")

    html_lines.append("        <div class='card'>")
    html_lines.append("          <h3>Sample Issues</h3>")
    if lint_data:
        html_lines.append("          <table>")
        html_lines.append("            <tr><th>Severity</th><th>Rule</th><th>Description</th></tr>")
        for item in lint_data[:20]:
            sev = item.get("severity", "")
            rule_id = item.get("rule_id", "")
            desc = item.get("description", "")
            html_lines.append(f"            <tr><td>{sev}</td><td>{rule_id}</td><td>{desc}</td></tr>")
        html_lines.append("          </table>")
        html_lines.append(
            "          <div style='margin-top:6px;'><a href='lint_results.json'>Open lint_results.json</a> • <a href='violations_todo.csv'>Open violations_todo.csv</a></div>"
        )
    else:
        html_lines.append("          <div>No lint issues recorded yet.</div>")
    html_lines.append("        </div>")
    html_lines.append("      </div>")
    html_lines.append("    </section>")

    # Packs & key files
    html_lines.append("    <section>")
    html_lines.append("      <h2>6. Packs & Key Files</h2>")
    html_lines.append("      <div class='grid'>")
    html_lines.append("        <div class='card'>")
    html_lines.append("          <h3>Litigation Packs</h3>")
    if packs:
        html_lines.append("          <ul>")
        for p in sorted(packs):
            rel = safe_relpath(p, OUTPUT_DIR.parent)
            html_lines.append(f"            <li><a href='../{rel}'>{p.name}</a></li>")
        html_lines.append("          </ul>")
    else:
        html_lines.append("          <div>No packs built yet. Run 'pack' or 'all'.</div>")
    html_lines.append("        </div>")

    html_lines.append("        <div class='card'>")
    html_lines.append("          <h3>Core CSVs & JSON</h3>")
    html_lines.append("          <ul>")
    html_lines.append("            <li><a href='../vault/INVENTORY.csv'>vault/INVENTORY.csv</a></li>")
    html_lines.append("            <li><a href='../vault/NORMALIZED_DOCS.jsonl'>vault/NORMALIZED_DOCS.jsonl</a></li>")
    html_lines.append("            <li><a href='../vault/EVIDENCE_STORE.json'>vault/EVIDENCE_STORE.json</a></li>")
    html_lines.append(
        "            <li><a href='../vault/AUTHORITY_UNIVERSE.json'>vault/AUTHORITY_UNIVERSE.json</a></li>"
    )
    html_lines.append("            <li><a href='mindeye2_nodes.csv'>output/mindeye2_nodes.csv</a></li>")
    html_lines.append("            <li><a href='mindeye2_edges.csv'>output/mindeye2_edges.csv</a></li>")
    html_lines.append("            <li><a href='neo4j_nodes.csv'>output/neo4j_nodes.csv</a></li>")
    html_lines.append("            <li><a href='neo4j_relationships.csv'>output/neo4j_relationships.csv</a></li>")
    html_lines.append("          </ul>")
    html_lines.append("        </div>")
    html_lines.append("      </div>")
    html_lines.append("    </section>")

    # Affidavit preview
    html_lines.append("    <section>")
    html_lines.append("      <h2>7. Affidavit Fact Stubs</h2>")
    html_lines.append("      <div class='card'>")
    if affidavit_lines:
        html_lines.append(
            "        <div style='margin-bottom:6px;'><a href='affidavit_blocks.txt'>Open affidavit_blocks.txt</a></div>"
        )
        html_lines.append(
            "        <pre class='mono' style='white-space:pre-wrap;font-size:11px;max-height:260px;overflow:auto;'>"
        )
        for line in affidavit_lines[:80]:
            safe_line = line.replace("<", "&lt;").replace(">", "&gt;")
            html_lines.append(safe_line)
        html_lines.append("        </pre>")
    else:
        html_lines.append("        <div>No affidavit_blocks.txt yet. Run the 'affidavit' tool.</div>")
    html_lines.append("      </div>")
    html_lines.append("    </section>")

    # Neo4j import instructions
    html_lines.append("    <section>")
    html_lines.append("      <h2>8. Neo4j Nucleus Wheel Import</h2>")
    html_lines.append("      <div class='card'>")
    html_lines.append(
        "        <p>Use these files with Neo4j's <code>LOAD CSV</code> to build the nucleus wheel graph:</p>"
    )
    html_lines.append("        <ul>")
    html_lines.append("          <li><a href='neo4j_nodes.csv'>neo4j_nodes.csv</a> (nodes)</li>")
    html_lines.append(
        "          <li><a href='neo4j_relationships.csv'>neo4j_relationships.csv</a> (relationships)</li>"
    )
    html_lines.append("        </ul>")
    cypher_example = (
        "LOAD CSV WITH HEADERS FROM 'file:///neo4j_nodes.csv' AS row\n"
        "MERGE (n:Node {id: row.`nodeId:ID`})\n"
        "SET n.name = row.name,\n"
        "    n.node_type = row.node_type,\n"
        "    n.case_anchor = row.case_anchor,\n"
        "    n.track_tags = row.track_tags,\n"
        "    n.date_primary = row.date_primary,\n"
        "    n.severity = row.severity;\n\n"
        "LOAD CSV WITH HEADERS FROM 'file:///neo4j_relationships.csv' AS row\n"
        "MATCH (s:Node {id: row.`:START_ID`}),\n"
        "      (t:Node {id: row.`:END_ID`})\n"
        "MERGE (s)-[r:REL {type: row.`:TYPE`}]->(t)\n"
        "SET r.weight = toInteger(row.`weight:long`),\n"
        "    r.track_tags = row.track_tags,\n"
        "    r.notes = row.notes;"
    )
    html_lines.append(
        "        <pre class='mono' style='white-space:pre-wrap;font-size:11px;'>" + cypher_example + "</pre>"
    )
    html_lines.append("      </div>")
    html_lines.append("    </section>")

    # Command cheat sheet
    html_lines.append("    <section>")
    html_lines.append("      <h2>9. Command Cheat Sheet</h2>")
    html_lines.append("      <div class='card'>")
    cmd_cheat = (
        "python LITIGATION_OS_MASTER_MONOLITH.py all\n"
        "python LITIGATION_OS_MASTER_MONOLITH.py phase1\n"
        "python LITIGATION_OS_MASTER_MONOLITH.py phase2\n"
        "python LITIGATION_OS_MASTER_MONOLITH.py phase3\n"
        "python LITIGATION_OS_MASTER_MONOLITH.py phase4\n"
        "python LITIGATION_OS_MASTER_MONOLITH.py evidence\n"
        "python LITIGATION_OS_MASTER_MONOLITH.py authorities\n"
        "python LITIGATION_OS_MASTER_MONOLITH.py timeline\n"
        "python LITIGATION_OS_MASTER_MONOLITH.py graph\n"
        "python LITIGATION_OS_MASTER_MONOLITH.py neo4j\n"
        "python LITIGATION_OS_MASTER_MONOLITH.py lint\n"
        "python LITIGATION_OS_MASTER_MONOLITH.py pack [pack_name.zip]\n"
        'python LITIGATION_OS_MASTER_MONOLITH.py search "query text"\n'
        "python LITIGATION_OS_MASTER_MONOLITH.py status\n"
        "python LITIGATION_OS_MASTER_MONOLITH.py affidavit\n"
        "python LITIGATION_OS_MASTER_MONOLITH.py index\n"
        "python LITIGATION_OS_MASTER_MONOLITH.py violations\n"
        "python LITIGATION_OS_MASTER_MONOLITH.py runmeta\n"
        "python LITIGATION_OS_MASTER_MONOLITH.py dashboard\n"
    )
    html_lines.append("        <pre class='mono' style='white-space:pre-wrap;font-size:11px;'>" + cmd_cheat + "</pre>")
    html_lines.append("      </div>")
    html_lines.append("    </section>")

    html_lines.append("  </main>")
    html_lines.append("</body>")
    html_lines.append("</html>")

    out_path = OUTPUT_DIR / "index.html"
    out_path.write_text("\n".join(html_lines), encoding="utf-8")
    logging.info("Wrote HTML dashboard: %s", out_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def print_usage() -> None:
    print("Usage:")
    print("  python LITIGATION_OS_MASTER_MONOLITH.py all")
    print("  python LITIGATION_OS_MASTER_MONOLITH.py phase1")
    print("  python LITIGATION_OS_MASTER_MONOLITH.py phase2")
    print("  python LITIGATION_OS_MASTER_MONOLITH.py phase3")
    print("  python LITIGATION_OS_MASTER_MONOLITH.py phase4")
    print("  python LITIGATION_OS_MASTER_MONOLITH.py evidence")
    print("  python LITIGATION_OS_MASTER_MONOLITH.py authorities")
    print("  python LITIGATION_OS_MASTER_MONOLITH.py timeline")
    print("  python LITIGATION_OS_MASTER_MONOLITH.py graph")
    print("  python LITIGATION_OS_MASTER_MONOLITH.py neo4j")
    print("  python LITIGATION_OS_MASTER_MONOLITH.py lint")
    print("  python LITIGATION_OS_MASTER_MONOLITH.py pack [pack_name.zip]")
    print('  python LITIGATION_OS_MASTER_MONOLITH.py search "query text"')
    print("  python LITIGATION_OS_MASTER_MONOLITH.py status")
    print("  python LITIGATION_OS_MASTER_MONOLITH.py affidavit")
    print("  python LITIGATION_OS_MASTER_MONOLITH.py index")
    print("  python LITIGATION_OS_MASTER_MONOLITH.py violations")
    print("  python LITIGATION_OS_MASTER_MONOLITH.py runmeta")
    print("  python LITIGATION_OS_MASTER_MONOLITH.py dashboard")


def main(argv: Optional[List[str]] = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    setup_logging()
    ensure_dirs()

    if not argv:
        print_usage()
        return

    cmd = argv[0].lower()

    if cmd == "phase1":
        phase1_intake_scan()
    elif cmd == "phase2":
        phase2_text_and_metadata()
    elif cmd == "phase3":
        phase3_coc_and_sor()
    elif cmd == "phase4":
        phase4_normalization()
    elif cmd == "evidence":
        build_evidence_store()
    elif cmd == "authorities":
        build_authority_universe()
    elif cmd == "timeline":
        events = build_timeline_from_normalized()
        write_timeline_csv(events)
    elif cmd == "graph":
        build_graph_from_normalized_and_timeline()
    elif cmd == "neo4j":
        build_neo4j_exports_from_graph()
    elif cmd == "lint":
        run_lint_checks()
    elif cmd == "pack":
        pack_name = argv[1] if len(argv) > 1 else None
        pack_path = build_zip_pack(pack_name)
        print("Pack built:", pack_path)
    elif cmd == "search":
        if len(argv) < 2:
            print("Please provide a search query.")
        else:
            out_path = tool_search_text(" ".join(argv[1:]))
            print("Search results written to:", out_path)
    elif cmd == "status":
        tool_status_report()
    elif cmd == "affidavit":
        tool_affidavit_blocks()
    elif cmd == "index":
        tool_case_track_index()
    elif cmd == "violations":
        tool_violations_table()
    elif cmd == "runmeta":
        build_run_manifest()
    elif cmd == "dashboard":
        build_html_dashboard()
    elif cmd == "all":
        phase1_intake_scan()
        phase2_text_and_metadata()
        phase3_coc_and_sor()
        phase4_normalization()
        build_evidence_store()
        build_authority_universe()
        events = build_timeline_from_normalized()
        write_timeline_csv(events)
        build_graph_from_normalized_and_timeline()
        build_neo4j_exports_from_graph()
        run_lint_checks()
        build_run_manifest()
        pack_path = build_zip_pack()
        print("Full pipeline complete. Pack:", pack_path)
    else:
        print_usage()


if __name__ == "__main__":
    main()
