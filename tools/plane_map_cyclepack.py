#!/usr/bin/env python3
"""Build a LitigationOS Plane-Map CyclePack from input archives.

This script inventories known inputs, extracts openable zip archives, classifies
files into planes, and emits derived indices, glossaries, and Neo4j import CSVs.
"""

import argparse
import ast
import csv
import hashlib
import json
import platform
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional


DEFAULT_INPUT_NAMES = [
    "PYTHON.zip",
    "Expanded_Chronological_Events_Table (1) (1).zip",
    "JSON.zip",
    "upgrade_pack_v1_1.zip",
    "litigation_graph_tokens (2).zip",
    "Catalog1.edb",
    "MasterGraph_IncrementPack_v4(3).3I0l10Xq.zip.part",
    "LITIGATION_OS_forged_full_code_bundle.pVMBqrmv.zip.part",
]

PLANE_DEFS = [
    ("P0", "INVARIANTS_SPEC", "Directives/specs/schemas/constraints/token grammars/READMEs"),
    ("P1", "INTAKE_INVENTORY", "Inventory/manifests/registries/corpora loaders/EDB and catalog surfaces"),
    ("P2", "NORMALIZE_EXTRACT", "Parsers/extractors/converters/OCR/unzip/text normalization"),
    ("P3", "INDEX_RETRIEVE", "Index builders/search/retrieval/embedding/vector lookup"),
    ("P4", "ENRICH_ATOMIZE", "Atomizers/timeline/event extraction/entity linking/scoring/contradictions"),
    ("P5", "VALIDATE_GATE", "Validators/self-tests/lint/gates/verification"),
    ("P6", "PACKAGE_EXPORT", "Bundlers/exporters/ZIP/CyclePack writers/report generators"),
    ("P7", "EXECUTE_ORCHESTRATE", "Launchers/runners/hypervisor loops/schedulers/queue/worker"),
    ("P8", "UI_VIEWER", "HTML/UI dashboards/viewers/GUI surfaces"),
    ("P9", "GRAPH_NEO4J", "Nodes/edges/cypher/graph schemas/import packs"),
    ("P10", "DATA_TABLES", "Data assets (CSV/JSON shards/tables/sqlite/etc.)"),
    ("P11", "MISC_DOCS", "Notes/examples/misc"),
]

PLANE_KEYWORDS = {
    "P0": ["superpin", "directive", "spec", "schema", "constraints", "readme", "grammar", "token", "lexicon", "authority", "benchbook"],
    "P1": ["manifest", "inventory", "registry", "catalog", "intake", "harvest", "edb", "catalog1"],
    "P2": ["normalize", "extract", "parse", "ocr", "convert", "unzip", "decode", "html2", "pdf2", "json2", "csv2"],
    "P3": ["index", "search", "retriev", "embed", "vector", "qdrant", "lookup", "fts", "sqlite"],
    "P4": ["enrich", "atom", "timeline", "event", "entity", "link", "score", "contradiction", "chronological", "bitemp", "graphrax"],
    "P5": ["validate", "verify", "gate", "lint", "selftest", "assert", "vrpt", "check", "integrity"],
    "P6": ["package", "bundle", "export", "zip", "cyclepack", "pack", "report"],
    "P7": ["launch", "runner", "hypervisor", "orchestr", "pipeline", "queue", "worker", "task", "scheduler", "batch", "daemon"],
    "P8": ["ui", "viewer", "dashboard", "html", "react", "electron", "bloom"],
    "P9": ["neo4j", "cypher", "nodes", "edges", "graph", "import", "apoc"],
}

CODE_EXTS = {".py", ".ps1", ".bat", ".cmd", ".js", ".ts", ".tsx", ".jsx", ".sh", ".java", ".cs", ".rs", ".cpp", ".c"}
DATA_EXTS = {".csv", ".json", ".jsonl", ".parquet", ".sqlite", ".db", ".edb"}
DOC_EXTS = {".md", ".txt", ".rst", ".pdf", ".docx"}

TERM_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
TOKEN_GRAMMAR_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\:[A-Z0-9_\/\-\.\$]+\b")

MAX_TEXT_BYTES = 1_000_000
MAX_SHA256_EXTRACTED = 10_000_000


@dataclass
class RunPaths:
    base: Path
    out_root: Path
    extract_root: Path
    derived_root: Path
    log_root: Path


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def jdump(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def ledger(path: Path, event: dict) -> None:
    event = dict(event)
    event["ts"] = datetime.now().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(jdump(event) + "\n")


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            hasher.update(block)
    return hasher.hexdigest()


def crc32_file(path: Path, chunk: int = 1024 * 1024) -> str:
    import zlib

    crc = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            crc = zlib.crc32(block, crc)
    return f"{crc & 0xFFFFFFFF:08x}"


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_zip_slip(member: str) -> bool:
    member = member.replace("\\", "/")
    return member.startswith("/") or member.startswith("../") or "/../" in member


def safe_extract_zip(zip_path: Path, target_dir: Path) -> List[str]:
    members: List[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if is_zip_slip(info.filename):
                continue
            members.append(info.filename)
        zf.extractall(target_dir, members=[member for member in members])
    return members


def classify_plane(rel_path: str) -> str:
    low = rel_path.lower()
    ext = Path(rel_path).suffix.lower()
    if ext in {".html", ".htm"}:
        return "P8"
    if ext in {".cypher", ".cql"} or "neo4j" in low or "cypher" in low:
        return "P9"
    if ext in DATA_EXTS:
        if any(key in low for key in PLANE_KEYWORDS["P9"]):
            return "P9"
        return "P10"
    if ext in DOC_EXTS:
        if any(key in low for key in PLANE_KEYWORDS["P0"]):
            return "P0"
        return "P11"
    if ext in CODE_EXTS:
        scores = {pid: 0 for pid, _, _ in PLANE_DEFS}
        for pid, keys in PLANE_KEYWORDS.items():
            for key in keys:
                if key in low:
                    scores[pid] += 2
        scores["P7"] += 1
        best = max(scores.items(), key=lambda item: item[1])
        return best[0] if best[1] > 0 else "P7"
    scores = {pid: 0 for pid, _, _ in PLANE_DEFS}
    for pid, keys in PLANE_KEYWORDS.items():
        for key in keys:
            if key in low:
                scores[pid] += 2
    best = max(scores.items(), key=lambda item: item[1])
    return best[0] if best[1] > 0 else "P11"


def read_text_safe(path: Path, max_bytes: int = MAX_TEXT_BYTES) -> str:
    data = path.read_bytes()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


def fast_newline_estimate(path: Path, max_bytes: int = 512 * 1024) -> Optional[int]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            buffer = handle.read(max_bytes)
        newline_count = buffer.count(b"\n")
        if newline_count <= 1:
            return None
        if size <= max_bytes:
            return max(newline_count - 1, 0)
        return int((newline_count / max_bytes) * size) - 1
    except Exception:
        return None


def file_kind(ext: str) -> str:
    if ext in CODE_EXTS or ext in DATA_EXTS or ext in DOC_EXTS or ext in {".html", ".htm", ".cypher", ".cql", ".yaml", ".yml"}:
        return "text"
    return "binary"


def add_occ(map_: Dict[str, dict], key: str, occ: dict, cap: int = 20) -> None:
    if key not in map_:
        map_[key] = {"count": 0, "occ": []}
    map_[key]["count"] += 1
    if len(map_[key]["occ"]) < cap:
        map_[key]["occ"].append(occ)


def nid(prefix: str, value: str) -> str:
    digest = hashlib.sha1(f"{prefix}|{value}".encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:16]}"


def initialize_paths(base: Path, run_id: str) -> RunPaths:
    out_root = base / f"LITIGATIONOS_PLANE_MAP_CYCLEPACK_{run_id}"
    extract_root = out_root / "extracted"
    derived_root = out_root / "derived"
    log_root = out_root / "logs"
    for directory in (out_root, extract_root, derived_root, log_root):
        ensure_dir(directory)
    return RunPaths(base=base, out_root=out_root, extract_root=extract_root, derived_root=derived_root, log_root=log_root)


def build_cyclepack(base: Path, input_names: Iterable[str]) -> Path:
    run_id = f"cycle_{now_id()}"
    paths = initialize_paths(base, run_id)
    cycle_ledger = paths.log_root / "CYCLE_LEDGER.jsonl"
    run_ledger = paths.log_root / "RUN_LEDGER.jsonl"

    blockers: List[dict] = []
    manifest_inputs: List[dict] = []
    archives: List[dict] = []

    ledger(
        cycle_ledger,
        {
            "event": "cycle_start",
            "run_id": run_id,
            "cycle": 1,
            "stage": "inventory_inputs",
            "params": {"K": 3, "EPS": 0.01, "STABLE_N": 3, "MAX_CYCLES": 25, "STRICT": True},
        },
    )

    start_time = time.time()

    for name in input_names:
        path = base / name
        if not path.exists():
            blockers.append({"type": "MISSING_INPUT", "path": str(path), "detail": "File not found in base directory."})
            continue
        stat = path.stat()
        rec = {
            "scope": "input",
            "path": path.name,
            "abs_path": str(path),
            "size_bytes": stat.st_size,
            "mtime_iso": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "sha256": sha256_file(path) if stat.st_size <= 200_000_000 else None,
            "crc32": crc32_file(path),
            "ext": path.suffix.lower(),
            "zip_openable": None,
            "zip_error": None,
        }
        if path.suffix.lower() == ".zip" or path.name.lower().endswith(".zip.part"):
            try:
                with zipfile.ZipFile(path, "r") as zf:
                    _ = zf.namelist()[:1]
                rec["zip_openable"] = True
            except Exception as exc:
                rec["zip_openable"] = False
                rec["zip_error"] = f"{type(exc).__name__}: {exc}"
                if path.name.lower().endswith(".zip.part"):
                    blockers.append(
                        {
                            "type": "INCOMPLETE_ARCHIVE",
                            "path": path.name,
                            "detail": "Looks like a split ZIP fragment but is not openable; likely missing remaining parts or requires reassembly.",
                            "error": rec["zip_error"],
                        }
                    )
        manifest_inputs.append(rec)

    manifest_json = paths.derived_root / "MANIFEST_INPUTS.json"
    manifest_csv = paths.derived_root / "MANIFEST_INPUTS.csv"
    manifest_json.write_text(
        json.dumps({"run_id": run_id, "generated_at": datetime.now().isoformat(timespec="seconds"), "records": manifest_inputs}, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    fieldnames = sorted({key for record in manifest_inputs for key in record.keys()}) if manifest_inputs else ["scope", "path"]
    with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in manifest_inputs:
            writer.writerow(record)
    ledger(run_ledger, {"event": "manifest_inputs_written", "run_id": run_id, "records": len(manifest_inputs)})

    ledger(cycle_ledger, {"event": "cycle_step", "run_id": run_id, "cycle": 1, "stage": "extract_openable_zips"})
    for record in manifest_inputs:
        path = base / record["path"]
        if record.get("zip_openable") and path.suffix.lower() == ".zip":
            target = paths.extract_root / path.stem
            ensure_dir(target)
            try:
                members = safe_extract_zip(path, target)
                archives.append({"archive": record["path"], "extracted_to": safe_relpath(target, paths.out_root), "members": len(members)})
                ledger(run_ledger, {"event": "zip_extracted", "archive": record["path"], "members": len(members)})
            except Exception as exc:
                blockers.append({"type": "EXTRACT_ERROR", "path": record["path"], "detail": f"{type(exc).__name__}: {exc}"})

    extracted_files = [fp for fp in paths.extract_root.rglob("*") if fp.is_file()]
    ledger(run_ledger, {"event": "extracted_scan", "files": len(extracted_files), "archives": len(archives)})

    terms: Dict[str, dict] = {}
    tokens: Dict[str, dict] = {}
    symbols: List[dict] = []
    tables: List[dict] = []
    file_records: List[dict] = []

    ledger(cycle_ledger, {"event": "cycle_step", "run_id": run_id, "cycle": 2, "stage": "scan_and_classify_extracted"})

    for fp in extracted_files:
        rel = safe_relpath(fp, paths.out_root)
        ext = fp.suffix.lower()
        plane = classify_plane(rel)
        size = fp.stat().st_size
        record = {
            "scope": "extracted",
            "path": rel,
            "size_bytes": size,
            "ext": ext,
            "plane_id": plane,
            "kind": file_kind(ext),
            "sha256": None,
            "crc32": None,
        }
        try:
            record["crc32"] = crc32_file(fp)
            if size <= MAX_SHA256_EXTRACTED:
                record["sha256"] = sha256_file(fp)
        except Exception as exc:
            blockers.append({"type": "HASH_ERROR", "path": rel, "detail": f"{type(exc).__name__}: {exc}"})
        file_records.append(record)

        if record["kind"] != "text":
            continue
        try:
            text = read_text_safe(fp)
        except Exception:
            text = ""
        if not text:
            continue

        for match in TERM_RE.finditer(text):
            add_occ(terms, match.group(0), {"path": rel, "pos": match.start()})
        for match in TOKEN_GRAMMAR_RE.finditer(text):
            add_occ(tokens, match.group(0), {"path": rel, "pos": match.start()})
        if ext == ".py" and "extracted/PYTHON/" in rel.replace("\\", "/"):
            try:
                tree = ast.parse(text)
            except Exception as exc:
                blockers.append({"type": "PY_PARSE_ERROR", "path": rel, "detail": f"{type(exc).__name__}: {exc}"})
                continue
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    symbols.append({"symbol_type": "function", "name": node.name, "file": rel, "lineno": node.lineno})
                elif isinstance(node, ast.AsyncFunctionDef):
                    symbols.append({"symbol_type": "async_function", "name": node.name, "file": rel, "lineno": node.lineno})
                elif isinstance(node, ast.ClassDef):
                    symbols.append({"symbol_type": "class", "name": node.name, "file": rel, "lineno": node.lineno})
        if ext == ".csv":
            try:
                with fp.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                    header = handle.readline().strip("\n\r")
                cols = [col.strip() for col in header.split(",")] if header else []
                row_est = fast_newline_estimate(fp)
                tables.append(
                    {
                        "table_path": rel,
                        "size_bytes": size,
                        "columns": cols[:300],
                        "col_count": len(cols),
                        "row_count_est": row_est,
                        "analysis_level": "header+row_est",
                    }
                )
            except Exception as exc:
                blockers.append({"type": "CSV_PARSE_ERROR", "path": rel, "detail": f"{type(exc).__name__}: {exc}"})

    ledger(
        run_ledger,
        {
            "event": "content_scan_complete",
            "files": len(file_records),
            "symbols": len(symbols),
            "terms": len(terms),
            "tokens": len(tokens),
            "tables": len(tables),
        },
    )

    plane_table_csv = paths.derived_root / "PLANE_TABLE.csv"
    with plane_table_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["plane_id", "plane_name", "objective"])
        for pid, name, obj in PLANE_DEFS:
            writer.writerow([pid, name, obj])

    archives_json = paths.derived_root / "ARCHIVES_EXTRACTED.json"
    archives_json.write_text(json.dumps({"archives": archives}, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")

    manifest_ex_json = paths.derived_root / "MANIFEST_EXTRACTED.json"
    manifest_ex_csv = paths.derived_root / "MANIFEST_EXTRACTED.csv"
    manifest_ex_json.write_text(
        json.dumps({"run_id": run_id, "generated_at": datetime.now().isoformat(timespec="seconds"), "records": file_records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
        newline="\n",
    )
    ex_fieldnames = sorted({key for record in file_records for key in record.keys()}) if file_records else ["scope", "path"]
    with manifest_ex_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ex_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in file_records:
            writer.writerow(record)

    index_md = paths.derived_root / "INDEX.md"
    ext_counts: Dict[str, int] = {}
    plane_counts: Dict[str, int] = {}
    for record in file_records:
        ext_counts[record["ext"]] = ext_counts.get(record["ext"], 0) + 1
        plane_counts[record["plane_id"]] = plane_counts.get(record["plane_id"], 0) + 1
    with index_md.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Corpus Index (Extracted)\n\n")
        handle.write(f"- run_id: `{run_id}`\n")
        handle.write(f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`\n")
        handle.write(f"- extracted_files: `{len(file_records)}`\n")
        handle.write(f"- python_symbols: `{len(symbols)}`\n")
        handle.write(f"- unique_terms: `{len(terms)}`\n")
        handle.write(f"- token_grammar_entries: `{len(tokens)}`\n")
        handle.write(f"- csv_tables: `{len(tables)}`\n\n")
        handle.write("## Archives extracted\n\n")
        for archive in archives:
            handle.write(f"- `{archive['archive']}` → `{archive['extracted_to']}` (members: {archive['members']})\n")
        handle.write("\n## Plane counts\n\n")
        for pid, _, _ in PLANE_DEFS:
            handle.write(f"- `{pid}`: {plane_counts.get(pid, 0)}\n")
        handle.write("\n## File type counts\n\n")
        for ext, count in sorted(ext_counts.items(), key=lambda kv: (-kv[1], kv[0])):
            handle.write(f"- `{ext or '(noext)'}`: {count}\n")

    plane_map_md = paths.derived_root / "PLANE_MAP.md"
    by_plane: Dict[str, List[dict]] = {pid: [] for pid, _, _ in PLANE_DEFS}
    for record in file_records:
        by_plane.setdefault(record["plane_id"], []).append(record)
    for pid in by_plane:
        by_plane[pid].sort(key=lambda item: (-item["size_bytes"], item["path"]))
    with plane_map_md.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Plane Map\n\n")
        handle.write(f"- run_id: `{run_id}`\n")
        handle.write(f"- extracted_files_indexed: `{len(file_records)}`\n")
        handle.write(f"- archives: `{len(archives)}`\n\n")
        for pid, name, obj in PLANE_DEFS:
            files = by_plane.get(pid, [])
            handle.write(f"## {pid} — {name}\n\n")
            handle.write(f"Objective: {obj}\n\n")
            handle.write(f"Files mapped: {len(files)}\n\n")
            for record in files[:80]:
                handle.write(f"- `{record['path']}` ({record['size_bytes']} bytes, ext `{record['ext']}`)\n")
            if len(files) > 80:
                handle.write(f"- (plus {len(files) - 80} more)\n")
            handle.write("\n")

    script_rows = [record for record in file_records if record["ext"] in CODE_EXTS]
    scripts_catalog_csv = paths.derived_root / "SCRIPTS_CATALOG.csv"
    with scripts_catalog_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "ext", "size_bytes", "sha256", "crc32", "plane_id"])
        for record in sorted(script_rows, key=lambda item: item["path"]):
            writer.writerow([record["path"], record["ext"], record["size_bytes"], record["sha256"] or "", record["crc32"] or "", record["plane_id"]])

    symbols_csv = paths.derived_root / "SCRIPT_SYMBOLS.csv"
    with symbols_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol_type", "name", "file", "lineno"])
        writer.writeheader()
        for symbol in sorted(symbols, key=lambda item: (item["file"], item["lineno"], item["name"])):
            writer.writerow(symbol)

    tables_csv = paths.derived_root / "TABLE_CATALOG.csv"
    with tables_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["table_path", "size_bytes", "col_count", "row_count_est", "analysis_level", "columns_json"])
        for table in sorted(tables, key=lambda item: item["table_path"]):
            writer.writerow(
                [
                    table["table_path"],
                    table["size_bytes"],
                    table["col_count"],
                    table["row_count_est"] if table["row_count_est"] is not None else "",
                    table["analysis_level"],
                    json.dumps(table["columns"], ensure_ascii=False),
                ]
            )

    glossary_jsonl = paths.derived_root / "GLOSSARY_TERMS.jsonl"
    glossary_md = paths.derived_root / "GLOSSARY_TERMS.md"
    ranked_terms = sorted(terms.items(), key=lambda kv: (-kv[1]["count"], kv[0]))
    with glossary_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for term, meta in ranked_terms[:3000]:
            handle.write(json.dumps({"term": term, "count": meta["count"], "occurrences": meta["occ"]}, ensure_ascii=False) + "\n")
    with glossary_md.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Glossary — Observed Terms\n\n")
        handle.write("Corpus-derived term list with occurrence pointers only.\n\n")
        handle.write(f"- unique_terms_indexed: `{len(terms)}`\n")
        handle.write("- exported_terms: `3000`\n\n")
        for term, meta in ranked_terms[:250]:
            handle.write(f"## {term}\n\n")
            handle.write(f"- count: `{meta['count']}`\n")
            for occ in meta["occ"][:8]:
                handle.write(f"- occurrence: `{occ['path']}` @pos `{occ['pos']}`\n")
            handle.write("\n")

    token_jsonl = paths.derived_root / "TOKEN_GRAMMAR.jsonl"
    token_md = paths.derived_root / "TOKEN_GRAMMAR.md"
    ranked_tokens = sorted(tokens.items(), key=lambda kv: (-kv[1]["count"], kv[0]))
    with token_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for token, meta in ranked_tokens[:3000]:
            handle.write(json.dumps({"token": token, "count": meta["count"], "occurrences": meta["occ"]}, ensure_ascii=False) + "\n")
    with token_md.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Token Grammar Index — Observed TOKEN:SCOPE Strings\n\n")
        handle.write("Extracted from the corpus using a conservative regex. No invented semantics.\n\n")
        handle.write(f"- unique_token_strings: `{len(tokens)}`\n")
        handle.write("- exported_tokens: `3000`\n\n")
        for token, meta in ranked_tokens[:200]:
            handle.write(f"## {token}\n\n")
            handle.write(f"- count: `{meta['count']}`\n")
            for occ in meta["occ"][:8]:
                handle.write(f"- occurrence: `{occ['path']}` @pos `{occ['pos']}`\n")
            handle.write("\n")

    nodes_csv = paths.derived_root / "GRAPH_NODES.csv"
    edges_csv = paths.derived_root / "GRAPH_EDGES.csv"

    node_rows: List[dict] = []
    edge_rows: List[dict] = []

    plane_nodes: Dict[str, str] = {}
    for pid, name, obj in PLANE_DEFS:
        pid_id = nid("plane", pid)
        plane_nodes[pid] = pid_id
        node_rows.append(
            {
                "node_id": pid_id,
                "labels": "Plane",
                "name": f"{pid}:{name}",
                "path": "",
                "ext": "",
                "size_bytes": "",
                "sha256": "",
                "crc32": "",
                "meta_json": json.dumps({"plane_id": pid, "plane_name": name, "objective": obj}, ensure_ascii=False),
            }
        )

    archive_nodes: Dict[str, str] = {}
    for archive in archives:
        archive_id = nid("archive", archive["archive"])
        archive_nodes[archive["archive"]] = archive_id
        node_rows.append(
            {
                "node_id": archive_id,
                "labels": "Archive",
                "name": archive["archive"],
                "path": archive["archive"],
                "ext": ".zip",
                "size_bytes": "",
                "sha256": "",
                "crc32": "",
                "meta_json": json.dumps(archive, ensure_ascii=False),
            }
        )

    file_nodes: Dict[str, str] = {}
    for record in file_records:
        fid = nid("file", record["path"])
        file_nodes[record["path"]] = fid
        node_rows.append(
            {
                "node_id": fid,
                "labels": "File",
                "name": Path(record["path"]).name,
                "path": record["path"],
                "ext": record["ext"],
                "size_bytes": str(record["size_bytes"]),
                "sha256": record["sha256"] or "",
                "crc32": record["crc32"] or "",
                "meta_json": json.dumps({"plane_id": record["plane_id"], "kind": record["kind"]}, ensure_ascii=False),
            }
        )
        edge_rows.append(
            {
                "edge_id": nid("edge", f"in_plane|{record['path']}|{record['plane_id']}"),
                "type": "IN_PLANE",
                "src_id": fid,
                "dst_id": plane_nodes.get(record["plane_id"], plane_nodes["P11"]),
                "meta_json": "{}",
            }
        )

    for archive in archives:
        arc = archive["archive"]
        arc_id = archive_nodes[arc]
        extracted_folder = f"extracted/{Path(arc).stem}/"
        for record in file_records:
            if record["path"].startswith(extracted_folder):
                edge_rows.append(
                    {
                        "edge_id": nid("edge", f"contains|{arc}|{record['path']}"),
                        "type": "CONTAINS",
                        "src_id": arc_id,
                        "dst_id": file_nodes[record["path"]],
                        "meta_json": "{}",
                    }
                )

    term_node_limit = 1800
    for term, meta in ranked_terms[:term_node_limit]:
        term_id = nid("term", term)
        node_rows.append(
            {
                "node_id": term_id,
                "labels": "Term",
                "name": term,
                "path": "",
                "ext": "",
                "size_bytes": "",
                "sha256": "",
                "crc32": "",
                "meta_json": json.dumps({"count": meta["count"]}, ensure_ascii=False),
            }
        )
        for occ in meta["occ"][:12]:
            fpath = occ["path"]
            if fpath in file_nodes:
                edge_rows.append(
                    {
                        "edge_id": nid("edge", f"mentions|{fpath}|{term}|{occ['pos']}"),
                        "type": "MENTIONS",
                        "src_id": file_nodes[fpath],
                        "dst_id": term_id,
                        "meta_json": json.dumps({"pos": occ["pos"]}, ensure_ascii=False),
                    }
                )

    token_node_limit = 1200
    for token, meta in ranked_tokens[:token_node_limit]:
        token_id = nid("tok", token)
        node_rows.append(
            {
                "node_id": token_id,
                "labels": "TokenString",
                "name": token,
                "path": "",
                "ext": "",
                "size_bytes": "",
                "sha256": "",
                "crc32": "",
                "meta_json": json.dumps({"count": meta["count"]}, ensure_ascii=False),
            }
        )
        for occ in meta["occ"][:12]:
            fpath = occ["path"]
            if fpath in file_nodes:
                edge_rows.append(
                    {
                        "edge_id": nid("edge", f"mentions_token|{fpath}|{token}|{occ['pos']}"),
                        "type": "MENTIONS_TOKEN",
                        "src_id": file_nodes[fpath],
                        "dst_id": token_id,
                        "meta_json": json.dumps({"pos": occ["pos"]}, ensure_ascii=False),
                    }
                )

    for symbol in symbols[:12000]:
        sym_id = nid("sym", f"{symbol['file']}|{symbol['symbol_type']}|{symbol['name']}|{symbol['lineno']}")
        node_rows.append(
            {
                "node_id": sym_id,
                "labels": "Symbol",
                "name": symbol["name"],
                "path": symbol["file"],
                "ext": "",
                "size_bytes": "",
                "sha256": "",
                "crc32": "",
                "meta_json": json.dumps({"symbol_type": symbol["symbol_type"], "lineno": symbol["lineno"]}, ensure_ascii=False),
            }
        )
        if symbol["file"] in file_nodes:
            edge_rows.append(
                {
                    "edge_id": nid("edge", f"defines|{symbol['file']}|{symbol['name']}|{symbol['lineno']}"),
                    "type": "DEFINES",
                    "src_id": file_nodes[symbol["file"]],
                    "dst_id": sym_id,
                    "meta_json": "{}",
                }
            )

    for table in tables[:8000]:
        table_id = nid("table", table["table_path"])
        node_rows.append(
            {
                "node_id": table_id,
                "labels": "Table",
                "name": Path(table["table_path"]).name,
                "path": table["table_path"],
                "ext": ".csv",
                "size_bytes": "",
                "sha256": "",
                "crc32": "",
                "meta_json": json.dumps({"row_count_est": table["row_count_est"], "col_count": table["col_count"]}, ensure_ascii=False),
            }
        )
        if table["table_path"] in file_nodes:
            edge_rows.append(
                {
                    "edge_id": nid("edge", f"is_table|{table['table_path']}"),
                    "type": "IS_TABLE",
                    "src_id": file_nodes[table["table_path"]],
                    "dst_id": table_id,
                    "meta_json": "{}",
                }
            )

    with nodes_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_id", "labels", "name", "path", "ext", "size_bytes", "sha256", "crc32", "meta_json"])
        writer.writeheader()
        for row in node_rows:
            writer.writerow(row)

    with edges_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["edge_id", "type", "src_id", "dst_id", "meta_json"])
        writer.writeheader()
        for row in edge_rows:
            writer.writerow(row)

    import_cypher = paths.derived_root / "IMPORT.cypher"
    import_cypher.write_text(
        """// Neo4j Import Pack (CSV in Neo4j import/ directory required)
// Recommended: install APOC. If APOC is not available, you can still LOAD CSV and MERGE manually by label.
// 1) Copy GRAPH_NODES.csv and GRAPH_EDGES.csv into Neo4j's import/ folder.
// 2) Run this file in cypher-shell or Neo4j Browser.
//
// Constraints
CREATE CONSTRAINT plane_node_id IF NOT EXISTS FOR (n:Plane) REQUIRE n.node_id IS UNIQUE;
CREATE CONSTRAINT file_node_id  IF NOT EXISTS FOR (n:File)  REQUIRE n.node_id IS UNIQUE;
CREATE CONSTRAINT term_node_id  IF NOT EXISTS FOR (n:Term)  REQUIRE n.node_id IS UNIQUE;
CREATE CONSTRAINT token_node_id IF NOT EXISTS FOR (n:TokenString) REQUIRE n.node_id IS UNIQUE;
CREATE CONSTRAINT sym_node_id   IF NOT EXISTS FOR (n:Symbol) REQUIRE n.node_id IS UNIQUE;
CREATE CONSTRAINT arch_node_id  IF NOT EXISTS FOR (n:Archive) REQUIRE n.node_id IS UNIQUE;
CREATE CONSTRAINT table_node_id IF NOT EXISTS FOR (n:Table) REQUIRE n.node_id IS UNIQUE;
// Nodes
LOAD CSV WITH HEADERS FROM 'file:///GRAPH_NODES.csv' AS row
WITH row
CALL {
  WITH row
  WITH row, split(row.labels,'|') AS lbs
  CALL apoc.create.node(lbs, {
    node_id: row.node_id,
    name: row.name,
    path: row.path,
    ext: row.ext,
    size_bytes: CASE WHEN row.size_bytes = '' THEN null ELSE toInteger(row.size_bytes) END,
    sha256: CASE WHEN row.sha256 = '' THEN null ELSE row.sha256 END,
    crc32: CASE WHEN row.crc32 = '' THEN null ELSE row.crc32 END,
    meta_json: row.meta_json
  }) YIELD node
  RETURN node
}
RETURN count(*) AS nodes_loaded;
// Relationships
LOAD CSV WITH HEADERS FROM 'file:///GRAPH_EDGES.csv' AS row
MATCH (a {node_id: row.src_id})
MATCH (b {node_id: row.dst_id})
CALL apoc.create.relationship(a, row.type, {meta_json: row.meta_json}, b) YIELD rel
RETURN count(*) AS rels_loaded;
""",
        encoding="utf-8",
        newline="\n",
    )

    blockers_md = paths.derived_root / "BLOCKERS_AND_ACQUISITION_PLAN.md"
    with blockers_md.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Blockers and Acquisition Plan\n\n")
        handle.write("This list is **fail-closed**: only confirmed blockers are included.\n\n")
        if not blockers:
            handle.write("- None detected.\n")
        else:
            handle.write("## Blockers\n\n")
            for blocker in blockers:
                handle.write(f"- **{blocker.get('type')}** — `{blocker.get('path', '')}`\n")
                if blocker.get("detail"):
                    handle.write(f"  - detail: {blocker['detail']}\n")
                if blocker.get("error"):
                    handle.write(f"  - error: {blocker['error']}\n")
            handle.write("\n## Acquisition plan\n\n")
            incomplete = [blocker for blocker in blockers if blocker.get("type") == "INCOMPLETE_ARCHIVE"]
            if incomplete:
                handle.write("### Split ZIP parts (incomplete / not reassemblable from current corpus)\n\n")
                for blocker in incomplete:
                    handle.write(f"- `{blocker['path']}`\n")
                    handle.write("  - Provide **all parts** belonging to this split set (same base name). If possible, upload the reconstructed full `.zip` instead.\n")
                    handle.write("  - After upload: rerun the builder script; it will attempt open + extract and will record `zip_openable=true` only if it passes.\n")
            else:
                handle.write("- If a file is unreadable, upload it again as a standard `.zip`.\n")

    provenance = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "inputs_present": [record["path"] for record in manifest_inputs],
        "archives_extracted": archives,
        "counts": {
            "extracted_files": len(file_records),
            "symbols": len(symbols),
            "terms": len(terms),
            "token_strings": len(tokens),
            "tables": len(tables),
            "graph_nodes": len(node_rows),
            "graph_edges": len(edge_rows),
        },
    }
    prov_path = paths.derived_root / "PROVENANCE_INDEX.json"
    prov_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")

    readme = paths.derived_root / "README.md"
    readme.write_text(
        f"""# LitigationOS Plane-Map CyclePack
- run_id: `{run_id}`
- purpose: non-destructive extraction + plane classification + index + glossary + Neo4j import pack
- inputs: {len(manifest_inputs)} present
- extracted archives: {len(archives)}
- extracted files indexed: {len(file_records)}
## What is inside
- `derived/INDEX.md` — corpus summary (counts, planes, types)
- `derived/PLANE_MAP.md` — files mapped into planes
- `derived/MANIFEST_INPUTS.*` — hashes/metadata for original inputs (non-destructive)
- `derived/MANIFEST_EXTRACTED.*` — hashes/metadata for extracted files
- `derived/SCRIPTS_CATALOG.csv` + `derived/SCRIPT_SYMBOLS.csv` — code inventory
- `derived/TABLE_CATALOG.csv` — CSV table inventory (header + row-count estimate)
- `derived/GLOSSARY_TERMS.*` — observed terms (uppercase tokens) with pointers
- `derived/TOKEN_GRAMMAR.*` — observed TOKEN:SCOPE-like strings with pointers
- `derived/GRAPH_NODES.csv` + `derived/GRAPH_EDGES.csv` — Neo4j-ready import tables
- `derived/IMPORT.cypher` — APOC-assisted import script
- `derived/PROVENANCE_INDEX.json` — run provenance + counts
- `derived/BLOCKERS_AND_ACQUISITION_PLAN.md` — fail-closed blockers
## Neo4j import (quick)
1) Copy `GRAPH_NODES.csv` and `GRAPH_EDGES.csv` into Neo4j's `import/` folder.
2) Run `IMPORT.cypher` in Neo4j Browser or `cypher-shell -f IMPORT.cypher`.
If APOC is not installed, load nodes/edges manually by label; this pack still gives you the normalized CSVs.
""",
        encoding="utf-8",
        newline="\n",
    )

    elapsed = time.time() - start_time
    ledger(cycle_ledger, {"event": "cycle_step", "run_id": run_id, "cycle": 3, "stage": "export_artifacts", "metrics": {"elapsed_sec": round(elapsed, 3)}})
    ledger(cycle_ledger, {"event": "cycle_complete", "run_id": run_id, "cycle": 3, "status": "PASS_WITH_BLOCKERS" if blockers else "PASS", "delta_eps_est": 0.0})

    cyclepack_zip = base / f"{paths.out_root.name}.zip"
    with zipfile.ZipFile(cyclepack_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fp in paths.out_root.rglob("*"):
            if fp.is_file():
                zf.write(fp, fp.relative_to(base))
    with zipfile.ZipFile(cyclepack_zip, "r") as zf:
        bad_member = zf.testzip()
    ledger(run_ledger, {"event": "cyclepack_zip_test", "zip": str(cyclepack_zip), "bad_member": bad_member or ""})

    return cyclepack_zip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Plane-Map CyclePack from LitigationOS inputs.")
    parser.add_argument("--base", type=Path, default=Path("/mnt/data"), help="Base directory containing input archives.")
    parser.add_argument("--inputs", nargs="*", default=DEFAULT_INPUT_NAMES, help="Input filenames to inventory/extract.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cyclepack = build_cyclepack(args.base, args.inputs)
    print(f"CyclePack written: {cyclepack}")


if __name__ == "__main__":
    main()
