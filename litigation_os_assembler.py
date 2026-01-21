#!/usr/bin/env python3
"""FredPrime Litigation OS Assembler.

This script consolidates legal system artifacts into a unified, market-ready
bundle with manifests, checksums, resumability, and robust logging.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import logging.handlers
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import zipfile

# ---------------------------------------------------------------------------
# Constants & Configuration
# ---------------------------------------------------------------------------
TOOL_NAME = "litigation_os"
DEFAULT_EXTENSIONS = {".py", ".ps1", ".json", ".html"}
DEFAULT_ROOTS = [Path("F:/"), Path("D:/"), Path("Z:/")]
OPTIONAL_ROOT = Path("R:/")
CHUNK_SIZE = 1024 * 1024  # 1 MiB for hashing
STATE_SAVE_INTERVAL = 10
RETRY_ATTEMPTS = 3
RETRY_BACKOFF = 2.0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class FileRecord:
    """Represents metadata for a scanned file."""

    source_path: Path
    root_label: str
    relative_path: Path
    size: int
    mtime_utc: str
    sha256: str
    duplicate_of: Optional[str]
    bates_number: Optional[str]
    evidence_stamp: Optional[str]
    mifile_ready: bool

    def manifest_dict(self, identifier: str) -> Dict[str, object]:
        return {
            "id": identifier,
            "root": self.root_label,
            "relative_path": self.relative_path.as_posix(),
            "absolute_path": str(self.source_path),
            "size_bytes": self.size,
            "mtime_utc": self.mtime_utc,
            "sha256": self.sha256,
            "duplicate_of": self.duplicate_of,
            "bates_number": self.bates_number,
            "evidence_stamp": self.evidence_stamp,
            "mifile_ready": self.mifile_ready,
        }


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def generate_token(provided: Optional[str] = None) -> str:
    if provided:
        return provided
    return uuid.uuid4().hex[:16]


def utc_now() -> dt.datetime:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)


def isoformat(ts: dt.datetime) -> str:
    return ts.astimezone(dt.timezone.utc).isoformat(timespec="seconds")


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_root_label(root: Path) -> str:
    label = str(root).replace(":", "").replace("/", "_").replace("\\", "_")
    label = label.strip("_")
    return label or "root"


def is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts if part not in {".", ".."})


def should_skip_dir(path: Path) -> bool:
    skip_names = {"System Volume Information", "$RECYCLE.BIN"}
    return is_hidden(path) or path.name in skip_names


def path_with_prefix(path: Path) -> Path:
    # Handle Windows long paths by applying \\?\ prefix when necessary.
    raw = str(path)
    if sys.platform.startswith("win") and not raw.startswith("\\\\?\\"):
        return Path("\\\\?\\" + raw)
    return path


def open_with_retries(path: Path, mode: str) -> Tuple[object, os.stat_result]:
    last_exc: Optional[Exception] = None
    prefixed = path_with_prefix(path)
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            stat_result = prefixed.stat()
            handle = prefixed.open(mode)
            return handle, stat_result
        except (OSError, PermissionError) as exc:
            last_exc = exc
            if attempt == RETRY_ATTEMPTS:
                raise
            time.sleep(RETRY_BACKOFF**attempt)
    raise last_exc  # type: ignore[misc]


def compute_sha256(path: Path, logger: logging.Logger) -> Tuple[str, int, float]:
    handle, stat_result = open_with_retries(path, "rb")
    try:
        digest = hashlib.sha256()
        total_read = 0
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            total_read += len(chunk)
    finally:
        handle.close()
    if total_read != stat_result.st_size:
        logger.warning(
            "Read size mismatch for %s: expected %s, read %s",
            path,
            stat_result.st_size,
            total_read,
        )
    return digest.hexdigest(), stat_result.st_size, stat_result.st_mtime


def save_json_atomic(data: object, target_path: Path, temp_root: Path) -> None:
    ensure_directory(target_path.parent)
    temp_path = temp_root / f"{target_path.name}.tmp"
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, target_path)


def save_text_atomic(text: str, target_path: Path, temp_root: Path) -> None:
    ensure_directory(target_path.parent)
    temp_path = temp_root / f"{target_path.name}.tmp"
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, target_path)


def save_binary_atomic(data: bytes, target_path: Path, temp_root: Path) -> None:
    ensure_directory(target_path.parent)
    temp_path = temp_root / f"{target_path.name}.tmp"
    with temp_path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, target_path)


def load_state(state_path: Path) -> Dict[str, object]:
    if not state_path.exists():
        return {}
    with state_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(state: Dict[str, object], state_path: Path, temp_root: Path) -> None:
    ensure_directory(state_path.parent)
    save_json_atomic(state, state_path, temp_root)


def estimated_required_space(total_bytes: int) -> int:
    # Add 20% overhead plus 200 MiB buffer for archives and logs.
    buffer = 200 * 1024 * 1024
    overhead = int(total_bytes * 1.2)
    return total_bytes + overhead + buffer


def check_free_space(path: Path, required: int) -> None:
    usage = shutil.disk_usage(path)
    if usage.free < required:
        raise RuntimeError(f"Insufficient free space on {path}: required {required:,} bytes, available {usage.free:,}")


def detect_secrets(path: Path, logger: logging.Logger) -> List[Dict[str, str]]:
    try:
        handle, _stat = open_with_retries(path, "rb")
        try:
            sample = handle.read(512_000)
        finally:
            handle.close()
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Secret scan skipped for %s (%s)", path, exc)
        return []
    text: Optional[str] = None
    try:
        text = sample.decode("utf-8", errors="ignore")
    except Exception:  # pylint: disable=broad-except
        return []
    findings: List[Dict[str, str]] = []
    patterns = {
        "AWS Key": r"AKIA[0-9A-Z]{16}",
        "Slack Token": r"xox[baprs]-[0-9A-Za-z-]+",
        "Generic Secret": r"(?i)(api_key|secret|password)\s*[:=]\s*[\'\"]?[A-Za-z0-9\-_/]{8,}",
    }
    import re

    for label, pattern in patterns.items():
        for match in re.finditer(pattern, text):
            start = max(match.start() - 8, 0)
            end = min(match.end() + 8, len(text))
            snippet = text[start:end]
            findings.append(
                {
                    "type": label,
                    "match": snippet[:4] + "***REDACTED***" + snippet[-4:],
                }
            )
    if findings:
        logger.warning("Potential secrets detected in %s", path)
    return findings


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class JsonLinesHandler(logging.Handler):
    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        ensure_directory(self.path.parent)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        log_entry = {
            "ts": isoformat(utc_now()),
            "level": record.levelname,
            "message": self.format(record),
            "logger": record.name,
        }
        line = json.dumps(log_entry, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def setup_logging(logs_root: Path, token: str) -> logging.Logger:
    ensure_directory(logs_root)
    today = utc_now().strftime("%Y%m%d")
    human_log = logs_root / f"{TOOL_NAME}{today}{token}.log"
    jsonl_log = logs_root / f"{TOOL_NAME}.jsonl"

    logger = logging.getLogger(TOOL_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)sZ | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    rotating_handler = logging.handlers.RotatingFileHandler(  # type: ignore[attr-defined]
        human_log, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    rotating_handler.setFormatter(formatter)
    logger.addHandler(rotating_handler)

    json_handler = JsonLinesHandler(jsonl_log)
    json_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(json_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Logging initialized (token=%s)", token)
    return logger


# ---------------------------------------------------------------------------
# Scanning Logic
# ---------------------------------------------------------------------------


def gather_candidates(
    roots: Sequence[Path],
    include_hidden: bool,
    extensions: Sequence[str],
    logger: logging.Logger,
) -> List[Tuple[Path, Path, str]]:
    candidates: List[Tuple[Path, Path, str]] = []
    normalized_exts = {ext.lower() for ext in extensions}
    for root in roots:
        if not root.exists():
            logger.warning("Root %s unavailable; skipping", root)
            continue
        root_label = sanitize_root_label(root)
        for current_root, dirs, files in os.walk(root, topdown=True):
            current_path = Path(current_root)
            if not include_hidden:
                dirs[:] = [d for d in dirs if not should_skip_dir(current_path / d)]
            for filename in files:
                candidate_path = current_path / filename
                if not include_hidden and is_hidden(candidate_path):
                    continue
                if candidate_path.suffix.lower() not in normalized_exts:
                    continue
                relative_path = candidate_path.relative_to(root)
                candidates.append((candidate_path, relative_path, root_label))
    logger.info("Discovered %s candidate files", len(candidates))
    return candidates


def process_candidates(
    candidates: Sequence[Tuple[Path, Path, str]],
    args: argparse.Namespace,
    logger: logging.Logger,
    state: Dict[str, object],
    state_path: Path,
    temp_root: Path,
) -> Tuple[int, Dict[str, List[Dict[str, str]]], int]:
    processed: Dict[str, Dict[str, object]] = state.setdefault("processed", {})  # type: ignore[assignment]
    duplicates: Dict[str, str] = state.setdefault("duplicates", {})  # type: ignore[assignment]
    secrets_report: Dict[str, List[Dict[str, str]]] = state.setdefault("secrets", {})  # type: ignore[assignment]
    hash_to_id: Dict[str, str] = state.setdefault("hash_to_id", {})  # type: ignore[assignment]
    new_entries = 0
    skipped_existing = 0
    counter = len(processed)
    bates_counter = state.setdefault("bates_counter", 1)  # type: ignore[assignment]
    lock = threading.Lock()

    def worker(item: Tuple[Path, Path, str]) -> Optional[str]:
        nonlocal counter, bates_counter, new_entries
        path, relative_path, root_label = item
        key = str(path)
        if key in processed:
            return None
        try:
            sha256, size, mtime = compute_sha256(path, logger)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to hash %s: %s", path, exc)
            return None
        duplicate_of = hash_to_id.get(sha256)
        bates_number: Optional[str] = None
        with lock:
            counter += 1
            identifier = f"DOC-{counter:06d}"
            if duplicate_of is None:
                hash_to_id[sha256] = identifier
            else:
                duplicates[key] = duplicate_of
            if args.bates_prefix:
                bates_number = f"{args.bates_prefix}-{bates_counter:07d}"
                bates_counter += 1
        evidence_stamp = args.evidence_stamp
        mifile_ready = bool(args.mifile_mode)
        record = FileRecord(
            source_path=path,
            root_label=root_label,
            relative_path=relative_path,
            size=size,
            mtime_utc=isoformat(dt.datetime.utcfromtimestamp(mtime).replace(tzinfo=dt.timezone.utc)),
            sha256=sha256,
            duplicate_of=duplicate_of,
            bates_number=bates_number,
            evidence_stamp=evidence_stamp,
            mifile_ready=mifile_ready,
        )
        entry_dict = record.manifest_dict(identifier)
        with lock:
            processed[key] = entry_dict
            new_entries += 1
            if duplicate_of is None:
                total = state.setdefault("total_bytes", 0)  # type: ignore[assignment]
                state["total_bytes"] = total + size  # type: ignore[index]
            else:
                state.setdefault("duplicate_bytes", 0)
                state["duplicate_bytes"] = state.get("duplicate_bytes", 0) + size  # type: ignore[index]
        if args.scan_secrets:
            findings = detect_secrets(path, logger)
            if findings:
                secrets_report[key] = findings
        return key

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(worker, item): item for item in candidates}
        processed_since_save = 0
        for future in as_completed(future_map):
            result = future.result()
            if result is None:
                skipped_existing += 1
                continue
            processed_since_save += 1
            if processed_since_save >= STATE_SAVE_INTERVAL:
                with lock:
                    save_state(state, state_path, temp_root)
                processed_since_save = 0
        if processed_since_save:
            with lock:
                save_state(state, state_path, temp_root)
    return new_entries, secrets_report, skipped_existing


# ---------------------------------------------------------------------------
# Output Writers
# ---------------------------------------------------------------------------


def write_manifests(
    manifest_rows: Sequence[Dict[str, object]],
    output_dir: Path,
    temp_root: Path,
) -> Tuple[Path, Path]:
    manifest_json = output_dir / "manifest.json"
    manifest_csv = output_dir / "manifest.csv"
    sorted_rows = sorted(manifest_rows, key=lambda row: row["id"])  # type: ignore[index]
    save_json_atomic(sorted_rows, manifest_json, temp_root)

    csv_lines: List[List[str]] = []
    headers = [
        "id",
        "root",
        "relative_path",
        "absolute_path",
        "size_bytes",
        "mtime_utc",
        "sha256",
        "duplicate_of",
        "bates_number",
        "evidence_stamp",
        "mifile_ready",
    ]
    csv_lines.append(headers)
    for row in sorted_rows:
        csv_lines.append([str(row.get(header, "")) for header in headers])
    ensure_directory(manifest_csv.parent)
    temp_path = temp_root / "manifest.csv.tmp"
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(csv_lines)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, manifest_csv)
    return manifest_json, manifest_csv


def create_bundle(
    entries: Sequence[FileRecord],
    output_dir: Path,
    temp_root: Path,
    logger: logging.Logger,
    bundle_name: str,
) -> Path:
    bundle_path = output_dir / bundle_name
    unique_entries = [entry for entry in entries if entry.duplicate_of is None]
    unique_entries.sort(key=lambda item: (item.root_label, item.relative_path.as_posix()))
    ensure_directory(bundle_path.parent)
    temp_zip_path = temp_root / "bundle.zip.tmp"
    with zipfile.ZipFile(temp_zip_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as bundle:
        for entry in unique_entries:
            arcname = Path(entry.root_label) / entry.relative_path
            source_path = path_with_prefix(entry.source_path)
            bundle.write(source_path, arcname=arcname.as_posix())
    with zipfile.ZipFile(temp_zip_path, "r") as validation:
        validation.testzip()
    os.replace(temp_zip_path, bundle_path)
    logger.info("Bundle created at %s", bundle_path)
    return bundle_path


def write_checksums(paths: Sequence[Path], output_dir: Path, temp_root: Path) -> Path:
    lines: List[str] = []
    for path in paths:
        sha = sha256_file(path)
        lines.append(f"{sha}  {path.name}")
    checksums_path = output_dir / "checksums.sha256"
    save_text_atomic("\n".join(lines) + "\n", checksums_path, temp_root)
    return checksums_path


def write_sqlite_index(manifest_rows: Sequence[Dict[str, object]], output_dir: Path, temp_root: Path) -> Path:
    db_path = output_dir / "manifest.sqlite"
    temp_db = temp_root / "manifest.sqlite.tmp"
    if temp_db.exists():
        temp_db.unlink()
    connection = sqlite3.connect(temp_db)
    try:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                root TEXT,
                relative_path TEXT,
                absolute_path TEXT,
                size_bytes INTEGER,
                mtime_utc TEXT,
                sha256 TEXT,
                duplicate_of TEXT,
                bates_number TEXT,
                evidence_stamp TEXT,
                mifile_ready INTEGER
            )
            """)
        cursor.executemany(
            """
            INSERT OR REPLACE INTO files (
                id, root, relative_path, absolute_path, size_bytes, mtime_utc,
                sha256, duplicate_of, bates_number, evidence_stamp, mifile_ready
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(row.get("id")),
                    str(row.get("root")),
                    str(row.get("relative_path")),
                    str(row.get("absolute_path")),
                    int(row.get("size_bytes", 0)),
                    str(row.get("mtime_utc")),
                    str(row.get("sha256")),
                    str(row.get("duplicate_of")) if row.get("duplicate_of") else None,
                    str(row.get("bates_number")) if row.get("bates_number") else None,
                    str(row.get("evidence_stamp")) if row.get("evidence_stamp") else None,
                    1 if row.get("mifile_ready") else 0,
                )
                for row in manifest_rows
            ],
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(FULL)")
        connection.close()
        os.replace(temp_db, db_path)
    finally:
        if connection:
            connection.close()
    return db_path


def write_secrets_report(secrets: Dict[str, List[Dict[str, str]]], output_dir: Path, temp_root: Path) -> Optional[Path]:
    if not secrets:
        return None
    report_path = output_dir / "potential_secrets.json"
    save_json_atomic(secrets, report_path, temp_root)
    return report_path


def summarize(entries: Sequence[FileRecord], secrets: Dict[str, List[Dict[str, str]]]) -> Dict[str, object]:
    total_size = sum(entry.size for entry in entries)
    duplicates = sum(1 for entry in entries if entry.duplicate_of is not None)
    unique = len(entries) - duplicates
    return {
        "total_files": len(entries),
        "unique_files": unique,
        "duplicate_files": duplicates,
        "total_bytes": total_size,
        "secrets_detected": len(secrets),
        "timestamp": isoformat(utc_now()),
    }


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble FredPrime Litigation OS bundle",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--roots",
        nargs="*",
        type=Path,
        default=DEFAULT_ROOTS,
        help="Root directories to scan",
    )
    parser.add_argument(
        "--include-optional-root",
        action="store_true",
        help="Include optional R:/ mirror if available",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=sorted(DEFAULT_EXTENSIONS),
        help="File extensions to include",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("OUTPUT"),
        help="Output directory root",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("LOGS"),
        help="Logs directory root",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Idempotency token (reuse to resume)",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from saved state")
    parser.add_argument("--dry-run", action="store_true", help="Perform scan without writing outputs")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden/system files")
    parser.add_argument("--max-workers", type=int, default=max(4, os.cpu_count() or 4), help="Thread workers")
    parser.add_argument("--bates-prefix", type=str, help="Assign Bates numbers with provided prefix")
    parser.add_argument("--evidence-stamp", type=str, help="Evidence stamp value for manifest entries")
    parser.add_argument("--mifile-mode", action="store_true", help="Flag manifest entries as MiFILE-ready")
    parser.add_argument("--scan-secrets", action="store_true", help="Enable secret detection report")
    parser.add_argument("--bundle-name", type=str, default="litigation_bundle.zip", help="Bundle file name")
    parser.add_argument(
        "--sqlite-index",
        action="store_true",
        help="Write SQLite manifest index",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    token = generate_token(args.token)

    if args.include_optional_root:
        args.roots.append(OPTIONAL_ROOT)

    output_root = args.output_root
    logs_root = args.logs_root
    ensure_directory(output_root)
    ensure_directory(logs_root)

    logger = setup_logging(logs_root, token)

    run_output_dir = output_root / f"bundle_{utc_now().strftime('%Y%m%d_%H%M%S')}_{token}"
    state_dir = output_root / "state"
    ensure_directory(state_dir)
    state_path = state_dir / f"{token}.json"

    temp_root = Path(tempfile.gettempdir()) / "TEMP" / token
    ensure_directory(temp_root)

    state = load_state(state_path) if args.resume else {"token": token, "started": isoformat(utc_now())}
    state["token"] = token
    state.setdefault("started", isoformat(utc_now()))

    logger.info("Scan roots: %s", ", ".join(str(root) for root in args.roots))
    candidates = gather_candidates(args.roots, args.include_hidden, args.extensions, logger)
    if not candidates:
        logger.warning("No files discovered. Exiting.")
        return 0

    new_entries, secrets_report, skipped_existing = process_candidates(
        candidates, args, logger, state, state_path, temp_root
    )

    manifest_rows: List[Dict[str, object]] = list(state.get("processed", {}).values())  # type: ignore[arg-type]

    def rows_to_records(rows: Sequence[Dict[str, object]]) -> List[FileRecord]:
        records: List[FileRecord] = []
        for row in rows:
            records.append(
                FileRecord(
                    source_path=Path(str(row.get("absolute_path"))),
                    root_label=str(row.get("root")),
                    relative_path=Path(str(row.get("relative_path"))),
                    size=int(row.get("size_bytes", 0)),
                    mtime_utc=str(row.get("mtime_utc")),
                    sha256=str(row.get("sha256")),
                    duplicate_of=str(row.get("duplicate_of")) if row.get("duplicate_of") else None,
                    bates_number=str(row.get("bates_number")) if row.get("bates_number") else None,
                    evidence_stamp=str(row.get("evidence_stamp")) if row.get("evidence_stamp") else None,
                    mifile_ready=bool(row.get("mifile_ready")),
                )
            )
        return records

    all_records = rows_to_records(manifest_rows)

    state_summary = summarize(all_records, secrets_report)
    state["summary"] = state_summary
    save_state(state, state_path, temp_root)

    logger.info("Processed %s new files (%s previously processed)", new_entries, skipped_existing)
    logger.info("Total bytes captured: %s", sum(entry.size for entry in all_records))

    if args.dry_run:
        logger.info("Dry-run requested; skipping artifact generation")
        return 0

    ensure_directory(run_output_dir)
    if run_output_dir.exists() and any(run_output_dir.iterdir()) and not args.force:
        logger.error("Output directory %s already contains files; use --force to overwrite", run_output_dir)
        return 1

    required_space = estimated_required_space(sum(entry.size for entry in all_records))
    check_free_space(run_output_dir, required_space)

    manifest_json, manifest_csv = write_manifests(manifest_rows, run_output_dir, temp_root)
    bundle_path = create_bundle(all_records, run_output_dir, temp_root, logger, args.bundle_name)
    checksum_targets = [manifest_json, manifest_csv, bundle_path]
    sqlite_path: Optional[Path] = None
    if args.sqlite_index:
        sqlite_path = write_sqlite_index(manifest_rows, run_output_dir, temp_root)
        checksum_targets.append(sqlite_path)
    secrets_path = write_secrets_report(secrets_report, run_output_dir, temp_root)
    if secrets_path:
        checksum_targets.append(secrets_path)

    checksums_path = run_output_dir / "checksums.sha256"
    state["artifacts"] = {
        "manifest_json": str(manifest_json),
        "manifest_csv": str(manifest_csv),
        "bundle_zip": str(bundle_path),
        "checksums": str(checksums_path),
        "sqlite": str(sqlite_path) if sqlite_path else None,
        "secrets_report": str(secrets_path) if secrets_path else None,
        "output_dir": str(run_output_dir),
    }
    save_state(state, state_path, temp_root)

    summary = {
        **state_summary,
        "output_dir": str(run_output_dir),
        "artifacts": state["artifacts"],
    }
    summary_path = run_output_dir / "summary.json"
    save_json_atomic(summary, summary_path, temp_root)
    checksum_targets.append(summary_path)

    checksums_path = write_checksums(checksum_targets, run_output_dir, temp_root)
    checksum_targets.append(checksums_path)
    save_state(state, state_path, temp_root)

    logger.info("Artifacts written to %s", run_output_dir)
    logger.info("Summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
