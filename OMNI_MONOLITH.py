#!/usr/bin/env python3
"""Court-deployable monolithic bundle builder for Litigation OS."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ----------------------------- Constants -----------------------------------
DEFAULT_EXTENSIONS = {".py", ".ps1", ".json", ".html", ".htm"}
DEFAULT_ROOTS = [Path("F:/"), Path("D:/"), Path("Z:/"), Path("R:/")]
OUTPUT_DIR = Path("OUTPUT")
LOG_DIR = Path("LOGS")
STATE_DIR = OUTPUT_DIR / "STATE"
TEMP_ROOT = OUTPUT_DIR / "TEMP"
SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ASIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]+PRIVATE KEY-----"),
    re.compile(r"(?i)secret[_-]?key\s*[:=]\s*['\"][A-Za-z0-9/+]{8,}['\"]"),
]
CSV_FIELDS = [
    "sha256",
    "relative_path",
    "original_path",
    "size",
    "mtime_utc",
    "extension",
    "normalized",
    "notes",
]
BUNDLE_NAME = "OMNI_BUNDLE.zip"
CHECKSUM_FILE = "checksums.sha256"
SECRET_REPORT = "secret_findings.json"
JSON_MANIFEST = "manifest.json"
CSV_MANIFEST = "manifest.csv"
SUMMARY_FILE = "SUMMARY.txt"
JSON_LOG_SUFFIX = ".jsonl"
HUMAN_LOG_SUFFIX = ".log"
IDEMPOTENCY_ENV = "OMNI_TOKEN"
DEFAULT_MAX_WORKERS = max(4, (os.cpu_count() or 4) // 2)
STATE_FLUSH_INTERVAL = 10


# ----------------------------- Data Models ---------------------------------
@dataclass
class RunConfig:
    token: str
    timestamp: str
    roots: List[Path]
    extensions: List[str]
    include_hidden: bool
    normalize: bool
    dry_run: bool
    force: bool
    max_workers: int
    resume: bool


@dataclass
class FileRecord:
    sha256: str
    original_path: str
    relative_path: str
    size: int
    mtime_utc: str
    extension: str
    normalized: bool
    notes: List[str]
    secrets: List[str]


@dataclass
class RunContext:
    config: RunConfig
    output_dir: Path
    log_path: Path
    json_log_path: Path
    state_path: Path
    logger: logging.Logger
    processed: Dict[str, FileRecord]


# ----------------------------- Utilities -----------------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_directories(context: RunContext) -> None:
    for path in [OUTPUT_DIR, LOG_DIR, STATE_DIR, TEMP_ROOT, context.output_dir.parent]:
        path.mkdir(parents=True, exist_ok=True)


def generate_token(user_token: Optional[str]) -> str:
    if user_token:
        return re.sub(r"[^A-Za-z0-9_-]", "", user_token)[:48] or "token"
    if IDEMPOTENCY_ENV in os.environ:
        env_token = os.environ[IDEMPOTENCY_ENV].strip()
        if env_token:
            return re.sub(r"[^A-Za-z0-9_-]", "", env_token)[:48] or "token"
    entropy = hashlib.sha256(f"{time.time_ns()}_{os.getpid()}".encode()).hexdigest()
    return entropy[:24]


def default_tokenised_output(token: str, timestamp: str) -> Path:
    folder_name = f"OMNI_BUNDLE_OUT_{timestamp}_{token}"
    return OUTPUT_DIR / folder_name


def safe_windows_path(path: Path) -> Path:
    if os.name == "nt":
        raw = str(path)
        if not raw.startswith("\\\\?\\") and len(raw) >= 240:
            return Path("\\\\?\\" + raw)
    return path


def is_hidden(path: Path) -> bool:
    name = path.name
    if name.startswith("."):
        return True
    if os.name == "nt":
        try:
            import ctypes

            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
            if attrs == -1:
                return False
            return bool(attrs & 2)
        except Exception:
            return False
    return False


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=str(path.parent), delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
    Path(tmp.name).replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write(path, text.encode("utf-8"))


def configure_logging(context: RunContext) -> None:
    logger = logging.getLogger("OMNI_MONOLITH")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)sZ [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        context.log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    json_handler = RotatingFileHandler(
        context.json_log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    json_handler.setLevel(logging.DEBUG)
    json_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(JsonLogHandler(json_handler))

    context.logger = logger


class JsonLogHandler(logging.Handler):
    """Wrapper that writes structured logs to a rotating handler."""

    def __init__(self, handler: RotatingFileHandler) -> None:
        super().__init__(level=logging.DEBUG)
        self._handler = handler
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        raw = json.dumps(log_entry, ensure_ascii=False)
        with self._lock:
            self._handler.emit(logging.makeLogRecord({"msg": raw}))

    def flush(self) -> None:
        with self._lock:
            self._handler.flush()


def discover_files(
    roots: Iterable[Path], extensions: Sequence[str], include_hidden: bool, logger: logging.Logger
) -> List[Path]:
    candidates: List[Path] = []
    for root in roots:
        if not root.exists():
            logger.debug("Skipping missing root %s", root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            current = Path(dirpath)
            if not include_hidden and is_hidden(current):
                dirnames[:] = []
                continue
            filtered_dirs = []
            for name in dirnames:
                child = current / name
                if include_hidden or not is_hidden(child):
                    filtered_dirs.append(name)
            dirnames[:] = filtered_dirs
            for filename in filenames:
                path = current / filename
                try:
                    if not path.is_file():
                        continue
                except OSError:
                    continue
                if not include_hidden and is_hidden(path):
                    continue
                if extensions and path.suffix.lower() not in extensions:
                    continue
                candidates.append(path)
    logger.info("Discovered %d candidate files", len(candidates))
    return candidates


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(safe_windows_path(path), "rb", buffering=1024 * 1024) as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path, limit: int = 5_000_000) -> Tuple[str, List[str]]:
    notes: List[str] = []
    try:
        data = path.read_bytes()
    except OSError as exc:
        return "", [f"read_error:{exc}"]
    truncated = data[:limit]
    if len(data) > limit:
        notes.append("truncated_for_analysis")
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return truncated.decode(encoding, errors="replace"), notes
        except UnicodeDecodeError:
            continue
    return truncated.decode("utf-8", errors="replace"), notes


def normalize_content(path: Path, data: bytes) -> Tuple[bytes, bool, List[str]]:
    ext = path.suffix.lower()
    if ext not in {".py", ".ps1", ".json", ".html", ".htm"}:
        return data, False, []
    text, notes = read_text(path)
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if ext == ".json":
        try:
            obj = json.loads(normalized)
            normalized = json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
        except json.JSONDecodeError:
            notes.append("json_unparsed")
    return normalized.encode("utf-8"), True, notes


def detect_secrets(text: str) -> List[str]:
    hits: List[str] = []
    for pattern in SECRET_PATTERNS:
        match = pattern.search(text)
        if match:
            hits.append(pattern.pattern)
    return hits


def format_bytes(value: int) -> str:
    suffixes = ["B", "KB", "MB", "GB", "TB"]
    result = float(value)
    for suffix in suffixes:
        if result < 1024 or suffix == suffixes[-1]:
            return f"{result:.2f} {suffix}"
        result /= 1024
    return f"{value} B"


def load_state(context: RunContext) -> Dict[str, FileRecord]:
    if not context.state_path.exists():
        return {}
    data = json.loads(context.state_path.read_text(encoding="utf-8"))
    records = {}
    for entry in data.get("records", []):
        records[entry["sha256"]] = FileRecord(
            sha256=entry["sha256"],
            original_path=entry["original_path"],
            relative_path=entry["relative_path"],
            size=entry["size"],
            mtime_utc=entry["mtime_utc"],
            extension=entry["extension"],
            normalized=entry.get("normalized", False),
            notes=entry.get("notes", []),
            secrets=entry.get("secrets", []),
        )
    context.logger.info("Loaded %d records from state for resumability", len(records))
    return records


def flush_state(context: RunContext) -> None:
    payload = {
        "token": context.config.token,
        "timestamp": context.config.timestamp,
        "records": [asdict(rec) for rec in context.processed.values()],
    }
    atomic_write_text(context.state_path, json.dumps(payload, ensure_ascii=False, indent=2))


def check_free_space(target: Path, required: int, logger: logging.Logger) -> None:
    usage = shutil.disk_usage(str(target))
    if usage.free < required:
        raise RuntimeError(
            f"Insufficient free space on {target}: required {format_bytes(required)}, available {format_bytes(usage.free)}"
        )
    logger.info(
        "Free space OK on %s (required %s, available %s)",
        target,
        format_bytes(required),
        format_bytes(usage.free),
    )


def inspect_file(path: Path, normalize: bool) -> Tuple[Optional[FileRecord], Optional[str], bytes, bool, List[str]]:
    try:
        stat = path.stat()
    except OSError as exc:
        return None, f"stat_error:{exc}", b"", False, []
    try:
        sha = compute_sha256(path)
    except OSError as exc:
        return None, f"hash_error:{exc}", b"", False, []
    raw_bytes = path.read_bytes()
    normalized_bytes, normalized, notes = (raw_bytes, False, [])
    if normalize:
        normalized_bytes, normalized, normalize_notes = normalize_content(path, raw_bytes)
        notes.extend(normalize_notes)
    text_preview = ""
    secrets: List[str] = []
    if normalized or path.suffix.lower() in {".py", ".ps1", ".json", ".html", ".htm"}:
        text_preview, extra_notes = read_text(path, limit=1_000_000)
        notes.extend(extra_notes)
        secrets = detect_secrets(text_preview)
    record = FileRecord(
        sha256=sha,
        original_path=str(path),
        relative_path="",
        size=stat.st_size,
        mtime_utc=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        extension=path.suffix.lower(),
        normalized=normalized,
        notes=notes,
        secrets=secrets,
    )
    return record, None, normalized_bytes, normalized, secrets


def sanitize_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return safe[:120]


def store_file(record: FileRecord, payload: bytes, context: RunContext) -> FileRecord:
    ext = record.extension or ""
    subdir = {
        ".json": "json",
        ".html": "html",
        ".htm": "html",
        ".py": "py",
        ".ps1": "ps1",
    }.get(ext, "misc")
    filename = f"{record.sha256[:12]}_{sanitize_filename(Path(record.original_path).name)}"
    destination = context.output_dir / subdir / filename
    atomic_write(destination, payload)
    record.relative_path = str(destination.relative_to(context.output_dir))
    return record


def write_manifest(context: RunContext, records: List[FileRecord], elapsed: float, total_bytes: int) -> None:
    manifest = {
        "run": asdict(context.config),
        "generated_at": utc_now_iso(),
        "file_count": len(records),
        "total_bytes": total_bytes,
        "files": [asdict(r) for r in records],
    }
    atomic_write_text(context.output_dir / JSON_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))

    csv_path = context.output_dir / CSV_MANIFEST
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=str(csv_path.parent), delete=False) as tmp:
        writer = csv.DictWriter(tmp, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "sha256": record.sha256,
                    "relative_path": record.relative_path,
                    "original_path": record.original_path,
                    "size": record.size,
                    "mtime_utc": record.mtime_utc,
                    "extension": record.extension,
                    "normalized": record.normalized,
                    "notes": "|".join(record.notes),
                }
            )
        tmp.flush()
        os.fsync(tmp.fileno())
    Path(tmp.name).replace(csv_path)

    checksums_path = context.output_dir / CHECKSUM_FILE
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(checksums_path.parent), delete=False) as tmp:
        for record in records:
            tmp.write(f"{record.sha256}  {record.relative_path}\n")
        tmp.flush()
        os.fsync(tmp.fileno())
    Path(tmp.name).replace(checksums_path)

    summary_lines = [
        f"Run token: {context.config.token}",
        f"Generated: {datetime.fromisoformat(context.config.timestamp).isoformat()}",
        f"Processed files: {len(records)}",
        f"Total size: {format_bytes(total_bytes)}",
        f"Elapsed: {elapsed:.2f}s",
        f"Output: {context.output_dir}",
    ]
    atomic_write_text(context.output_dir / SUMMARY_FILE, "\n".join(summary_lines) + "\n")


def write_secret_report(context: RunContext, records: List[FileRecord]) -> None:
    findings = []
    for record in records:
        if record.secrets:
            findings.append(
                {
                    "sha256": record.sha256,
                    "original_path": record.original_path,
                    "relative_path": record.relative_path,
                    "patterns": record.secrets,
                }
            )
    atomic_write_text(context.output_dir / SECRET_REPORT, json.dumps(findings, ensure_ascii=False, indent=2))


def build_bundle(context: RunContext, records: List[FileRecord]) -> None:
    bundle_path = context.output_dir / BUNDLE_NAME
    with tempfile.NamedTemporaryFile(dir=str(bundle_path.parent), delete=False) as tmp:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for record in sorted(records, key=lambda r: (r.extension, r.relative_path)):
                source = context.output_dir / record.relative_path
                archive.write(source, arcname=record.relative_path)
        tmp.flush()
        os.fsync(tmp.fileno())
    Path(tmp.name).replace(bundle_path)
    with zipfile.ZipFile(bundle_path, "r") as archive:
        archive.testzip()


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OMNI bundle from local drives")
    parser.add_argument("--roots", nargs="*", type=Path, default=None, help="Roots to scan")
    parser.add_argument("--extensions", nargs="*", default=None, help="File extensions to include")
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden files and folders")
    parser.add_argument("--normalize", action="store_true", help="Normalize text files to UTF-8/LF")
    parser.add_argument("--dry-run", action="store_true", help="Scan only; no copies or bundles")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directory")
    parser.add_argument("--token", help="Idempotency token to reuse run outputs")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Thread pool size")
    parser.add_argument("--resume", action="store_true", help="Resume from previous run state")
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console log level"
    )
    return parser.parse_args(argv)


def prepare_context(args: argparse.Namespace) -> RunContext:
    initial_timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    token = generate_token(args.token)
    if args.resume and not args.token:
        raise RuntimeError("--resume requires --token to reference a prior run")
    state_path = STATE_DIR / f"state_{token}.json"
    timestamp = initial_timestamp
    if args.resume:
        if not state_path.exists():
            raise RuntimeError(f"Resume requested but state file {state_path} not found")
        try:
            previous = json.loads(state_path.read_text(encoding="utf-8"))
            timestamp = previous.get("timestamp", initial_timestamp)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse state file {state_path}: {exc}") from exc
    roots = args.roots if args.roots else [root for root in DEFAULT_ROOTS if root.exists()]
    if not roots:
        roots = [Path.cwd()]
    extensions = [
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in (args.extensions or sorted(DEFAULT_EXTENSIONS))
    ]
    config = RunConfig(
        token=token,
        timestamp=timestamp,
        roots=list(roots),
        extensions=extensions,
        include_hidden=args.include_hidden,
        normalize=args.normalize,
        dry_run=args.dry_run,
        force=args.force,
        max_workers=max(1, args.max_workers),
        resume=args.resume,
    )
    output_dir = default_tokenised_output(token, timestamp)
    if output_dir.exists() and not config.force and not config.resume:
        raise RuntimeError(f"Output directory {output_dir} already exists. Use --force or --resume.")
    log_name = f"OMNI_MONOLITH{timestamp}{token}{HUMAN_LOG_SUFFIX}"
    json_log_name = f"OMNI_MONOLITH{timestamp}{token}{JSON_LOG_SUFFIX}"
    context = RunContext(
        config=config,
        output_dir=output_dir,
        log_path=LOG_DIR / log_name,
        json_log_path=LOG_DIR / json_log_name,
        state_path=state_path,
        logger=logging.getLogger("OMNI_MONOLITH"),
        processed={},
    )
    ensure_directories(context)
    configure_logging(context)
    console_level = getattr(logging, args.log_level.upper(), logging.INFO)
    for handler in context.logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(console_level)
    if config.resume:
        context.processed = load_state(context)
    return context


def validate_roots(roots: Iterable[Path]) -> None:
    for root in roots:
        if not root.exists():
            raise RuntimeError(f"Root {root} does not exist")
        if not root.is_dir():
            raise RuntimeError(f"Root {root} is not a directory")


def run_pipeline(context: RunContext) -> int:
    logger = context.logger
    validate_roots(context.config.roots)
    start = time.time()
    candidates = discover_files(
        roots=context.config.roots,
        extensions=context.config.extensions,
        include_hidden=context.config.include_hidden,
        logger=logger,
    )
    if not candidates:
        logger.warning("No files discovered; exiting")
        return 0

    total_bytes = sum((path.stat().st_size for path in candidates if path.exists()))
    check_free_space(context.output_dir.parent, total_bytes * (2 if not context.config.dry_run else 1), logger)

    dedupe: Dict[str, FileRecord] = dict(context.processed)
    payloads: Dict[str, bytes] = {}

    def process(path: Path) -> Tuple[Optional[FileRecord], Optional[str], bytes]:
        record, error, data, _normalized, _secrets = inspect_file(path, context.config.normalize)
        return record, error, data

    with ThreadPoolExecutor(max_workers=context.config.max_workers) as executor:
        future_map = {executor.submit(process, path): path for path in candidates}
        processed_count = 0
        for future in as_completed(future_map):
            path = future_map[future]
            try:
                record, error, data = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to process %s: %s", path, exc)
                continue
            if error:
                logger.warning("Skipped %s due to %s", path, error)
                continue
            assert record is not None
            if record.sha256 in dedupe:
                existing = dedupe[record.sha256]
                duplicate_note = f"duplicate:{path}"
                if duplicate_note not in existing.notes:
                    existing.notes.append(duplicate_note)
                continue
            dedupe[record.sha256] = record
            payloads[record.sha256] = data
            processed_count += 1
            if processed_count % STATE_FLUSH_INTERVAL == 0:
                context.processed = dedupe
                flush_state(context)
    context.processed = dedupe
    flush_state(context)

    unique_records = sorted(dedupe.values(), key=lambda item: item.sha256)

    if context.config.dry_run:
        elapsed = time.time() - start
        write_manifest(context, unique_records, elapsed, total_bytes)
        write_secret_report(context, unique_records)
        logger.info("Dry run complete with %d unique files", len(unique_records))
        return 0

    stored_records: List[FileRecord] = []
    for record in unique_records:
        payload = payloads.get(record.sha256)
        if payload is None:
            try:
                payload = Path(record.original_path).read_bytes()
            except OSError as exc:
                logger.error("Missing payload for %s: %s", record.original_path, exc)
                continue
        stored = store_file(record, payload, context)
        stored_records.append(stored)
        context.processed[stored.sha256] = stored
        if len(stored_records) % STATE_FLUSH_INTERVAL == 0:
            flush_state(context)
    flush_state(context)

    elapsed = time.time() - start
    stored_records_sorted = sorted(stored_records, key=lambda item: item.sha256)
    write_manifest(context, stored_records_sorted, elapsed, total_bytes)
    write_secret_report(context, stored_records_sorted)
    build_bundle(context, stored_records_sorted)
    elapsed_summary = f"Processed {len(stored_records_sorted)} unique files in {elapsed:.2f}s"
    logger.info(elapsed_summary)
    context.logger.info("Artifacts at %s", context.output_dir)
    return len(stored_records_sorted)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_arguments(argv)
        context = prepare_context(args)
        result = run_pipeline(context)
        return 0 if result >= 0 else 2
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("OMNI_MONOLITH").error("Fatal error: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
