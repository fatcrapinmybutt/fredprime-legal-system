"""windows_drive_organizer
=================================

Court-safe Windows evidence organizer with persistent dedupe, resumable runs,
rotating logs, manifest generation, and bundle packaging.

This module is intentionally self-contained and Windows-aware, but it also runs
on POSIX hosts for development/testing purposes.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import shutil
import sqlite3
import sys
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from core.assets_registry import AssetsRegistry, AssetsRegistryError
from core.network_policy import NetworkPolicy, NetworkPolicyError


logging.Formatter.converter = time.gmtime


TOOL_NAME = "drive_organizer"
REQUIRED_DRIVE_LETTERS = {"Q", "D", "Z"}
DEFAULT_DRIVES = [Path(r"Q:/"), Path(r"D:/"), Path(r"Z:/")]
DEFAULT_DRIVE_SCAN_ORDER = ["F", "D", "Z", "Q", "E"]
DEFAULT_EXTENSIONS = [".py", ".ps1", ".json", ".html"]
DEFAULT_SECRET_EXTENSIONS = {".py", ".ps1", ".json", ".txt", ".cfg", ".ini"}
DEFAULT_DENY_DIRS = {
    "windows",
    "program files",
    "program files (x86)",
    "$recycle.bin",
    "system volume information",
    "node_modules",
    ".git",
    "__pycache__",
    "venv",
    "dist",
    "build",
}
LOG_BYTES = 5 * 1024 * 1024
JSONL_LOG_NAME = f"{TOOL_NAME}.jsonl"
DEFAULT_OUTPUT_ROOT_BASE = Path(r"Z:/LitigationOS/Runs")
DEFAULT_TEMP_ROOT = Path(r"Z:/LitigationOS/_TMP")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_timestamp() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def long_path(path: Path) -> Path:
    if os.name == "nt":
        raw = str(path)
        if raw.startswith("\\\\?\\"):
            return path
        if path.is_absolute():
            return Path(r"\\\\?\\" + raw)
    return path


def normalize_extension(ext: str) -> str:
    ext = ext.strip().lower()
    if not ext.startswith("."):
        ext = f".{ext}"
    return ext


def human_bytes(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.2f} {unit}"
        num /= 1024
    return f"{num:.2f} PB"


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(chunk_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


def is_hidden(path: Path) -> bool:
    if path.name.startswith("."):
        return True
    if os.name != "nt":
        return False
    try:
        import ctypes

        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        return bool(attrs != -1 and attrs & 2)
    except Exception:
        return False


def atomic_write_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def _drive_letter(path: Path) -> str:
    drive = path.drive or path.anchor
    return drive.strip(":\\/").upper()


def _is_c_drive(path: Path) -> bool:
    return _drive_letter(path) == "C"


def discover_eligible_drives(
    scan_order: Sequence[str] = DEFAULT_DRIVE_SCAN_ORDER,
    exclude: Iterable[str] = ("C",),
) -> List[Path]:
    if os.name != "nt":
        return []
    excluded = {letter.upper() for letter in exclude}
    candidates: List[Path] = []
    for letter in scan_order:
        upper = letter.upper()
        if upper in excluded:
            continue
        root = Path(f"{upper}:/")
        if root.exists():
            candidates.append(root)
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if letter in excluded or letter in scan_order:
            continue
        root = Path(f"{letter}:/")
        if root.exists():
            candidates.append(root)
    return candidates


def ensure_roots(drives: Iterable[Path]) -> None:
    base_roots = [
        Path("LitigationOS/Runs"),
        Path("LitigationOS/State"),
        Path("LitigationOS/Logs"),
        Path("LitigationOS/Quarantine"),
        Path("LitigationOS/_TMP"),
    ]
    specialized_roots = [
        (Path("Q:/"), Path("LitigationOS/Evidence/Originals")),
        (Path("Q:/"), Path("LitigationOS/Evidence/Intake")),
        (Path("Q:/"), Path("LitigationOS/Evidence/Media")),
        (Path("Z:/"), Path("LitigationOS/Vault")),
        (Path("Z:/"), Path("LitigationOS/Graph/Neo4j")),
        (Path("Z:/"), Path("LitigationOS/Authority")),
        (Path("D:/"), Path("LitigationOS/Revenue")),
        (Path("D:/"), Path("LitigationOS/Builds")),
        (Path("D:/"), Path("LitigationOS/Releases")),
    ]
    for drive in drives:
        for root in base_roots:
            ensure_dir(drive / root)
    for drive, root in specialized_roots:
        ensure_dir(drive / root)


class JsonlLogHandler(logging.Handler):
    def __init__(self, jsonl_path: Path) -> None:
        super().__init__()
        self.jsonl_path = jsonl_path
        ensure_dir(jsonl_path.parent)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        payload = {
            "ts": utc_now().isoformat(),
            "level": record.levelname,
            "message": self.format(record),
        }
        line = json.dumps(payload, ensure_ascii=False)
        with self._lock:
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class DedupeIndex:
    def __init__(self, index_path: Path) -> None:
        self.path = index_path
        self._lock = threading.Lock()
        self._index: Dict[str, dict] = {}
        if index_path.exists():
            try:
                self._index = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                self._index = {}

    def lookup(self, sha: str) -> Optional[dict]:
        with self._lock:
            return self._index.get(sha)

    def add(self, sha: str, record: dict) -> None:
        with self._lock:
            if sha not in self._index:
                self._index[sha] = record

    def save(self) -> None:
        atomic_write_json(self.path, self._index)


class StateTracker:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: Dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                    if "source" in rec:
                        self._data[rec["source"]] = rec
                except Exception:
                    continue
        ensure_dir(path.parent)
        self._lock = threading.Lock()

    def should_skip(self, src: Path, size: int, mtime: float, force: bool) -> bool:
        if force:
            return False
        key = str(src)
        existing = self._data.get(key)
        if not existing:
            return False
        return existing.get("size") == size and existing.get("mtime") == mtime

    def append(self, record: dict) -> None:
        key = record.get("source")
        if key:
            self._data[key] = record
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


class SQLiteIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        ensure_dir(path.parent)
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence (
                sha256 TEXT PRIMARY KEY,
                source_path TEXT,
                dest_path TEXT,
                size INTEGER,
                mtime REAL,
                recorded_at TEXT
            )
            """
        )
        self.conn.commit()
        self._lock = threading.Lock()

    def upsert(self, sha: str, source: str, dest: str, size: int, mtime: float) -> None:
        with self._lock:
            self.conn.execute(
                "REPLACE INTO evidence (sha256, source_path, dest_path, size, mtime, recorded_at) VALUES (?, ?, ?, ?, ?, ?)",
                (sha, source, dest, size, mtime, utc_now().isoformat()),
            )
            self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


class SecretScanner:
    SECRET_PATTERNS = [
        ("aws_access_key", r"AKIA[0-9A-Z]{16}"),
        ("generic_secret", r"(?i)(password|api_key|secret)[\s:=]+[\w\-]{8,}"),
        ("token", r"(?i)token[\s:=]+[A-Za-z0-9\-_]{12,}"),
    ]

    def __init__(self) -> None:
        import re

        self._compiled = [(name, re.compile(pattern)) for name, pattern in self.SECRET_PATTERNS]

    def scan(self, path: Path) -> List[dict]:
        findings: List[dict] = []
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return findings
        for label, regex in self._compiled:
            for match in regex.finditer(text):
                snippet = match.group(0)
                redacted = snippet[:2] + "***" + snippet[-2:]
                findings.append(
                    {
                        "path": str(path),
                        "pattern": label,
                        "snippet": redacted,
                    }
                )
        return findings


def sanitize_label(value: str) -> str:
    cleaned = [ch for ch in value if ch.isalnum() or ch in {"_", "-", "."}]
    label = "".join(cleaned) or "branch"
    return label[:64]


@dataclass
class FileTask:
    source: Path
    drive_root: Path
    extension: str
    branch: Optional[str] = None


class EvidenceOrganizer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.args.extensions = [normalize_extension(ext) for ext in self.args.extensions]
        self.args.secret_exts = {normalize_extension(ext) for ext in self.args.secret_exts}
        if self.args.auto_drives and os.name == "nt":
            discovered = discover_eligible_drives()
            if not discovered:
                raise RuntimeError("No eligible drives discovered for auto-drive mode.")
            self.args.drives = [str(path) for path in discovered]
        self.args.drives = [Path(str(d)) for d in self.args.drives]
        self.args.deny_dirs = [d.lower() for d in self.args.deny_dirs]
        self.args.max_workers = max(1, int(self.args.max_workers))
        self.token = args.token or uuid.uuid4().hex[:10]
        self.timestamp = utc_timestamp()
        self.run_id = f"RUN_{self.timestamp}_{self.token}"
        output_root_input = Path(args.output_root)
        if output_root_input == DEFAULT_OUTPUT_ROOT_BASE:
            output_root_input = DEFAULT_OUTPUT_ROOT_BASE / self.run_id
        self.output_root = output_root_input.resolve()
        self.logs_dir = self.output_root / "LOGS"
        temp_root = Path(args.temp_root).resolve()
        self.temp_dir = temp_root / f"{TOOL_NAME}_{self.token}"
        self.collect_dir = self.output_root / "COLLECTED"
        self.state_tracker = StateTracker(self.logs_dir / "run_state.jsonl")
        self.dedupe_index = DedupeIndex(self.output_root / "dedupe_index.json")
        self.secret_scanner = SecretScanner()
        self.sqlite_index = SQLiteIndex(self.logs_dir / "evidence.db") if args.sqlite_index else None
        self.network_policy = self._load_network_policy()
        self.assets_registry = self._load_assets_registry()
        self.bates_counter = max(1, int(args.bates_start))
        self._bates_lock = threading.Lock()
        self.branch_roots = self._load_branch_roots()
        self._setup_logging()
        self.manifest_entries: List[dict] = []
        self.secret_findings: List[dict] = []
        self._registry_lock = threading.Lock()
        self.stats = {
            "scanned": 0,
            "copied": 0,
            "duplicates": 0,
            "skipped": 0,
            "dry": 0,
            "errors": 0,
            "bytes_copied": 0,
            "bates_assigned": 0,
            "branch_sources": len(self.branch_roots),
        }
        self._validate_paths()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> None:
        ensure_dir(self.logs_dir)
        datestamp = self.timestamp[:8]
        log_name = f"{TOOL_NAME}{datestamp}{self.token}.log"
        log_path = self.logs_dir / log_name
        jsonl_path = self.logs_dir / JSONL_LOG_NAME
        handler = RotatingFileHandler(log_path, maxBytes=LOG_BYTES, backupCount=3, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)sZ | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        level = getattr(logging, self.args.log_level.upper(), logging.INFO)
        logging.basicConfig(level=level, handlers=[handler], force=True)
        jsonl_handler = JsonlLogHandler(jsonl_path)
        jsonl_handler.setLevel(logging.INFO)
        jsonl_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(jsonl_handler)
        logging.info("Run token: %s", self.token)
        self.log_path = log_path
        self.jsonl_log_path = jsonl_path

    def _guard_non_c_drive(self, path: Path, label: str) -> None:
        if self.args.allow_c_drive:
            return
        if _is_c_drive(path):
            raise RuntimeError(f"{label} is on C: which is disallowed by policy: {path}")

    def _validate_drives(self) -> None:
        if os.name != "nt":
            return
        drive_letters = {_drive_letter(path) for path in self.args.drives}
        if "C" in drive_letters and not self.args.allow_c_drive:
            raise RuntimeError("C: is disallowed by policy for drive roots.")
        missing = REQUIRED_DRIVE_LETTERS - drive_letters
        if missing:
            raise RuntimeError(f"Missing required drives: {', '.join(sorted(missing))}")
        for drive in self.args.drives:
            if not drive.exists():
                raise RuntimeError(f"Required drive root is unavailable: {drive}")

    def _validate_paths(self) -> None:
        self._guard_non_c_drive(self.output_root, "Output root")
        self._guard_non_c_drive(self.temp_dir, "Temp root")
        self._validate_drives()

    def _load_network_policy(self) -> Optional[NetworkPolicy]:
        if not self.args.network_policy:
            return None
        try:
            policy = NetworkPolicy.from_path(Path(self.args.network_policy))
            logging.info("Loaded network policy from %s", policy.path)
            return policy
        except NetworkPolicyError as exc:
            raise RuntimeError(f"Network policy error: {exc}") from exc

    def _load_assets_registry(self) -> Optional[AssetsRegistry]:
        if not self.args.assets_registry:
            return None
        try:
            registry = AssetsRegistry.from_path(Path(self.args.assets_registry))
            return registry
        except AssetsRegistryError as exc:
            raise RuntimeError(f"Assets registry error: {exc}") from exc

    def _finalize_run(self) -> None:
        if self.sqlite_index:
            self.sqlite_index.close()
            self.sqlite_index = None
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _serializable_args(self) -> Dict[str, object]:
        serializable: Dict[str, object] = {}
        for key, value in vars(self.args).items():
            if isinstance(value, Path):
                serializable[key] = str(value)
            elif isinstance(value, (list, tuple, set)):
                serializable[key] = [str(v) if isinstance(v, Path) else v for v in value]
            else:
                serializable[key] = value
        return serializable

    def validate_space(self, estimated_bytes: int) -> None:
        usage = shutil.disk_usage(str(self.output_root))
        logging.info(
            "Free space: %s, Needed: %s",
            human_bytes(usage.free),
            human_bytes(estimated_bytes),
        )
        if estimated_bytes == 0:
            return
        if usage.free < estimated_bytes and not self.args.force:
            raise RuntimeError("Insufficient disk space. Use --force to override.")

    def _load_branch_roots(self) -> Dict[str, Path]:
        branches: Dict[str, Path] = {}
        specs: List[str] = list(self.args.branch or [])
        if self.args.branches_file:
            branch_file = Path(self.args.branches_file)
            if branch_file.exists():
                try:
                    payload = json.loads(branch_file.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        for key, value in payload.items():
                            specs.append(f"{key}={value}")
                    elif isinstance(payload, list):
                        for entry in payload:
                            if isinstance(entry, dict) and entry.get("label") and entry.get("path"):
                                specs.append(f"{entry['label']}={entry['path']}")
                except Exception as exc:
                    logging.warning("Failed to parse branches file %s: %s", branch_file, exc)
        for spec in specs:
            label, path = self._parse_branch_spec(spec)
            if not label:
                continue
            branches[label] = path
        if branches:
            logging.info("Loaded %d branch expansions", len(branches))
        return branches

    def _parse_branch_spec(self, spec: str) -> Tuple[str, Path]:
        if "=" not in spec:
            logging.warning("Invalid branch spec (expected label=path): %s", spec)
            return "", Path(".")
        label, raw_path = spec.split("=", 1)
        label = sanitize_label(label.strip())
        path = Path(raw_path.strip()).expanduser().resolve()
        return label, path

    def discover_files(self) -> List[FileTask]:
        tasks: List[FileTask] = []
        sources: List[Tuple[Path, Optional[str]]] = [(Path(drive), None) for drive in self.args.drives]
        for label, path in self.branch_roots.items():
            sources.append((Path(path), label))
        for base_root, branch_label in sources:
            if not base_root.exists():
                if branch_label:
                    logging.warning("Branch %s path %s is unavailable", branch_label, base_root)
                else:
                    logging.warning("Drive %s is unavailable", base_root)
                continue
            for root, dirs, files in os.walk(base_root, topdown=True):
                current_root = Path(root)
                if not self.args.include_hidden:
                    dirs[:] = [d for d in dirs if not is_hidden(current_root / d)]
                dirs[:] = [d for d in dirs if d.lower() not in self.args.deny_dirs]
                for name in files:
                    candidate = current_root / name
                    if not self.args.include_hidden and is_hidden(candidate):
                        continue
                    ext = candidate.suffix.lower()
                    if ext not in self.args.extensions:
                        continue
                    try:
                        candidate.stat()
                    except (PermissionError, FileNotFoundError):
                        logging.warning("Skipping inaccessible file %s", candidate)
                        self.stats["skipped"] += 1
                        continue
                    tasks.append(FileTask(candidate, base_root, ext, branch=branch_label))
        logging.info("Discovered %d candidate files", len(tasks))
        return tasks

    def relative_destination(self, task: FileTask) -> Path:
        try:
            rel = task.source.relative_to(task.drive_root)
        except ValueError:
            rel = Path(task.source.name)
        drive_label = getattr(task.drive_root, "drive", "") or task.drive_root.anchor.strip(":/\\") or task.drive_root.name
        if task.branch:
            branch_label = sanitize_label(task.branch)
            return Path("BRANCHES") / branch_label / rel
        return Path(drive_label) / rel

    def safe_copy(self, src: Path, dest: Path) -> None:
        ensure_dir(dest.parent)
        tmp_file = self.temp_dir / f"copy_{uuid.uuid4().hex}"
        attempts = 0
        while attempts < 5:
            try:
                shutil.copy2(long_path(src), long_path(tmp_file))
                os.replace(long_path(tmp_file), long_path(dest))
                return
            except PermissionError:
                attempts += 1
                with contextlib.suppress(Exception):
                    if tmp_file.exists():
                        tmp_file.unlink()
                time.sleep(1.0 * attempts)
            except Exception as exc:
                logging.error("Copy failure %s -> %s: %s", src, dest, exc)
                with contextlib.suppress(Exception):
                    if tmp_file.exists():
                        tmp_file.unlink()
                raise
        if tmp_file.exists():
            with contextlib.suppress(Exception):
                tmp_file.unlink()
        raise RuntimeError(f"Failed to copy {src} after retries")

    def _next_bates(self) -> Optional[str]:
        if not self.args.bates_prefix:
            return None
        with self._bates_lock:
            current = self.bates_counter
            self.bates_counter += 1
        bates = f"{self.args.bates_prefix}{current:06d}"
        self.stats["bates_assigned"] += 1
        return bates

    def process_task(self, task: FileTask) -> dict:
        record = {
            "source": str(task.source),
            "extension": task.extension,
            "branch": task.branch or "",
            "status": "PENDING",
            "error": "",
            "size": None,
            "mtime": None,
            "sha256": None,
            "dest": None,
            "duplicate_of": None,
            "bates_id": None,
            "stamp": self.args.evidence_stamp or None,
        }
        try:
            stat = task.source.stat()
            record["size"] = stat.st_size
            record["mtime"] = stat.st_mtime
            if self.state_tracker.should_skip(task.source, stat.st_size, stat.st_mtime, self.args.force):
                record["status"] = "SKIPPED_CACHED"
                self.stats["skipped"] += 1
                return record
            sha = hash_file(task.source)
            record["sha256"] = sha
            duplicate = self.dedupe_index.lookup(sha)
            if duplicate:
                record["status"] = "DUPLICATE"
                record["duplicate_of"] = duplicate.get("dest")
                record["dest"] = duplicate.get("dest")
                self.stats["duplicates"] += 1
                return record
            rel_dest = self.relative_destination(task)
            dest_path = self.collect_dir / task.extension.strip(".") / rel_dest
            record["dest"] = str(dest_path)
            if self.args.dry_run:
                record["status"] = "DRY_RUN"
                self.stats["dry"] += 1
            else:
                self.safe_copy(task.source, dest_path)
                record["status"] = "COPIED"
                self.stats["copied"] += 1
                self.stats["bytes_copied"] += stat.st_size
                bates_id = self._next_bates()
                if bates_id:
                    record["bates_id"] = bates_id
                self.dedupe_index.add(
                    sha,
                    {
                        "dest": str(dest_path),
                        "size": stat.st_size,
                        "recorded_at": utc_now().isoformat(),
                    },
                )
                if self.sqlite_index:
                    self.sqlite_index.upsert(sha, str(task.source), str(dest_path), stat.st_size, stat.st_mtime)
                if task.extension in self.args.secret_exts and stat.st_size <= self.args.secret_scan_limit:
                    findings = self.secret_scanner.scan(task.source)
                    self.secret_findings.extend(findings)
        except Exception as exc:
            logging.exception("Processing error for %s", task.source)
            record["status"] = "ERROR"
            record["error"] = str(exc)
            self.stats["errors"] += 1
        finally:
            self.stats["scanned"] += 1
            self.state_tracker.append(
                {
                    "source": str(task.source),
                    "size": record.get("size"),
                    "mtime": record.get("mtime"),
                    "sha256": record.get("sha256"),
                    "status": record.get("status"),
                    "dest": record.get("dest"),
                    "bates_id": record.get("bates_id"),
                }
            )
        return record

    def build_manifests(self) -> Tuple[Path, Path]:
        ensure_dir(self.output_root)
        manifest = {
            "tool": TOOL_NAME,
            "token": self.token,
            "run_id": self.run_id,
            "generated": utc_now().isoformat(),
            "summary": self.stats,
            "run": {
                "run_id": self.run_id,
                "token": self.token,
                "timestamp": self.timestamp,
                "arguments": self._serializable_args(),
                "log_path": str(self.log_path),
                "jsonl_log": str(self.jsonl_log_path),
            },
            "entries": self.manifest_entries,
        }
        json_path = self.output_root / f"manifest_{self.timestamp}_{self.token}.json"
        csv_path = self.output_root / f"manifest_{self.timestamp}_{self.token}.csv"
        atomic_write_json(json_path, manifest)
        fieldnames = [
            "source",
            "dest",
            "extension",
            "branch",
            "status",
            "size",
            "sha256",
            "duplicate_of",
            "bates_id",
            "stamp",
            "error",
        ]
        tmp_csv = csv_path.with_suffix(".tmp")
        with tmp_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.manifest_entries:
                writer.writerow({key: row.get(key, "") for key in fieldnames})
            fh.flush()
            os.fsync(fh.fileno())
        tmp_csv.replace(csv_path)
        return json_path, csv_path

    def append_registry_entry(self, payload: dict) -> Path:
        registry_path = self.output_root / "REGISTRY.jsonl"
        ensure_dir(registry_path.parent)
        line = json.dumps(payload, ensure_ascii=False)
        with self._registry_lock:
            with registry_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return registry_path

    def write_checksums(self) -> Optional[Path]:
        if self.args.dry_run:
            return None
        checksum_path = self.output_root / f"checksums_{self.timestamp}_{self.token}.sha256"
        lines = []
        for entry in sorted(self.manifest_entries, key=lambda r: r.get("dest", "")):
            if entry.get("status") == "COPIED" and entry.get("sha256"):
                dest = entry.get("dest", "")
                rel = os.path.relpath(dest, self.output_root)
                lines.append(f"{entry['sha256']}  {rel}")
        atomic_write_text(checksum_path, "\n".join(lines))
        return checksum_path

    def build_zip(self) -> Optional[Path]:
        if self.args.dry_run:
            return None
        bundle_path = self.output_root / f"bundle_{self.timestamp}_{self.token}.zip"
        copied_files = [Path(entry["dest"]) for entry in self.manifest_entries if entry.get("status") == "COPIED" and entry.get("dest")]
        copied_files.sort()
        ensure_dir(bundle_path.parent)
        tmp_zip = bundle_path.with_suffix(".tmp")
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for file_path in copied_files:
                rel = file_path.relative_to(self.collect_dir)
                zf.write(file_path, arcname=str(rel).replace("\\", "/"))
        tmp_zip.replace(bundle_path)
        with zipfile.ZipFile(bundle_path, "r") as zf:
            zf.namelist()
        return bundle_path

    def build_mifile_package(
        self,
        manifest_json: Path,
        manifest_csv: Path,
        checksum_path: Optional[Path],
        bundle_path: Optional[Path],
    ) -> Optional[Path]:
        if self.args.dry_run or not self.args.mifile_ready:
            return None
        package_path = self.output_root / f"mifile_{self.timestamp}_{self.token}.zip"
        tmp_zip = package_path.with_suffix(".tmp")
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_json, arcname="manifest.json")
            zf.write(manifest_csv, arcname="manifest.csv")
            if checksum_path:
                zf.write(checksum_path, arcname="checksums.sha256")
            if bundle_path:
                zf.write(bundle_path, arcname=bundle_path.name)
        tmp_zip.replace(package_path)
        with zipfile.ZipFile(package_path, "r") as zf:
            zf.namelist()
        return package_path

    def write_findings(self) -> Optional[Path]:
        if not self.secret_findings:
            return None
        findings_path = self.logs_dir / f"findings_{self.timestamp}_{self.token}.json"
        atomic_write_json(
            findings_path,
            {
                "generated": utc_now().isoformat(),
                "count": len(self.secret_findings),
                "findings": self.secret_findings,
            },
        )
        return findings_path

    def run(self) -> int:
        start = time.time()
        ensure_dir(self.output_root)
        ensure_dir(self.collect_dir)
        if os.name == "nt":
            try:
                ensure_roots([Path(f"{letter}:/") for letter in sorted(REQUIRED_DRIVE_LETTERS)])
            except Exception as exc:
                logging.error("Failed to ensure required roots: %s", exc)
                return 2
        if self.network_policy:
            if not self.network_policy.is_offline():
                logging.warning("Network policy permits outbound calls; ensure broker enforcement.")
        if self.assets_registry:
            report = self.assets_registry.validate()
            if report["missing"] or report["hash_mismatch"]:
                logging.error("Assets registry validation failed: %s", report)
                return 2
            logging.info("Assets registry validated: %s", report)
        tasks = self.discover_files()
        total_bytes = 0
        for task in tasks:
            try:
                total_bytes += task.source.stat().st_size
            except Exception:
                continue
        from concurrent.futures import ThreadPoolExecutor, as_completed

        try:
            try:
                self.validate_space(total_bytes)
            except RuntimeError as exc:
                logging.error(str(exc))
                return 2

            with ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
                futures = {executor.submit(self.process_task, task): task for task in tasks}
                for future in as_completed(futures):
                    task = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        logging.exception("Worker failure for %s", task.source)
                        result = {
                            "source": str(task.source),
                            "extension": task.extension,
                            "status": "ERROR",
                            "error": str(exc),
                            "size": None,
                            "mtime": None,
                            "sha256": None,
                            "dest": None,
                            "duplicate_of": None,
                            "bates_id": None,
                            "stamp": self.args.evidence_stamp or None,
                        }
                        self.stats["errors"] += 1
                    self.manifest_entries.append(result)
            if not self.args.dry_run:
                self.dedupe_index.save()
            json_manifest, csv_manifest = self.build_manifests()
            checksum_path = self.write_checksums()
            bundle_path = self.build_zip()
            mifile_bundle = self.build_mifile_package(json_manifest, csv_manifest, checksum_path, bundle_path)
            findings_path = self.write_findings()
            registry_path = self.append_registry_entry(
                {
                    "run_id": self.run_id,
                    "token": self.token,
                    "timestamp": self.timestamp,
                    "generated": utc_now().isoformat(),
                    "summary": self.stats,
                    "outputs": {
                        "manifest_json": str(json_manifest),
                        "manifest_csv": str(csv_manifest),
                        "checksums": str(checksum_path) if checksum_path else None,
                        "bundle": str(bundle_path) if bundle_path else None,
                        "mifile_bundle": str(mifile_bundle) if mifile_bundle else None,
                        "findings": str(findings_path) if findings_path else None,
                    },
                }
            )
            elapsed = time.time() - start
            summary = {
                "manifest_json": str(json_manifest),
                "manifest_csv": str(csv_manifest),
                "checksums": str(checksum_path) if checksum_path else "n/a",
                "bundle": str(bundle_path) if bundle_path else "n/a",
                "mifile_bundle": str(mifile_bundle) if mifile_bundle else "n/a",
                "findings": str(findings_path) if findings_path else "none",
                "registry": str(registry_path),
                "elapsed_seconds": round(elapsed, 2),
            }
            logging.info("Run summary: %s", summary)
            if self.stats["errors"] > 0:
                return 1
            return 0
        finally:
            self._finalize_run()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Windows evidence organizer with dedupe, manifests, and bundle packaging.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--drives", nargs="*", default=[str(p) for p in DEFAULT_DRIVES], help="Drive roots to scan (use Windows-style paths).")
    parser.add_argument("--auto-drives", action="store_true", help="Auto-discover eligible drives (Windows only).")
    parser.add_argument("--extensions", nargs="*", default=DEFAULT_EXTENSIONS, help="Extensions to include (lowercase, with dots).")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT_BASE), help="Root folder for OUTPUT artifacts.")
    parser.add_argument("--temp-root", default=str(DEFAULT_TEMP_ROOT), help="Root folder for temporary staging files.")
    parser.add_argument("--max-workers", type=int, default=4, help="Thread pool size.")
    parser.add_argument("--dry-run", action="store_true", help="Log actions without copying.")
    parser.add_argument("--force", action="store_true", help="Force operations despite warnings.")
    parser.add_argument("--include-hidden", action="store_true", help="Include hidden and system items.")
    parser.add_argument("--allow-c-drive", action="store_true", help="Allow C: drive usage despite policy warnings.")
    parser.add_argument("--token", help="Optional idempotency token.")
    parser.add_argument("--sqlite-index", action="store_true", help="Persist evidence records to SQLite.")
    parser.add_argument("--network-policy", help="Path to network_policy.json for broker enforcement.")
    parser.add_argument("--assets-registry", help="Path to external assets registry json.")
    parser.add_argument("--secret-scan-limit", type=int, default=1024 * 1024, help="Max bytes for secret scanning per file.")
    parser.add_argument("--secret-exts", nargs="*", default=sorted(DEFAULT_SECRET_EXTENSIONS), help="Extensions to scan for secrets.")
    parser.add_argument(
        "--deny-dirs",
        nargs="*",
        default=sorted(DEFAULT_DENY_DIRS),
        help="Directory names to skip during discovery (case-insensitive).",
    )
    parser.add_argument(
        "--branch",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Additional branch roots to scan (label=absolute_path).",
    )
    parser.add_argument(
        "--branches-file",
        help="JSON file containing branch specs as {'label': 'Case', 'path': 'R:/Case'} entries or a label:path mapping.",
    )
    parser.add_argument("--bates-prefix", default="", help="Optional Bates prefix (e.g., LIT-); numbering is zero-padded.")
    parser.add_argument("--bates-start", type=int, default=1, help="Starting number for Bates labels when prefix is provided.")
    parser.add_argument("--evidence-stamp", default="", help="Stamp label to record with each copied entry.")
    parser.add_argument("--mifile-ready", action="store_true", help="Emit a MiFILE-ready bundle zip containing manifests and bundle.")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity for both text and JSONL logs.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    organizer = EvidenceOrganizer(args)
    try:
        exit_code = organizer.run()
    except KeyboardInterrupt:
        logging.error("Interrupted by user")
        exit_code = 130
    except Exception as exc:
        logging.exception("Fatal error: %s", exc)
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
