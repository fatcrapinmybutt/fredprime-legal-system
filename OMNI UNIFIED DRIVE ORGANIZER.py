#!/usr/bin/env python3
"""
OMNI DRIVE ORGANIZER + LEGAL INTEL MONOLITH

Stage 1 (Organizer):
  - Scan one or more roots (drives/folders).
  - Bucket files under <root>/BUCKETS/<BUCKET_NAME>/.
  - Detect duplicates across all roots using size + binary compare (no hashes).
  - Write per-root manifests and a global drive_index_<run_id>.csv.

Stage 2 + 3 (Legal Intel):
  - Ingest drive_index_<run_id>.csv.
  - Sample text from doc-like files (txt/json/csv/html/pdf/docx) if requested.
  - Use structural signals + optional ML (if scikit-learn is installed) to score legal-likelihood.
  - Emit:
      * LEGAL_INDEX_<run_id>/legal_docs_index_<run_id>.csv
      * LEGAL_INDEX_<run_id>/case_index_<run_id>.csv
      * LEGAL_INDEX_<run_id>/case_timeline_<run_id>.csv
      * LEGAL_INDEX_<run_id>/legal_stats_<run_id>.txt

CLI modes:
  - Default (no mode flags): run Stage 1 then Stage 2+3 (full pipeline).
  - --organize-only: run Stage 1 only.
  - --intel-only: run Stage 2+3 only (requires --index path to existing drive_index_*.csv).

Dry-run vs execute:
  - Stage 1 uses --doit flag: without it, no moves are performed (dry-run) but index/manifest can still be created.
  - Stage 2 never mutates files; it only reads and writes indices/stats.
"""

import argparse
import csv
import ctypes
import os
import platform
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

CHUNK = 8 * 1024 * 1024
MAX_SAMPLE_BYTES = 200_000
MAX_TEXT_CHARS = 6000

# ---------------------------------------------------------------------------
# SHARED CONSTANTS / HEADERS
# ---------------------------------------------------------------------------

# Extension -> bucket mapping (fine-grained)
EXT_BUCKET_MAP: Dict[str, str] = {
    "pdf": "PDFs",
    "txt": "TXT",
    "docx": "DOCX",
    "doc": "DOC",
    "rtf": "DOC",
    "odt": "DOC",
    "xlsx": "XLSX",
    "xls": "XLS",
    "csv": "CSV",
    "tsv": "CSV",
    "pptx": "PPTX",
    "ppt": "PPT",
    "jpg": "IMAGES",
    "jpeg": "IMAGES",
    "png": "IMAGES",
    "gif": "IMAGES",
    "bmp": "IMAGES",
    "tif": "IMAGES",
    "tiff": "IMAGES",
    "webp": "IMAGES",
    "mp3": "AUDIO",
    "wav": "AUDIO",
    "flac": "AUDIO",
    "ogg": "AUDIO",
    "m4a": "AUDIO",
    "mp4": "VIDEO",
    "mkv": "VIDEO",
    "mov": "VIDEO",
    "avi": "VIDEO",
    "wmv": "VIDEO",
    "webm": "VIDEO",
    "zip": "ARCHIVES",
    "tar": "ARCHIVES",
    "gz": "ARCHIVES",
    "7z": "ARCHIVES",
    "rar": "ARCHIVES",
    "json": "JSON",
    "ndjson": "JSON",
    "html": "HTML",
    "htm": "HTML",
    "xml": "HTML",
    "ps1": "SCRIPTS",
    "py": "SCRIPTS",
    "ipynb": "SCRIPTS",
    "js": "SCRIPTS",
    "java": "SCRIPTS",
    "c": "SCRIPTS",
    "cpp": "SCRIPTS",
    "cs": "SCRIPTS",
    "sh": "SCRIPTS",
    "bash": "SCRIPTS",
    "zsh": "SCRIPTS",
    "bat": "SCRIPTS",
    "cmd": "SCRIPTS",
    "ini": "CONFIG",
    "cfg": "CONFIG",
    "conf": "CONFIG",
    "yaml": "CONFIG",
    "yml": "CONFIG",
    "toml": "CONFIG",
    "db": "DATA",
    "sqlite": "DATA",
    "sqlite3": "DATA",
}

DEFAULT_OTHER_BUCKET = "OTHERS"
DUPLICATES_BUCKET = "duplicates"
MANIFEST_FILENAME = "manifest.csv"

# Optional large/media/system skip set for Stage 1
SKIP_EXTS_DEFAULT: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".heic",
    ".heif",
    ".gif",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".wmv",
    ".webm",
    ".m4v",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".oga",
    ".wav",
    ".flac",
    ".apk",
    ".obb",
    ".dex",
    ".so",
    ".oat",
    ".vdex",
    ".tmp",
    ".temp",
    ".cache",
    ".thumb",
    ".nomedia",
}

# Unified schema for per-root manifests and global index
MANIFEST_HEADER: List[str] = [
    "run_id",
    "timestamp",
    "root",
    "original_path",
    "rel_path",
    "dest_path",
    "dest_exists",
    "bucket",
    "ext",
    "size_bytes",
    "mtime_utc",
    "ctime_utc",
    "atime_utc",
    "is_hidden",
    "action",
    "duplicate_of",
]

# Legal-doc index output schema
LEGAL_OUTPUT_HEADER: List[str] = [
    "run_id",
    "root",
    "dest_path",
    "bucket",
    "ext",
    "size_bytes",
    "mtime_utc",
    "doc_type",
    "legal_prob",
    "flags",
    "case_numbers",
    "seed_reason",
    "ml_used",
    "original_path",
    "rel_path",
]

# Case coverage index schema
CASE_INDEX_HEADER: List[str] = [
    "run_id",
    "case_number",
    "root",
    "doc_type",
    "count_docs",
    "example_paths",
]

# Timeline schema
TIMELINE_HEADER: List[str] = [
    "run_id",
    "case_number",
    "root",
    "doc_type",
    "mtime_utc",
    "dest_path",
    "bucket",
    "size_bytes",
]

# MI-style case number detection
CASE_RE_LIST = [
    re.compile(r"\b\d{2,4}-\d{4,}-[A-Z]{2}\b"),  # 24-01507-DC
    re.compile(r"\b\d{4}-\d{7}-[A-Z]{2}\b"),  # 2024-001507-DC
    re.compile(r"(?:Case\s+No\.?|No\.)\s*([0-9]{2,4}-[0-9]{4,}-[A-Za-z]{2})", re.I),
]

TEXT_EXTS: Set[str] = {
    "txt",
    "md",
    "json",
    "csv",
    "tsv",
    "log",
    "html",
    "htm",
    "xml",
}

DOC_LIKE_EXTS: Set[str] = {
    "pdf",
    "docx",
    "doc",
    "rtf",
    "odt",
}

MEDIA_BUCKET_TYPES: Dict[str, str] = {
    "IMAGES": "EVIDENCE_IMAGE",
    "AUDIO": "EVIDENCE_AUDIO",
    "VIDEO": "EVIDENCE_VIDEO",
}

# ---------------------------------------------------------------------------
# SHARED UTILS
# ---------------------------------------------------------------------------


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def unique_filename(path: Path) -> Path:
    if not path.exists():
        return path
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def is_hidden(path: Path) -> bool:
    """Cross-platform hidden detection."""
    try:
        if platform.system() == "Windows":
            ctypes.windll.kernel32.GetFileAttributesW.argtypes = [ctypes.c_wchar_p]
            ctypes.windll.kernel32.GetFileAttributesW.restype = ctypes.c_uint32
            attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
            if attrs == 0xFFFFFFFF:
                return False
            return bool(attrs & 0x2)
        return path.name.startswith(".")
    except Exception:
        return path.name.startswith(".")


def move_file_atomic(src: Path, dest: Path) -> None:
    """Move file safely with best-effort atomicity."""
    ensure_dir(dest.parent)
    try:
        try:
            if src.resolve() == dest.resolve():
                return
        except Exception:
            pass
        try:
            src_dev = src.stat().st_dev
            dest_parent = dest.parent
            dest_parent_stat = dest_parent.stat()
            if src_dev == dest_parent_stat.st_dev:
                os.replace(str(src), str(dest))
                return
        except Exception:
            pass
        shutil.move(str(src), str(dest))
    except Exception:
        shutil.copy2(str(src), str(dest))
        try:
            src.unlink()
        except Exception:
            pass


def uniquesafe(folder: Path, name: str) -> Path:
    return unique_filename(folder / name)


def iso_now_z() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def iso_from_timestamp_utc(ts: float) -> str:
    return datetime.utcfromtimestamp(ts).replace(microsecond=0).isoformat() + "Z"


def _write_manifest_atomic(rows: List[List[Any]], manifest_path: Path) -> None:
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    ensure_dir(manifest_path.parent)
    with tmp.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        for r in rows:
            writer.writerow(r)
    if manifest_path.exists():
        bak = manifest_path.with_name(
            manifest_path.stem + "." + datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + manifest_path.suffix + ".bak"
        )
        try:
            os.replace(str(manifest_path), str(bak))
        except Exception:
            try:
                manifest_path.replace(bak)
            except Exception:
                pass
    os.replace(str(tmp), str(manifest_path))


def compute_total_files(root: Path, skip_hidden: bool, skip_exts: Set[str]) -> int:
    count = 0
    buckets_root = root / "BUCKETS"
    try:
        buckets_root_resolved: Optional[Path] = buckets_root.resolve()
    except Exception:
        buckets_root_resolved = None

    for dirpath, dirnames, filenames in os.walk(root):
        dirp = Path(dirpath)
        try:
            dp_resolved = dirp.resolve()
        except Exception:
            dp_resolved = dirp

        if buckets_root_resolved is not None:
            try:
                if dp_resolved == buckets_root_resolved or buckets_root_resolved in dp_resolved.parents:
                    dirnames[:] = []
                    continue
            except Exception:
                pass
        else:
            if str(dirp).startswith(str(buckets_root)):
                dirnames[:] = []
                continue

        for name in filenames:
            p = dirp / name
            if skip_hidden:
                try:
                    if is_hidden(p):
                        continue
                except Exception:
                    continue
            try:
                if not p.is_file():
                    continue
            except Exception:
                continue
            if p.suffix.lower() in skip_exts:
                continue
            count += 1
    return count


def print_progress(processed: int, total: int, start_time: float, current_file: str, bar_len: int = 40) -> None:
    if total <= 0:
        return
    percent = processed / total if total else 0
    filled = int(bar_len * percent)
    bar = "█" * filled + "-" * (bar_len - filled)
    elapsed = time.time() - start_time
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = total - processed
    eta = int(remaining / rate) if rate > 0 else None
    eta_str = f"{eta}s" if eta is not None else "??s"
    cur = current_file
    if len(cur) > 60:
        cur = "..." + cur[-57:]
    sys.stdout.write(f"\r[{bar}] {percent*100:5.1f}% {processed}/{total} {rate:5.1f} f/s ETA: {eta_str} {cur}")
    sys.stdout.flush()


def files_equal(p1: Path, p2: Path) -> bool:
    if not p1.exists() or not p2.exists():
        return False
    try:
        if p1.stat().st_size != p2.stat().st_size:
            return False
        with p1.open("rb") as f1, p2.open("rb") as f2:
            while True:
                b1 = f1.read(CHUNK)
                b2 = f2.read(CHUNK)
                if b1 != b2:
                    return False
                if not b1:
                    break
        return True
    except Exception:
        return False


def delete_empty_dirs(root: Path, buckets_root: Path) -> None:
    protected: Set[Path] = {buckets_root}
    if buckets_root.exists():
        for d, _, _ in os.walk(buckets_root):
            protected.add(Path(d))

    for d, _, _ in os.walk(root, topdown=False):
        p = Path(d)
        if p == root or p in protected:
            continue
        try:
            next(p.iterdir())
        except StopIteration:
            try:
                p.rmdir()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# STAGE 1: ORGANIZER (BUCKET + DUPLICATES + GLOBAL INDEX)
# ---------------------------------------------------------------------------


def run_scan(
    root: Path,
    doit: bool,
    skip_hidden: bool,
    skip_exts: Set[str],
    manifest_path: Path,
    run_id: str,
    global_state: Dict[str, Any],
    global_index_writer: Optional[csv.writer],
) -> Dict[str, Any]:
    buckets_root = root / "BUCKETS"
    ensure_dir(buckets_root)
    duplicates_dir = buckets_root / DUPLICATES_BUCKET
    ensure_dir(duplicates_dir)

    rows: List[List[Any]] = [MANIFEST_HEADER[:]]

    size_map: Dict[int, List[str]] = global_state.setdefault("size_map", {})

    processed = moved = duplicates = skipped = errors = 0
    start_time = iso_now_z()
    wall_start = time.time()

    total_count = compute_total_files(root, skip_hidden, skip_exts)
    last_update = 0.0

    try:
        buckets_root_resolved: Optional[Path] = buckets_root.resolve()
    except Exception:
        buckets_root_resolved = None

    for dirpath, dirnames, filenames in os.walk(root):
        dirp = Path(dirpath)
        try:
            dp_resolved = dirp.resolve()
        except Exception:
            dp_resolved = dirp

        if buckets_root_resolved is not None:
            try:
                if dp_resolved == buckets_root_resolved or buckets_root_resolved in dp_resolved.parents:
                    dirnames[:] = []
                    skipped += len(filenames)
                    continue
            except Exception:
                pass
        else:
            if str(dirp).startswith(str(buckets_root)):
                dirnames[:] = []
                skipped += len(filenames)
                continue

        for name in filenames:
            src = dirp / name
            if skip_hidden:
                try:
                    if is_hidden(src):
                        skipped += 1
                        continue
                except Exception:
                    skipped += 1
                    continue
            try:
                is_file = src.is_file()
            except Exception:
                skipped += 1
                continue
            if not is_file:
                skipped += 1
                continue

            if src.suffix.lower() in skip_exts:
                skipped += 1
                continue

            processed += 1
            try:
                stat = src.stat()
                size = stat.st_size
                mtime_utc = iso_from_timestamp_utc(stat.st_mtime)
                ctime_utc = iso_from_timestamp_utc(stat.st_ctime)
                atime_utc = iso_from_timestamp_utc(stat.st_atime)
                hidden_flag = "yes" if is_hidden(src) else "no"

                ext = src.suffix.lower().lstrip(".") or ""
                bucket_guess = EXT_BUCKET_MAP.get(ext, DEFAULT_OTHER_BUCKET)
                bucket_dir = buckets_root / bucket_guess
                ensure_dir(bucket_dir)

                try:
                    rel_path = str(src.relative_to(root))
                except Exception:
                    rel_path = str(src)

                def record_and_move(
                    source_path: Path,
                    final_dest: Path,
                    bucket_name: str,
                    action_label: str,
                    duplicate_of: str,
                ) -> None:
                    nonlocal moved, duplicates
                    if doit and (action_label.startswith("moved") or action_label.startswith("duplicate")):
                        try:
                            if source_path.resolve() != final_dest.resolve():
                                move_file_atomic(source_path, final_dest)
                        except Exception:
                            pass
                    dest_exists = final_dest.exists()
                    row: List[Any] = [
                        run_id,
                        iso_now_z(),
                        str(root),
                        str(source_path),
                        rel_path,
                        str(final_dest),
                        "yes" if dest_exists else "no",
                        bucket_name,
                        ext,
                        size,
                        mtime_utc,
                        ctime_utc,
                        atime_utc,
                        hidden_flag,
                        action_label,
                        duplicate_of,
                    ]
                    rows.append(row)
                    if global_index_writer is not None:
                        global_index_writer.writerow(row)
                    if doit:
                        if action_label.startswith("moved") and dest_exists:
                            moved += 1
                        if action_label.startswith("duplicate") and dest_exists:
                            duplicates += 1

                duplicate_of = ""
                is_dup = False
                if size in size_map:
                    for canon_path_str in size_map[size]:
                        canon_path = Path(canon_path_str)
                        if not canon_path.exists():
                            continue
                        if files_equal(src, canon_path):
                            duplicate_of = str(canon_path)
                            dup_dest = uniquesafe(duplicates_dir, src.name)
                            action = "duplicate (moved)" if doit else "duplicate (dry-run)"
                            record_and_move(src, dup_dest, DUPLICATES_BUCKET, action, duplicate_of)
                            is_dup = True
                            break

                if not is_dup:
                    dest = uniquesafe(bucket_dir, src.name)
                    action = "moved" if doit else "move (dry-run)"
                    record_and_move(src, dest, bucket_guess, action, "")
                    canonical_path = dest if doit else src
                    size_map.setdefault(size, []).append(str(canonical_path))

                now = time.time()
                if now - last_update > 0.2:
                    print_progress(processed, total_count, wall_start, str(src))
                    last_update = now

            except Exception as e:
                errors += 1
                row_err: List[Any] = [
                    run_id,
                    iso_now_z(),
                    str(root),
                    str(src),
                    str(src),
                    "",
                    "no",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "no",
                    f"error: {e}",
                    "",
                ]
                rows.append(row_err)
                if global_index_writer is not None:
                    global_index_writer.writerow(row_err)

    print()
    end_time = iso_now_z()
    summary_row: List[Any] = [
        run_id,
        end_time,
        str(root),
        str(root / "*"),
        "*",
        str(root / "BUCKETS"),
        "",
        "",
        "",
        processed,
        "",
        "",
        "",
        "",
        f"summary moved={moved};duplicates={duplicates};skipped={skipped};errors={errors}",
        "",
    ]
    rows.append(summary_row)
    _write_manifest_atomic(rows, manifest_path)
    return {
        "root": str(root),
        "total": processed,
        "moved": moved,
        "duplicates": duplicates,
        "skipped": skipped,
        "errors": errors,
        "manifest": str(manifest_path),
        "start": start_time,
        "end": end_time,
    }


# ---------------------------------------------------------------------------
# STAGE 2 + 3: LEGAL INTEL (INGEST + ML + AUDIT)
# ---------------------------------------------------------------------------


def read_index(index_path: Path, roots_filter: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with index_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "run_id",
            "root",
            "dest_path",
            "bucket",
            "ext",
            "size_bytes",
            "mtime_utc",
            "action",
            "original_path",
            "rel_path",
        }
        if reader.fieldnames is None:
            raise ValueError("Index file has no header row")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Global index missing required columns: {missing}")
        for row in reader:
            root = row.get("root", "")
            if roots_filter and root not in roots_filter:
                continue
            action = (row.get("action") or "").lower()
            if not (action.startswith("moved") or action.startswith("duplicate")):
                continue
            dest_path = row.get("dest_path") or ""
            original_path = row.get("original_path") or ""
            if not dest_path and not original_path:
                continue
            rows.append(row)
    return rows


def _read_bytes(path: Path, limit: int) -> bytes:
    with path.open("rb") as f:
        return f.read(limit)


def _safe_decode(raw: bytes) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return raw.decode(enc, errors="ignore")
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def extract_text_snippet(path: Path, ext: str) -> str:
    ext = ext.lower()
    try:
        if ext in TEXT_EXTS:
            raw = _read_bytes(path, MAX_SAMPLE_BYTES)
            return _safe_decode(raw)[:MAX_TEXT_CHARS]

        if ext == "pdf":
            try:
                from pypdf import PdfReader  # type: ignore

                text_parts: List[str] = []
                with path.open("rb") as f:
                    reader = PdfReader(f)
                    for page in reader.pages[:5]:
                        try:
                            text_parts.append(page.extract_text() or "")
                        except Exception:
                            continue
                return "\n".join(text_parts)[:MAX_TEXT_CHARS]
            except Exception:
                try:
                    from io import StringIO  # type: ignore

                    from pdfminer.high_level import extract_text_to_fp  # type: ignore

                    output = StringIO()
                    with path.open("rb") as f:
                        extract_text_to_fp(f, output, maxpages=5)
                    return output.getvalue()[:MAX_TEXT_CHARS]
                except Exception:
                    return ""

        if ext == "docx":
            try:
                import docx  # type: ignore

                doc = docx.Document(str(path))
                full_text: List[str] = []
                for p in doc.paragraphs[:200]:
                    full_text.append(p.text)
                return "\n".join(full_text)[:MAX_TEXT_CHARS]
            except Exception:
                return ""

        return ""
    except Exception:
        return ""


def detect_case_numbers(name: str, text: str) -> List[str]:
    s = name + "\n" + text
    found: List[str] = []
    for cre in CASE_RE_LIST:
        for m in cre.findall(s):
            if isinstance(m, tuple):
                for g in m:
                    if g:
                        found.append(g)
            else:
                found.append(m)
    normed: List[str] = []
    seen: Set[str] = set()
    for c in found:
        c2 = c.strip()
        if not c2:
            continue
        if c2 not in seen:
            seen.add(c2)
            normed.append(c2)
    return normed


def compute_structural_features(name: str, text: str) -> Dict[str, float]:
    features: Dict[str, float] = {}
    text_len = len(text)
    features["text_len"] = float(text_len)
    if text_len == 0:
        features["line_count"] = 0.0
        features["upper_ratio"] = 0.0
        features["digit_ratio"] = 0.0
    else:
        lines = text.splitlines()
        features["line_count"] = float(len(lines))
        upper = sum(1 for c in text if c.isupper())
        letters = sum(1 for c in text if c.isalpha())
        digits = sum(1 for c in text if c.isdigit())
        features["upper_ratio"] = float(upper) / float(letters) if letters > 0 else 0.0
        features["digit_ratio"] = float(digits) / float(text_len)
    haystack = (name + "\n" + text).lower()
    has_v_pattern = (" v. " in haystack) or (" vs " in haystack)
    features["has_v_pattern"] = 1.0 if has_v_pattern else 0.0
    hyphen_digits = len(re.findall(r"[0-9]-[0-9]", haystack))
    features["hyphen_digit_count"] = float(hyphen_digits)
    return features


def seed_label(row: Dict[str, Any], path: Path, text: str, case_numbers: List[str]) -> Tuple[bool, str]:
    bucket = (row.get("bucket") or "").upper()
    ext = (row.get("ext") or "").lower()
    name_lower = path.name.lower()
    haystack = name_lower + "\n" + text.lower()
    has_v = (" v. " in haystack) or (" vs " in haystack)

    if case_numbers:
        return True, "CASE_NUMBER"
    if has_v:
        return True, "V_PATTERN"
    if bucket in {"PDFS", "DOCX", "DOC", "TXT", "CSV", "JSON"} and ext in (DOC_LIKE_EXTS | TEXT_EXTS):
        size_str = row.get("size_bytes") or "0"
        try:
            size = int(size_str)
        except Exception:
            size = 0
        if 0 < size < 50_000_000:
            return True, "BUCKET_DOC"
    return False, ""


def try_build_ml_classifier(docs: List[Dict[str, Any]]) -> Optional[Tuple[Any, Any]]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.linear_model import LogisticRegression  # type: ignore
    except Exception:
        return None

    texts: List[str] = []
    labels: List[int] = []

    for d in docs:
        text = d.get("sample_text", "")
        if not text:
            continue
        is_seed = d.get("seed_is_legal", False)
        labels.append(1 if is_seed else 0)
        texts.append((d.get("name", "") + "\n" + text))

    if len(texts) < 20:
        return None
    if len(set(labels)) < 2:
        return None

    max_train = 4000
    if len(texts) > max_train:
        texts = texts[:max_train]
        labels = labels[:max_train]

    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.linear_model import LogisticRegression  # type: ignore

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50000,
    )
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression(max_iter=200, n_jobs=-1 if hasattr(os, "cpu_count") else None)
    clf.fit(X, labels)
    return vectorizer, clf


def predict_legal_prob(
    vectorizer_clf: Optional[Tuple[Any, Any]],
    name: str,
    text: str,
    seed_is_legal: bool,
) -> float:
    if vectorizer_clf is None:
        return 0.9 if seed_is_legal else 0.1
    vectorizer, clf = vectorizer_clf
    try:
        text_full = name + "\n" + text
        X = vectorizer.transform([text_full])
        proba = clf.predict_proba(X)[0][1]
        return float(proba)
    except Exception:
        return 0.5 if seed_is_legal else 0.1


def process_rows(index_rows: List[Dict[str, Any]], scan_content: bool) -> Dict[str, Any]:
    if not index_rows:
        return {
            "run_id": "",
            "docs": [],
            "legal_rows": [],
            "case_map": {},
            "timeline": [],
        }

    docs: List[Dict[str, Any]] = []
    for row in index_rows:
        root = row.get("root", "")
        dest_path = row.get("dest_path") or row.get("original_path") or ""
        if not dest_path:
            continue
        path_obj = Path(dest_path)
        ext = (row.get("ext") or "").lower()
        bucket = (row.get("bucket") or "").upper()
        name = path_obj.name

        text_sample = ""
        if scan_content and (ext in TEXT_EXTS or ext in DOC_LIKE_EXTS):
            text_sample = extract_text_snippet(path_obj, ext)

        case_numbers = detect_case_numbers(name, text_sample)
        features = compute_structural_features(name, text_sample)
        seed_is_legal, seed_reason = seed_label(row, path_obj, text_sample, case_numbers)

        d = {
            "row": row,
            "root": root,
            "path": path_obj,
            "name": name,
            "ext": ext,
            "bucket": bucket,
            "sample_text": text_sample,
            "case_numbers": case_numbers,
            "features": features,
            "seed_is_legal": seed_is_legal,
            "seed_reason": seed_reason,
        }
        docs.append(d)

    if not docs:
        return {
            "run_id": index_rows[0].get("run_id", ""),
            "docs": [],
            "legal_rows": [],
            "case_map": {},
            "timeline": [],
        }

    run_id = index_rows[0].get("run_id", "")

    vectorizer_clf = try_build_ml_classifier(docs)
    ml_used_flag = vectorizer_clf is not None

    legal_rows: List[List[Any]] = []
    case_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    timeline_rows: List[List[Any]] = []

    for d in docs:
        row = d["row"]
        root = d["root"]
        path_obj = d["path"]
        name = d["name"]
        ext = d["ext"]
        bucket = d["bucket"]
        text_sample = d["sample_text"]
        case_numbers = d["case_numbers"]
        seed_is_legal = d["seed_is_legal"]
        seed_reason = d["seed_reason"]

        size_str = row.get("size_bytes") or "0"
        try:
            size_bytes = int(size_str)
        except Exception:
            size_bytes = 0
        mtime_utc = row.get("mtime_utc") or ""
        original_path = row.get("original_path") or str(path_obj)
        rel_path = row.get("rel_path") or ""

        legal_prob = predict_legal_prob(vectorizer_clf, name, text_sample, seed_is_legal)
        flags: List[str] = []
        if seed_is_legal:
            flags.append("SEED")
        if case_numbers:
            flags.append("CASE_NUMBER")
        if d["features"].get("has_v_pattern", 0.0) > 0:
            flags.append("V_PATTERN")
        if bucket in {"PDFS", "DOCX", "DOC", "TXT", "CSV", "JSON"}:
            flags.append("DOC_BUCKET")
        if legal_prob >= 0.85:
            flags.append("ML_HIGH")
        elif legal_prob >= 0.6:
            flags.append("ML_MED")

        doc_type = "UNKNOWN"
        bucket_upper = bucket
        name_lower = name.lower()
        if bucket_upper in MEDIA_BUCKET_TYPES:
            doc_type = MEDIA_BUCKET_TYPES[bucket_upper]
        else:
            if "transcript" in name_lower:
                doc_type = "TRANSCRIPT"
            elif any(k in name_lower for k in ["order", "judgment", "opinion", "decree"]):
                doc_type = "COURT_ORDER"
            elif "motion" in name_lower or "mot." in name_lower:
                doc_type = "MOTION"
            elif any(k in name_lower for k in ["complaint", "petition", "application"]):
                doc_type = "COMPLAINT_OR_PETITION"
            elif any(k in name_lower for k in ["brief", "memorandum", "memo"]):
                doc_type = "BRIEF_OR_MEMO"
            elif "ppo" in name_lower:
                doc_type = "PPO_DOC"
            elif any(k in name_lower for k in ["hearing", "show cause", "notice"]):
                doc_type = "HEARING_DOC"
            elif "appclose" in name_lower or "conversation" in name_lower or "convlog" in name_lower:
                doc_type = "COMM_LOG"

        keep = (legal_prob >= 0.6) or seed_is_legal or bool(case_numbers)
        if not keep:
            continue

        case_numbers_str = "|".join(case_numbers)

        legal_row: List[Any] = [
            run_id,
            root,
            str(path_obj),
            bucket,
            ext,
            size_bytes,
            mtime_utc,
            doc_type,
            f"{legal_prob:.4f}",
            ";".join(sorted(set(flags))),
            case_numbers_str,
            seed_reason,
            "yes" if ml_used_flag else "no",
            original_path,
            rel_path,
        ]
        legal_rows.append(legal_row)

        for cn in case_numbers:
            key = (cn, root, doc_type)
            if key not in case_map:
                case_map[key] = {
                    "run_id": run_id,
                    "case_number": cn,
                    "root": root,
                    "doc_type": doc_type,
                    "count_docs": 0,
                    "example_paths": [],
                }
            entry = case_map[key]
            entry["count_docs"] += 1
            if len(entry["example_paths"]) < 5:
                entry["example_paths"].append(str(path_obj))

        if case_numbers:
            for cn in case_numbers:
                trow: List[Any] = [
                    run_id,
                    cn,
                    root,
                    doc_type,
                    mtime_utc,
                    str(path_obj),
                    bucket,
                    size_bytes,
                ]
                timeline_rows.append(trow)
        else:
            trow_nc: List[Any] = [
                run_id,
                "NO_CASE",
                root,
                doc_type,
                mtime_utc,
                str(path_obj),
                bucket,
                size_bytes,
            ]
            timeline_rows.append(trow_nc)

    return {
        "run_id": run_id,
        "docs": docs,
        "legal_rows": legal_rows,
        "case_map": case_map,
        "timeline": timeline_rows,
    }


def write_outputs(
    base_out_dir: Path,
    run_id: str,
    legal_rows: List[List[Any]],
    case_map: Dict[Tuple[str, str, str], Dict[str, Any]],
    timeline_rows: List[List[Any]],
) -> Dict[str, str]:
    out_dir = base_out_dir / f"LEGAL_INDEX_{run_id}"
    ensure_dir(out_dir)

    legal_index_path = out_dir / f"legal_docs_index_{run_id}.csv"
    case_index_path = out_dir / f"case_index_{run_id}.csv"
    timeline_path = out_dir / f"case_timeline_{run_id}.csv"
    stats_path = out_dir / f"legal_stats_{run_id}.txt"

    with legal_index_path.open("w", newline="", encoding="utf-8") as lf:
        w = csv.writer(lf)
        w.writerow(LEGAL_OUTPUT_HEADER)
        for r in legal_rows:
            w.writerow(r)

    with case_index_path.open("w", newline="", encoding="utf-8") as cf:
        w = csv.writer(cf)
        w.writerow(CASE_INDEX_HEADER)
        for key, entry in sorted(case_map.items(), key=lambda kv: kv[0]):
            ex_paths = "||".join(entry["example_paths"])
            w.writerow(
                [
                    entry["run_id"],
                    entry["case_number"],
                    entry["root"],
                    entry["doc_type"],
                    entry["count_docs"],
                    ex_paths,
                ]
            )

    with timeline_path.open("w", newline="", encoding="utf-8") as tf:
        w = csv.writer(tf)
        w.writerow(TIMELINE_HEADER)
        for r in sorted(timeline_rows, key=lambda x: (x[1], x[4], x[5])):
            w.writerow(r)

    with stats_path.open("w", encoding="utf-8") as sf:
        total_legal = len(legal_rows)
        sf.write(f"Run ID: {run_id}\n")
        sf.write(f"Generated at: {iso_now_z()}\n")
        sf.write(f"Total legal-ish documents identified: {total_legal}\n\n")

        by_doc_type: Dict[str, int] = {}
        by_root: Dict[str, int] = {}
        for r in legal_rows:
            _, root, _, _, _, _, _, doc_type, _, _, _, _, _, _, _ = r
            by_doc_type[doc_type] = by_doc_type.get(doc_type, 0) + 1
            by_root[root] = by_root.get(root, 0) + 1

        sf.write("Counts by doc_type:\n")
        for dt, count in sorted(by_doc_type.items(), key=lambda kv: (-kv[1], kv[0])):
            sf.write(f"  {dt}: {count}\n")
        sf.write("\nCounts by root:\n")
        for rt, count in sorted(by_root.items(), key=lambda kv: (-kv[1], kv[0])):
            sf.write(f"  {rt}: {count}\n")

        sf.write("\nCase coverage summary:\n")
        by_case: Dict[str, int] = {}
        for (_, _, _), entry in case_map.items():
            cn = entry["case_number"]
            by_case[cn] = by_case.get(cn, 0) + entry["count_docs"]
        for cn, count in sorted(by_case.items(), key=lambda kv: (-kv[1], kv[0])):
            sf.write(f"  {cn}: {count} docs\n")

    return {
        "legal_index": str(legal_index_path),
        "case_index": str(case_index_path),
        "timeline": str(timeline_path),
        "stats": str(stats_path),
    }


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------


def run_tests() -> None:
    """Minimal combined test: Stage 1 + Stage 2+3"""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        root = td_path / "root"
        root.mkdir()
        d1 = root / "dir1"
        d2 = root / "dir2"
        d1.mkdir()
        d2.mkdir()
        a = d1 / "order_24-01507-DC.pdf"
        b = d2 / "order_copy.pdf"
        a.write_bytes(b"hello world")
        b.write_bytes(b"hello world")

        run_id = "TEST_RUN"
        manifest = root / "BUCKETS" / MANIFEST_FILENAME
        global_index = td_path / f"drive_index_{run_id}.csv"

        with global_index.open("w", newline="", encoding="utf-8") as gf:
            giw = csv.writer(gf)
            giw.writerow(MANIFEST_HEADER)
            global_state = {"size_map": {}}
            stats = run_scan(
                root,
                doit=True,
                skip_hidden=False,
                skip_exts=set(),
                manifest_path=manifest,
                run_id=run_id,
                global_state=global_state,
                global_index_writer=giw,
            )
        assert stats["total"] == 2
        assert (root / "BUCKETS" / "PDFs").exists()
        assert (root / "BUCKETS" / DUPLICATES_BUCKET).exists()
        assert manifest.exists()
        assert global_index.exists()

        index_rows = read_index(global_index, roots_filter=None)
        result = process_rows(index_rows, scan_content=False)
        legal_rows = result["legal_rows"]
        case_map = result["case_map"]
        timeline_rows = result["timeline"]
        assert len(legal_rows) >= 1

        out_paths = write_outputs(td_path, run_id, legal_rows, case_map, timeline_rows)
        for p in out_paths.values():
            assert Path(p).exists()

    print("tests passed")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "OMNI monolith: Stage 1 drive organizer + Stage 2/3 legal intel pipeline.\n"
            "Default: run full pipeline (organize then legal intel).\n"
            "Use --organize-only or --intel-only to restrict stages."
        )
    )

    # Stage 1 arguments
    ap.add_argument(
        "--drive",
        action="append",
        help=(
            "Drive or root folder to scan (Stage 1). Can be specified multiple times. "
            "Examples: F:\\ or D:\\ or /storage/emulated/0/Download. "
            "If omitted, defaults to F:\\ on Windows, /storage/emulated/0 on Android, or . on Unix."
        ),
    )
    ap.add_argument(
        "--doit",
        action="store_true",
        help=("Perform moves in Stage 1. Default is dry-run (no moves, but manifests and index still written)."),
    )
    ap.add_argument(
        "--skip-hidden",
        action="store_true",
        help=("Stage 1: skip hidden files. On Unix this checks dotfiles. On Windows this checks hidden attribute."),
    )
    ap.add_argument(
        "--include-media",
        action="store_true",
        help="Stage 1: include media/system extensions that are skipped by default.",
    )
    ap.add_argument(
        "--delete-empties",
        action="store_true",
        help="Stage 1: after organizing, prune empty directories (excluding BUCKETS/).",
    )
    ap.add_argument(
        "--global-index",
        help=("Stage 1: path to a global CSV index file. " "Default: ./drive_index_<UTC timestamp>.csv"),
    )

    # Stage 2+3 arguments
    ap.add_argument(
        "--index",
        help=(
            "When using --intel-only, path to an existing drive_index_*.csv. "
            "If not provided in full pipeline, index from Stage 1 is used."
        ),
    )
    ap.add_argument(
        "--root",
        action="append",
        help="Stage 2+3: restrict processing to specific roots (exact match to 'root' column).",
    )
    ap.add_argument(
        "--intel-scan-content",
        action="store_true",
        help=(
            "Stage 2: also read snippets from text/doc-like files (txt, json, csv, html, pdf, docx) "
            "to improve legal classification and case-number extraction."
        ),
    )
    ap.add_argument(
        "--intel-out-dir",
        help=(
            "Stage 2+3: base output directory for legal indices and stats. "
            "Default: current directory (creates LEGAL_INDEX_<run_id>/ under it)."
        ),
    )

    # Mode switches
    ap.add_argument(
        "--organize-only",
        action="store_true",
        help="Run Stage 1 only (organizer).",
    )
    ap.add_argument(
        "--intel-only",
        action="store_true",
        help="Run Stage 2+3 only (legal intel) using an existing --index.",
    )

    # Tests
    ap.add_argument(
        "--run-tests",
        action="store_true",
        help="Run integrated tests (Stage 1 + Stage 2+3) and exit.",
    )

    args = ap.parse_args()

    if args.run_tests:
        run_tests()
        return

    if args.organize_only and args.intel_only:
        print("[ERROR] Cannot use --organize-only and --intel-only at the same time.", file=sys.stderr)
        sys.exit(1)

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    mode_label = "FULL_PIPELINE"
    if args.organize_only:
        mode_label = "ORGANIZE_ONLY"
    elif args.intel_only:
        mode_label = "INTEL_ONLY"

    print(f"Run ID (initial): {run_id}")
    print(f"Mode: {mode_label}")

    # ------------------
    # STAGE 1: ORGANIZE
    # ------------------
    index_path: Optional[Path] = None

    if not args.intel_only:
        roots: List[Path] = []
        if args.drive:
            for d in args.drive:
                p = Path(d)
                if not p.exists():
                    print(f"[WARN] Drive/root not found, skipping: {p}", file=sys.stderr)
                    continue
                roots.append(p)
        else:
            if os.name == "nt":
                default_root = Path("F:/")
            else:
                candidate = Path("/storage/emulated/0")
                if candidate.exists():
                    default_root = candidate
                else:
                    default_root = Path(".")
            if not default_root.exists():
                print(f"[ERROR] Default root not found: {default_root}", file=sys.stderr)
                sys.exit(1)
            roots.append(default_root)

        if not roots:
            print("[ERROR] No valid roots to scan.", file=sys.stderr)
            sys.exit(1)

        print("Roots to scan (Stage 1):")
        for r in roots:
            print("  ", r)

        if args.global_index:
            index_path = Path(args.global_index)
        else:
            index_path = Path.cwd() / f"drive_index_{run_id}.csv"
        ensure_dir(index_path.parent)

        with index_path.open("w", newline="", encoding="utf-8") as gf:
            global_index_writer = csv.writer(gf)
            global_index_writer.writerow(MANIFEST_HEADER)

            global_state: Dict[str, Any] = {"size_map": {}}
            global_totals: Dict[str, int] = {
                "total": 0,
                "moved": 0,
                "duplicates": 0,
                "skipped": 0,
                "errors": 0,
            }

            for root in roots:
                print("\n" + "=" * 80)
                print(f"STAGE 1: SCANNING ROOT: {root}")
                print("=" * 80)
                manifest_path = root / "BUCKETS" / MANIFEST_FILENAME

                skip_exts = set() if args.include_media else set(SKIP_EXTS_DEFAULT)

                stats = run_scan(
                    root,
                    doit=args.doit,
                    skip_hidden=args.skip_hidden,
                    skip_exts=skip_exts,
                    manifest_path=manifest_path,
                    run_id=run_id,
                    global_state=global_state,
                    global_index_writer=global_index_writer,
                )

                print("  Start UTC:", stats["start"])
                print("  End   UTC:", stats["end"])
                print("  Total files scanned:", stats["total"])
                print("  Moved:", stats["moved"])
                print("  Duplicates:", stats["duplicates"])
                print("  Skipped:", stats["skipped"])
                print("  Errors:", stats["errors"])
                print("  Manifest:", stats["manifest"])

                global_totals["total"] += stats["total"]
                global_totals["moved"] += stats["moved"]
                global_totals["duplicates"] += stats["duplicates"]
                global_totals["skipped"] += stats["skipped"]
                global_totals["errors"] += stats["errors"]

                if args.delete_empties:
                    print("  Pruning empty directories...")
                    delete_empty_dirs(root, root / "BUCKETS")

        print("\n" + "#" * 80)
        print("STAGE 1 GLOBAL SUMMARY")
        print("#" * 80)
        print("Total files scanned:", global_totals["total"])
        print("Moved:", global_totals["moved"])
        print("Duplicates:", global_totals["duplicates"])
        print("Skipped:", global_totals["skipped"])
        print("Errors:", global_totals["errors"])
        print("Global index written to:", index_path)

        if args.organize_only:
            return

    # -----------------------------
    # STAGE 2 + 3: LEGAL INTEL
    # -----------------------------

    if args.intel_only:
        if not args.index:
            print("[ERROR] --intel-only requires --index pointing to an existing drive_index_*.csv", file=sys.stderr)
            sys.exit(1)
        index_path = Path(args.index)
        if not index_path.exists():
            print(f"[ERROR] Index file not found: {index_path}", file=sys.stderr)
            sys.exit(1)

    if index_path is None:
        print("[ERROR] Internal: index_path is None before Stage 2+3.", file=sys.stderr)
        sys.exit(1)

    roots_filter: Optional[Set[str]] = None
    if args.root:
        roots_filter = set(args.root)

    print("\n" + "=" * 80)
    print(f"STAGE 2+3: LEGAL INTEL USING INDEX: {index_path}")
    print("=" * 80)

    index_rows = read_index(index_path, roots_filter=roots_filter)
    if not index_rows:
        print("[WARN] No matching rows in index after filtering by roots/action. Nothing to do.", file=sys.stderr)
        sys.exit(0)

    index_run_id = index_rows[0].get("run_id", run_id)
    if index_run_id:
        run_id = index_run_id
    print(f"Detected run_id from index: {run_id}")
    print(f"Rows to analyze: {len(index_rows)}")
    print(f"Scan content: {'yes' if args.intel_scan_content else 'no'}")

    result = process_rows(index_rows, scan_content=args.intel_scan_content)
    legal_rows = result["legal_rows"]
    case_map = result["case_map"]
    timeline_rows = result["timeline"]

    print(f"Legal-ish documents kept: {len(legal_rows)}")
    print(f"Distinct (case_number, root, doc_type) combos: {len(case_map)}")
    print(f"Timeline entries: {len(timeline_rows)}")

    if not legal_rows:
        print("[WARN] No legal-looking documents detected; nothing to write.", file=sys.stderr)
        sys.exit(0)

    base_out_dir = Path(args.intel_out_dir) if args.intel_out_dir else Path.cwd()
    paths = write_outputs(base_out_dir, run_id, legal_rows, case_map, timeline_rows)

    print("Outputs written:")
    for label, path_str in paths.items():
        print(f"  {label}: {path_str}")


if __name__ == "__main__":
    main()
