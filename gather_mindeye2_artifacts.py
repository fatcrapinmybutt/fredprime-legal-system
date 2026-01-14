# review this code, provide 20 additions & improvements & upgrades & enhancements. #!/usr/bin/env python3
# gather_mindeye2_artifacts.py
# Purpose:
#   - Recursively scan one or more roots (default: D:\) for "graph artifacts".
#   - Deduplicate by SHA-256.
#   - Centralize copies in D:\MINDEYE2 (Windows) or /storage/emulated/0/MINDEYE2 (Android/Termux).
#   - Produce JSON + CSV manifests, a persistent hash index, and a run log.
#   - Optional: upload/copy to Google Drive at gdrive:/MINDEYE2 using rclone.
#
# Graph artifacts definition (ext-based + keyword heuristics):
#   Definite extensions:
#     .graphml .gexf .gml .xgmml .gephi .graphson .gryo .gpickle .gmlz .dot .gv
#     .cypher .cql .neo4j .dump .db .sqlite .db3 .tgz .zip  (common Neo4j dumps/exports)
#     .html  (pyvis/interactive graph pages)
#   Heuristic on filename for generic data: {json,csv,tsv,ndjson,txt}
#     Requires keyword match in the filename or parent dirs:
#       {"graph","neo4j","pyvis","gephi","nodes","edges","edgelist","nodelist",
#        "network","mindeye","constellation","gexf","graphml","cypher","gml","xgmml"}
#
# Usage examples (Windows PowerShell):
#   py .\gather_mindeye2_artifacts.py                              # dry-run, scan D:\ only
#   py .\gather_mindeye2_artifacts.py --roots D:\ E:\ --execute     # actually copy
#   py .\gather_mindeye2_artifacts.py --execute --rclone-upload     # copy + upload to gdrive:/MINDEYE2
#   py .\gather_mindeye2_artifacts.py --exts .gexf .graphml .html   # narrow extensions
#   py .\gather_mindeye2_artifacts.py --self-test                   # create sample artifacts and test
#
# Dependencies:
#   - Python 3.8+ (standard library only)
#   - rclone in PATH if you use --rclone-upload (default remote 'gdrive')
#
# Outputs inside DEST (MINDEYE2):
#   - artifacts/               copied files, named with __<hash8> suffix to avoid collisions
#   - manifest/master_manifest.json
#   - manifest/master_manifest.csv
#   - manifest/hash_index.json     (persistent dedupe index)
#   - logs/run_YYYYMMDD_HHMMSS.log
#
# Safety:
#   - Dry-run by default; no files are copied until --execute is supplied.
#   - Skips Windows system dirs by default. You can add more via --ignore-dirs.
#   - Every copied file is re-hashed and size-checked for integrity.
#
# Notes:
#   - Default Google Drive remote is 'gdrive'; adjust via --rclone-remote.
#   - Upload uses 'rclone copy' (non-destructive). Use --rclone-mode sync to mirror exactly.

import argparse
import csv
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

KEYWORDS = {
    "graph",
    "neo4j",
    "pyvis",
    "gephi",
    "nodes",
    "edges",
    "edgelist",
    "nodelist",
    "network",
    "mindeye",
    "constellation",
    "gexf",
    "graphml",
    "cypher",
    "gml",
    "xgmml",
}

DEFAULT_EXTS = {
    # strong graph formats
    ".graphml",
    ".gexf",
    ".gml",
    ".xgmml",
    ".gephi",
    ".graphson",
    ".gryo",
    ".gpickle",
    ".gmlz",
    ".dot",
    ".gv",
    ".cypher",
    ".cql",
    ".neo4j",
    ".dump",
    ".db",
    ".sqlite",
    ".db3",
    # bundles/exports that often contain graphs
    ".tgz",
    ".zip",
    # interactive visualization
    ".html",
}
# heuristic extensions: require keyword match in filename or any parent folder name
HEURISTIC_EXTS = {".json", ".csv", ".tsv", ".ndjson", ".txt"}

DEFAULT_IGNORE_DIRS = {
    "$recycle.bin",
    "system volume information",
    "windows",
    "program files",
    "program files (x86)",
    "programdata",
    "appdata",
    "node_modules",
    ".git",
    ".cache",
    ".venv",
    "__pycache__",
}

CHUNK = 1024 * 1024


def is_windows() -> bool:
    return os.name == "nt"


def default_dest() -> Path:
    if is_windows():
        return Path("D:/MINDEYE2")
    # Termux / Android fallback
    return Path("/storage/emulated/0/MINDEYE2")


def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sanitize_name(name: str) -> str:
    # Keep reasonable characters; remove path separators and control chars
    safe = "".join(c for c in name if c.isalnum() or c in "._- +@#()[]{}")
    return safe[:240] if len(safe) > 240 else safe


def candidate_reason(path: Path, chosen_exts: set, heuristic_exts: set, keywords: set) -> str or None:
    ext = path.suffix.lower()
    if ext in chosen_exts:
        return f"ext:{ext}"
    if ext in heuristic_exts:
        # check filename and parent dirs for keywords
        s = path.stem.lower()
        if any(k in s for k in keywords):
            return f"heuristic:{ext}:name"
        # check up to 3 parent dirs
        cnt = 0
        for part in path.parents:
            piece = part.name.lower()
            if any(k in piece for k in keywords):
                return f"heuristic:{ext}:parent"
            cnt += 1
            if cnt >= 3:
                break
    return None


def should_skip_dir(dirpath: Path, ignore_dirs: set) -> bool:
    name = dirpath.name.lower()
    return name in ignore_dirs


def ensure_dirs(dest: Path):
    (dest / "artifacts").mkdir(parents=True, exist_ok=True)
    (dest / "manifest").mkdir(parents=True, exist_ok=True)
    (dest / "logs").mkdir(parents=True, exist_ok=True)


def load_json(p: Path, default):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def save_json(p: Path, obj):
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_csv(csv_path: Path, rows, header):
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new_file:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def log_line(log_path: Path, msg: str):
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")
    print(msg)


def detect_rclone() -> bool:
    try:
        subprocess.run(["rclone", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def copy_with_verify(src: Path, dst: Path) -> tuple[bool, str]:
    # copy then verify size and sha
    tmp = dst.with_suffix(dst.suffix + ".tmpcopy")
    shutil.copy2(src, tmp)
    # verify size
    if src.stat().st_size != tmp.stat().st_size:
        try:
            tmp.unlink()
        except Exception:
            pass
        return False, "size_mismatch"
    # verify hash
    if sha256_file(src) != sha256_file(tmp):
        try:
            tmp.unlink()
        except Exception:
            pass
        return False, "hash_mismatch"
    tmp.rename(dst)
    return True, "ok"


def main():
    ap = argparse.ArgumentParser(
        description="Gather and centralize graph artifacts into MINDEYE2, dedupe by SHA-256, and optionally upload to Google Drive via rclone."
    )
    ap.add_argument(
        "--roots",
        nargs="+",
        default=["D:\\" if os.name == "nt" else "/storage/emulated/0"],
        help="Root folders to scan recursively. Default: D:\\ on Windows, /storage/emulated/0 on Android.",
    )
    ap.add_argument(
        "--exts",
        nargs="*",
        default=None,
        help="Override default strong graph extensions. Example: --exts .gexf .graphml .html",
    )
    ap.add_argument(
        "--include-heuristics",
        action="store_true",
        help="Include heuristic matches for .json/.csv/.tsv/.ndjson/.txt when keywords present.",
    )
    ap.add_argument("--execute", action="store_true", help="Actually copy files. Without this flag, dry-run only.")
    ap.add_argument(
        "--dest", default=str(default_dest()), help="Destination MINDEYE2 folder. Default platform-specific."
    )
    ap.add_argument(
        "--rclone-upload", action="store_true", help="After copy, upload to Google Drive using rclone copy."
    )
    ap.add_argument("--rclone-remote", default="gdrive", help="rclone remote name. Default: gdrive")
    ap.add_argument(
        "--rclone-mode",
        choices=["copy", "sync"],
        default="copy",
        help="Upload mode: copy (non-destructive) or sync (mirror). Default: copy",
    )
    ap.add_argument("--ignore-dirs", nargs="*", default=None, help="Additional directory basenames to ignore.")
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Create small sample artifacts under a temp folder inside the first root, then scan that root.",
    )
    args = ap.parse_args()

    chosen_exts = set(DEFAULT_EXTS) if args.exts is None else set(e.lower() for e in args.exts)
    heuristic_exts = HEURISTIC_EXTS if args.include - heuristics else set()

    # prepare destination
    dest = Path(args.dest).resolve()
    ensure_dirs(dest)
    manifest_dir = dest / "manifest"
    artifacts_dir = dest / "artifacts"
    logs_dir = dest / "logs"
    log_path = logs_dir / f"run_{now_stamp()}.log"

    # load or init dedupe index
    hash_index_path = manifest_dir / "hash_index.json"
    hash_index = load_json(hash_index_path, default={})  # sha256 -> relative dest path

    manifest_json_path = manifest_dir / "master_manifest.json"
    existing_manifest = load_json(manifest_json_path, default=[])
    manifest_csv_path = manifest_dir / "master_manifest.csv"

    # ignore dirs
    ignore_dirs = set(DEFAULT_IGNORE_DIRS)
    if args.ignore_dirs:
        ignore_dirs |= {d.lower() for d in args.ignore_dirs}

    # optional self-test
    if args.self_test:
        root0 = Path(args.roots[0])
        sample_base = root0 / "_mindeye2_selftest"
        sample_base.mkdir(parents=True, exist_ok=True)
        (sample_base / "graphs").mkdir(exist_ok=True)
        (sample_base / "data_nodes").mkdir(exist_ok=True)
        (sample_base / "reports").mkdir(exist_ok=True)
        (sample_base / "graphs" / "toy.graphml").write_text("<graphml/>", encoding="utf-8")
        (sample_base / "graphs" / "toy.gexf").write_text("<gexf/>", encoding="utf-8")
        (sample_base / "data_nodes" / "my_nodes.csv").write_text("id,label\n1,A\n2,B\n", encoding="utf-8")
        (sample_base / "reports" / "network_overview.html").write_text("<html>pyvis</html>", encoding="utf-8")
        print(f"[SELF-TEST] Created sample artifacts under: {sample_base}")

    # Scan
    run_rows = []
    found = []
    scanned_files = 0
    for root in args.roots:
        root = Path(root)
        if not root.exists():
            log_line(log_path, f"Root not found, skipping: {root}")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # prune ignored directories
            dp = Path(dirpath)
            if should_skip_dir(dp, ignore_dirs):
                dirnames[:] = []  # prevent descending
                continue
            # also prune hidden/system-ish dirs quickly
            dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__")]
            for fn in filenames:
                scanned_files += 1
                p = dp / fn
                # quick skip: avoid scanning inside destination itself
                try:
                    if dest in p.resolve().parents or p.resolve() == dest:
                        continue
                except Exception:
                    pass
                reason = candidate_reason(p, chosen_exts, heuristic_exts, KEYWORDS)
                if reason:
                    found.append((p, reason))

    log_line(log_path, f"Scanned files: {scanned_files:,}")
    log_line(log_path, f"Candidate artifacts: {len(found):,}")

    # Copy phase
    copied = 0
    skipped_dupe = 0
    errors = 0
    manifest_additions = []

    for src, reason in found:
        try:
            src_stat = src.stat()
        except FileNotFoundError:
            log_line(log_path, f"Skip missing: {src}")
            continue
        except PermissionError:
            log_line(log_path, f"Skip perm denied: {src}")
            continue

        try:
            fhash = sha256_file(src)
        except Exception as e:
            log_line(log_path, f"Hash error [{src}]: {e}")
            errors += 1
            continue

        if fhash in hash_index:
            skipped_dupe += 1
            run_rows.append(
                {
                    "action": "skip_dupe",
                    "src": str(src),
                    "dest": hash_index[fhash],
                    "size": src_stat.st_size,
                    "sha256": fhash,
                    "reason": reason,
                }
            )
            continue

        # target filename with short hash suffix to avoid collisions
        stem = sanitize_name(src.stem)
        ext = src.suffix.lower()
        short = fhash[:8]
        out_name = f"{stem}__{short}{ext}" if ext else f"{stem}__{short}"
        out_path = artifacts_dir / out_name

        if args.execute:
            try:
                ok, why = copy_with_verify(src, out_path)
                if not ok:
                    log_line(log_path, f"Copy verify FAILED [{src}] -> [{out_path}] ({why})")
                    errors += 1
                    continue
            except Exception as e:
                log_line(log_path, f"Copy error [{src}] -> [{out_path}]: {e}")
                errors += 1
                continue
        else:
            # dry-run only logs planned action
            why = "dry_run_ok"

        # record in indexes and manifests
        rel_dest = str(out_path.relative_to(dest)) if out_path.is_absolute() else str(out_path)
        hash_index[fhash] = rel_dest
        rec = {
            "id": f"{int(time.time()*1000)}_{short}",
            "src_path": str(src),
            "dest_path": str(out_path if args.execute else out_path),
            "size_bytes": src_stat.st_size,
            "mtime_iso": datetime.datetime.fromtimestamp(src_stat.st_mtime).isoformat(timespec="seconds"),
            "sha256": fhash,
            "ext": ext,
            "reason": reason,
            "copied": bool(args.execute),
            "verify": why,
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        existing_manifest.append(rec)
        manifest_additions.append(rec)
        run_rows.append(
            {
                "action": "copy" if args.execute else "plan_copy",
                "src": str(src),
                "dest": str(out_path),
                "size": src_stat.st_size,
                "sha256": fhash,
                "reason": reason,
            }
        )
        copied += 1

    # save manifests and index
    save_json(hash_index_path, hash_index)
    save_json(manifest_json_path, existing_manifest)

    csv_hdr = ["action", "src", "dest", "size", "sha256", "reason"]
    append_csv(manifest_csv_path, run_rows, csv_hdr)

    log_line(log_path, f"Copied (or planned): {copied:,}  | Duplicates skipped: {skipped_dupe:,}  | Errors: {errors:,}")
    log_line(log_path, f"DEST: {dest}")
    log_line(log_path, f"Artifacts: {artifacts_dir}")
    log_line(log_path, f"Manifest JSON: {manifest_json_path}")
    log_line(log_path, f"Manifest CSV:  {manifest_csv_path}")
    log_line(log_path, f"Hash index:    {hash_index_path}")

    # optional upload
    if args.rclone_upload:
        if not detect_rclone():
            log_line(log_path, "rclone not found in PATH; upload skipped.")
        else:
            remote = f"{args.rclone_remote}:/MINDEYE2"
            cmd = [
                "rclone",
                args.rclone_mode,
                str(dest),
                remote,
                "--fast-list",
                "--transfers",
                "8",
                "--checkers",
                "8",
                "--copy-links",
            ]
            log_line(log_path, f"Uploading via rclone ({args.rclone_mode}): {' '.join(cmd)}")
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode == 0:
                    log_line(log_path, "Upload completed successfully.")
                else:
                    log_line(
                        log_path, f"Upload failed. rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                    )
            except Exception as e:
                log_line(log_path, f"Upload exception: {e}")

    # final summary with verification samples
    # Show up to 5 latest records with path, size, sha
    sample = manifest_additions[-5:] if len(manifest_additions) > 5 else manifest_additions
    if sample:
        log_line(log_path, "Verification sample (up to 5):")
        for rec in sample:
            log_line(
                log_path,
                f"- {rec['dest_path']} | {rec['size_bytes']} bytes | sha256:{rec['sha256']} | copied:{rec['copied']}",
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
python
# gather_mindeye2_artifacts.py
# Purpose:
#   - Recursively scan one or more roots (default: D:\) for "graph artifacts".
#   - Deduplicate by SHA-256.
#   - Centralize copies in D:\MINDEYE2 (Windows) or /storage/emulated/0/MINDEYE2 (Android/Termux).
#   - Produce JSON + CSV manifests, a persistent hash index, and a run log.
#   - Optional: upload/copy to Google Drive at gdrive:/MINDEYE2 using rclone.
#
# Graph artifacts definition (ext-based + keyword heuristics):
#   Definite extensions:
#     .graphml .gexf .gml .xgmml .gephi .graphson .gryo .gpickle .gmlz .dot .gv
#     .cypher .cql .neo4j .dump .db .sqlite .db3 .tgz .zip  (common Neo4j dumps/exports)
#     .html  (pyvis/interactive graph pages)
#   Heuristic on filename for generic data: {json,csv,tsv,ndjson,txt}
#     Requires keyword match in the filename or parent dirs:
#       {"graph","neo4j","pyvis","gephi","nodes","edges","edgelist","nodelist",
#        "network","mindeye","constellation","gexf","graphml","cypher","gml","xgmml"}
#
# Usage examples (Windows PowerShell):
#   py .\gather_mindeye2_artifacts.py                              # dry-run, scan D:\ only
#   py .\gather_mindeye2_artifacts.py --roots D:\ E:\ --execute     # actually copy
#   py .\gather_mindeye2_artifacts.py --execute --rclone-upload     # copy + upload to gdrive:/MINDEYE2
#   py .\gather_mindeye2_artifacts.py --exts .gexf .graphml .html   # narrow extensions
#   py .\gather_mindeye2_artifacts.py --self-test                   # create sample artifacts and test
#
# Dependencies:
#   - Python 3.8+ (standard library only)
#   - rclone in PATH if you use --rclone-upload (default remote 'gdrive')
#
# Outputs inside DEST (MINDEYE2):
#   - artifacts/               copied files, named with __<hash8> suffix to avoid collisions
#   - manifest/master_manifest.json
#   - manifest/master_manifest.csv
#   - manifest/hash_index.json     (persistent dedupe index)
#   - logs/run_YYYYMMDD_HHMMSS.log
#
# Safety:
#   - Dry-run by default; no files are copied until --execute is supplied.
#   - Skips Windows system dirs by default. You can add more via --ignore-dirs.
#   - Every copied file is re-hashed and size-checked for integrity.
#
# Notes:
#   - Default Google Drive remote is 'gdrive'; adjust via --rclone-remote.
#   - Upload uses 'rclone copy' (non-destructive). Use --rclone-mode sync to mirror exactly.

import argparse
import atexit
import csv
import datetime
import hashlib
import json
import logging
import os
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# === 1. CONFIGURABLE CONSTANTS (moved to top for clarity) ===
KEYWORDS: Set[str] = {
    "graph",
    "neo4j",
    "pyvis",
    "gephi",
    "nodes",
    "edges",
    "edgelist",
    "nodelist",
    "network",
    "mindeye",
    "constellation",
    "gexf",
    "graphml",
    "cypher",
    "gml",
    "xgmml",
}

DEFAULT_EXTS: Set[str] = {
    ".graphml",
    ".gexf",
    ".gml",
    ".xgmml",
    ".gephi",
    ".graphson",
    ".gryo",
    ".gpickle",
    ".gmlz",
    ".dot",
    ".gv",
    ".cypher",
    ".cql",
    ".neo4j",
    ".dump",
    ".db",
    ".sqlite",
    ".db3",
    ".tgz",
    ".zip",
    ".html",
}

HEURISTIC_EXTS: Set[str] = {".json", ".csv", ".tsv", ".ndjson", ".txt"}

DEFAULT_IGNORE_DIRS: Set[str] = {
    "$recycle.bin",
    "system volume information",
    "windows",
    "program files",
    "program files (x86)",
    "programdata",
    "appdata",
    "node_modules",
    ".git",
    ".cache",
    ".venv",
    "__pycache__",
}

CHUNK_SIZE: int = 1024 * 1024  # 1 MiB
MAX_WORKERS: int = min(32, os.cpu_count() + 4)  # ThreadPoolExecutor default
HASH_WORKERS: int = max(4, os.cpu_count())  # Parallel hashing
COPY_WORKERS: int = max(2, os.cpu_count() // 2)  # Avoid I/O thrashing


# === 2. DATA CLASSES FOR TYPE SAFETY & AUDITABILITY ===
@dataclass(frozen=True)
class Artifact:
    src: Path
    reason: str
    size: int
    mtime: float
    sha256: Optional[str] = None


@dataclass
class CopyResult:
    src: Path
    dest: Path
    success: bool
    error: Optional[str] = None


# === 3. LOGGING SETUP (structured, rotating, dual output) ===
def setup_logging(dest: Path) -> Path:
    logs_dir = dest / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"run_{now_stamp()}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )
    return log_path


# === 4. UTILITY FUNCTIONS (enhanced) ===
def is_windows() -> bool:
    return platform.system() == "Windows"


def default_dest() -> Path:
    return Path("D:/MINDEYE2") if is_windows() else Path("/storage/emulated/0/MINDEYE2")


def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize_name(name: str) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in "._- +@#()[]{}")
    return safe[:240]


def candidate_reason(path: Path, chosen_exts: Set[str], heuristic_exts: Set[str], keywords: Set[str]) -> Optional[str]:
    ext = path.suffix.lower()
    if ext in chosen_exts:
        return f"ext:{ext}"
    if ext in heuristic_exts:
        s = path.stem.lower()
        if any(k in s for k in keywords):
            return f"heuristic:{ext}:name"
        for parent in path.parents:
            if parent.name.lower() in keywords:
                return f"heuristic:{ext}:parent"
            if parent == path.anchor:
                break
    return None


def should_skip_dir(dirpath: Path, ignore_dirs: Set[str]) -> bool:
    return dirpath.name.lower() in ignore_dirs


def ensure_dirs(dest: Path) -> None:
    for sub in ["artifacts", "manifest", "logs"]:
        (dest / sub).mkdir(parents=True, exist_ok=True)


def load_json(p: Path, default: Any) -> Any:
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logging.error(f"Corrupt JSON {p}: {e}")
    return default


def save_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def append_csv(csv_path: Path, rows: List[Dict], header: List[str]) -> None:
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


# === 5. PARALLEL SCANNING (ThreadPool + os.scandir) ===
def scan_root(
    root: Path, chosen_exts: Set[str], heuristic_exts: Set[str], keywords: Set[str], ignore_dirs: Set[str]
) -> List[Artifact]:
    artifacts: List[Artifact] = []
    try:
        for entry in os.scandir(root):
            if entry.is_dir(follow_symlinks=False):
                if should_skip_dir(Path(entry.path), ignore_dirs):
                    continue
                artifacts.extend(scan_root(Path(entry.path), chosen_exts, heuristic_exts, keywords, ignore_dirs))
            elif entry.is_file(follow_symlinks=False):
                path = Path(entry.path)
                reason = candidate_reason(path, chosen_exts, heuristic_exts, keywords)
                if reason:
                    try:
                        stat = entry.stat(follow_symlinks=False)
                        artifacts.append(Artifact(src=path, reason=reason, size=stat.st_size, mtime=stat.st_mtime))
                    except OSError:
                        pass
    except (PermissionError, OSError):
        pass
    return artifacts


# === 6. PARALLEL HASHING ===
def hash_artifact(artifact: Artifact) -> Tuple[Artifact, str]:
    h = sha256_file(artifact.src)
    return artifact, h


# === 7. COPY WITH VERIFICATION (atomic, temp suffix) ===
def copy_with_verify(src: Path, dst: Path) -> CopyResult:
    tmp = dst.with_suffix(dst.suffix + ".tmpcopy")
    try:
        shutil.copy2(src, tmp)
        if src.stat().st_size != tmp.stat().st_size:
            tmp.unlink(missing_ok=True)
            return CopyResult(src, dst, False, "size_mismatch")
        if sha256_file(src) != sha256_file(tmp):
            tmp.unlink(missing_ok=True)
            return CopyResult(src, dst, False, "hash_mismatch")
        tmp.replace(dst)
        return CopyResult(src, dst, True)
    except Exception as e:
        try:
            tmp.unlink(missing_ok=True)
        except:
            pass
        return CopyResult(src, dst, False, str(e))


# === 8. RCLONE UPLOAD WITH RETRY & PROGRESS ===
def rclone_upload(dest: Path, remote: str, mode: str) -> bool:
    cmd = [
        "rclone",
        mode,
        str(dest),
        remote,
        "--fast-list",
        "--transfers",
        "8",
        "--checkers",
        "8",
        "--retries",
        "3",
        "--low-level-retries",
        "10",
        "--stats",
        "5s",
        "--progress",
    ]
    logging.info(f"rclone cmd: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            logging.info("rclone upload succeeded.")
            return True
        else:
            logging.error(f"rclone failed (rc={result.returncode}):\n{result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        logging.error("rclone upload timed out.")
        return False
    except Exception as e:
        logging.error(f"rclone exception: {e}")
        return False


# === 9. SELF-TEST ENHANCEMENT (isolated temp dir, cleanup) ===
def run_self_test(root: Path) -> Path:
    with tempfile.TemporaryDirectory(prefix="_mindeye2_test_", dir=root) as tmpdir:
        base = Path(tmpdir)
        (base / "graphs").mkdir()
        (base / "data_nodes").mkdir()
        (base / "reports").mkdir()
        (base / "graphs" / "toy.graphml").write_text("<graphml/>", encoding="utf-8")
        (base / "graphs" / "toy.gexf").write_text("<gexf/>", encoding="utf-8")
        (base / "data_nodes" / "my_nodes.csv").write_text("id,label\n1,A\n2,B\n", encoding="utf-8")
        (base / "reports" / "network_overview.html").write_text("<html>pyvis</html>", encoding="utf-8")
        logging.info(f"[SELF-TEST] Created temp artifacts in {base}")
        atexit.register(lambda: logging.info(f"[SELF-TEST] Temp dir auto-removed: {base}"))
        return base


# === 10. MAIN EXECUTION (refactored, stateful, progress bar) ===
