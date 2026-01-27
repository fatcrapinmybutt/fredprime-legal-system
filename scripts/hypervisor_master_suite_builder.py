#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Builds the Hypervisor Master Suite bundle.

- Writes suite files to a target directory.
- Creates a manifest.json with CRC32+SHA256 receipts.
- Emits RUN_MASTER.bat and WATCH_MASTER.bat launchers.
- Exports a suite ZIP alongside the manifest.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

DEFAULT_BUCKET_SOURCE = Path("/mnt/data/LITIGATIONOS_DRIVE_BUCKET_FORGER_v2026_01_27D.py")
DEFAULT_TARGET = Path(r"F:\\CAPSTONE\\Litigation_OS\\HYPERVISOR_MASTER_SUITE")
DEFAULT_TAG = "HYP_MASTER"


@dataclass(frozen=True)
class BuildContext:
    stamp: str
    tag: str
    builder_name: str
    zip_name: str
    detroit_ts: dt.datetime


def detroit_timestamp() -> dt.datetime:
    tz = ZoneInfo("America/Detroit")
    return dt.datetime.now(tz=tz).replace(microsecond=0)


def make_context(tag: str) -> BuildContext:
    detroit = detroit_timestamp()
    stamp = detroit.strftime("%Y%m%d_%H%M")
    builder_name = f"{stamp}_{tag}_BUILDER.py"
    zip_name = f"{stamp}_{tag}_SUITE.zip"
    return BuildContext(
        stamp=stamp,
        tag=tag,
        builder_name=builder_name,
        zip_name=zip_name,
        detroit_ts=detroit,
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8", newline="\n")


def file_receipt(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "rel": str(path).replace("\\", "/"),
        "bytes": len(payload),
        "crc32_uint32": zlib.crc32(payload) & 0xFFFFFFFF,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def load_bucket_source(bucket_source: Path) -> str:
    if not bucket_source.exists():
        raise FileNotFoundError(f"Bucket source not found: {bucket_source}")
    return bucket_source.read_text(encoding="utf-8", errors="replace")


def build_runbook(context: BuildContext) -> str:
    detroit_str = context.detroit_ts.strftime("%Y-%m-%d %H:%M")
    return (
        f"# RUNBOOK — HYPERVISOR MASTER SUITE ({detroit_str} America/Detroit)\n\n"
        "## What this suite is\n"
        "This suite is a single integrated runner that performs, per cycle:\n"
        "1) BUCKET organizer (inventory, bucket plan, duplicate grouping, version families, optional safe copy; no move by default)\n"
        "2) ADVERSARIAL signal mining (structured MV signals + fastlane hits; no OCR)\n"
        "3) MERGE_PIPELINE (creates new merged artifacts; originals untouched)\n"
        "4) APPLY_GATE (token gates destructive apply like move/cleanup)\n"
        "5) GRAPH_FUSER (combined graph pack: jsonl/csv/neo4j load + html viewer + sqlite)\n\n"
        "It also includes:\n"
        "- WATCHER_DAEMON: file-change watcher that triggers mini-runs.\n\n"
        "## Install to your durable path (recommended)\n"
        "Run the builder you downloaded:\n"
        f"- `python {context.builder_name}`\n\n"
        "Default target:\n"
        "- `F:\\CAPSTONE\\Litigation_OS\\HYPERVISOR_MASTER_SUITE\\`\n\n"
        "The builder writes:\n"
        "- `SUITE\\hypervisor_master_orchestrator.py`\n"
        "- `SUITE\\engines\\*`\n"
        "- `RUN_MASTER.bat` and `WATCH_MASTER.bat`\n"
        "- `manifest.json` with CRC32+SHA256 receipts\n\n"
        "## Run (plan-first, safe)\n"
        "Double click:\n"
        "- `RUN_MASTER.bat`\n\n"
        "Or CLI:\n"
        "```powershell\n"
        "python SUITE\\hypervisor_master_orchestrator.py run --roots F:\\CAPSTONE\\Litigation_OS --out OUT --cycles 10 --stable-n 2 --eps 0 --bucket-apply plan --bucket-zip-inventory --bucket-content-scan --graph-physics fast\n\n"
        "Copy mirror into buckets (still non-destructive to originals)\n"
        "python SUITE\\hypervisor_master_orchestrator.py run --roots F:\\CAPSTONE\\Litigation_OS --out OUT --cycles 5 --bucket-apply copy --bucket-out-root BUCKETS --bucket-zip-inventory\n\n"
        "Destructive move or cleanup (gated)\n\n"
        "Run once in plan mode to generate apply token in:\n\n"
        "OUT\\RUNS\\<run_id>\\CYCLE_001\\APPLY_GATE\\apply_signature.json\n\n"
        "Rerun with:\n\n"
        "--bucket-apply move --apply-confirm <token>\n"
        "or\n\n"
        "--bucket-apply plan --bucket-apply-cleanup --apply-confirm <token>\n\n"
        "Watcher daemon\n"
        "python SUITE\\hypervisor_master_orchestrator.py watch --roots F:\\CAPSTONE\\Litigation_OS --out OUT --poll-seconds 10 --debounce-seconds 15\n\n"
        "Optional deps (increase extraction and watcher quality)\n"
        "pip install pdfminer.six python-docx striprtf watchdog\n\n"
        "No dependency is mandatory. Missing deps are logged to STOPPERS and the run continues.\n"
    )


def suite_sources(bucket_src: str) -> dict[str, str]:
    bucket_cycle_src = r'''\n\ndef bucket_cycle(\n    roots,\n    out_dir,\n    bucket_out_root,\n    plan_only,\n    copy_mode,\n    hash_on,\n    quick_sig,\n    verify_existing_hash,\n    zip_inventory,\n    zip_hash_members,\n    ocr_queue,\n    content_scan,\n    content_terms_file,\n    content_extra_terms,\n    content_include_exts,\n    content_max_bytes,\n    content_max_terms,\n    plan_cleanup,\n    apply_cleanup,\n    halt_on_hard,\n    apply_dir,\n    stoppers,\n    max_files,\n):\n    out_dir = Path(out_dir)\n    out_dir.mkdir(parents=True, exist_ok=True)\n    bumpers = []\n\n    recs = []\n    count = 0\n    for fr in iter_files(roots):\n        recs.append(fr)\n        count += 1\n        if max_files and count >= max_files:\n            break\n\n    if hash_on:\n        for r in recs:\n            try:\n                if (not r.sha256) or verify_existing_hash:\n                    r.sha256 = sha256_file(Path(r.path))\n            except Exception as exc:\n                stoppers.append({"code": "HASH_FAIL", "target": str(r.path), "detail": {"error": repr(exc)}})\n    if quick_sig:\n        for r in recs:\n            try:\n                r.quick_sig = quick_signature(Path(r.path))\n            except Exception as exc:\n                stoppers.append({"code": "QUICK_SIG_FAIL", "target": str(r.path), "detail": {"error": repr(exc)}})\n\n    for r in recs:\n        try:\n            r.integrity_key = integrity_key(Path(r.path))\n            r.bucket = classify_bucket(Path(r.path))\n            r.version_family = detect_version_family(Path(r.path))\n        except Exception as exc:\n            stoppers.append({"code": "RECORD_ENRICH_FAIL", "target": str(r.path), "detail": {"error": repr(exc)}})\n\n    catalog_jsonl = out_dir / "catalog_files.jsonl"\n    if catalog_jsonl.exists():\n        catalog_jsonl.unlink()\n    with catalog_jsonl.open("w", encoding="utf-8", newline="\n") as handle:\n        for r in recs:\n            handle.write(json.dumps({\n                "path": str(r.path),\n                "size_bytes": int(r.size_bytes),\n                "mtime_epoch": float(r.mtime_epoch),\n                "ext": str(r.ext),\n                "bucket": str(r.bucket),\n                "sha256": str(r.sha256),\n                "quick_sig": str(r.quick_sig),\n                "version_family": str(r.version_family),\n                "integrity_key": str(r.integrity_key),\n            }, ensure_ascii=False) + "\n")\n\n    actions = plan_bucket_paths(Path(bucket_out_root), recs, bumpers, copy_mode, hash_on, verify_existing_hash)\n    plan_moves = out_dir / "plan_moves.jsonl"\n    if plan_moves.exists():\n        plan_moves.unlink()\n    with plan_moves.open("w", encoding="utf-8", newline="\n") as handle:\n        for a in actions:\n            handle.write(json.dumps({"op": a.op, "src": str(a.src), "dst": str(a.dst), "reason": a.reason}, ensure_ascii=False) + "\n")\n\n    apply_executed = False\n    apply_counts = {}\n    if (not plan_only) and (not copy_mode):\n        apply_executed = False\n    elif copy_mode and (not plan_only):\n        try:\n            apply_counts = apply_plan(actions, bumpers, True, apply_dir, halt_on_hard)\n            apply_executed = True\n        except Exception as exc:\n            stoppers.append({"code": "COPY_APPLY_FAIL", "target": str(apply_dir), "detail": {"error": repr(exc)}})\n\n    dups = dedupe_groups(recs)\n    dups_sum = build_duplicates_summary(dups)\n    dup_json = out_dir / "duplicates_summary.json"\n    dup_json.write_text(json.dumps(dups_sum, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")\n\n    fams = group_versions(recs)\n    ver_sum = build_versions_summary(fams)\n    ver_json = out_dir / "versions_summary.json"\n    ver_json.write_text(json.dumps(ver_sum, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")\n\n    zip_inventory_jsonl = out_dir / "zip_inventory.jsonl"\n    zip_members_total = 0\n    if zip_inventory:\n        if zip_inventory_jsonl.exists():\n            zip_inventory_jsonl.unlink()\n        with zip_inventory_jsonl.open("w", encoding="utf-8", newline="\n") as handle:\n            for r in recs:\n                if str(r.ext).lower() == ".zip":\n                    try:\n                        inv = inventory_zip(Path(r.path), bumpers, hash_members=zip_hash_members)\n                        for item in inv:\n                            item["_zip_path"] = str(r.path)\n                            handle.write(json.dumps(item, ensure_ascii=False) + "\n")\n                        zip_members_total += len(inv)\n                    except Exception as exc:\n                        stoppers.append({"code": "ZIP_INV_FAIL", "target": str(r.path), "detail": {"error": repr(exc)}})\n\n    ocr_jsonl = out_dir / "ocr_queue.jsonl"\n    if ocr_queue:\n        if ocr_jsonl.exists():\n            ocr_jsonl.unlink()\n        with ocr_jsonl.open("w", encoding="utf-8", newline="\n") as handle:\n            for r in recs:\n                if str(r.ext).lower() == ".pdf":\n                    try:\n                        if needs_ocr_pdf(Path(r.path)):\n                            handle.write(json.dumps({"path": str(r.path), "reason": "needs_ocr_pdf_heuristic"}, ensure_ascii=False) + "\n")\n                    except Exception as exc:\n                        stoppers.append({"code": "OCR_HEUR_FAIL", "target": str(r.path), "detail": {"error": repr(exc)}})\n\n    content_flags_jsonl = out_dir / "content_flags.jsonl"\n    content_flags_count = 0\n    if content_scan:\n        try:\n            extra_terms = content_extra_terms or []\n            _summary, flags, adversarial_hits = content_scan_and_adversarial(\n                recs,\n                bumpers,\n                terms_file=(Path(content_terms_file) if content_terms_file else None),\n                include_exts=content_include_exts,\n                max_bytes=content_max_bytes,\n                max_terms=content_max_terms,\n                extra_terms=extra_terms,\n            )\n            if content_flags_jsonl.exists():\n                content_flags_jsonl.unlink()\n            with content_flags_jsonl.open("w", encoding="utf-8", newline="\n") as handle:\n                for item in flags:\n                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")\n                content_flags_count = len(flags)\n            adv_hits_path = out_dir / "adversarial_hits.jsonl"\n            if adv_hits_path.exists():\n                adv_hits_path.unlink()\n            with adv_hits_path.open("w", encoding="utf-8", newline="\n") as handle:\n                for item in adversarial_hits:\n                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")\n        except Exception as exc:\n            stoppers.append({"code": "CONTENT_SCAN_FAIL", "target": str(out_dir), "detail": {"error": repr(exc)}})\n\n    cleanup_plan_json = out_dir / "cleanup_plan.json"\n    if plan_cleanup:\n        try:\n            plan = plan_empty_folder_cleanup(roots, bumpers)\n            cleanup_plan_json.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")\n        except Exception as exc:\n            stoppers.append({"code": "CLEANUP_PLAN_FAIL", "target": str(out_dir), "detail": {"error": repr(exc)}})\n\n    bumpers_jsonl = out_dir / "bumpers.jsonl"\n    if bumpers_jsonl.exists():\n        bumpers_jsonl.unlink()\n    with bumpers_jsonl.open("w", encoding="utf-8", newline="\n") as handle:\n        for b in bumpers:\n            handle.write(json.dumps({"code": b.code, "severity": b.severity, "target": b.target, "detail": b.detail}, ensure_ascii=False) + "\n")\n\n    return {\n        "file_count": len(recs),\n        "planned_actions": len(actions),\n        "catalog_jsonl": str(catalog_jsonl),\n        "plan_moves_jsonl": str(plan_moves),\n        "duplicates_summary_json": str(dup_json),\n        "versions_summary_json": str(ver_json),\n        "zip_inventory_jsonl": str(zip_inventory_jsonl) if zip_inventory else "",\n        "zip_members_total": zip_members_total,\n        "content_flags_jsonl": str(content_flags_jsonl) if content_scan else "",\n        "content_flags": content_flags_count,\n        "cleanup_plan_json": str(cleanup_plan_json) if plan_cleanup else "",\n        "apply_executed": apply_executed,\n        "apply_counts": apply_counts,\n    }\n'''

    orchestrator_src = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HYPERVISOR_MASTER_ORCHESTRATOR
Adds: MERGE_PIPELINE + APPLY_GATE + WATCHER_DAEMON + GRAPH_FUSER
Wires: Bucket Organizer (Drive Bucket Forger) + Adversarial Signal Mining + FASTLANE-style detectors
Mode: chained cycles until convergence. Non-destructive defaults. No OCR.

Output contract:
OUT/RUNS/<run_id>/
  RUN/run_ledger.jsonl
  RUN/provenance_index.json
  RUN/convergence_report.json
  RUN/stoppers_log.json
  RUN/blockers_and_acquisition_plan.json
  CYCLE_###/
    BUCKET/ (inventory, bucket plan, optional apply outputs, zip inventory, content flags)
    ADVERSARIAL/ (events.jsonl, hits.jsonl, summaries)
    MERGE/ (merge_plan.jsonl, merged outputs, provenance_map.json)
    GRAPH/ (nodes+edges jsonl/csv, neo4j pack, html viewer, sqlite, dashboard)

Commands:
  run     : run chained cycles until convergence
  watch   : watcher daemon (polling; optional watchdog if installed)
  token   : compute apply-confirm token for an existing apply plan
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from engines import adversarial_signal_engine as ase
from engines import apply_gate as ag
from engines import drive_bucket_forger as dbf
from engines import graph_fuser as gf
from engines import merge_pipeline as mp
from engines import watcher_daemon as wd

APP_ID = "HYPERVISOR_MASTER_ORCHESTRATOR"
APP_VER = "v2026-01-27.2"


def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def append_jsonl(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def sha256_json(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_roots(roots: List[str]) -> List[Path]:
    out = []
    for root in roots:
        if not root:
            continue
        out.append(Path(root).expanduser())
    if out:
        return out
    if os.name == "nt":
        return [Path(r"F:\\CAPSTONE\\Litigation_OS")]
    return [Path.cwd()]


def convergence_signature(summary: Dict[str, Any]) -> str:
    keep = {
        "bucket": {
            "file_count": summary["bucket"].get("file_count", 0),
            "planned_actions": summary["bucket"].get("planned_actions", 0),
            "zip_members": summary["bucket"].get("zip_members_total", 0),
            "content_flags": summary["bucket"].get("content_flags", 0),
        },
        "adversarial": {
            "events": summary["adversarial"].get("events", 0),
            "hits": summary["adversarial"].get("hits", 0),
        },
        "merge": {
            "merge_candidates": summary["merge"].get("merge_candidates", 0),
            "merged_outputs": summary["merge"].get("merged_outputs", 0),
        },
        "graph": {
            "nodes": summary["graph"].get("nodes", 0),
            "edges": summary["graph"].get("edges", 0),
        },
        "apply": {
            "applied": summary["apply"].get("applied", False),
            "skipped_reason": summary["apply"].get("skipped_reason", "")[:120],
        },
    }
    return sha256_json(keep)


def run_one_cycle(
    roots: List[Path],
    cycle_dir: Path,
    out_run_dir: Path,
    args: argparse.Namespace,
    stoppers: List[dict],
) -> Dict[str, Any]:
    ensure_dir(cycle_dir)
    bucket_dir = cycle_dir / "BUCKET"
    adv_dir = cycle_dir / "ADVERSARIAL"
    merge_dir = cycle_dir / "MERGE"
    graph_dir = cycle_dir / "GRAPH"
    apply_dir = cycle_dir / "APPLY_GATE"
    for directory in [bucket_dir, adv_dir, merge_dir, graph_dir, apply_dir]:
        ensure_dir(directory)

    bucket_res = dbf.bucket_cycle(
        roots=roots,
        out_dir=bucket_dir,
        bucket_out_root=args.bucket_out_root,
        plan_only=(args.bucket_apply == "plan"),
        copy_mode=(args.bucket_apply == "copy"),
        hash_on=args.bucket_hash,
        quick_sig=args.bucket_quick_sig,
        verify_existing_hash=args.bucket_verify_existing_hash,
        zip_inventory=args.bucket_zip_inventory,
        zip_hash_members=args.bucket_zip_hash_members,
        ocr_queue=args.bucket_ocr_queue,
        content_scan=args.bucket_content_scan,
        content_terms_file=args.bucket_content_terms_file,
        content_extra_terms=args.bucket_content_extra_terms,
        content_include_exts=args.bucket_content_include_exts,
        content_max_bytes=args.bucket_content_max_bytes,
        content_max_terms=args.bucket_content_max_terms,
        plan_cleanup=args.bucket_plan_cleanup,
        apply_cleanup=args.bucket_apply_cleanup,
        halt_on_hard=args.halt_on_hard,
        apply_dir=bucket_dir / "APPLY",
        stoppers=stoppers,
        max_files=args.max_files,
    )

    adv_res = ase.scan_adversarial(
        roots=roots,
        out_dir=adv_dir,
        allow_exts=args.adv_allow_exts,
        max_files=args.adv_max_files,
        max_chars=args.adv_max_chars,
        stoppers=stoppers,
        bank=args.adv_bank,
    )

    merge_res = mp.run_merge_pipeline(
        records_jsonl=Path(bucket_res["catalog_jsonl"]),
        versions_summary_json=Path(bucket_res["versions_summary_json"]),
        duplicates_summary_json=Path(bucket_res["duplicates_summary_json"]),
        out_dir=merge_dir,
        max_group_size=args.merge_max_group_size,
        max_bytes_per_file=args.merge_max_bytes_per_file,
        stoppers=stoppers,
        enable_text_merge=not args.merge_disable_text,
        enable_table_merge=not args.merge_disable_tables,
        enable_code_merge=not args.merge_disable_code,
    )

    apply_res = ag.apply_gate_and_maybe_execute(
        bucket_apply=args.bucket_apply,
        apply_cleanup=args.bucket_apply_cleanup,
        planned_actions_jsonl=Path(bucket_res["plan_moves_jsonl"]),
        cleanup_plan_json=Path(bucket_res["cleanup_plan_json"]) if bucket_res.get("cleanup_plan_json") else None,
        apply_token_path=apply_dir / "apply_signature.json",
        apply_summary_path=apply_dir / "apply_summary.md",
        confirm_token=args.apply_confirm,
        policy_allow_copy_without_confirm=True,
        stoppers=stoppers,
    )

    if apply_res.get("skip_apply_execution", False):
        bucket_res["apply_executed"] = False

    graph_res = gf.fuse_graph(
        run_id=args.run_id,
        out_dir=graph_dir,
        bucket_catalog_jsonl=Path(bucket_res["catalog_jsonl"]),
        bucket_plan_jsonl=Path(bucket_res["plan_moves_jsonl"]),
        zip_inventory_jsonl=Path(bucket_res["zip_inventory_jsonl"]) if bucket_res.get("zip_inventory_jsonl") else None,
        content_flags_jsonl=Path(bucket_res["content_flags_jsonl"]) if bucket_res.get("content_flags_jsonl") else None,
        adversarial_events_jsonl=Path(adv_res["events_jsonl"]),
        adversarial_hits_jsonl=Path(adv_res["hits_jsonl"]),
        merge_provenance_json=Path(merge_res["provenance_map_json"]),
        physics_mode=args.graph_physics,
        stoppers=stoppers,
    )

    cycle_summary = {
        "ts": utc_now_iso(),
        "bucket": bucket_res,
        "adversarial": adv_res,
        "merge": merge_res,
        "apply": apply_res,
        "graph": graph_res,
    }
    write_json(cycle_dir / "cycle_summary.json", cycle_summary)
    return cycle_summary


def cmd_run(args: argparse.Namespace) -> int:
    roots = normalize_roots(args.roots)
    out_root = Path(args.out).expanduser().resolve()
    run_id = args.run_id.strip() or (
        f"RUN_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{hashlib.sha256(os.urandom(16)).hexdigest()[:6]}"
    )
    args.run_id = run_id
    run_dir = out_root / "RUNS" / run_id
    ensure_dir(run_dir / "RUN")
    ledger = run_dir / "RUN" / "run_ledger.jsonl"
    stoppers: List[dict] = []

    prov = {
        "app": APP_ID,
        "ver": APP_VER,
        "ts": utc_now_iso(),
        "run_id": run_id,
        "roots": [str(r) for r in roots],
        "out": str(out_root),
        "args": {k: getattr(args, k) for k in vars(args).keys() if k not in {"cmd"}},
    }
    write_json(run_dir / "RUN" / "provenance_index.json", prov)

    stable = 0
    last_sig = ""
    converged = False
    for cyc in range(1, max(1, args.cycles) + 1):
        cycle_dir = run_dir / f"CYCLE_{cyc:03d}"
        summary = run_one_cycle(roots, cycle_dir, run_dir, args, stoppers)
        sig = convergence_signature(summary)
        delta = 0 if (sig == last_sig) else 1
        stable = stable + 1 if delta <= args.eps else 0
        last_sig = sig
        converged = stable >= args.stable_n

        append_jsonl(
            ledger,
            {
                "ts": utc_now_iso(),
                "run_id": run_id,
                "cycle": cyc,
                "signature": sig,
                "delta": delta,
                "eps": args.eps,
                "stable": stable,
                "stable_n": args.stable_n,
                "converged": converged,
            },
        )
        if converged:
            break
        time.sleep(max(0, args.sleep_seconds))

    write_json(run_dir / "RUN" / "stoppers_log.json", {"count": len(stoppers), "stoppers": stoppers})
    blockers = {
        "missing_optional_deps": list(
            sorted(
                {
                    s.get("detail", {}).get("pip", "")
                    for s in stoppers
                    if s.get("code", "").endswith("_DEP_MISSING") and s.get("detail", {}).get("pip")
                }
            )
        ),
        "notes": [
            "System is bumpers-not-blockers: missing deps do not halt run.",
            "Destructive apply is gated by APPLY_GATE and requires --apply-confirm token.",
        ],
        "acquisition_plan": [
            "Optional deps to increase extraction: pip install pdfminer.six python-docx striprtf watchdog",
            "If you want move/cleanup apply: run once to generate apply token, then rerun with --apply-confirm <token>",
        ],
    }
    write_json(run_dir / "RUN" / "blockers_and_acquisition_plan.json", blockers)
    write_json(
        run_dir / "RUN" / "convergence_report.json",
        {
            "ts": utc_now_iso(),
            "run_id": run_id,
            "converged": converged,
            "cycles_ran": cyc,
            "stable": stable,
            "stable_n": args.stable_n,
            "eps": args.eps,
        },
    )

    final_md = "\n".join(
        [
            f"# FINAL_DELIVERABLE — {APP_ID} {APP_VER}",
            "",
            f"- Run ID: `{run_id}`",
            f"- Output root: `{str(run_dir)}`",
            f"- Converged: `{converged}`",
            "",
            "## Key outputs",
            f"- Run ledger: `{str(run_dir / 'RUN' / 'run_ledger.jsonl')}`",
            f"- Stoppers: `{str(run_dir / 'RUN' / 'stoppers_log.json')}`",
            f"- Graph pack: `{str(run_dir / f'CYCLE_{cyc:03d}' / 'GRAPH')}`",
            f"- Merge outputs: `{str(run_dir / f'CYCLE_{cyc:03d}' / 'MERGE')}`",
            "",
            "## Notes",
            "- Non-destructive defaults. No OCR.",
            "- Apply gate required for move/cleanup.",
        ]
    )
    (run_dir / "FINAL_DELIVERABLE.md").write_text(final_md + "\n", encoding="utf-8", newline="\n")
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    roots = normalize_roots(args.roots)
    out_root = Path(args.out).expanduser().resolve()
    ensure_dir(out_root)
    return wd.watch_loop(
        roots=roots,
        out_root=out_root,
        poll_seconds=args.poll_seconds,
        debounce_seconds=args.debounce_seconds,
        max_files=args.max_files,
        trigger_args=args,
    )


def cmd_token(args: argparse.Namespace) -> int:
    plan = Path(args.apply_plan).expanduser()
    if not plan.exists():
        print("APPLY_PLAN_NOT_FOUND:", str(plan))
        return 2
    token = ag.compute_confirm_token_from_plan(plan)
    print(token)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypervisor_master_orchestrator.py",
        description=(
            "Hypervisor Master Orchestrator (bucket + adversarial + merge + apply gate + graph fuser + watcher)."
        ),
    )
    sp = parser.add_subparsers(dest="cmd", required=True)

    pr = sp.add_parser("run", help="Run chained cycles until convergence.")
    pr.add_argument("--roots", nargs="*", default=[])
    pr.add_argument("--out", default="OUT")
    pr.add_argument("--run-id", default="")
    pr.add_argument("--cycles", type=int, default=10)
    pr.add_argument("--eps", type=int, default=0)
    pr.add_argument("--stable-n", type=int, default=2)
    pr.add_argument("--sleep-seconds", type=int, default=1)

    pr.add_argument("--max-files", type=int, default=250000)

    pr.add_argument("--bucket-apply", choices=["plan", "copy", "move"], default="plan")
    pr.add_argument("--bucket-out-root", default="BUCKETS")
    pr.add_argument("--bucket-hash", action="store_true")
    pr.add_argument("--bucket-quick-sig", action="store_true")
    pr.add_argument("--bucket-verify-existing-hash", action="store_true")
    pr.add_argument("--bucket-zip-inventory", action="store_true")
    pr.add_argument("--bucket-zip-hash-members", action="store_true")
    pr.add_argument("--bucket-ocr-queue", action="store_true")
    pr.add_argument("--bucket-content-scan", action="store_true")
    pr.add_argument("--bucket-content-terms-file", default="")
    pr.add_argument("--bucket-content-extra-terms", nargs="*", default=[])
    pr.add_argument("--bucket-content-include-exts", default="")
    pr.add_argument("--bucket-content-max-bytes", type=int, default=200000)
    pr.add_argument("--bucket-content-max-terms", type=int, default=2500)
    pr.add_argument("--bucket-plan-cleanup", action="store_true")
    pr.add_argument("--bucket-apply-cleanup", action="store_true")
    pr.add_argument("--halt-on-hard", action="store_true")

    pr.add_argument(
        "--adv-allow-exts",
        default=".txt,.md,.log,.json,.jsonl,.csv,.tsv,.html,.htm,.py,.ps1,.bat,.rtf,.docx,.pdf",
    )
    pr.add_argument("--adv-max-files", type=int, default=10000)
    pr.add_argument("--adv-max-chars", type=int, default=2000000)
    pr.add_argument("--adv-bank", type=int, default=3)

    pr.add_argument("--merge-max-group-size", type=int, default=12)
    pr.add_argument("--merge-max-bytes-per-file", type=int, default=2500000)
    pr.add_argument("--merge-disable-text", action="store_true")
    pr.add_argument("--merge-disable-tables", action="store_true")
    pr.add_argument("--merge-disable-code", action="store_true")

    pr.add_argument("--apply-confirm", default="")

    pr.add_argument("--graph-physics", choices=["off", "fast", "grid", "full"], default="fast")

    pw = sp.add_parser("watch", help="Watcher daemon (polling; optional watchdog if installed).")
    pw.add_argument("--roots", nargs="*", default=[])
    pw.add_argument("--out", default="OUT")
    pw.add_argument("--poll-seconds", type=int, default=10)
    pw.add_argument("--debounce-seconds", type=int, default=15)
    pw.add_argument("--max-files", type=int, default=250000)
    pw.add_argument("--bucket-apply", choices=["plan", "copy", "move"], default="plan")
    pw.add_argument("--bucket-out-root", default="BUCKETS")
    pw.add_argument("--bucket-hash", action="store_true")
    pw.add_argument("--bucket-quick-sig", action="store_true")
    pw.add_argument("--bucket-zip-inventory", action="store_true")
    pw.add_argument("--bucket-ocr-queue", action="store_true")
    pw.add_argument("--bucket-content-scan", action="store_true")
    pw.add_argument("--adv-bank", type=int, default=3)
    pw.add_argument("--apply-confirm", default="")
    pw.add_argument("--graph-physics", choices=["off", "fast", "grid", "full"], default="fast")

    pt = sp.add_parser("token", help="Compute apply-confirm token from an apply plan jsonl.")
    pt.add_argument("--apply-plan", required=True)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "watch":
        return cmd_watch(args)
    if args.cmd == "token":
        return cmd_token(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''

    adversarial_engine_src = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADVERSARIAL_SIGNAL_ENGINE
- Scans allowed file types and extracts text (no OCR).
- Emits:
  - adversarial_events.jsonl (structured signals)
  - adversarial_hits.jsonl (fastlane-style pattern hits)
  - adversarial_summary.json

Optional deps (never required):
  pip install pdfminer.six python-docx striprtf
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import os
import re
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def append_jsonl(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


EXCLUDE_DIR = {
    "$recycle.bin",
    "system volume information",
    "windows",
    "program files",
    "program files (x86)",
    "programdata",
    "appdata",
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
}
EXCLUDE_PATH = [
    r"\\Windows\\WinSxS\\",
    r"\\Windows\\Installer\\",
    r"\\Windows\\SoftwareDistribution\\",
    r"\\Program Files\\WindowsApps\\",
]


def should_ex_dir(name: str) -> bool:
    return name.lower() in EXCLUDE_DIR


def should_ex_path(path: str) -> bool:
    return any(re.search(pat, path, flags=re.IGNORECASE) for pat in EXCLUDE_PATH)


HAS_DOCX = importlib.util.find_spec("docx") is not None
if HAS_DOCX:
    import docx

HAS_STRIPRTF = importlib.util.find_spec("striprtf") is not None
if HAS_STRIPRTF:
    from striprtf.striprtf import rtf_to_text

HAS_PDFMINER = importlib.util.find_spec("pdfminer") is not None
if HAS_PDFMINER:
    from pdfminer.high_level import extract_text as pdfminer_extract_text


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def safe_read_text(path: Path, stoppers: List[dict]) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        stoppers.append({"code": "READ_TEXT_FAIL", "target": str(path), "detail": {"error": repr(exc)}})
        try:
            return path.read_text(encoding="latin-1", errors="replace")
        except Exception as exc2:
            stoppers.append({"code": "READ_TEXT_FAIL_2", "target": str(path), "detail": {"error": repr(exc2)}})
            return ""


def strip_html_to_text(text: str) -> str:
    text = re.sub(r"(?is)<(script|style)\b.*?>.*?</\1>", " ", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p\s*>", "\n", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    import html

    text = html.unescape(text)
    return normalize_ws(text)


def extract_text_no_ocr(path: Path, stoppers: List[dict], max_chars: int) -> Tuple[str, str]:
    ext = path.suffix.lower()
    try:
        if ext in (
            ".txt",
            ".log",
            ".md",
            ".markdown",
            ".py",
            ".js",
            ".ts",
            ".ps1",
            ".bat",
            ".sh",
            ".cypher",
            ".cql",
            ".cy",
            ".json",
            ".jsonl",
            ".ndjson",
            ".csv",
            ".tsv",
            ".yaml",
            ".yml",
            ".xml",
            ".html",
            ".htm",
        ):
            text = safe_read_text(path, stoppers)
            if ext in (".html", ".htm"):
                return (strip_html_to_text(text)[:max_chars], "html")
            return (text[:max_chars], "text")
        if ext == ".docx":
            if not HAS_DOCX:
                stoppers.append({"code": "DOCX_DEP_MISSING", "target": str(path), "detail": {"pip": "pip install python-docx"}})
                return ("", "docx")
            doc = docx.Document(str(path))
            parts = [p.text for p in doc.paragraphs if p.text.strip()]
            return ("\n".join(parts)[:max_chars], "docx")
        if ext == ".rtf":
            if not HAS_STRIPRTF:
                stoppers.append({"code": "RTF_DEP_MISSING", "target": str(path), "detail": {"pip": "pip install striprtf"}})
                return ("", "rtf")
            raw = path.read_text(encoding="utf-8", errors="replace")
            return (rtf_to_text(raw)[:max_chars], "rtf")
        if ext == ".pdf":
            if not HAS_PDFMINER:
                stoppers.append({"code": "PDF_DEP_MISSING", "target": str(path), "detail": {"pip": "pip install pdfminer.six"}})
                return ("", "pdf")
            text = (pdfminer_extract_text(str(path)) or "")
            if not text.strip():
                stoppers.append({"code": "PDF_TEXT_EMPTY_OCR_DEFERRED", "target": str(path), "detail": {"note": "No OCR performed."}})
                return ("", "pdf")
            return (text[:max_chars], "pdf")
        return ("", "unsupported")
    except Exception as exc:
        stoppers.append({"code": "EXTRACT_FAIL", "target": str(path), "detail": {"error": repr(exc)}})
        return ("", "error")


MV_RULES = [
    ("MV01", ["bias", "partial", "favorit", "prejudic", "hostile", "credibility"]),
    ("MV02", ["ppo", "personal protection order", "ex parte", "extend ppo", "weaponiz"]),
    ("MV03", ["show cause", "contempt", "jail", "bench warrant", "sanction"]),
    ("MV04", ["due process", "not allowed", "refused", "denied", "would not let", "no chance", "not heard"]),
    ("MV05", ["excluded", "refused evidence", "wouldn't take evidence", "not admitted", "not allowed to present", "not allowed to show"]),
    ("MV06", ["served", "notice", "short notice", "improper service", "no service"]),
    ("MV07", ["vague", "unclear order", "impossible", "cannot comply"]),
    ("MV08", ["transcript", "record missing", "omitted", "not on the record"]),
    ("MV09", ["withhold", "denied parenting", "no parenting time", "suspended parenting", "interfere"]),
    ("MV10", ["retaliat", "because you filed", "protected speech", "punished for"]),
]

NEG_PATS = [
    r"\bmanic\b",
    r"\bunstable\b",
    r"\bdangerous\b",
    r"\bstalking\b",
    r"\bharass(ing|ment)\b",
    r"\bthreat(en|ening)\b",
    r"\bviolent\b",
    r"\bdrug(s)?\b",
    r"\bcrazy\b",
]


def classify_mv(text: str) -> List[str]:
    lower = text.lower()
    out = []
    for code, kws in MV_RULES:
        if any(kw in lower for kw in kws):
            out.append(code)
    return out[:6]


def find_signal_snips(text: str, bank: int, max_snips: int) -> List[Dict]:
    parts = re.split(r"(?<=[\.\!\?])\s+|\n{2,}", text.replace("\r", "\n"))
    out = []
    for idx, part in enumerate(parts):
        snippet = normalize_ws(part)
        if len(snippet) < 12:
            continue
        lower = snippet.lower()
        hit = False
        if bank >= 1 and any(kw in lower for kw in ["not allowed", "refused", "denied", "would not let", "no chance", "not heard", "excluded", "wouldn't take evidence"]):
            hit = True
        if bank >= 2 and any(re.search(pat, lower, flags=re.IGNORECASE) for pat in NEG_PATS):
            hit = True
        if bank >= 3 and any(kw in lower for kw in ["ex parte", "without hearing", "without notice", "no opportunity", "reconsideration denied", "recusal denied"]):
            hit = True
        if hit:
            out.append({"idx": idx, "text": snippet[:900], "mv": classify_mv(snippet)})
            if len(out) >= max_snips:
                break
    return out


FASTLANE_PATTERNS = {
    "NEGATIVE_CHARACTERIZATION": [
        r"\bunfit\b",
        r"\babusive\b",
        r"\bdangerous\b",
        r"\bunstable\b",
        r"\bharass",
        r"\bstalk",
        r"\bthreat",
        r"\bviolent\b",
        r"\bfraud\b",
        r"\bperjur",
        r"\bfalse\b",
    ],
    "RIGHTS_PROCESS_VIOLATION": [
        r"due process",
        r"no notice",
        r"without notice",
        r"denied hearing",
        r"refused evidence",
        r"excluded evidence",
        r"not allowed",
        r"\bbias\b",
        r"\bpartial",
        r"ex parte",
        r"no opportunity",
    ],
    "RECORD_DEFECT": [
        r"no transcript",
        r"missing transcript",
        r"off the record",
        r"not recorded",
        r"inaudible",
        r"omitted",
        r"record mismatch",
    ],
}


def parse_allow_exts(text: str) -> Optional[set]:
    text = (text or "").strip()
    if not text:
        return None
    parts = [p.strip().lower() for p in re.split(r"[,\s]+", text) if p.strip()]
    return set(parts)


def iter_files(roots: List[Path], max_files: int, stoppers: List[dict], allow_exts: Optional[set]) -> List[Path]:
    out = []
    for root in roots:
        if not root.exists():
            stoppers.append({"code": "ROOT_NOT_FOUND", "target": str(root), "detail": {}})
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not should_ex_dir(d)]
            if should_ex_path(dirpath):
                dirnames[:] = []
                continue
            for filename in filenames:
                path = Path(dirpath) / filename
                if not path.is_file():
                    continue
                if allow_exts and (path.suffix.lower() not in allow_exts):
                    continue
                out.append(path)
                if max_files and len(out) >= max_files:
                    return out
    return out


def scan_adversarial(
    roots: List[Path],
    out_dir: Path,
    allow_exts: str,
    max_files: int,
    max_chars: int,
    stoppers: List[dict],
    bank: int,
) -> Dict:
    ensure_dir(out_dir)
    allow = parse_allow_exts(allow_exts)
    files = iter_files(roots, max_files, stoppers, allow)

    events_path = out_dir / "adversarial_events.jsonl"
    hits_path = out_dir / "adversarial_hits.jsonl"
    events_csv = out_dir / "adversarial_events.csv"
    summary_path = out_dir / "adversarial_summary.json"

    for path in [events_path, hits_path, events_csv]:
        if path.exists():
            path.unlink()

    compiled = {key: [re.compile(pat, re.IGNORECASE) for pat in pats] for key, pats in FASTLANE_PATTERNS.items()}
    with events_csv.open("w", encoding="utf-8", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=["file", "extract_mode", "mv_codes", "snippet_idx", "snippet"])
        writer.writeheader()
        events = 0
        hits = 0
        for path in files:
            text, mode = extract_text_no_ocr(path, stoppers, max_chars)
            if not text.strip():
                continue
            snips = find_signal_snips(text, bank, 30)
            for snip in snips:
                rec = {
                    "ts": utc_now_iso(),
                    "file": str(path),
                    "extract_mode": mode,
                    "mv_codes": snip["mv"],
                    "snippet_idx": snip["idx"],
                    "snippet": snip["text"],
                }
                append_jsonl(events_path, rec)
                writer.writerow(
                    {
                        "file": rec["file"],
                        "extract_mode": rec["extract_mode"],
                        "mv_codes": ",".join(rec["mv_codes"]),
                        "snippet_idx": rec["snippet_idx"],
                        "snippet": rec["snippet"],
                    }
                )
                events += 1
            lower = text.lower()
            for cat, regs in compiled.items():
                for rgx in regs:
                    match = rgx.search(lower)
                    if match:
                        start = max(0, match.start() - 180)
                        end = min(len(text), match.end() + 180)
                        excerpt = normalize_ws(text[start:end])[:700]
                        append_jsonl(
                            hits_path,
                            {
                                "ts": utc_now_iso(),
                                "file": str(path),
                                "category": cat,
                                "pattern": rgx.pattern,
                                "excerpt": excerpt,
                            },
                        )
                        hits += 1
                        break

    write_json(
        summary_path,
        {
            "ts": utc_now_iso(),
            "files_considered": len(files),
            "events": events,
            "hits": hits,
            "events_jsonl": str(events_path),
            "hits_jsonl": str(hits_path),
        },
    )
    return {
        "events": events,
        "hits": hits,
        "events_jsonl": str(events_path),
        "hits_jsonl": str(hits_path),
        "summary_json": str(summary_path),
    }
'''

    merge_pipeline_src = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MERGE_PIPELINE
Non-destructive merge planner and merged-output generator.
- Consumes bucket catalog + versions_summary + duplicates_summary.
- Emits:
  - merge_plan.jsonl (actions)
  - merged/ (new merged artifacts)
  - provenance_map.json (merged output -> sources + method)

Merge strategy (deterministic and conservative):
- Duplicate (same content hash): pick canonical only, no merged artifact.
- Version families:
  - For text-like files: create merged artifact with header blocks per source (provenance preserved).
  - For CSV/TSV: union by concatenation with source header and normalized row trimming.
  - For JSON/JSONL: concatenation (JSONL) or bundle list (JSON) if parseable; otherwise text merge.
  - For code: textual merge by concatenation with file headers (no AST rewriting).

This stage does not modify originals. It generates new merged outputs only.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


TEXT_EXT = {".txt", ".md", ".log", ".rtf", ".html", ".htm"}
CODE_EXT = {".py", ".js", ".ts", ".ps1", ".bat", ".sh", ".c", ".cpp", ".rs", ".java", ".cs", ".go"}
TABLE_EXT = {".csv", ".tsv"}
JSON_EXT = {".json", ".jsonl", ".ndjson"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def append_jsonl(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def iter_jsonl(path: Path) -> List[dict]:
    out = []
    if not path or (not path.exists()):
        return out
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def safe_read_bytes(path: Path, max_bytes: int) -> bytes:
    payload = path.read_bytes()
    if max_bytes and len(payload) > max_bytes:
        return payload[:max_bytes]
    return payload


def safe_read_text(path: Path, max_bytes: int) -> str:
    try:
        payload = safe_read_bytes(path, max_bytes)
        return payload.decode("utf-8", errors="replace")
    except Exception:
        try:
            return path.read_text(encoding="latin-1", errors="replace")
        except Exception:
            return ""


def canonical_choice(paths: List[str], meta: Dict[str, Any]) -> str:
    best = ""
    best_key = (-1, -1)
    for path in paths:
        meta_row = meta.get(path, {}) or {}
        key = (int(meta_row.get("mtime_epoch", 0)), int(meta_row.get("size_bytes", 0)))
        if key > best_key:
            best_key = key
            best = path
    return best or (paths[0] if paths else "")


def merge_text_block(sources: List[Dict[str, Any]], max_bytes_per_file: int) -> str:
    parts = []
    for src in sources:
        path = Path(src["path"])
        text = safe_read_text(path, max_bytes_per_file)
        parts.append(
            "\n".join(
                [
                    "===== SOURCE_BEGIN =====",
                    f"PATH: {src['path']}",
                    f"SIZE_BYTES: {src.get('size_bytes', '')}",
                    f"MTIME: {src.get('mtime', '')}",
                    "----- CONTENT -----",
                    text.rstrip(),
                    "===== SOURCE_END =====",
                    "",
                ]
            )
        )
    return "\n".join(parts)


def merge_csv_concat(sources: List[Dict[str, Any]], max_bytes_per_file: int) -> str:
    out = []
    for src in sources:
        path = Path(src["path"])
        raw = safe_read_text(path, max_bytes_per_file)
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line for line in raw.split("\n") if line.strip()]
        out.append(f"# SOURCE: {src['path']}")
        out.extend(lines)
        out.append("")
    return "\n".join(out)


def merge_json_smart(sources: List[Dict[str, Any]], max_bytes_per_file: int) -> Tuple[str, str]:
    items = []
    for src in sources:
        path = Path(src["path"])
        text = safe_read_text(path, max_bytes_per_file).strip()
        if not text:
            continue
        ext = path.suffix.lower()
        if ext in (".jsonl", ".ndjson"):
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append({"source": src["path"], "row": json.loads(line)})
                except Exception:
                    items.append({"source": src["path"], "row_raw": line})
        else:
            try:
                obj = json.loads(text)
                items.append({"source": src["path"], "json": obj})
            except Exception:
                items.append({"source": src["path"], "raw": text})
    payload = json.dumps(items, indent=2, ensure_ascii=False)
    return payload, "json_bundle"


def run_merge_pipeline(
    records_jsonl: Path,
    versions_summary_json: Path,
    duplicates_summary_json: Path,
    out_dir: Path,
    max_group_size: int,
    max_bytes_per_file: int,
    stoppers: List[dict],
    enable_text_merge: bool,
    enable_table_merge: bool,
    enable_code_merge: bool,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    merged_dir = out_dir / "merged"
    ensure_dir(merged_dir)
    plan_path = out_dir / "merge_plan.jsonl"
    prov_path = out_dir / "provenance_map.json"
    summary_path = out_dir / "merge_summary.json"

    if plan_path.exists():
        plan_path.unlink()

    recs = iter_jsonl(records_jsonl)
    meta = {}
    for record in recs:
        try:
            meta[record["path"]] = {
                "size_bytes": record.get("size_bytes", 0),
                "mtime": record.get("mtime", ""),
                "mtime_epoch": record.get("mtime_epoch", 0),
                "ext": record.get("ext", ""),
            }
        except Exception:
            continue

    versions = read_json(versions_summary_json) if versions_summary_json and versions_summary_json.exists() else {}
    dups = read_json(duplicates_summary_json) if duplicates_summary_json and duplicates_summary_json.exists() else {}

    provenance = {"merged_outputs": [], "canonical_only": [], "skipped": []}
    merge_candidates = 0
    merged_outputs = 0

    for gkey, group in (dups or {}).items():
        paths = group.get("paths") or []
        if not paths:
            continue
        canon = canonical_choice(paths, meta)
        provenance["canonical_only"].append({"group": "dup:" + gkey, "canonical": canon, "sources": paths})
        append_jsonl(plan_path, {"action": "CANONICAL_SELECT", "group": "dup:" + gkey, "canonical": canon, "sources": paths})

    for fam_key, fam in (versions or {}).items():
        paths = fam.get("paths") or []
        if len(paths) < 2:
            continue
        merge_candidates += 1
        paths = paths[: max(2, max_group_size)]
        sources = [{"path": p, **(meta.get(p, {}))} for p in paths]
        canon = canonical_choice(paths, meta)
        ext = Path(canon).suffix.lower()

        method = ""
        out_path = None
        if ext in TABLE_EXT and enable_table_merge:
            method = "csv_concat"
            out_path = merged_dir / f"MERGED_{safe_name(fam_key)}.csv"
            out_path.write_text(merge_csv_concat(sources, max_bytes_per_file) + "\n", encoding="utf-8", newline="\n")
        elif ext in JSON_EXT and enable_text_merge:
            method = "json_bundle"
            out_path = merged_dir / f"MERGED_{safe_name(fam_key)}.json"
            payload, _mode = merge_json_smart(sources, max_bytes_per_file)
            out_path.write_text(payload + "\n", encoding="utf-8", newline="\n")
        elif (ext in CODE_EXT) and enable_code_merge:
            method = "code_concat"
            out_path = merged_dir / f"MERGED_{safe_name(fam_key)}{ext if ext else '.txt'}"
            out_path.write_text(merge_text_block(sources, max_bytes_per_file) + "\n", encoding="utf-8", newline="\n")
        elif (ext in TEXT_EXT) and enable_text_merge:
            method = "text_block"
            out_path = merged_dir / f"MERGED_{safe_name(fam_key)}.txt"
            out_path.write_text(merge_text_block(sources, max_bytes_per_file) + "\n", encoding="utf-8", newline="\n")
        else:
            provenance["skipped"].append(
                {
                    "family": fam_key,
                    "reason": "unsupported_or_disabled",
                    "canonical": canon,
                    "sources": paths,
                }
            )
            append_jsonl(
                plan_path,
                {
                    "action": "MERGE_SKIPPED",
                    "family": fam_key,
                    "reason": "unsupported_or_disabled",
                    "canonical": canon,
                    "sources": paths,
                },
            )
            continue

        merged_outputs += 1
        provenance["merged_outputs"].append(
            {
                "family": fam_key,
                "method": method,
                "output": str(out_path),
                "canonical": canon,
                "sources": paths,
            }
        )
        append_jsonl(
            plan_path,
            {
                "action": "MERGE_CREATE",
                "family": fam_key,
                "method": method,
                "output": str(out_path),
                "canonical": canon,
                "sources": paths,
            },
        )

    write_json(prov_path, provenance)
    write_json(
        summary_path,
        {
            "merge_candidates": merge_candidates,
            "merged_outputs": merged_outputs,
            "plan_jsonl": str(plan_path),
            "merged_dir": str(merged_dir),
            "provenance_map_json": str(prov_path),
        },
    )
    return {
        "merge_candidates": merge_candidates,
        "merged_outputs": merged_outputs,
        "merge_plan_jsonl": str(plan_path),
        "merged_dir": str(merged_dir),
        "provenance_map_json": str(prov_path),
        "merge_summary_json": str(summary_path),
    }


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", (text or "")).strip("_")
    return text[:120] if text else "FAMILY"
'''

    apply_gate_src = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APPLY_GATE
Produces apply token and gating artifacts, and decides whether to execute destructive apply.
This module does not execute filesystem changes; it is an orchestrator gate.

Token scheme:
- token = SHA256( first 2MB of plan_jsonl bytes )[:16]

Outputs:
- apply_signature.json (token + hash receipts)
- apply_summary.md (human-readable summary)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, List, Dict, Any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8", newline="\n")


def compute_confirm_token_from_plan(plan_jsonl: Path) -> str:
    payload = plan_jsonl.read_bytes()
    payload = payload[:2_000_000]
    return hashlib.sha256(payload).hexdigest()[:16]


def apply_gate_and_maybe_execute(
    bucket_apply: str,
    apply_cleanup: bool,
    planned_actions_jsonl: Path,
    cleanup_plan_json: Optional[Path],
    apply_token_path: Path,
    apply_summary_path: Path,
    confirm_token: str,
    policy_allow_copy_without_confirm: bool,
    stoppers: List[dict],
) -> Dict[str, Any]:
    destructive = (bucket_apply == "move") or bool(apply_cleanup)
    if planned_actions_jsonl and planned_actions_jsonl.exists():
        token = compute_confirm_token_from_plan(planned_actions_jsonl)
        write_json(apply_token_path, {"token": token, "plan": str(planned_actions_jsonl)})
        summary_lines = [
            "# APPLY_GATE",
            "",
            f"- planned_actions_jsonl: {str(planned_actions_jsonl)}",
            f"- computed_token: {token}",
            f"- bucket_apply: {bucket_apply}",
            f"- apply_cleanup: {apply_cleanup}",
            "",
            "## Policy",
            "- Non-destructive defaults. Destructive apply requires explicit confirm token.",
        ]
        write_text(apply_summary_path, "\n".join(summary_lines) + "\n")
    else:
        stoppers.append(
            {
                "code": "APPLY_PLAN_MISSING",
                "target": str(planned_actions_jsonl),
                "detail": {"note": "No plan file to compute token."},
            }
        )
        token = ""

    if not destructive:
        if bucket_apply == "copy" and (not policy_allow_copy_without_confirm):
            if confirm_token.strip() != token:
                return {
                    "applied": False,
                    "skip_apply_execution": True,
                    "skipped_reason": "COPY_GATED_TOKEN_MISMATCH",
                    "token": token,
                }
        return {"applied": True, "skip_apply_execution": False, "token": token, "skipped_reason": ""}

    if (not confirm_token) or (confirm_token.strip() != token):
        return {
            "applied": False,
            "skip_apply_execution": True,
            "token": token,
            "skipped_reason": "DESTRUCTIVE_APPLY_REQUIRES_CONFIRM_TOKEN",
        }

    return {"applied": True, "skip_apply_execution": False, "token": token, "skipped_reason": ""}
'''

    graph_fuser_src = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GRAPH_FUSER
Builds a combined graph pack from:
- Bucket catalog + planned actions + zip inventory + content flags
- Adversarial events + hits
- Merge provenance (merged outputs and edges)
Emits:
- nodes.jsonl, edges.jsonl
- NEO4J_IMPORT/nodes.csv, edges.csv, constraints.cypher, indexes.cypher, load.cypher
- virtual_graph.sqlite
- graph_viewer.html
- dashboard_index.html

Uses the graph helpers embedded in drive_bucket_forger engine for consistency.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import drive_bucket_forger as dbf


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def append_jsonl(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def iter_jsonl(path: Path) -> List[dict]:
    out = []
    if not path or (not path.exists()):
        return out
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def to_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def fuse_graph(
    run_id: str,
    out_dir: Path,
    bucket_catalog_jsonl: Path,
    bucket_plan_jsonl: Path,
    zip_inventory_jsonl: Optional[Path],
    content_flags_jsonl: Optional[Path],
    adversarial_events_jsonl: Path,
    adversarial_hits_jsonl: Path,
    merge_provenance_json: Path,
    physics_mode: str,
    stoppers: List[dict],
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    nodes_path = out_dir / "nodes.jsonl"
    edges_path = out_dir / "edges.jsonl"
    neo_dir = out_dir / "NEO4J_IMPORT"
    ensure_dir(neo_dir)

    for path in [nodes_path, edges_path]:
        if path.exists():
            path.unlink()

    recs_raw = iter_jsonl(bucket_catalog_jsonl)
    recs = []
    for record in recs_raw:
        try:
            recs.append(
                dbf.FileRecord(
                    path=Path(record["path"]),
                    size_bytes=int(record.get("size_bytes", 0)),
                    mtime_epoch=float(record.get("mtime_epoch", 0)),
                    ext=str(record.get("ext", "")),
                    bucket=str(record.get("bucket", "")),
                    sha256=str(record.get("sha256", "")),
                    quick_sig=str(record.get("quick_sig", "")),
                    version_family=str(record.get("version_family", "")),
                    integrity_key=str(record.get("integrity_key", "")),
                )
            )
        except Exception as exc:
            stoppers.append({"code": "GRAPH_REC_PARSE_FAIL", "target": str(record.get("path", "")), "detail": {"error": repr(exc)}})

    actions_raw = iter_jsonl(bucket_plan_jsonl)
    actions = []
    for action in actions_raw:
        try:
            actions.append({"src": action.get("src", ""), "dst": action.get("dst", ""), "op": action.get("op", "")})
        except Exception:
            continue

    zip_inv = iter_jsonl(zip_inventory_jsonl) if zip_inventory_jsonl else []
    content_flags = iter_jsonl(content_flags_jsonl) if content_flags_jsonl else []
    adv_events = iter_jsonl(adversarial_events_jsonl)
    adv_hits = iter_jsonl(adversarial_hits_jsonl)

    merge_edges = []
    merge_nodes = []
    if merge_provenance_json and merge_provenance_json.exists():
        try:
            prov = json.loads(merge_provenance_json.read_text(encoding="utf-8", errors="replace"))
            for merged in prov.get("merged_outputs", []):
                outp = merged.get("output", "")
                if outp:
                    merge_nodes.append({"id": "MERGED:" + outp, "label": "MergedArtifact", "path": outp})
                    for src in merged.get("sources", []):
                        merge_edges.append({"src": "FILE:" + src, "dst": "MERGED:" + outp, "type": "MERGED_INTO"})
        except Exception as exc:
            stoppers.append({"code": "MERGE_PROV_PARSE_FAIL", "target": str(merge_provenance_json), "detail": {"error": repr(exc)}})

    nodes, edges, metrics = dbf.build_graph_model(run_id, recs, actions, zip_inv, content_flags, adv_hits)

    for ev in adv_events:
        eid = "EVT:" + hash_id(ev)
        nodes.append(
            {
                "id": eid,
                "label": "AdversarialEvent",
                "file": ev.get("file", ""),
                "mv_codes": ",".join(ev.get("mv_codes", [])),
                "snippet": ev.get("snippet", "")[:240],
            }
        )
        fpath = ev.get("file", "")
        if fpath:
            edges.append({"src": "FILE:" + fpath, "dst": eid, "type": "HAS_EVENT"})

    nodes.extend(merge_nodes)
    edges.extend(merge_edges)

    uniq_nodes = {}
    for node in nodes:
        node_id = node.get("id")
        if not node_id:
            continue
        if node_id not in uniq_nodes:
            uniq_nodes[node_id] = node
    nodes = list(uniq_nodes.values())

    for node in nodes:
        append_jsonl(nodes_path, node)
    for edge in edges:
        append_jsonl(edges_path, edge)

    node_fields = sorted(list(set().union(*[set(node.keys()) for node in nodes])))
    edge_fields = sorted(list(set().union(*[set(edge.keys()) for edge in edges])))
    to_csv(neo_dir / "nodes.csv", nodes, node_fields)
    to_csv(neo_dir / "edges.csv", edges, edge_fields)

    constraints = "CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;\n"
    indexes = "CREATE INDEX node_label IF NOT EXISTS FOR (n:Node) ON (n.label);\n"
    load = "\n".join(
        [
            "LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row",
            "MERGE (n:Node {id: row.id})",
            "SET n += row;",
            "",
            "LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row",
            "MATCH (a:Node {id: row.src})",
            "MATCH (b:Node {id: row.dst})",
            "MERGE (a)-[r:REL {type: row.type}]->(b)",
            "SET r += row;",
            "",
        ]
    )
    (neo_dir / "constraints.cypher").write_text(constraints, encoding="utf-8", newline="\n")
    (neo_dir / "indexes.cypher").write_text(indexes, encoding="utf-8", newline="\n")
    (neo_dir / "load.cypher").write_text(load, encoding="utf-8", newline="\n")

    try:
        dbf.emit_virtual_graph_sqlite(out_dir / "virtual_graph.sqlite", nodes, edges)
    except Exception as exc:
        stoppers.append({"code": "VIRTUAL_GRAPH_FAIL", "target": str(out_dir / "virtual_graph.sqlite"), "detail": {"error": repr(exc)}})
    try:
        positions = {}
        dbf.emit_html_graph_viewer(out_dir / "graph_viewer.html", nodes, edges, positions, title=f"Graph — {run_id}")
    except Exception as exc:
        stoppers.append({"code": "HTML_GRAPH_FAIL", "target": str(out_dir / "graph_viewer.html"), "detail": {"error": repr(exc)}})
    try:
        dbf.emit_dashboard_index(
            out_dir / "dashboard_index.html",
            {
                "Neo4j Import (nodes.csv)": str((neo_dir / "nodes.csv").name),
                "Neo4j Import (edges.csv)": str((neo_dir / "edges.csv").name),
                "Neo4j LOAD.cypher": str((neo_dir / "load.cypher").name),
                "Graph Viewer": "graph_viewer.html",
                "Virtual Graph SQLite": "virtual_graph.sqlite",
            },
            title=f"Dashboard — {run_id}",
        )
    except Exception as exc:
        stoppers.append({"code": "DASHBOARD_FAIL", "target": str(out_dir / "dashboard_index.html"), "detail": {"error": repr(exc)}})

    write_json(out_dir / "graph_metrics.json", {"metrics": metrics, "nodes": len(nodes), "edges": len(edges)})
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "nodes_jsonl": str(nodes_path),
        "edges_jsonl": str(edges_path),
        "neo4j_dir": str(neo_dir),
        "dashboard": str(out_dir / "dashboard_index.html"),
    }


def hash_id(ev: dict) -> str:
    payload = json.dumps(ev, sort_keys=True, ensure_ascii=False).encode("utf-8")
    import hashlib

    return hashlib.sha256(payload).hexdigest()[:16]
'''

    watcher_daemon_src = r'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WATCHER_DAEMON
Default: polling snapshot (mtime+size) with debounce.
Optional: watchdog if installed (used automatically when available and selected).

Emits:
OUT/WATCHER/
  events.jsonl
  snapshots/
  last_trigger.json

When changes detected: triggers a 1-cycle orchestrator run by invoking the orchestrator module in-process.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from . import adversarial_signal_engine as ase
from . import apply_gate as ag
from . import drive_bucket_forger as dbf
from . import graph_fuser as gf
from . import merge_pipeline as mp


def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


EXCLUDE_DIR = {
    "$recycle.bin",
    "system volume information",
    "windows",
    "program files",
    "program files (x86)",
    "programdata",
    "appdata",
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
}


def should_ex_dir(name: str) -> bool:
    return name.lower() in EXCLUDE_DIR


def iter_files_poll(roots: List[Path], max_files: int) -> List[Path]:
    out = []
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not should_ex_dir(d)]
            for filename in filenames:
                path = Path(dirpath) / filename
                if path.is_file():
                    out.append(path)
                    if max_files and len(out) >= max_files:
                        return out
    return out


def snapshot(roots: List[Path], max_files: int) -> Dict[str, Tuple[int, int]]:
    snap = {}
    for path in iter_files_poll(roots, max_files):
        try:
            stat = path.stat()
            snap[str(path)] = (int(stat.st_size), int(stat.st_mtime))
        except Exception:
            continue
    return snap


def diff_snap(a: Dict[str, Tuple[int, int]], b: Dict[str, Tuple[int, int]]) -> Dict[str, Any]:
    changed = []
    added = []
    removed = []
    for key, value in b.items():
        if key not in a:
            added.append(key)
        else:
            if a[key] != value:
                changed.append(key)
    for key in a.keys():
        if key not in b:
            removed.append(key)
    return {"added": added, "changed": changed, "removed": removed}


def watch_loop(
    roots: List[Path],
    out_root: Path,
    poll_seconds: int,
    debounce_seconds: int,
    max_files: int,
    trigger_args: Any,
) -> int:
    watch_dir = out_root / "WATCHER"
    ensure_dir(watch_dir / "snapshots")
    events = watch_dir / "events.jsonl"
    last_trigger = watch_dir / "last_trigger.json"

    last_snap = snapshot(roots, max_files)
    write_json(watch_dir / "snapshots" / "snap_000.json", {"ts": utc_now_iso(), "count": len(last_snap)})
    last_fire = 0.0
    tick = 0
    while True:
        time.sleep(max(1, poll_seconds))
        tick += 1
        cur = snapshot(roots, max_files)
        diff = diff_snap(last_snap, cur)
        last_snap = cur
        any_change = bool(diff["added"] or diff["changed"] or diff["removed"])
        if any_change:
            append_jsonl(events, {"ts": utc_now_iso(), "event": "FS_CHANGE", "diff": {k: len(v) for k, v in diff.items()}})
            now = time.time()
            if (now - last_fire) < debounce_seconds:
                append_jsonl(
                    events,
                    {"ts": utc_now_iso(), "event": "DEBOUNCE_SKIP", "seconds_since_last": (now - last_fire)},
                )
                continue
            last_fire = now
            run_id = f"WATCHRUN_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{hashlib.sha256(os.urandom(8)).hexdigest()[:6]}"
            write_json(
                last_trigger,
                {"ts": utc_now_iso(), "run_id": run_id, "diff_counts": {k: len(v) for k, v in diff.items()}},
            )

            stoppers = []
            run_dir = out_root / "RUNS" / run_id
            ensure_dir(run_dir / "RUN")
            cycle_dir = run_dir / "CYCLE_001"
            ensure_dir(cycle_dir)

            bucket_res = dbf.bucket_cycle(
                roots=roots,
                out_dir=cycle_dir / "BUCKET",
                bucket_out_root=getattr(trigger_args, "bucket_out_root", "BUCKETS"),
                plan_only=(getattr(trigger_args, "bucket_apply", "plan") == "plan"),
                copy_mode=(getattr(trigger_args, "bucket_apply", "plan") == "copy"),
                hash_on=getattr(trigger_args, "bucket_hash", False),
                quick_sig=getattr(trigger_args, "bucket_quick_sig", False),
                verify_existing_hash=False,
                zip_inventory=getattr(trigger_args, "bucket_zip_inventory", False),
                zip_hash_members=False,
                ocr_queue=getattr(trigger_args, "bucket_ocr_queue", False),
                content_scan=getattr(trigger_args, "bucket_content_scan", False),
                content_terms_file="",
                content_extra_terms=[],
                content_include_exts="",
                content_max_bytes=200000,
                content_max_terms=2500,
                plan_cleanup=False,
                apply_cleanup=False,
                halt_on_hard=False,
                apply_dir=cycle_dir / "BUCKET" / "APPLY",
                stoppers=stoppers,
                max_files=max_files,
            )

            adv_res = ase.scan_adversarial(
                roots=roots,
                out_dir=cycle_dir / "ADVERSARIAL",
                allow_exts=getattr(
                    trigger_args,
                    "adv_allow_exts",
                    ".txt,.md,.log,.json,.jsonl,.csv,.tsv,.html,.htm,.py,.ps1,.bat,.rtf,.docx,.pdf",
                ),
                max_files=10000,
                max_chars=2000000,
                stoppers=stoppers,
                bank=getattr(trigger_args, "adv_bank", 3),
            )

            merge_res = mp.run_merge_pipeline(
                records_jsonl=Path(bucket_res["catalog_jsonl"]),
                versions_summary_json=Path(bucket_res["versions_summary_json"]),
                duplicates_summary_json=Path(bucket_res["duplicates_summary_json"]),
                out_dir=cycle_dir / "MERGE",
                max_group_size=12,
                max_bytes_per_file=2500000,
                stoppers=stoppers,
                enable_text_merge=True,
                enable_table_merge=True,
                enable_code_merge=True,
            )

            _apply_res = ag.apply_gate_and_maybe_execute(
                bucket_apply=getattr(trigger_args, "bucket_apply", "plan"),
                apply_cleanup=False,
                planned_actions_jsonl=Path(bucket_res["plan_moves_jsonl"]),
                cleanup_plan_json=None,
                apply_token_path=cycle_dir / "APPLY_GATE" / "apply_signature.json",
                apply_summary_path=cycle_dir / "APPLY_GATE" / "apply_summary.md",
                confirm_token=getattr(trigger_args, "apply_confirm", ""),
                policy_allow_copy_without_confirm=True,
                stoppers=stoppers,
            )

            graph_res = gf.fuse_graph(
                run_id=run_id,
                out_dir=cycle_dir / "GRAPH",
                bucket_catalog_jsonl=Path(bucket_res["catalog_jsonl"]),
                bucket_plan_jsonl=Path(bucket_res["plan_moves_jsonl"]),
                zip_inventory_jsonl=Path(bucket_res["zip_inventory_jsonl"]) if bucket_res.get("zip_inventory_jsonl") else None,
                content_flags_jsonl=Path(bucket_res["content_flags_jsonl"]) if bucket_res.get("content_flags_jsonl") else None,
                adversarial_events_jsonl=Path(adv_res["events_jsonl"]),
                adversarial_hits_jsonl=Path(adv_res["hits_jsonl"]),
                merge_provenance_json=Path(merge_res["provenance_map_json"]),
                physics_mode=getattr(trigger_args, "graph_physics", "fast"),
                stoppers=stoppers,
            )

            write_json(run_dir / "RUN" / "stoppers_log.json", {"count": len(stoppers), "stoppers": stoppers})
            write_json(run_dir / "RUN" / "summary.json", {"bucket": bucket_res, "adversarial": adv_res, "merge": merge_res, "graph": graph_res})
            append_jsonl(events, {"ts": utc_now_iso(), "event": "TRIGGER_RUN_COMPLETE", "run_id": run_id, "stoppers": len(stoppers)})
'''

    return {
        "SUITE/hypervisor_master_orchestrator.py": orchestrator_src,
        "SUITE/engines/__init__.py": "from . import drive_bucket_forger\n",
        "SUITE/engines/drive_bucket_forger.py": bucket_src + bucket_cycle_src,
        "SUITE/engines/adversarial_signal_engine.py": adversarial_engine_src,
        "SUITE/engines/merge_pipeline.py": merge_pipeline_src,
        "SUITE/engines/apply_gate.py": apply_gate_src,
        "SUITE/engines/graph_fuser.py": graph_fuser_src,
        "SUITE/engines/watcher_daemon.py": watcher_daemon_src,
    }


def build_suite(
    bucket_source: Path,
    target: Path,
    tag: str,
    emit_zip: bool,
    verbose: bool,
) -> Path:
    context = make_context(tag)
    bucket_src = load_bucket_source(bucket_source)
    suite_files = suite_sources(bucket_src)
    suite_files["SUITE/README_RUNBOOK.md"] = build_runbook(context)

    ensure_dir(target)
    written: list[Path] = []
    for rel, content in suite_files.items():
        outp = target / rel
        write_text(outp, content)
        written.append(outp)
        if verbose:
            print(f"WROTE: {outp}")

    run_bat = "\r\n".join(
        [
            "@echo off",
            "setlocal",
            "cd /d %~dp0",
            "python SUITE\\hypervisor_master_orchestrator.py run --roots F:\\CAPSTONE\\Litigation_OS --out OUT --cycles 10 --stable-n 2 --eps 0 --bucket-apply plan --bucket-zip-inventory --bucket-content-scan --graph-physics fast",
            "endlocal",
            "",
        ]
    )
    watch_bat = "\r\n".join(
        [
            "@echo off",
            "setlocal",
            "cd /d %~dp0",
            "python SUITE\\hypervisor_master_orchestrator.py watch --roots F:\\CAPSTONE\\Litigation_OS --out OUT --poll-seconds 10 --debounce-seconds 15",
            "endlocal",
            "",
        ]
    )
    write_text(target / "RUN_MASTER.bat", run_bat)
    write_text(target / "WATCH_MASTER.bat", watch_bat)
    written.extend([target / "RUN_MASTER.bat", target / "WATCH_MASTER.bat"])

    manifest = {"files": [], "count": 0}
    for path in written:
        manifest["files"].append(file_receipt(path))
    manifest["count"] = len(manifest["files"])
    write_text(target / "manifest.json", json.dumps(manifest, indent=2))

    if emit_zip:
        zip_path = target / context.zip_name
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            for rel in suite_files.keys():
                path = target / rel
                zipf.write(path, arcname=rel)
            zipf.write(target / "RUN_MASTER.bat", arcname="RUN_MASTER.bat")
            zipf.write(target / "WATCH_MASTER.bat", arcname="WATCH_MASTER.bat")
            zipf.write(target / "manifest.json", arcname="manifest.json")
        if verbose:
            print(f"ZIP: {zip_path}")

    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Hypervisor Master Suite bundle.")
    parser.add_argument("--bucket-source", type=Path, default=DEFAULT_BUCKET_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    target = args.target.expanduser()
    emit_zip = not args.no_zip
    verbose = not args.quiet

    output = build_suite(
        bucket_source=args.bucket_source,
        target=target,
        tag=args.tag,
        emit_zip=emit_zip,
        verbose=verbose,
    )

    if verbose:
        print(f"BUILT: {output}")
        print(f"RUN: {output / 'RUN_MASTER.bat'}")
        print(f"WATCH: {output / 'WATCH_MASTER.bat'}")
        if emit_zip:
            stamp = make_context(args.tag).zip_name
            print(f"ZIP: {output / stamp}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
