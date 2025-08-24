"""Modular runner for Golden Litigator OS."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from agents import (
    generate_filings,
    generate_narrative,
    init_llm_from_config,
    run_analysis_agents,
)
from file_processors import dispatch_get_text
from utils import (
    ensure_dirs,
    excerpt,
    init_db,
    insert_evidence,
    insert_exhibit,
    insert_source,
    insert_timeline,
    load_config,
    register_code_file,
    safe_rename_done,
    sha256_file,
    is_excluded_dir,
)

try:
    from vector_memory import get_vm

    VM = get_vm()
except Exception:  # pragma: no cover
    VM = None

CODE_TYPES = {".py", ".ps1", ".psm1", ".json", ".txt"}


def crawl(cfg: Dict[str, Any]) -> None:
    drives: List[str] = list(cfg["drives"])
    excludes: set[str] = set(cfg["exclude_dirs"])
    db_path = cfg["db_path"]
    tag = cfg["processed_tag"]
    tess = cfg.get("tesseract_cmd", "")
    wb = cfg.get("whisper_backend", "faster-whisper")
    wm = cfg.get("whisper_model", "medium")

    for root_drive in drives:
        for root, dirs, files in os.walk(root_drive):
            if is_excluded_dir(root, excludes):
                continue
            for fname in files:
                path = Path(root) / fname
                if tag in path.stem or fname.startswith("~$"):
                    continue
                ext = path.suffix.lower()
                stat = path.stat()
                sha = sha256_file(path)

                if ext in CODE_TYPES:
                    register_code_file(db_path, path, sha)

                text, source_type = dispatch_get_text(
                    str(path),
                    tesseract_cmd=tess,
                    whisper_backend=wb,
                    whisper_model=wm,
                )

                insert_source(
                    db_path,
                    sha,
                    source_type,
                    {
                        "filename": path.name,
                        "filepath": str(path),
                        "size_bytes": stat.st_size,
                        "modified_ts": Path(path).stat().st_mtime,
                    },
                )

                llm = run_analysis_agents(text)
                parties = llm.get("parties", [])
                claims = llm.get("claims", [])
                statutes = llm.get("statutes", [])
                rules = llm.get("court_rules", [])
                timeline = llm.get("timeline", [])
                exhibits = llm.get("exhibits", [])

                rel = 0.0
                if claims:
                    rel += 1.0
                rel += 0.5 * min(3, len(statutes))
                rel += 0.5 * min(3, len(rules))

                insert_evidence(
                    db_path,
                    {
                        "sha256": sha,
                        "filename": path.name,
                        "filepath": str(path),
                        "ext": ext,
                        "size_bytes": stat.st_size,
                        "modified_ts": Path(path).stat().st_mtime,
                        "content_excerpt": excerpt(text, 1000),
                        "party": (
                            parties[0]["name"]
                            if parties and isinstance(parties[0], dict)
                            else ""
                        ),
                        "parties": parties,
                        "claims": claims,
                        "statutes": statutes,
                        "court_rules": rules,
                        "relevance_score": rel,
                        "timeline_refs": timeline,
                        "exhibit_tag": (
                            exhibits[0].get("label", "")
                            if exhibits and isinstance(exhibits[0], dict)
                            else ""
                        ),
                        "exhibit_label": (
                            exhibits[0].get("label", "")
                            if exhibits and isinstance(exhibits[0], dict)
                            else ""
                        ),
                    },
                )
                if VM is not None:
                    VM.upsert_doc(
                        doc_id=sha,
                        text=(text or "")[:8000],
                        meta={
                            "filename": path.name,
                            "filepath": str(path),
                            "ext": ext,
                            "claims": ",".join(
                                [
                                    c if isinstance(c, str) else c.get("name", "")
                                    for c in claims
                                ]
                            ),
                            "statutes": ",".join(
                                [
                                    s if isinstance(s, str) else s.get("cite", "")
                                    for s in statutes
                                ]
                            ),
                            "rules": ",".join(
                                [
                                    r if isinstance(r, str) else r.get("rule", "")
                                    for r in rules
                                ]
                            ),
                        },
                    )

                for ev in timeline:
                    insert_timeline(db_path, sha, ev)
                for ex in exhibits:
                    insert_exhibit(db_path, sha, ex)

                if rel >= 0.5:
                    safe_rename_done(path, tag)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config()
    init_llm_from_config(cfg)
    results_dir = Path(cfg["results_dir"])
    ensure_dirs(results_dir)
    init_db(cfg["db_path"])

    crawl(cfg)

    generate_narrative(cfg["db_path"], results_dir / "Narratives")
    generate_filings(
        cfg["db_path"],
        results_dir,
        motion_types=[
            "Motion to Set Aside / Stay Enforcement",
            "Motion for Sanctions (MCR 1.109(E)/2.114)",
            "Federal Complaint Draft (42 USC §1983/§1985 + IIED + Abuse of Process + Malicious Prosecution)",
        ],
    )


if __name__ == "__main__":
    main()
