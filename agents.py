"""LLM helpers and high-level operations."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from utils import db_conn

OPENAI_MODEL = "gpt-4o-mini"
ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"
PROVIDER_ORDER: List[str] = ["openai", "anthropic"]


def init_llm_from_config(cfg: Dict[str, Any]) -> None:
    global OPENAI_MODEL, ANTHROPIC_MODEL, PROVIDER_ORDER
    PROVIDER_ORDER = cfg.get("llm", {}).get("provider_order", PROVIDER_ORDER)
    OPENAI_MODEL = cfg.get("llm", {}).get("openai_model", OPENAI_MODEL)
    ANTHROPIC_MODEL = cfg.get("llm", {}).get("anthropic_model", ANTHROPIC_MODEL)


def call_llm(prompt: str) -> str:
    """Call configured LLM provider chain; return response text."""
    text = ""
    try:
        if "openai" in PROVIDER_ORDER and os.environ.get("OPENAI_API_KEY"):
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "Michigan litigation expert. Cite precisely; no speculation.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1400,
            )
            text = resp.choices[0].message.content or ""
            if text.strip():
                return text
    except Exception as exc:  # pragma: no cover
        logging.warning("OpenAI fail: %s", exc)

    try:
        if "anthropic" in PROVIDER_ORDER and os.environ.get("ANTHROPIC_API_KEY"):
            import anthropic  # type: ignore[import-not-found]

            c = anthropic.Anthropic()
            msg = c.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=1400,
                temperature=0.2,
                system="Michigan litigation expert. Cite precisely; no speculation.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = (
                "".join(blk.text for blk in msg.content if hasattr(blk, "text")) or ""
            )
    except Exception as exc:  # pragma: no cover
        logging.warning("Anthropic fail: %s", exc)
    return text


def run_analysis_agents(text: str) -> Dict[str, Any]:
    if not text.strip():
        return {
            "parties": [],
            "claims": [],
            "statutes": [],
            "court_rules": [],
            "timeline": [],
            "exhibits": [],
        }
    prompt = (
        "Extract STRICT JSON with keys: parties, claims, statutes, court_rules, "
        "timeline, exhibits. Use Michigan MCL/MCR and WDMI/FRCP if implicated.\n"
        f"CONTENT:\n{text[:10000]}"
    )
    raw = call_llm(prompt)
    try:
        data = json.loads(raw)
        return {
            "parties": data.get("parties", []),
            "claims": data.get("claims", []),
            "statutes": data.get("statutes", []),
            "court_rules": data.get("court_rules", []),
            "timeline": data.get("timeline", []),
            "exhibits": data.get("exhibits", []),
        }
    except Exception:
        return {
            "parties": [],
            "claims": [],
            "statutes": [],
            "court_rules": [],
            "timeline": [],
            "exhibits": [],
        }


def generate_narrative(db_path: str, out_dir: Path) -> None:
    conn = db_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        """SELECT filename, content_excerpt, parties_json, claims_json, timeline_refs_json
                   FROM evidence ORDER BY id ASC LIMIT 1000"""
    )
    rows = cur.fetchall()
    conn.close()
    bundle = []
    for r in rows:
        bundle.append(
            {
                "filename": r[0],
                "excerpt": r[1],
                "parties": json.loads(r[2] or "[]"),
                "claims": json.loads(r[3] or "[]"),
                "timeline": json.loads(r[4] or "[]"),
            }
        )
    narrative = call_llm(
        "Build a Michigan court-compliant, fact-only affidavit (numbered) from:\n"
        + json.dumps(bundle, ensure_ascii=False)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"Master_Narrative_{Path(db_path).stem}.txt"
    p.write_text(narrative, encoding="utf-8")


def generate_filings(db_path: str, results_dir: Path, motion_types: List[str]) -> None:
    from filings import make_motion_docx

    conn = db_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        """SELECT court_name, case_number, caption_plaintiff, caption_defendant, judge, jurisdiction, division
                   FROM case_meta ORDER BY id DESC LIMIT 1"""
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return
    case_meta = {
        "court_name": row[0] or "",
        "case_number": row[1] or "",
        "caption_plaintiff": row[2] or "",
        "caption_defendant": row[3] or "",
        "judge": row[4] or "",
        "jurisdiction": row[5] or "",
        "division": row[6] or "",
    }

    conn = db_conn(db_path)
    cur = conn.cursor()
    cur.execute(
        """SELECT filename, filepath, statutes_json, court_rules_json, claims_json, timeline_refs_json, content_excerpt
                   FROM evidence ORDER BY relevance_score DESC, id DESC LIMIT 200"""
    )
    rows = cur.fetchall()
    conn.close()
    pool = []
    for r in rows:
        pool.append(
            {
                "filename": r[0],
                "filepath": r[1],
                "statutes": json.loads(r[2] or "[]"),
                "rules": json.loads(r[3] or "[]"),
                "claims": json.loads(r[4] or "[]"),
                "timeline": json.loads(r[5] or "[]"),
                "excerpt": r[6] or "",
            }
        )
    mats = {"materials": pool}
    for m in motion_types:
        make_motion_docx(results_dir, m, case_meta, mats)
