#!/usr/bin/env python3
"""Benchbook / MCR Rules Plugin + LEXVAULT + Violation Graph Toolkit

modular script providing:

1. Benchbook / MCR rule hooks
   - BENCHBOOK_RULES and VIOLATION_RULES
   - register_benchbook_rules(engine)
   - register_violation_rules(vrules)

2. LEXVAULT Large-Input → Program Synthesizer (LM Studio Edition)
   - Build corpus from file/folder/zip
   - Call local LM Studio (OpenAI-compatible)
   - Materialize project files from LLM output markers

3. Violation Graph Tools
   - merge-violations: merge base nodes/edges CSVs with violations.json
   - decorate-violations: decorate violations.json with labels + viz hints
   - violation-wheel: build a standalone HTML "Violation Wheel" viewer

Usage (subcommands):

  # Default: LEXVAULT synthesizer
  python script.py --input INPUT --outdir OUTDIR [other args]

  # Merge violations into CSV graph
  python script.py merge-violations --nodes nodes.csv --edges edges.csv \
      --violations violations.json --outdir outdir

  # Decorate violations.json
  python script.py decorate-violations --in violations.json \
      --out violations_decorated.json --severity-filter critical,warning

  # Build Violation Wheel HTML
  python script.py violation-wheel --in violations_decorated.json \
      --out Violation_Wheel.html
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

# ==============================
# Benchbook / MCR Rules Helpers
# ==============================

RuleResult = Dict[str, str]

GRAPH_VIOLATION_NODE_TYPE = "BenchbookViolation"
GRAPH_VIOLATION_EDGE_TYPE = "benchbook_flags"


def _rule_ex_parte_parenting_time(text: str, meta: Dict[str, Any]) -> Optional[RuleResult]:
    lower = text.lower()
    if (
        "ex parte" in lower
        and "parenting time" in lower
        and ("suspend" in lower or "suspended" in lower or "terminate" in lower)
    ):
        return {
            "rule": "MCL 722.27a / MCR 3.207",
            "detail": (
                "Ex parte parenting-time suspension detected. "
                "Check for findings that parenting time would endanger the child "
                "and compliance with MCR 3.207 emergency/interim relief standards."
            ),
        }
    return None


def _rule_custody_change_without_bi(text: str, meta: Dict[str, Any]) -> Optional[RuleResult]:
    lower = text.lower()
    custody_keywords = (
        "change of custody",
        "modify custody",
        "sole legal custody",
        "sole physical custody",
    )
    if any(k in lower for k in custody_keywords) and "best interest" not in lower and "mcl 722.23" not in lower:
        return {
            "rule": "MCL 722.23 / Family Benchbook",
            "detail": (
                "Custody modification language detected without explicit best-interest factor discussion. "
                "Verify that the order or transcript contains findings on all applicable MCL 722.23 factors."
            ),
        }
    return None


def _rule_contempt_incarceration_without_ability(text: str, meta: Dict[str, Any]) -> Optional[RuleResult]:
    lower = text.lower()
    if (
        "show cause" in lower
        and ("jail" in lower or "incarceration" in lower)
        and "ability to pay" not in lower
        and "ability to comply" not in lower
    ):
        return {
            "rule": "MCR 3.606 / Contempt Benchbook",
            "detail": (
                "Contempt with incarceration language detected but no explicit reference to ability to pay/comply. "
                "Verify that the record contains findings on willfulness and present ability to comply."
            ),
        }
    return None


def _rule_ppo_custody_interaction(text: str, meta: Dict[str, Any]) -> Optional[RuleResult]:
    lower = text.lower()
    if ("personal protection order" in lower or "ppo" in lower) and (
        "custody" in lower or "parenting time" in lower or "parenting-time" in lower
    ):
        return {
            "rule": "MCL 600.2950 / 600.2950a / PPO Benchbook",
            "detail": (
                "PPO language appears intertwined with custody/parenting-time issues. "
                "Check that PPO relief is not being used as a substitute for custody modification "
                "without compliance with MCL 722.23 and 722.27a."
            ),
        }
    return None


def _rule_small_pdf_evidence(text: str, meta: Dict[str, Any]) -> Optional[RuleResult]:
    if meta.get("suffix") == ".pdf" and meta.get("size", 0) < 1024:
        return {
            "rule": "Evidence Benchbook / MRE 401–403",
            "detail": (
                "Very small PDF detected. Confirm that the exhibit is complete, legible, and properly scanned "
                "before relying on it for findings."
            ),
        }
    return None


BENCHBOOK_RULES: List[Callable[[str, Dict[str, Any]], Optional[RuleResult]]] = [
    _rule_ex_parte_parenting_time,
    _rule_custody_change_without_bi,
    _rule_ppo_custody_interaction,
    _rule_small_pdf_evidence,
]

VIOLATION_RULES: List[Callable[[str, Dict[str, Any]], Optional[RuleResult]]] = [
    _rule_ex_parte_parenting_time,
    _rule_custody_change_without_bi,
    _rule_contempt_incarceration_without_ability,
    _rule_ppo_custody_interaction,
]


def register_benchbook_rules(engine: Any) -> None:
    """Monkey-patch BenchbookRulesEngine.apply_rules to add extra rules."""

    import types

    original_apply = getattr(engine, "apply_rules", None)
    if original_apply is None:
        return

    def _extra(text: str, meta: Dict[str, Any]) -> List[RuleResult]:
        out: List[RuleResult] = []
        for fn in BENCHBOOK_RULES:
            try:
                res = fn(text, meta)
                if res is not None:
                    out.append(res)
            except Exception:
                continue
        return out

    def new_apply(self: Any, text: str, meta: Dict[str, Any]) -> List[RuleResult]:
        base = original_apply(text, meta)
        extra = _extra(text, meta)
        return list(base) + extra

    engine.apply_rules = types.MethodType(new_apply, engine)


def register_violation_rules(vrules: Any) -> None:
    """Monkey-patch ViolationRules.analyze to add Benchbook-aware checks."""

    import types

    original_analyze = getattr(vrules, "analyze", None)
    if original_analyze is None:
        return

    def _extra(text: str, meta: Dict[str, Any]) -> List[RuleResult]:
        out: List[RuleResult] = []
        for fn in VIOLATION_RULES:
            try:
                res = fn(text, meta)
                if res is not None:
                    out.append(res)
            except Exception:
                continue
        return out

    def new_analyze(self: Any, text: str, meta: Dict[str, Any]) -> List[RuleResult]:
        base = original_analyze(text, meta)
        extra = _extra(text, meta)
        return list(base) + extra

    vrules.analyze = types.MethodType(new_analyze, vrules)


# ===============================
# LEXVAULT: Large Input → Project
# ===============================

ALLOWED_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".csv",
    ".json",
    ".log",
    ".ini",
    ".cfg",
    ".html",
    ".htm",
    ".docx",
    ".pdf",
}

FILE_START_PREFIX = "<<FILE "
FILE_END_MARK = "<<END FILE>>"


@dataclass
class LMClientConfig:
    base_url: str
    model: Optional[str]
    temperature: float
    max_output_tokens: int


def setup_logging(outdir: Path) -> None:
    """Configure logging to stdout and a file in outdir."""

    outdir.mkdir(parents=True, exist_ok=True)
    log_file = outdir / "lexvault_build.log"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers = []

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
    logger.addHandler(fh)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logging.warning("Failed to read %s: %s", path, e)
        return ""


def _handle_zip_input(zip_path: Path, work_root: Path) -> Path:
    target_dir = work_root / "_unzipped_input"
    target_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Extracting ZIP %s → %s", zip_path, target_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    return target_dir


def _iter_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            p = Path(dirpath) / name
            if p.suffix.lower() in ALLOWED_EXTENSIONS:
                files.append(p)
    files.sort()
    return files


def build_corpus(input_path: Path, work_root: Path) -> Tuple[str, List[Path]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() == ".zip":
            scan_root = _handle_zip_input(input_path, work_root)
        else:
            logging.info("Building corpus from single file: %s", input_path)
            content = _read_text(input_path)
            header = f"=== BEGIN FILE: {input_path.name} ===\n"
            footer = f"\n=== END FILE: {input_path.name} ===\n"
            return header + content + footer, [input_path]
    else:
        scan_root = input_path

    logging.info("Scanning directory for allowed files: %s", scan_root)
    files = _iter_files(scan_root)
    if not files:
        raise RuntimeError(f"No allowed files found under: {scan_root}")

    pieces: List[str] = []
    for f in files:
        rel = f.relative_to(scan_root)
        header = f"=== BEGIN FILE: {rel.as_posix()} ===\n"
        footer = f"\n=== END FILE: {rel.as_posix()} ===\n"
        pieces.append(header + _read_text(f) + footer)

    corpus = "\n\n".join(pieces)
    logging.info("Corpus built from %d files, length=%d chars", len(files), len(corpus))
    return corpus, files


class LMStudioClient:
    """Thin client for LM Studio's /v1/chat/completions endpoint."""

    def __init__(self, cfg: LMClientConfig) -> None:
        self.base_url = cfg.base_url.rstrip("/")
        self.temperature = cfg.temperature
        self.max_output_tokens = cfg.max_output_tokens
        if cfg.model:
            self.model = cfg.model
            logging.info("Using configured model: %s", self.model)
        else:
            self.model = self._autodetect_model()
            logging.info("Auto-detected LM Studio model: %s", self.model)

    def _autodetect_model(self) -> str:
        url = f"{self.base_url}/models"
        logging.info("Autodetecting LM Studio model from: %s", url)
        try:
            resp = requests.get(url, timeout=15)
        except Exception as e:
            raise RuntimeError(f"Failed to query LM Studio models at {url}: {e}") from e
        if resp.status_code != 200:
            raise RuntimeError(f"LM Studio /models HTTP {resp.status_code}: {resp.text}")
        try:
            data = resp.json()
        except ValueError as e:
            raise RuntimeError(f"Invalid JSON from /models: {e}") from e
        models = data.get("data") or data.get("models") or []
        if not models:
            raise RuntimeError("No models reported by LM Studio /models endpoint.")
        first = models[0]
        if isinstance(first, dict) and first.get("id"):
            return str(first["id"])
        if isinstance(first, dict):
            for key in ("name", "model", "engine"):
                if first.get(key):
                    return str(first[key])
        raise RuntimeError("Could not determine model id from /models response.")

    def chat(self, messages: List[Dict[str, str]]) -> str:
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
        }
        logging.info("Calling LM Studio at %s", url)
        try:
            resp = requests.post(url, json=payload, timeout=600)
        except Exception as e:
            raise RuntimeError(f"Error calling LM Studio /chat/completions: {e}") from e
        if resp.status_code != 200:
            raise RuntimeError(f"LM Studio /chat/completions HTTP {resp.status_code}: {resp.text}")
        try:
            data = resp.json()
        except ValueError as e:
            raise RuntimeError(f"Invalid JSON from /chat/completions: {e}") from e
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("LM Studio returned no choices in /chat/completions.")
        message = choices[0].get("message") or {}
        content = message.get("content") or ""
        text = str(content)
        logging.info("LLM response received, length=%d chars", len(text))
        return text


def build_messages(corpus: str, project_name: str) -> List[Dict[str, str]]:
    system_prompt = textwrap.dedent(
        f"""
        You are LEXVAULT, an expert software architect and senior Python developer.

        Task:
        - Read the provided corpus of documents. They may include specifications,
          notes, code snippets, legal artifacts, or transcripts.
        - Design and emit a complete, working Python-based project that best
          operationalizes the intent of the corpus.

        Output format:
        - You MUST emit project files using the following exact markers:

            <<FILE relative/path/to/file.ext>>
            ... entire file contents ...
            <<END FILE>>

        - Use forward slashes in paths.
        - Do not include any commentary outside of FILE blocks, unless necessary
          at the very top in a short summary (< 20 lines). If you provide such a
          summary, keep it before the first <<FILE ...>> marker.

        Project constraints:
        - The project should be centered around this project name:
              {project_name}
        - Use Python 3.10+ compatible code.
        - Include a main entry point (e.g., cli.py or main.py) with clear argparse
          or similar interface.
        - Add logging and error handling throughout.
        - Structure the project as installable (package directory with __init__.py,
          clear module boundaries).
        - If the corpus references a "Litigation OS" or similar, design the code
          so that any external systems are accessed through a minimal service
          abstraction that can later be wired to a real implementation.

        Markers:
        - Do NOT nest FILE blocks.
        - Do NOT truncate file contents.
        """
    ).strip()

    user_prompt = (
        "Below is the aggregated corpus from the input files. "
        "Use it as the sole specification source for the project you are about to generate.\n\n"
        "=== START CORPUS ===\n"
        f"{corpus}\n"
        "=== END CORPUS ===\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def materialize_project_from_llm(text: str, project_root: Path) -> List[str]:
    project_root.mkdir(parents=True, exist_ok=True)
    created_paths: List[str] = []

    lines = text.splitlines()
    i = 0
    n = len(lines)

    def write_file(rel_path: str, contents: str) -> None:
        dest = project_root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(contents, encoding="utf-8")
        created_paths.append(rel_path)
        logging.info("Created file from LLM: %s", rel_path)

    while i < n:
        line = lines[i]
        if line.startswith(FILE_START_PREFIX) and line.strip().endswith(">"):
            inner = line.strip()[len(FILE_START_PREFIX) : -2].strip()
            rel_path = inner
            i += 1
            buf: List[str] = []
            while i < n and lines[i].strip() != FILE_END_MARK:
                buf.append(lines[i])
                i += 1
            if i < n:
                i += 1
            contents = "\n".join(buf).rstrip() + "\n"
            write_file(rel_path, contents)
        else:
            i += 1

    if not created_paths:
        logging.warning("No FILE markers found; writing the entire LLM output to PROJECT_PLAN_AND_CODE.txt")
        fallback = "PROJECT_PLAN_AND_CODE.txt"
        project_root.joinpath(fallback).write_text(text, encoding="utf-8")
        created_paths.append(fallback)

    return created_paths


def inject_litigation_service(project_root: Path, lit_api_path: Optional[Path]) -> None:
    (project_root / "__init__.py").touch()

    if lit_api_path is not None:
        if not lit_api_path.is_file():
            raise FileNotFoundError(f"litigation-api file not found: {lit_api_path}")
        target_api = project_root / "litigation_os_api.py"
        logging.info("Copying litigation_api %s → %s", lit_api_path, target_api)
        target_api.write_text(lit_api_path.read_text(encoding="utf-8"), encoding="utf-8")

    services_dir = project_root / "services"
    services_dir.mkdir(parents=True, exist_ok=True)
    (services_dir / "__init__.py").touch()
    service_path = services_dir / "litigation_service.py"

    service_code = (
        textwrap.dedent(
            '''
        """Lightweight facade into a litigation_os_api module."""

        from typing import Any, Dict

        try:
            import litigation_os_api as _lit_api
        except ImportError:  # pragma: no cover
            _lit_api = None

        class LitigationService:
            def __init__(self) -> None:
                if _lit_api is None:
                    raise RuntimeError(
                        "litigation_os_api is not available. Ensure it is on the PYTHONPATH "
                        "or copied into the project root."
                    )

            def list_capabilities(self) -> Dict[str, Any]:
                caps: Dict[str, Any] = {}
                for name in dir(_lit_api):
                    if name.startswith("_"):
                        continue
                    attr = getattr(_lit_api, name)
                    if callable(attr):
                        caps[name] = "callable"
                    else:
                        caps[name] = "value"
                return caps

            def has_function(self, name: str) -> bool:
                return hasattr(_lit_api, name) and callable(getattr(_lit_api, name))

            def run_task(self, task_name: str, **kwargs: Any) -> Any:
                if not hasattr(_lit_api, task_name):
                    raise AttributeError(
                        f"litigation_os_api has no attribute {task_name!r}"
                    )
                fn = getattr(_lit_api, task_name)
                if not callable(fn):
                    raise TypeError(
                        f"litigation_os_api.{task_name} is not callable"
                    )
                return fn(**kwargs)
        '''
        ).strip()
        + "\n"
    )

    service_path.write_text(service_code, encoding="utf-8")
    logging.info("Injected litigation_service.py into %s", service_path.relative_to(project_root))


def parse_lexvault_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LEXVAULT Large-Input → Program Synthesizer (LM Studio Edition)")
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input file, folder, or .zip to build the corpus from.",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory where the generated project will be created.",
    )
    parser.add_argument(
        "--project-name",
        default="lexvault_project",
        help="Logical project name used in prompts and folder naming.",
    )
    parser.add_argument(
        "--lm-url",
        default="http://127.0.0.1:1234/v1",
        help="Base LM Studio URL (OpenAI-compatible). Default: %(default)s",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model id to use. If omitted, the script attempts to autodetect.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for LM Studio. Default: %(default)s",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help=("Maximum tokens for the primary /chat/completions call. " "Default: %(default)s"),
    )
    parser.add_argument(
        "--litigation-api",
        default=None,
        help=(
            "Optional path to litigation_os_api.py. If provided, this file is copied "
            "into the project and a services/litigation_service.py facade is generated."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Build the corpus and prompt, but do not call the LLM. " "Writes DRY_RUN_PROMPT.txt into the outdir."),
    )
    return parser.parse_args(argv)


def main_lexvault(argv: Optional[List[str]] = None) -> None:
    args = parse_lexvault_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    project_name = args.project_name

    outdir.mkdir(parents=True, exist_ok=True)
    setup_logging(outdir)

    logging.info("LEXVAULT starting")
    logging.info("Input  : %s", input_path)
    logging.info("Outdir : %s", outdir)
    logging.info("Project: %s", project_name)

    work_root = outdir / "_lexvault_work"
    work_root.mkdir(parents=True, exist_ok=True)

    corpus, files = build_corpus(input_path, work_root)
    messages = build_messages(corpus, project_name)

    if args.dry_run:
        prompt_path = outdir / "DRY_RUN_PROMPT.txt"
        payload = {
            "messages": messages,
            "approx_corpus_length": len(corpus),
            "file_count": len(files),
        }
        prompt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logging.info("Dry run complete; prompt written to %s", prompt_path)
        return

    lm_cfg = LMClientConfig(
        base_url=args.lm_url,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
    )
    client = LMStudioClient(lm_cfg)

    response_text = client.chat(messages)

    project_root = outdir / project_name
    created_files = materialize_project_from_llm(response_text, project_root)

    lit_api_path = Path(args.litigation_api).expanduser().resolve() if args.litigation_api else None
    if args.litigation_api is not None:
        inject_litigation_service(project_root, lit_api_path)

    manifest = {
        "input_path": str(input_path),
        "outdir": str(outdir),
        "project_name": project_name,
        "lm_url": args.lm_url,
        "model": args.model,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "file_count": len(files),
        "created_files": created_files,
    }
    manifest_path = outdir / "lexvault_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logging.info("Manifest written to %s", manifest_path)

    logging.info("LEXVAULT completed successfully.")


# =================================
# Violation Graph Merge (CSV + JSON)
# =================================


def parse_violation_merge_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge base nodes/edges CSVs with a violations.json graph, "
            "producing merged nodes/edges CSVs with violation nodes/edges added."
        )
    )
    parser.add_argument("--nodes", required=True, help="Path to base nodes.csv")
    parser.add_argument("--edges", required=True, help="Path to base edges.csv")
    parser.add_argument(
        "--violations",
        required=True,
        help="Path to violations.json (from rule_results_to_graph).",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for merged_nodes.csv and merged_edges.csv",
    )
    return parser.parse_args(argv)


def load_base_graph_csv(nodes_path: Path, edges_path: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    nodes: List[Dict[str, str]] = []
    edges: List[Dict[str, str]] = []

    with nodes_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes.append(dict(row))

    with edges_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append(dict(row))

    return nodes, edges


def load_violations_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_node_schema_fields(rows: List[Dict[str, str]]) -> Dict[str, str]:
    """Infer id/label/type field names from an existing nodes.csv sample.

    Tuned for your graphs, preferring:
    - ':ID' or 'id_str' (Neo4j-style imports),
    - 'id' / 'label' for authorities/MCR nodes,
    - sane defaults otherwise.
    """

    if not rows:
        return {"id_field": "id", "label_field": "label", "type_field": "type"}

    keys = set(rows[0].keys())

    def pick(candidates: List[str], default: str) -> str:
        for c in candidates:
            if c in keys:
                return c
        return default

    id_field = pick([":ID", "id_str", "id", "node_id", "Id", "ID"], "id")
    label_field = pick(["label", ":LABEL", "name", "title", "text"], "label")
    type_field = pick(["type", "node_type", "category"], "type")

    return {
        "id_field": id_field,
        "label_field": label_field,
        "type_field": type_field,
    }


def infer_edge_schema_fields(rows: List[Dict[str, str]]) -> Dict[str, str]:
    if not rows:
        return {
            "source_field": "source",
            "target_field": "target",
            "type_field": "type",
        }

    keys = set(rows[0].keys())

    def pick(candidates: List[str], default: str) -> str:
        for c in candidates:
            if c in keys:
                return c
        return default

    source_field = pick(["source", "from", "src", "start"], "source")
    target_field = pick(["target", "to", "dst", "end"], "target")
    type_field = pick(["type", "edge_type", "relation", "rel"], "type")

    return {
        "source_field": source_field,
        "target_field": target_field,
        "type_field": type_field,
    }


def decorate_violation_nodes_for_viz(nodes: List[Dict[str, Any]]) -> None:
    """Add label and viz_* fields to violation nodes for nice visualization."""

    for node in nodes:
        if node.get("type") != GRAPH_VIOLATION_NODE_TYPE:
            continue

        if not node.get("label"):
            rule = str(node.get("rule") or "Violation")
            category = str(node.get("category") or "").strip()
            citations = node.get("citations") or []
            if isinstance(citations, (list, tuple)):
                cite_str = ", ".join(str(c) for c in citations)
            else:
                cite_str = str(citations)

            if category == "parenting_time":
                base_label = "Parenting-time issue"
            elif category == "custody":
                base_label = "Custody issue"
            elif category == "contempt":
                base_label = "Contempt / jail issue"
            elif category == "ppo_sequence":
                base_label = "PPO show-cause sequence"
            elif category == "foc":
                base_label = "FOC delegation issue"
            else:
                base_label = "Benchbook rule flag"

            if cite_str:
                label = f"{base_label} ({cite_str})"
            else:
                label = f"{base_label} ({rule})"
            node["label"] = label

        sev = str(node.get("severity") or "").lower()
        if sev == "critical":
            color, size = "#ff0000", 16
        elif sev == "warning":
            color, size = "#ff9900", 12
        elif sev == "info":
            color, size = "#0099ff", 10
        else:
            color, size = "#999999", 8

        node.setdefault("viz_color", color)
        node.setdefault("viz_size", size)


def merge_base_graph_with_violations(
    base_nodes: List[Dict[str, str]],
    base_edges: List[Dict[str, str]],
    violation_graph: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Merge base graph CSVs with violation graph, schema-aware."""

    merged_nodes: List[Dict[str, Any]] = [dict(n) for n in base_nodes]
    merged_edges: List[Dict[str, Any]] = [dict(e) for e in base_edges]

    node_schema = infer_node_schema_fields(base_nodes)
    edge_schema = infer_edge_schema_fields(base_edges)

    id_field = node_schema["id_field"]
    label_field = node_schema["label_field"]
    type_field = node_schema["type_field"]

    source_field = edge_schema["source_field"]
    target_field = edge_schema["target_field"]
    edge_type_field = edge_schema["type_field"]

    violation_nodes = [dict(node) for node in violation_graph.get("nodes", [])]
    decorate_violation_nodes_for_viz(violation_nodes)

    for node in violation_nodes:
        adapted = dict(node)

        if id_field and id_field not in adapted:
            if "id" in adapted:
                adapted[id_field] = str(adapted["id"])

        if label_field and label_field not in adapted:
            lbl = adapted.get("label") or adapted.get("rule") or adapted.get("detail") or "Benchbook violation"
            adapted[label_field] = str(lbl)

        if type_field and type_field not in adapted:
            adapted[type_field] = GRAPH_VIOLATION_NODE_TYPE

        merged_nodes.append(adapted)

    violation_edges = [dict(edge) for edge in violation_graph.get("edges", [])]

    for edge in violation_edges:
        adapted = dict(edge)

        if source_field and source_field not in adapted and "source" in adapted:
            adapted[source_field] = str(adapted["source"])
        if target_field and target_field not in adapted and "target" in adapted:
            adapted[target_field] = str(adapted["target"])

        if edge_type_field and edge_type_field not in adapted:
            adapted[edge_type_field] = adapted.get("type", GRAPH_VIOLATION_EDGE_TYPE)

        merged_edges.append(adapted)

    return merged_nodes, merged_edges


def write_merged_csvs(
    merged_nodes: List[Dict[str, Any]],
    merged_edges: List[Dict[str, Any]],
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    if merged_nodes:
        node_fieldnames = list(merged_nodes[0].keys())
        with (outdir / "merged_nodes.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=node_fieldnames)
            writer.writeheader()
            for row in merged_nodes:
                writer.writerow(row)

    if merged_edges:
        edge_fieldnames = list(merged_edges[0].keys())
        with (outdir / "merged_edges.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=edge_fieldnames)
            writer.writeheader()
            for row in merged_edges:
                writer.writerow(row)


def main_merge_violations(argv: Optional[List[str]] = None) -> None:
    args = parse_violation_merge_args(argv)

    nodes_path = Path(args.nodes).expanduser().resolve()
    edges_path = Path(args.edges).expanduser().resolve()
    violations_path = Path(args.violations).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()

    setup_logging(outdir)
    logging.info("Merging base graph with violations")
    logging.info("Nodes      : %s", nodes_path)
    logging.info("Edges      : %s", edges_path)
    logging.info("Violations : %s", violations_path)
    logging.info("Outdir     : %s", outdir)

    base_nodes, base_edges = load_base_graph_csv(nodes_path, edges_path)
    violation_graph = load_violations_json(violations_path)

    merged_nodes, merged_edges = merge_base_graph_with_violations(base_nodes, base_edges, violation_graph)

    write_merged_csvs(merged_nodes, merged_edges, outdir)
    logging.info("Merge complete; merged_nodes.csv and merged_edges.csv written to %s", outdir)


# -------------------------------
# Standalone Violations Decorator
# -------------------------------


def parse_decorate_violations_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decorate a violations.json graph with human-readable labels and "
            "viz_* hints (color/size and width) for graph viewers, with an optional severity filter."
        )
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help=("Input violations.json file (from rule_results_to_graph / " "merge_violation_graph_collections)."),
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output path for decorated violations JSON.",
    )
    parser.add_argument(
        "--severity-filter",
        dest="severity_filter",
        default=None,
        help=(
            "Comma separated list of severities to keep, for example: "
            "critical,warning. If omitted, all severities are kept."
        ),
    )
    return parser.parse_args(argv)


def decorate_violation_graph_json(
    path_in: Path,
    path_out: Path,
    severity_filter: Optional[List[str]] = None,
) -> None:
    """Load a violations.json file, decorate nodes and edges, and write a new JSON.

    Features:
    - Adds human readable labels and viz_* hints to violation nodes.
    - Derives viz_width and viz_color for edges based on severity.
    - Optional severity_filter controls which severities are kept (nodes and edges).
    """

    if not path_in.is_file():
        raise FileNotFoundError(f"violations.json not found: {path_in}")

    data = json.loads(path_in.read_text(encoding="utf-8"))
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []

    allowed_severities: Optional[set] = None
    if severity_filter:
        allowed_severities = {s.strip().lower() for s in severity_filter if s and s.strip()}
        if not allowed_severities:
            allowed_severities = None

    severity_rank = {"critical": 3, "warning": 2, "info": 1}

    node_severity: Dict[str, str] = {}

    for node in nodes:
        if node.get("type") != GRAPH_VIOLATION_NODE_TYPE:
            sev_raw = str(node.get("severity") or "").lower()
            node_id = (
                str(node.get("id")) if node.get("id") is not None else str(node.get(":ID") or node.get("id_str") or "")
            )
            if node_id:
                node_severity[node_id] = sev_raw
            continue

        if not node.get("label"):
            rule = str(node.get("rule") or "Violation")
            category = str(node.get("category") or "").strip()
            citations = node.get("citations") or []
            if isinstance(citations, (list, tuple)):
                cite_str = ", ".join(str(c) for c in citations)
            else:
                cite_str = str(citations)

            if category == "parenting_time":
                base_label = "Parenting-time issue"
            elif category == "custody":
                base_label = "Custody issue"
            elif category == "contempt":
                base_label = "Contempt or jail issue"
            elif category == "ppo_sequence":
                base_label = "PPO show-cause sequence"
            elif category == "foc":
                base_label = "FOC delegation issue"
            else:
                base_label = "Benchbook rule flag"

            if cite_str:
                label = f"{base_label} ({cite_str})"
            else:
                label = f"{base_label} ({rule})"
            node["label"] = label

        sev = str(node.get("severity") or "").lower()
        if sev == "critical":
            color, size = "#ff0000", 16
        elif sev == "warning":
            color, size = "#ff9900", 12
        elif sev == "info":
            color, size = "#0099ff", 10
        else:
            color, size = "#999999", 8

        node.setdefault("viz_color", color)
        node.setdefault("viz_size", size)

        node_id = (
            str(node.get("id")) if node.get("id") is not None else str(node.get(":ID") or node.get("id_str") or "")
        )
        if node_id:
            node_severity[node_id] = sev

    if allowed_severities is not None:
        filtered_nodes: List[Dict[str, Any]] = []
        kept_ids: set = set()
        for node in nodes:
            sev = str(node.get("severity") or "").lower()
            if not sev:
                continue
            if sev in allowed_severities:
                filtered_nodes.append(node)
                node_id = (
                    str(node.get("id"))
                    if node.get("id") is not None
                    else str(node.get(":ID") or node.get("id_str") or "")
                )
                if node_id:
                    kept_ids.add(node_id)
        nodes = filtered_nodes
    else:
        kept_ids = {
            str(node.get("id")) if node.get("id") is not None else str(node.get(":ID") or node.get("id_str") or "")
            for node in nodes
            if node.get("id") is not None or node.get(":ID") is not None or node.get("id_str") is not None
        }

    decorated_edges: List[Dict[str, Any]] = []
    for edge in edges:
        src = str(edge.get("source")) or str(edge.get("src") or edge.get("from") or "")
        tgt = str(edge.get("target")) or str(edge.get("dst") or edge.get("to") or "")

        if kept_ids and (src not in kept_ids and tgt not in kept_ids):
            continue

        edge_sev = str(edge.get("severity") or "").lower()
        if not edge_sev:
            src_sev = node_severity.get(src, "")
            tgt_sev = node_severity.get(tgt, "")

            def rank(sev: str) -> int:
                return severity_rank.get(sev, 0)

            edge_sev = src_sev if rank(src_sev) >= rank(tgt_sev) else tgt_sev

        if edge_sev == "critical":
            e_color, e_width = "#ff0000", 3.0
        elif edge_sev == "warning":
            e_color, e_width = "#ff9900", 2.0
        elif edge_sev == "info":
            e_color, e_width = "#0099ff", 1.5
        else:
            e_color, e_width = "#cccccc", 1.0

        edge.setdefault("viz_color", e_color)
        edge.setdefault("viz_width", e_width)

        decorated_edges.append(edge)

    data["nodes"] = nodes
    data["edges"] = decorated_edges

    path_out.parent.mkdir(parents=True, exist_ok=True)
    path_out.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ------------------------------
# Violation Wheel HTML Generator
# ------------------------------


def parse_violation_wheel_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a standalone HTML 'Violation Wheel' viewer from a violations.json "
            "graph, embedding the data directly so it can be opened from disk."
        )
    )
    parser.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input violations.json file (decorated or raw).",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output HTML file for the violation wheel viewer.",
    )
    return parser.parse_args(argv)


def build_violation_wheel_html(violations_path: Path, html_out_path: Path) -> None:
    """Generate a self-contained HTML violation wheel from violations.json.

    - Embeds the JSON directly into the page (no fetch / CORS issues).
    - Uses a force-directed layout via D3.js (CDN hosted).
    - Honors viz_color / viz_size / viz_width where present.
    """

    if not violations_path.is_file():
        raise FileNotFoundError(f"violations.json not found: {violations_path}")

    data = json.loads(violations_path.read_text(encoding="utf-8"))

    nodes = data.get("nodes") or []
    edges = data.get("edges") or data.get("links") or []
    payload = {"nodes": nodes, "edges": edges}

    json_str = json.dumps(payload)

    html = f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>Violation Wheel Viewer</title>
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <style>
      html, body {{ margin: 0; padding: 0; width: 100%; height: 100%; background: #05060a; color: #f5f5f5; font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
      #app {{ width: 100%; height: 100%; display: flex; flex-direction: column; }}
      header {{ padding: 8px 16px; background: #10121a; border-bottom: 1px solid #262a3a; display: flex; align-items: center; justify-content: space-between; }}
      header h1 {{ font-size: 16px; margin: 0; font-weight: 600; }}
      header .meta {{ font-size: 11px; opacity: 0.8; }}
      #graph {{ flex: 1; }}
      svg {{ width: 100%; height: 100%; display: block; }}
      .node circle {{ cursor: pointer; }}
      .node text {{ pointer-events: none; font-size: 10px; fill: #f5f5f5; text-shadow: 0 0 2px #000; }}
      .edge {{ stroke-opacity: 0.8; }}
      .hud {{ position: absolute; left: 12px; bottom: 12px; padding: 6px 10px; background: rgba(5, 6, 10, 0.85); border-radius: 8px; border: 1px solid #303548; font-size: 11px; max-width: 360px; }}
      .hud strong {{ color: #ffffff; }}
      .hud .severity-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }}
    </style>
  </head>
  <body>
    <div id=\"app\">
      <header>
        <h1>Violation Wheel</h1>
        <div class=\"meta\" id=\"meta-counts\"></div>
      </header>
      <div id=\"graph\"></div>
      <div class=\"hud\" id=\"hud\">Click a node to see details.</div>
    </div>

    <script src=\"https://d3js.org/d3.v7.min.js\"></script>
    <script>
      const graph = {json_str};

      const width = window.innerWidth;
      const height = window.innerHeight - 40;

      const svg = d3.select('#graph').append('svg')
        .attr('width', width)
        .attr('height', height);

      const hud = document.getElementById('hud');
      const metaEl = document.getElementById('meta-counts');

      function nodeId(n) {{
        return n.id ?? n[':ID'] ?? n.id_str ?? String(n.id ?? '');
      }}

      function nodeSeverity(n) {{
        const sev = (n.severity || '').toLowerCase();
        return sev || '';
      }}

      function nodeColor(n) {{
        if (n.viz_color) return n.viz_color;
        const sev = nodeSeverity(n);
        if (sev === 'critical') return '#ff0000';
        if (sev === 'warning') return '#ff9900';
        if (sev === 'info') return '#0099ff';
        return '#999999';
      }}

      function nodeSize(n) {{
        if (n.viz_size) return n.viz_size;
        const sev = nodeSeverity(n);
        if (sev === 'critical') return 16;
        if (sev === 'warning') return 12;
        if (sev === 'info') return 10;
        return 8;
      }}

      function edgeColor(e) {{
        if (e.viz_color) return e.viz_color;
        const sev = (e.severity || '').toLowerCase();
        if (sev === 'critical') return '#ff0000';
        if (sev === 'warning') return '#ff9900';
        if (sev === 'info') return '#0099ff';
        return '#666666';
      }}

      function edgeWidth(e) {{
        if (e.viz_width) return e.viz_width;
        const sev = (e.severity || '').toLowerCase();
        if (sev === 'critical') return 3.0;
        if (sev === 'warning') return 2.0;
        if (sev === 'info') return 1.5;
        return 1.0;
      }}

      const nodes = graph.nodes.map((n, idx) => Object.assign({{ index: idx }}, n));
      const nodeById = new Map(nodes.map(n => [nodeId(n), n]));

      const edges = (graph.edges || []).map(e => {{
        const srcId = e.source ?? e.src ?? e.from;
        const tgtId = e.target ?? e.dst ?? e.to;
        const src = nodeById.get(String(srcId));
        const tgt = nodeById.get(String(tgtId));
        return Object.assign({{ source: src, target: tgt }}, e);
      }}).filter(e => e.source && e.target);

      const sim = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(edges).id(nodeId).distance(80).strength(0.4))
        .force('charge', d3.forceManyBody().strength(-180))
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide().radius(d => nodeSize(d) * 1.6));

      const link = svg.append('g')
        .attr('stroke-linecap', 'round')
        .selectAll('line')
        .data(edges)
        .join('line')
        .attr('class', 'edge')
        .attr('stroke', edgeColor)
        .attr('stroke-width', edgeWidth);

      const node = svg.append('g')
        .selectAll('g')
        .data(nodes)
        .join('g')
        .attr('class', 'node')
        .call(d3.drag()
          .on('start', (event, d) => {{
            if (!event.active) sim.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
          }})
          .on('drag', (event, d) => {{
            d.fx = event.x;
            d.fy = event.y;
          }})
          .on('end', (event, d) => {{
            if (!event.active) sim.alphaTarget(0);
            d.fx = null;
            d.fy = null;
          }}));

      node.append('circle')
        .attr('r', d => nodeSize(d))
        .attr('fill', d => nodeColor(d));

      node.append('text')
        .attr('x', 10)
        .attr('y', 3)
        .text(d => d.label || d.name || nodeId(d));

      node.on('click', (_, d) => {{
        const sev = (d.severity || '').toLowerCase() || 'n/a';
        const rule = d.rule || 'N/A';
        const cat = d.category || 'N/A';
        const detail = d.detail || d.description || '';
        const cites = d.citations || [];
        const citeStr = Array.isArray(cites) ? cites.join(', ') : String(cites || '');

        const color = nodeColor(d);
        hud.innerHTML = `
          <div>
            <div><span class=\"severity-dot\" style=\"background:${{color}}\"></span>
              <strong>${{d.label || d.name || nodeId(d)}} </strong></div>
            <div>Severity: <strong>${{sev}}</strong></div>
            <div>Category: <strong>${{cat}}</strong></div>
            <div>Rule: <strong>${{rule}}</strong></div>
            <div>Citations: <span>${{citeStr || '—'}}</span></div>
            <div style=\"margin-top:4px;\">${{detail}}</div>
          </div>`;
      }});

      sim.on('tick', () => {{
        link
          .attr('x1', d => d.source.x)
          .attr('y1', d => d.source.y)
          .attr('x2', d => d.target.x)
          .attr('y2', d => d.target.y);

        node.attr('transform', d => `translate(${{d.x}}, ${{d.y}})`);
      }});

      const total = nodes.length;
      const sevCounts = {{ critical: 0, warning: 0, info: 0, other: 0 }};
      nodes.forEach(n => {{
        const sev = (n.severity || '').toLowerCase();
        if (sev === 'critical') sevCounts.critical++;
        else if (sev === 'warning') sevCounts.warning++;
        else if (sev === 'info') sevCounts.info++;
        else sevCounts.other++;
      }});

      metaEl.textContent = `Nodes: ${{total}}  |  critical: ${{sevCounts.critical}}, warning: ${{sevCounts.warning}}, info: ${{sevCounts.info}}, other: ${{sevCounts.other}}`;
    </script>
  </body>
</html>
"""

    html_out_path.parent.mkdir(parents=True, exist_ok=True)
    html_out_path.write_text(html, encoding="utf-8")


# ==========
# Entry Point
# ==========


if __name__ == "__main__":
    # Subcommand modes:
    #   merge-violations      → graph CSV + violations JSON merger
    #   decorate-violations   → decorate violations.json with labels/viz hints
    #   violation-wheel       → build standalone HTML viewer for violations.json
    #   (default)             → LEXVAULT synthesizer
    if len(sys.argv) > 1 and sys.argv[1] == "merge-violations":
        main_merge_violations(sys.argv[2:])
    elif len(sys.argv) > 1 and sys.argv[1] == "decorate-violations":
        args = parse_decorate_violations_args(sys.argv[2:])
        in_path = Path(args.in_path).expanduser().resolve()
        out_path = Path(args.out_path).expanduser().resolve()
        sev_filter: Optional[List[str]] = None
        if getattr(args, "severity_filter", None):
            sev_filter = [s.strip() for s in str(args.severity_filter).split(",") if s and s.strip()]
        setup_logging(out_path.parent)
        logging.info("Decorating violations graph JSON")
        logging.info("Input : %s", in_path)
        logging.info("Output: %s", out_path)
        if sev_filter:
            logging.info("Severity filter: %s", ",".join(sev_filter))
        decorate_violation_graph_json(in_path, out_path, severity_filter=sev_filter)
        logging.info("Decoration complete")
    elif len(sys.argv) > 1 and sys.argv[1] == "violation-wheel":
        args = parse_violation_wheel_args(sys.argv[2:])
        in_path = Path(args.in_path).expanduser().resolve()
        out_path = Path(args.out_path).expanduser().resolve()
        setup_logging(out_path.parent)
        logging.info("Building Violation Wheel HTML")
        logging.info("Input : %s", in_path)
        logging.info("Output: %s", out_path)
        build_violation_wheel_html(in_path, out_path)
        logging.info("Violation Wheel HTML written")
    else:
        main_lexvault(sys.argv[1:])
