"""Utilities for loading and extracting Codex instruction payloads."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PRIMARY_HEADER = "## 2) Primary Spec — EPOCH.2 (Verbatim)"
BRANCH_HEADER = "## 3) Branch Format v2 (Verbatim)"
CODE_FENCE = "```text"
CODE_FENCE_END = "```"


@dataclass(frozen=True)
class CodexInstructions:
    primary_spec: str
    branch_format: str
    source_path: Path


def load_text(path: Path) -> str:
    """Load a UTF-8 text file into memory."""

    return path.read_text(encoding="utf-8")


def extract_code_block(document: str, header: str) -> str:
    """Extract a fenced code block that follows a given header."""

    header_index = document.find(header)
    if header_index == -1:
        raise ValueError(f"Header not found: {header}")

    fence_index = document.find(CODE_FENCE, header_index)
    if fence_index == -1:
        raise ValueError(f"Code fence not found after header: {header}")

    content_start = fence_index + len(CODE_FENCE)
    content_start = document.find("\n", content_start) + 1
    fence_end = document.find(CODE_FENCE_END, content_start)
    if fence_end == -1:
        raise ValueError(f"Closing code fence not found after header: {header}")

    return document[content_start:fence_end].rstrip("\n")


def load_codex_instructions(path: Path) -> CodexInstructions:
    """Load Codex instructions from the merged markdown document."""

    document = load_text(path)
    primary = extract_code_block(document, PRIMARY_HEADER)
    branch = extract_code_block(document, BRANCH_HEADER)
    return CodexInstructions(primary_spec=primary, branch_format=branch, source_path=path)


def validate_required_markers(payload: str, required_markers: Iterable[str]) -> list[str]:
    """Validate that required markers exist in a payload."""

    missing = [marker for marker in required_markers if marker not in payload]
    return missing


def validate_codex_instructions(codex: CodexInstructions) -> dict[str, list[str]]:
    """Return missing marker details for the primary and branch payloads."""

    primary_missing = validate_required_markers(
        codex.primary_spec,
        ["SPEC=LITIGATIONOS_GRAPH-LEGAL-BRAIN@EPOCH.2", "LOCKS{", "REDTEAM{"],
    )
    branch_missing = validate_required_markers(
        codex.branch_format,
        ["BRANCH_FORMAT@v2", "HEADER|", "SELECT|winner="],
    )
    return {"primary": primary_missing, "branch": branch_missing}
