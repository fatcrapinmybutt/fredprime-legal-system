"""Ingestion helpers for Evidence Commander chat exports.

This module focuses on the lightweight processing that turns chat
transcripts emitted as `*.ndjson` files into aggregate counts that can be
used by higher-level orchestration code.  The ingestion itself is very
simple: each line is parsed as JSON and may contain ``nodes`` and
``edges`` collections representing additions to the knowledge graph.

Two utilities are provided:

``ingest_chat_ndjson``
    Parse a single NDJSON file and return how many nodes and edges were
    described by the file.  The helper is resilient to blank lines and
    missing keys so it can be used on partially populated exports.

``cycle_once``
    Convenience wrapper that iterates across one or more NDJSON files and
    accumulates totals in a mutable ``state`` mapping.  This mirrors the
    orchestration style used by the Evidence Commander utilities where a
    dispatcher tracks totals across polling cycles.
"""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
import json
from pathlib import Path
from typing import Any, Tuple


NDJSONEntry = dict[str, Any]
PathLike = str | Path


def _normalise_sequence(value: Any) -> list[Any]:
    """Coerce ``value`` into a list for counting purposes.

    Evidence Commander exports sometimes omit a collection entirely or use
    ``None``.  Treat those cases as an empty list so callers do not have to
    guard against ``None`` before counting.
    """

    if value is None:
        return []
    if isinstance(value, list):
        return value
    # Some callers may pass tuples/sets; convert to list for counting.
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def ingest_chat_ndjson(path: PathLike) -> Tuple[int, int]:
    """Read a chat NDJSON file and return ``(node_count, edge_count)``.

    Parameters
    ----------
    path:
        File system path (string or :class:`~pathlib.Path`) to the NDJSON
        export.  Each non-empty line should be a JSON object that may
        contain ``nodes`` and ``edges`` keys describing graph additions.

    Returns
    -------
    tuple[int, int]
        A tuple containing the number of nodes and edges described by the
        file.  Missing keys are treated as zero-length collections.
    """

    node_total = 0
    edge_total = 0
    path_obj = Path(path)

    with path_obj.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                entry: NDJSONEntry = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - error path
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path_obj}: {exc}"
                ) from exc

            nodes = _normalise_sequence(entry.get("nodes"))
            # Support alternate key ``links`` sometimes used for edges.
            edges = _normalise_sequence(entry.get("edges"))
            if not edges and "links" in entry:
                edges = _normalise_sequence(entry.get("links"))

            node_total += len(nodes)
            edge_total += len(edges)

    return node_total, edge_total


def cycle_once(
    ndjson_paths: Iterable[PathLike],
    *,
    state: MutableMapping[str, int] | None = None,
) -> MutableMapping[str, int]:
    """Process a set of NDJSON files and accumulate ingestion statistics.

    Parameters
    ----------
    ndjson_paths:
        Iterable of NDJSON file paths to process in this cycle.
    state:
        Mutable mapping holding counters such as ``nodes_added`` and
        ``edges_added``.  A new mapping is created automatically when one
        is not provided.

    Returns
    -------
    MutableMapping[str, int]
        The mapping supplied via ``state`` (or a newly created one) with
        ``nodes_added`` and ``edges_added`` updated to include the data
        from ``ndjson_paths``.  A ``files_processed`` counter is maintained
        as a convenience for callers that track throughput.
    """

    if state is None:
        state = {"nodes_added": 0, "edges_added": 0, "files_processed": 0}
    else:
        state.setdefault("nodes_added", 0)
        state.setdefault("edges_added", 0)
        state.setdefault("files_processed", 0)

    for path in ndjson_paths:
        node_count, edge_count = ingest_chat_ndjson(path)
        state["nodes_added"] += node_count
        state["edges_added"] += edge_count
        state["files_processed"] += 1

    return state


__all__ = ["ingest_chat_ndjson", "cycle_once"]

