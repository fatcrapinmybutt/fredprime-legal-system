from __future__ import annotations

import json
from pathlib import Path

from scripts.chat_ingest import cycle_once, ingest_chat_ndjson


def write_ndjson(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")


def test_ingest_chat_ndjson_counts_nodes_and_edges(tmp_path: Path) -> None:
    ndjson_path = tmp_path / "chat.ndjson"
    write_ndjson(
        ndjson_path,
        [
            {"nodes": [{"id": 1}, {"id": 2}], "edges": [{"src": 1, "dst": 2}]},
            {"nodes": [{"id": 3}], "links": [{"src": 2, "dst": 3}, {"src": 3, "dst": 1}]},
            {},
        ],
    )

    nodes, edges = ingest_chat_ndjson(ndjson_path)

    assert nodes == 3
    assert edges == 3


def test_cycle_once_accumulates_state(tmp_path: Path) -> None:
    first = tmp_path / "first.ndjson"
    second = tmp_path / "second.ndjson"
    write_ndjson(first, [{"nodes": [1, 2]}, {"edges": [1]}])
    write_ndjson(second, [{"nodes": [3]}, {"nodes": [4], "edges": [2, 3]}])

    state = cycle_once([first, second])

    assert state["nodes_added"] == 4
    assert state["edges_added"] == 3
    assert state["files_processed"] == 2

    # A subsequent cycle should continue accumulating into the provided state.
    cycle_once([first], state=state)
    assert state["nodes_added"] == 6
    assert state["edges_added"] == 4
    assert state["files_processed"] == 3
