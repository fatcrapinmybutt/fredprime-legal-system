"""Graph persistence utilities for the litigation operating system cycle."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class GraphStore:
    """Persist graph nodes and edges to JSONL files.

    Parameters
    ----------
    storage_dir:
        Directory where the graph JSONL files should live. The directory will be
        created automatically if it does not already exist.
    """

    def __init__(self, storage_dir: str | Path) -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.nodes_path = self.storage_dir / "nodes.jsonl"
        self.edges_path = self.storage_dir / "edges.jsonl"

        self.nodes_seen: set[str] = set()
        self.edges_seen: set[str] = set()

        self._bootstrap_seen_sets()

    def add_node(self, payload: Mapping[str, Any]) -> bool:
        """Add a single node to the store.

        Returns ``True`` if the node was written and ``False`` if the node was
        already present.
        """

        record = dict(payload)
        key = self._normalise_payload(record)
        if key in self.nodes_seen:
            return False

        self._append_json(self.nodes_path, record)
        self.nodes_seen.add(key)
        return True

    def add_edge(self, payload: Mapping[str, Any]) -> bool:
        """Add a single edge to the store.

        Returns ``True`` if the edge was written and ``False`` when the edge was
        already present.
        """

        record = dict(payload)
        key = self._normalise_payload(record)
        if key in self.edges_seen:
            return False

        self._append_json(self.edges_path, record)
        self.edges_seen.add(key)
        return True

    def _bootstrap_seen_sets(self) -> None:
        """Load any existing nodes and edges into the seen sets."""

        self._hydrate_seen_set(self.nodes_path, self.nodes_seen)
        self._hydrate_seen_set(self.edges_path, self.edges_seen)

    def _hydrate_seen_set(self, path: Path, seen: set[str]) -> None:
        if not path.exists():
            return

        with path.open("r", encoding="utf-8") as stream:
            for raw_line in stream:
                key = self._normalise_line(raw_line)
                if key is not None:
                    seen.add(key)

    def _normalise_line(self, raw_line: str) -> str | None:
        line = raw_line.strip()
        if not line:
            return None

        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return line

        return self._normalise_payload(payload)

    def _normalise_payload(self, payload: Any) -> str:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )

    def _append_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as stream:
            json.dump(dict(payload), stream, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
