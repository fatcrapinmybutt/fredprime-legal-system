import sys
import types
from typing import Any

from scripts import meek_pipeline_launcher


def test_pipeline_sequential_run(monkeypatch: Any) -> None:
    calls: list[str] = []

    ingest_mod: Any = types.ModuleType("meek_ingest_cli")

    def ingest_main() -> None:
        calls.append("ingest")

    ingest_mod.main = ingest_main
    monkeypatch.setitem(sys.modules, "meek_ingest_cli", ingest_mod)

    fts_mod: Any = types.ModuleType("fts_cli")

    def search_records(query: str) -> None:
        calls.append(query)

    fts_mod.search_records = search_records
    monkeypatch.setitem(sys.modules, "fts_cli", fts_mod)

    meek_pipeline_launcher.main(["--search-query", "dummy"])

    assert calls == ["ingest", "dummy"]
