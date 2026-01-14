import os
import tempfile
import importlib
from pathlib import Path


def create_csv(path: Path, header: str, rows: list[str]) -> None:
    path.write_text("\n".join([header] + rows), encoding="utf-8")


def test_graph_preview_generates_html() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir)
        graph_dir = outdir / "graph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        create_csv(graph_dir / "nodes_test.csv", "id:ID,label:string", ["1,Document", "2,Name"])
        create_csv(graph_dir / "edges_test.csv", ":START_ID,:END_ID,:TYPE", ["1,2,LINK"])
        os.environ["OUTDIR"] = str(outdir)
        gp = importlib.reload(importlib.import_module("scripts.graph_preview"))
        nodes, edges = gp.load_graph()
        assert len(nodes) == 2
        assert len(edges) == 1
        gp.write_html(nodes, edges)
        files = list((outdir / "graph").glob("preview_*.html"))
        assert files
