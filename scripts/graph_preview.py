#!/usr/bin/env python3
import csv
import os
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

if sys.version_info < (3, 8):
    raise SystemExit("Python 3.8+ required")


def _mk(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


OUTDIR = Path(os.environ.get("OUTDIR", str(Path.home() / "EC_OUT")))
GRAPH_DIR = OUTDIR / "graph"
MAX_NODES = int(os.environ.get("MAX_NODES", "10000"))
MAX_EDGES = int(os.environ.get("MAX_EDGES", "20000"))


def newest(pattern: str) -> Path | None:
    files = sorted(
        GRAPH_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True
    )
    return files[0] if files else None


def load_graph() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    nodes_csv = newest("nodes_*.csv")
    edges_csv = newest("edges_*.csv")
    if not nodes_csv or not edges_csv:
        raise SystemExit("No nodes_*.csv / edges_*.csv found. Run EC_NEO4J first.")
    nodes: list[dict[str, str]] = []
    edges: list[dict[str, str]] = []
    with nodes_csv.open(encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        for row in r:
            nodes.append(
                {
                    "id": row.get("id:ID") or row.get("id") or "",
                    "label": row.get("label:string") or row.get("label", ""),
                    "path": row.get("path", ""),
                    "value": row.get("value", ""),
                    "doc_type": row.get("doc_type", ""),
                    "case_no": row.get("case_no", ""),
                }
            )
    with edges_csv.open(encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        for row in r:
            edges.append(
                {
                    "source": row.get(":START_ID") or row.get("source") or "",
                    "target": row.get(":END_ID") or row.get("target") or "",
                    "type": row.get(":TYPE") or row.get("type", ""),
                }
            )
    doc_nodes = [n for n in nodes if n.get("label") == "Document"]
    other_nodes = [n for n in nodes if n.get("label") != "Document"]
    keep: set[str] = set()
    for n in doc_nodes[: MAX_NODES // 2]:
        keep.add(n["id"])
    neigh_edges = [e for e in edges if e["source"] in keep or e["target"] in keep]
    for e in neigh_edges:
        keep.add(e["source"])
        keep.add(e["target"])
    for n in other_nodes:
        if len(keep) >= MAX_NODES:
            break
        keep.add(n["id"])
    nodes_f = [n for n in nodes if n["id"] in keep][:MAX_NODES]
    idset = {n["id"] for n in nodes_f}
    edges_f = [e for e in edges if e["source"] in idset and e["target"] in idset]
    if len(edges_f) > MAX_EDGES:
        edges_f = edges_f[:MAX_EDGES]
    return nodes_f, edges_f


def write_html(nodes: list[dict[str, str]], edges: list[dict[str, str]]) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    _mk(GRAPH_DIR)
    out = GRAPH_DIR / f"preview_{ts}.html"
    data = {"nodes": nodes, "links": edges}
    template = """<!DOCTYPE html>
<html lang='en'><head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>Evidence Graph Preview</title>
<style>
body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
#toolbar { padding:10px; border-bottom:1px solid #eee; display:flex; gap:10px; align-items:center; }
#graph { width:100vw; height:calc(100vh - 52px); }
.badge { padding:2px 6px; border-radius:6px; background:#eef; margin-left:6px; font-size:12px; }
.node-doc { fill:#1f77b4; }
.node-email { fill:#ff7f0e; }
.node-phone { fill:#2ca02c; }
.node-money { fill:#9467bd; }
.node-name { fill:#8c564b; }
.node-citation { fill:#d62728; }
.node-caseid { fill:#17becf; }
.link { stroke:#999; stroke-opacity:.6; }
</style>
</head>
<body>
<div id='toolbar'>
  <strong>Evidence Graph</strong>
  <span class='badge' id='counts'></span>
  <input id='q' placeholder='filter (case no, email, doc_type...)' style='flex:1; padding:6px 8px; border:1px solid #ddd; border-radius:8px'/>
  <button id='reset'>Reset</button>
  <button id='export'>Export PNG</button>
</div>
<svg id='graph'></svg>
<!-- try local D3 first for offline; fall back to CDN dynamically -->
<script src="d3.v7.min.js"></script>
<script>
(function(){
  function withD3(fn){
    if (window.d3) return fn();
    var s=document.createElement('script');
    s.src='https://cdn.jsdelivr.net/npm/d3@7';
    s.onload=fn;
    document.head.appendChild(s);
  }
  function main(){
    const data = DATA_JSON;
    const width = window.innerWidth, height = window.innerHeight-52;
    const svg = d3.select('#graph').attr('width', width).attr('height', height);
    const g = svg.append('g');
    const zoom = d3.zoom().scaleExtent([0.1, 6]).on('zoom', (ev)=> g.attr('transform', ev.transform));
    svg.call(zoom);
    const colorByLabel = (lbl)=> ({
      'Document':'node-doc','Email':'node-email','Phone':'node-phone','Money':'node-money','Name':'node-name',
      'MCL':'node-citation','MCR':'node-citation','MRE':'node-citation','USC':'node-citation','FRCP':'node-citation','FRE':'node-citation',
      'MI_CASE':'node-citation','FED_CASE':'node-citation','CaseID':'node-caseid'
    }[lbl]||'node-name');
    document.getElementById('counts').textContent = data.nodes.length + ' nodes / ' + data.links.length + ' edges';
    const sim = d3.forceSimulation(data.nodes)
      .force('link', d3.forceLink(data.links).id(d=>d.id).distance(40).strength(0.2))
      .force('charge', d3.forceManyBody().strength(-60))
      .force('center', d3.forceCenter(width/2, height/2));

    const link = g.append('g').attr('stroke-width', 1).selectAll('line')
      .data(data.links).enter().append('line').attr('class','link');

    const node = g.append('g').selectAll('circle')
      .data(data.nodes).enter().append('circle')
      .attr('r', d=> d.label==='Document' ? 5 : 3)
      .attr('class', d=> colorByLabel(d.label))
      .call(d3.drag().on('start', dragstarted).on('drag', dragged).on('end', dragended));

    node.append('title').text(d=> `${d.label}${d.doc_type? ' · '+d.doc_type:''}${d.case_no? ' · '+d.case_no:''}${d.value? ' · '+d.value:''}${d.path? ' · '+d.path:''}`);

    sim.on('tick', ()=>{
      link.attr('x1', d=>d.source.x).attr('y1', d=>d.source.y)
          .attr('x2', d=>d.target.x).attr('y2', d=>d.target.y);
      node.attr('cx', d=>d.x).attr('cy', d=>d.y);
    });

    function dragstarted(ev, d){ if (!ev.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }
    function dragged(ev, d){ d.fx = ev.x; d.fy = ev.y; }
    function dragended(ev, d){ if (!ev.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }

    function filterGraph(q){
      const query = q.toLowerCase();
      const visible = new Set();
      data.nodes.forEach(d=>{
        const s = (d.label+' '+(d.doc_type||'')+' '+(d.case_no||'')+' '+(d.value||'')+' '+(d.path||'')).toLowerCase();
        if (!query || s.includes(query)) visible.add(d.id);
      });
      node.attr('opacity', d => visible.has(d.id) ? 1 : 0.1);
      link.attr('opacity', l => (visible.has(l.source.id) && visible.has(l.target.id)) ? 0.6 : 0.05);
    }

    document.getElementById('q').addEventListener('input', (e)=> filterGraph(e.target.value));
    document.getElementById('reset').onclick=()=>{
      svg.transition().duration(250).call(zoom.transform, d3.zoomIdentity);
      document.getElementById('q').value='';
      filterGraph('');
    };

    document.getElementById('export').onclick=()=>{
      const serializer = new XMLSerializer();
      const source = serializer.serializeToString(svg.node());
      const img = new Image();
      const svgBlob = new Blob([source], {type: 'image/svg+xml;charset=utf-8'});
      const url = URL.createObjectURL(svgBlob);
      img.onload = function(){
        const canvas = document.createElement('canvas');
        canvas.width = width; canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle='#ffffff'; ctx.fillRect(0,0,width,height);
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);
        const a = document.createElement('a');
        a.download = 'graph.png';
        a.href = canvas.toDataURL('image/png');
        a.click();
      };
      img.src = url;
    };
  }
  document.addEventListener('DOMContentLoaded', function(){ withD3(main); });
})();
</script>
</body></html>
"""
    html = template.replace("DATA_JSON", json.dumps(data))
    out.write_text(html, encoding="utf-8")
    print(str(out))


def main() -> None:
    nodes, edges = load_graph()
    write_html(nodes, edges)


if __name__ == "__main__":
    main()
