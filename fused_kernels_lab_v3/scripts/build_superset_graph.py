from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass(frozen=True)
class Node:
    id: str
    label: str
    kind: str
    path: str


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    rel: str


def build() -> Dict[str, Any]:
    # Canonical superset map: components and their relationships (maintained in-code, deterministic).
    nodes: List[Node] = []
    edges: List[Edge] = []

    def add_node(id: str, label: str, kind: str, path: str):
        nodes.append(Node(id=id, label=label, kind=kind, path=path))

    def add_edge(s: str, t: str, rel: str):
        edges.append(Edge(source=s, target=t, rel=rel))

    # Kernel nodes
    add_node("K_BIAS_GELU_TANH", "fused_bias_gelu_tanh", "kernel", "kernels/fused_bias_gelu_tanh.py")
    add_node("K_BIAS_GELU_EXACT", "fused_bias_gelu_exact", "kernel", "kernels/fused_bias_gelu_exact.py")
    add_node("K_LAYERNORM", "fused_layernorm", "kernel", "kernels/fused_layernorm.py")
    add_node("K_MASKED_SOFTMAX", "fused_masked_softmax", "kernel", "kernels/fused_masked_softmax.py")
    add_node("K_GEMM_EPILOGUE", "fused_gemm_epilogue", "kernel", "kernels/fused_gemm_epilogue.py")
    add_node("K_COMMON", "kernel_common", "module", "kernels/_common.py")

    for kid in ["K_BIAS_GELU_TANH", "K_BIAS_GELU_EXACT", "K_LAYERNORM", "K_MASKED_SOFTMAX", "K_GEMM_EPILOGUE"]:
        add_edge(kid, "K_COMMON", "uses")

    # Scripts
    add_node("S_ENV", "env_check", "script", "scripts/env_check.py")
    add_node("S_BENCH_BG", "bench_bias_gelu", "script", "scripts/bench_bias_gelu.py")
    add_node("S_BENCH_LN", "bench_layernorm", "script", "scripts/bench_layernorm.py")
    add_node("S_BENCH_SM", "bench_softmax", "script", "scripts/bench_softmax.py")
    add_node("S_BENCH_GEMM", "bench_gemm_epilogue", "script", "scripts/bench_gemm_epilogue.py")
    add_node("S_BUILD_GRAPH", "build_superset_graph", "script", "scripts/build_superset_graph.py")

    add_edge("S_BENCH_BG", "K_BIAS_GELU_TANH", "benchmarks")
    add_edge("S_BENCH_BG", "K_BIAS_GELU_EXACT", "benchmarks")
    add_edge("S_BENCH_LN", "K_LAYERNORM", "benchmarks")
    add_edge("S_BENCH_SM", "K_MASKED_SOFTMAX", "benchmarks")
    add_edge("S_BENCH_GEMM", "K_GEMM_EPILOGUE", "benchmarks")
    add_edge("S_BUILD_GRAPH", "S_ENV", "documents")
    add_edge("S_BUILD_GRAPH", "S_BENCH_BG", "documents")
    add_edge("S_BUILD_GRAPH", "S_BENCH_LN", "documents")
    add_edge("S_BUILD_GRAPH", "S_BENCH_SM", "documents")
    add_edge("S_BUILD_GRAPH", "S_BENCH_GEMM", "documents")

    # Tests
    add_node("T_BG", "test_bias_gelu", "test", "tests/test_bias_gelu.py")
    add_node("T_LN", "test_layernorm", "test", "tests/test_layernorm.py")
    add_node("T_SM", "test_softmax", "test", "tests/test_softmax.py")
    add_node("T_GEMM", "test_gemm_epilogue", "test", "tests/test_gemm_epilogue.py")

    add_edge("T_BG", "K_BIAS_GELU_TANH", "validates")
    add_edge("T_BG", "K_BIAS_GELU_EXACT", "validates")
    add_edge("T_LN", "K_LAYERNORM", "validates")
    add_edge("T_SM", "K_MASKED_SOFTMAX", "validates")
    add_edge("T_GEMM", "K_GEMM_EPILOGUE", "validates")

    # Docs
    add_node("D_SPEC", "SPEC", "doc", "docs/SPEC.md")
    add_node("D_VENDOR", "VENDOR_EPILOGUE_TRACK", "doc", "docs/VENDOR_EPILOGUE_TRACK.md")
    add_node("D_MAP", "SUPSET_MAP_GUIDE", "doc", "docs/SPEC.md")
    add_edge("D_SPEC", "K_BIAS_GELU_TANH", "specifies")
    add_edge("D_SPEC", "K_LAYERNORM", "specifies")
    add_edge("D_SPEC", "K_MASKED_SOFTMAX", "specifies")
    add_edge("D_SPEC", "K_GEMM_EPILOGUE", "specifies")
    add_edge("D_VENDOR", "K_GEMM_EPILOGUE", "extends")
    add_edge("D_MAP", "S_BUILD_GRAPH", "explains")

    return {
        "nodes": [n.__dict__ for n in nodes],
        "edges": [e.__dict__ for e in edges],
        "meta": {
            "name": "Fused Kernels Lab Superset Map",
            "version": "v3",
        },
    }


def emit(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build()
    (out_dir / "superset_map.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def emit_neo4j(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build()
    nodes = data["nodes"]
    edges = data["edges"]

    # CSV for Neo4j
    nodes_csv = "id:ID,label,kind,path\n" + "\n".join(
        f"{n['id']},{json.dumps(n['label'])},{n['kind']},{json.dumps(n['path'])}" for n in nodes
    ) + "\n"
    edges_csv = "source:START_ID,target:END_ID,rel\n" + "\n".join(
        f"{e['source']},{e['target']},{e['rel']}" for e in edges
    ) + "\n"

    (out_dir / "nodes.csv").write_text(nodes_csv, encoding="utf-8")
    (out_dir / "edges.csv").write_text(edges_csv, encoding="utf-8")

    cypher = """// Neo4j import (place CSVs in Neo4j import directory)
//
// :param nodesCsv => 'file:///nodes.csv';
// :param edgesCsv => 'file:///edges.csv';

CREATE CONSTRAINT fusedkernels_node_id IF NOT EXISTS
FOR (n:FusedKernelsNode) REQUIRE n.id IS UNIQUE;

LOAD CSV WITH HEADERS FROM $nodesCsv AS row
MERGE (n:FusedKernelsNode {id: row.id})
SET n.label = row.label,
    n.kind = row.kind,
    n.path = row.path;

LOAD CSV WITH HEADERS FROM $edgesCsv AS row
MATCH (s:FusedKernelsNode {id: row.source})
MATCH (t:FusedKernelsNode {id: row.target})
MERGE (s)-[r:REL {type: row.rel}]->(t);
"""
    (out_dir / "import.cypher").write_text(cypher, encoding="utf-8")


def emit_html(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build()

    # Minimal offline interactive graph (no external libs). Force layout is simplified for readability.
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>Fused Kernels Lab — Superset Map (Offline)</title>
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<style>
  body {{ margin: 0; font-family: Arial, sans-serif; }}
  #topbar {{ padding: 10px 12px; border-bottom: 1px solid #ddd; display: flex; gap: 10px; align-items: center; }}
  #search {{ width: 360px; padding: 6px 8px; }}
  #legend {{ display: flex; gap: 10px; flex-wrap: wrap; font-size: 12px; }}
  .pill {{ padding: 3px 8px; border-radius: 999px; border: 1px solid #ccc; }}
  #canvas {{ width: 100vw; height: calc(100vh - 52px); display: block; background: #fafafa; }}
  #info {{ position: absolute; top: 60px; right: 12px; width: 360px; max-height: 70vh; overflow: auto;
           background: white; border: 1px solid #ddd; border-radius: 8px; padding: 10px; box-shadow: 0 2px 10px rgba(0,0,0,.08); }}
  #info h3 {{ margin: 0 0 6px 0; font-size: 14px; }}
  #info pre {{ white-space: pre-wrap; word-wrap: break-word; margin: 0; font-size: 12px; }}
</style>
</head>
<body>
<div id=\"topbar\">
  <strong>Superset Map</strong>
  <input id=\"search\" placeholder=\"Search (id, label, kind, path) …\">
  <button id=\"fit\">Fit</button>
  <button id=\"toggleLinks\">Toggle links</button>
  <div id=\"legend\">
    <span class=\"pill\">kernel</span>
    <span class=\"pill\">script</span>
    <span class=\"pill\">test</span>
    <span class=\"pill\">doc</span>
    <span class=\"pill\">module</span>
  </div>
</div>
<canvas id=\"canvas\"></canvas>
<div id=\"info\" style=\"display:none\"></div>

<script>
// No Base64. Data is embedded as plain JSON.
const DATA = {json.dumps(data)};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const info = document.getElementById('info');

let showLinks = true;

function resize() {{
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight - 52;
}}
window.addEventListener('resize', resize);
resize();

const nodes = DATA.nodes.map(n => ({{...n}}));
const edges = DATA.edges.map(e => ({{...e}}));

const kindRadius = {{
  kernel: 18,
  script: 14,
  test: 12,
  doc: 12,
  module: 10,
}};

const kindColor = {{
  kernel: '#ffb000',
  script: '#4e79a7',
  test: '#59a14f',
  doc: '#9c755f',
  module: '#bab0ac',
}};

const id2node = new Map(nodes.map(n => [n.id, n]));

function randomInit() {{
  const w = canvas.width, h = canvas.height;
  nodes.forEach((n, i) => {{
    n.x = (w * 0.15) + (w * 0.7) * Math.random();
    n.y = (h * 0.15) + (h * 0.7) * Math.random();
    n.vx = 0; n.vy = 0;
    n.r = kindRadius[n.kind] || 10;
  }});
}}
randomInit();

// Basic force simulation (offline, compact)
// - repulsion
// - spring edges
// - mild center gravity
let alpha = 1.0;

function step() {{
  const w = canvas.width, h = canvas.height;
  const cx = w/2, cy = h/2;

  // repulsion
  for (let i = 0; i < nodes.length; i++) {{
    const a = nodes[i];
    for (let j = i+1; j < nodes.length; j++) {{
      const b = nodes[j];
      let dx = a.x - b.x, dy = a.y - b.y;
      let d2 = dx*dx + dy*dy + 0.01;
      let d = Math.sqrt(d2);
      const minD = (a.r + b.r) * 1.2;
      const force = 2000 / d2;
      const fx = force * dx / d;
      const fy = force * dy / d;
      a.vx += fx; a.vy += fy;
      b.vx -= fx; b.vy -= fy;

      // soft collision
      if (d < minD) {{
        const push = (minD - d) * 0.6;
        const px = push * dx / d;
        const py = push * dy / d;
        a.vx += px; a.vy += py;
        b.vx -= px; b.vy -= py;
      }}
    }}
  }}

  // springs
  edges.forEach(e => {{
    const s = id2node.get(e.source);
    const t = id2node.get(e.target);
    if (!s || !t) return;
    const dx = t.x - s.x, dy = t.y - s.y;
    const d = Math.sqrt(dx*dx + dy*dy) + 0.01;
    const desired = 140;
    const k = 0.06;
    const f = k * (d - desired);
    const fx = f * dx / d;
    const fy = f * dy / d;
    s.vx += fx; s.vy += fy;
    t.vx -= fx; t.vy -= fy;
  }});

  // center gravity and damping
  nodes.forEach(n => {{
    n.vx += (cx - n.x) * 0.001;
    n.vy += (cy - n.y) * 0.001;
    n.vx *= 0.82;
    n.vy *= 0.82;
    n.x += n.vx * alpha;
    n.y += n.vy * alpha;
  }});

  alpha *= 0.98;
}}

let offsetX = 0, offsetY = 0, scale = 1.0;
let dragging = false;
let dragNode = null;
let lastX = 0, lastY = 0;

function toWorld(x, y) {{
  return {{
    x: (x - offsetX) / scale,
    y: (y - offsetY) / scale
  }};
}}

function draw() {{
  ctx.clearRect(0,0,canvas.width, canvas.height);
  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);

  if (showLinks) {{
    ctx.strokeStyle = '#bbb';
    ctx.lineWidth = 1/scale;
    edges.forEach(e => {{
      const s = id2node.get(e.source);
      const t = id2node.get(e.target);
      if (!s || !t) return;
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(t.x, t.y);
      ctx.stroke();
    }});
  }}

  nodes.forEach(n => {{
    ctx.beginPath();
    ctx.fillStyle = kindColor[n.kind] || '#888';
    ctx.strokeStyle = '#333';
    ctx.lineWidth = 1/scale;
    ctx.arc(n.x, n.y, n.r, 0, Math.PI*2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = '#111';
    ctx.font = `${{12/scale}}px Arial`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(n.id, n.x, n.y - n.r - (10/scale));
  }});

  ctx.restore();
}}

function tick() {{
  if (alpha > 0.01 && !dragNode) {{
    step();
  }}
  draw();
  requestAnimationFrame(tick);
}}
tick();

function pickNode(mx, my) {{
  const p = toWorld(mx, my);
  for (let i = nodes.length-1; i >= 0; i--) {{
    const n = nodes[i];
    const dx = p.x - n.x, dy = p.y - n.y;
    if (dx*dx + dy*dy <= n.r*n.r) return n;
  }}
  return null;
}}

canvas.addEventListener('mousedown', (ev) => {{
  dragging = true;
  lastX = ev.clientX; lastY = ev.clientY;
  dragNode = pickNode(ev.clientX, ev.clientY);
}});

canvas.addEventListener('mousemove', (ev) => {{
  if (!dragging) return;
  const dx = ev.clientX - lastX;
  const dy = ev.clientY - lastY;
  lastX = ev.clientX; lastY = ev.clientY;

  if (dragNode) {{
    const p = toWorld(ev.clientX, ev.clientY);
    dragNode.x = p.x;
    dragNode.y = p.y;
    dragNode.vx = 0; dragNode.vy = 0;
    alpha = 0.2;
  }} else {{
    offsetX += dx;
    offsetY += dy;
  }}
}});

window.addEventListener('mouseup', () => {{
  dragging = false;
  dragNode = null;
}});

canvas.addEventListener('wheel', (ev) => {{
  ev.preventDefault();
  const mouseX = ev.clientX, mouseY = ev.clientY;
  const p0 = toWorld(mouseX, mouseY);
  const zoom = Math.exp(-ev.deltaY * 0.001);
  scale *= zoom;
  const p1 = toWorld(mouseX, mouseY);
  offsetX += (p1.x - p0.x) * scale;
  offsetY += (p1.y - p0.y) * scale;
}}, {{ passive: false }});

canvas.addEventListener('click', (ev) => {{
  const n = pickNode(ev.clientX, ev.clientY);
  if (!n) {{
    info.style.display = 'none';
    return;
  }}
  info.style.display = 'block';
  info.innerHTML = `
    <h3>${{n.id}}</h3>
    <pre>${{JSON.stringify(n, null, 2)}}</pre>
    <h3>Edges</h3>
    <pre>${{JSON.stringify(edges.filter(e => e.source===n.id || e.target===n.id), null, 2)}}</pre>
  `;
}});

document.getElementById('fit').addEventListener('click', () => {{
  // reset transform and reheat
  offsetX = 0; offsetY = 0; scale = 1.0;
  alpha = 1.0;
}});

document.getElementById('toggleLinks').addEventListener('click', () => {{
  showLinks = !showLinks;
}});

document.getElementById('search').addEventListener('input', (ev) => {{
  const q = ev.target.value.toLowerCase().trim();
  if (!q) {{
    nodes.forEach(n => n.r = kindRadius[n.kind] || 10);
    alpha = 0.2;
    return;
  }}
  nodes.forEach(n => {{
    const hay = `${{n.id}} ${{n.label}} ${{n.kind}} ${{n.path}}`.toLowerCase();
    n.r = (hay.includes(q) ? (kindRadius[n.kind] || 10) * 1.6 : (kindRadius[n.kind] || 10) * 0.7);
  }});
  alpha = 0.2;
}});
</script>
</body>
</html>
"""
    (out_dir / "superset_map.html").write_text(html, encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    graphs = repo_root / "graphs"
    emit(graphs)
    emit_neo4j(graphs / "neo4j")
    emit_html(graphs / "html")
    print("Wrote:")
    print("-", graphs / "superset_map.json")
    print("-", graphs / "neo4j" / "nodes.csv")
    print("-", graphs / "neo4j" / "edges.csv")
    print("-", graphs / "neo4j" / "import.cypher")
    print("-", graphs / "html" / "superset_map.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
