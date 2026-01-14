#!/data/data/com.termux/files/usr/bin/bash
# ec_graph_preview_addon.sh — installs D3 graph preview and menu entry
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
OUTDIR="${OUTDIR:-$HOME/EC_OUT}"
mkdir -p "$OUTDIR/graph" "$HOME/.local/bin"

# 1) install graph_preview.py
cat <<'PY' > "$OUTDIR/graph_preview.py"
#!/usr/bin/env python3
import csv
import os
from datetime import datetime
from pathlib import Path
import json

OUTDIR = Path(os.environ.get("OUTDIR", str(Path.home() / "EC_OUT")))
GRAPH_DIR = OUTDIR / "graph"
MAX_NODES = int(os.environ.get("MAX_NODES", "10000"))
MAX_EDGES = int(os.environ.get("MAX_EDGES", "20000"))

def newest(pattern: str) -> Path | None:
    files = sorted(GRAPH_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def load_graph():
    nodes_csv = newest("nodes_*.csv")
    edges_csv = newest("edges_*.csv")
    if not nodes_csv or not edges_csv:
        raise SystemExit("No nodes_*.csv / edges_*.csv found. Run EC_NEO4J first.")
    nodes = []
    edges = []
    with nodes_csv.open(encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        for row in r:
            nodes.append({
                "id": row.get("id:ID") or row.get("id") or "",
                "label": row.get("label:string") or row.get("label", ""),
                "path": row.get("path", ""),
                "value": row.get("value", ""),
                "doc_type": row.get("doc_type", ""),
                "case_no": row.get("case_no", ""),
            })
    with edges_csv.open(encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        for row in r:
            edges.append({
                "source": row.get(":START_ID") or row.get("source"),
                "target": row.get(":END_ID") or row.get("target"),
                "type": row.get(":TYPE") or row.get("type", ""),
            })
    doc_nodes = [n for n in nodes if n.get("label") == "Document"]
    other_nodes = [n for n in nodes if n.get("label") != "Document"]
    keep = set()
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

def write_html(nodes, edges):
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    out = OUTDIR / "graph" / f"preview_{ts}.html"
    data = {"nodes": nodes, "links": edges}
    html = f"""<!DOCTYPE html>
<html lang='en'><head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>Evidence Graph Preview</title>
<style>
body {{ margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
#toolbar {{ padding:10px; border-bottom:1px solid #eee; display:flex; gap:10px; align-items:center; }}
#graph {{ width:100vw; height:calc(100vh - 52px); }}
.badge {{ padding:2px 6px; border-radius:6px; background:#eef; margin-left:6px; font-size:12px; }}
.node-doc {{ fill:#1f77b4; }}
.node-email {{ fill:#ff7f0e; }}
.node-phone {{ fill:#2ca02c; }}
.node-money {{ fill:#9467bd; }}
.node-name {{ fill:#8c564b; }}
.node-citation {{ fill:#d62728; }}
.node-caseid {{ fill:#17becf; }}
.link {{ stroke:#999; stroke-opacity:.6; }}
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
<script src='https://cdn.jsdelivr.net/npm/d3@7'></script>
<script>
const data = {json.dumps(data)};
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

document.getElementById('q').addEventListener('input', (e)=>{
  const q = e.target.value.toLowerCase();
  node.attr('opacity', d=>{
    if (!q) return 1;
    const s = (d.label+' '+(d.doc_type||'')+' '+(d.case_no||'')+' '+(d.value||'')+' '+(d.path||'')).toLowerCase();
    return s.includes(q) ? 1 : 0.1;
  });
  link.attr('opacity', l=> (l.source.opacity!==0.1 && l.target.opacity!==0.1) ? .6 : .05);
});

document.getElementById('reset').onclick=()=>{ svg.transition().duration(250).call(zoom.transform, d3.zoomIdentity); document.getElementById('q').value=''; node.attr('opacity',1); link.attr('opacity',.6); };

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
</script>
</body></html>
"""
    out.write_text(html, encoding='utf-8')
    print(str(out))

def main():
    nodes, edges = load_graph()
    write_html(nodes, edges)

if __name__ == '__main__':
    main()
PY
chmod +x "$OUTDIR/graph_preview.py"
cp "$(dirname "$0")/graph_preview.py" "$OUTDIR/graph_preview.py"

# 1a) try to cache a local D3 for offline use
if command -v curl >/dev/null 2>&1; then
  curl -fsSL "https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js" -o "$OUTDIR/graph/d3.v7.min.js" || true
fi

# 2) EC_GRAPH wrapper
cat <<'H' > "$HOME/.local/bin/EC_GRAPH"
#!/data/data/com.termux/files/usr/bin/bash
set -e
OUTDIR="${OUTDIR:-$HOME/EC_OUT}"
MAX_NODES="${MAX_NODES:-10000}"
MAX_EDGES="${MAX_EDGES:-20000}"
EC_NEO4J >/dev/null 2>&1 || true
MAX_NODES="$MAX_NODES" MAX_EDGES="$MAX_EDGES" OUTDIR="$OUTDIR" python "$OUTDIR/graph_preview.py"
H
chmod +x "$HOME/.local/bin/EC_GRAPH"

# 3) Patch menu if exists
if [ -f "$HOME/EC_MENU_FZF.sh" ]; then
  cat <<'MENU' > "$HOME/EC_MENU_FZF.sh"
#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

cfg="$HOME/.ec_menu.conf"; touch "$cfg"
get_cfg(){ awk -F= -v k="$1" '$1==k{print substr($0,index($0,"=")+1)}' "$cfg" 2>/dev/null || true; }
set_cfg(){ if grep -q "^$1=" "$cfg" 2>/dev/null; then sed -i "s#^$1=.*#$1=$2#" "$cfg"; else echo "$1=$2" >> "$cfg"; fi; }
have(){ command -v "$1" >/dev/null 2>&1; }

THREADS="$(get_cfg THREADS)"; [ -n "${THREADS}" ] || THREADS=12
OCR_MAX_PAGES="$(get_cfg OCR_MAX_PAGES)"; [ -n "${OCR_MAX_PAGES}" ] || OCR_MAX_PAGES=250
OCR_LANGS="$(get_cfg OCR_LANGS)"; [ -n "${OCR_LANGS}" ] || OCR_LANGS="eng"
DEEP="$(get_cfg DEEP)"; [ -n "${DEEP}" ] || DEEP=0
DO_ORGANIZE="$(get_cfg DO_ORGANIZE)"; [ -n "${DO_ORGANIZE}" ] || DO_ORGANIZE=0
FORCE_ORGANIZE="$(get_cfg FORCE_ORGANIZE)"; [ -n "${FORCE_ORGANIZE}" ] || FORCE_ORGANIZE=0
DRY_RUN="$(get_cfg DRY_RUN)"; [ -n "${DRY_RUN}" ] || DRY_RUN=1
CASE_FILTER="$(get_cfg CASE_FILTER)"; [ -n "${CASE_FILTER}" ] || CASE_FILTER=""
DRIVE_PREFIX="$(get_cfg DRIVE_PREFIX)"; [ -n "${DRIVE_PREFIX}" ] || DRIVE_PREFIX=""
REMOTE="$(get_cfg REMOTE)"; [ -n "${REMOTE}" ] || REMOTE="gdrive"
DRIVE_PATH="$(get_cfg DRIVE_PATH)"; [ -n "${DRIVE_PATH}" ] || DRIVE_PATH="/"

apply_filters(){
  if [ -n "$DRIVE_PREFIX" ] && [ -f "$HOME/EC_OUT/indices/drive_full_pdf.json" ]; then
    cp "$HOME/EC_OUT/indices/drive_full_pdf.json"{,.bak} 2>/dev/null || true
    jq -c "map(select(.Path|startswith(\"$DRIVE_PREFIX\")))" "$HOME/EC_OUT/indices/drive_full_pdf.json" > "$HOME/EC_OUT/indices/drive_full_pdf.json.tmp" || true
    mv -f "$HOME/EC_OUT/indices/drive_full_pdf.json.tmp" "$HOME/EC_OUT/indices/drive_full_pdf.json" 2>/dev/null || true
  fi
}
run_env(){
  local cmd="$1"; shift
  echo -e "\n>>> THREADS=$THREADS OCR_MAX_PAGES=$OCR_MAX_PAGES OCR_LANGS=\"$OCR_LANGS\" DEEP=$DEEP DO_ORGANIZE=$DO_ORGANIZE FORCE_ORGANIZE=$FORCE_ORGANIZE DRY_RUN=$DRY_RUN REMOTE=$REMOTE DRIVE_PATH=$DRIVE_PATH"
  THREADS="$THREADS" OCR_MAX_PAGES="$OCR_MAX_PAGES" OCR_LANGS="$OCR_LANGS" DEEP="$DEEP" DO_ORGANIZE="$DO_ORGANIZE" FORCE_ORGANIZE="$FORCE_ORGANIZE" DRY_RUN="$DRY_RUN" REMOTE="$REMOTE" DRIVE_PATH="$DRIVE_PATH" "$cmd" "$@"
  echo -e "\n[press Enter to continue]"; read -r _
}
prompt(){ local label="$1" var="$2" def="$3"; read -rp "$label [$def]: " val || true; [ -z "${val:-}" ] && val="$def"; eval "$var=\"$val\""; }
toggle(){ local name="$1"; local val="$(eval echo "\$$1")"; if [ "$val" = "1" ]; then val=0; else val=1; fi; eval "$1=$val"; }
case_subset(){ local case="$1"; local out="$HOME/EC_OUT/case_timeline/$case/subset.csv"; mkdir -p "$(dirname "$out")"; awk -F, -v c="$case" 'NR==1||index($0,c){print}' "$HOME/EC_OUT"/ledger/analysis_master_part*.csv > "$out" 2>/dev/null || true; echo "Subset written: $out"; }

menu_fzf(){
  local items="
Run: OMEGA (all)
Run: FAST scan
Run: ACCURATE (force OCR)
Run: STORY only
Run: DRIVE-ONLY
Run: LOCAL-ONLY
Run: DEEP SWEEP
Run: ORGANIZE (PDF) — DRY RUN
Run: ORGANIZE (PDF) — EXECUTE
Run: AUTH REFRESH
Run: NEO4J EXPORT (nodes/edges)
Run: GRAPH PREVIEW (10k from latest ledger)
Run: MIFILE ZIP PACKER (prompt)
Run: PURGE CACHE
Run: CASE DOCKET MD
Set: THREADS ($THREADS)
Set: OCR_MAX_PAGES ($OCR_MAX_PAGES)
Set: OCR_LANGS ($OCR_LANGS)
Set: REMOTE ($REMOTE)
Set: DRIVE_PATH ($DRIVE_PATH)
Set: DRIVE_PREFIX filter ($DRIVE_PREFIX)
Set: CASE_FILTER ($CASE_FILTER)
Toggle: DEEP ($DEEP)
Toggle: DO_ORGANIZE ($DO_ORGANIZE)
Toggle: FORCE_ORGANIZE ($FORCE_ORGANIZE)
Toggle: DRY_RUN ($DRY_RUN)
Exit"
  if command -v fzf >/dev/null 2>&1; then
    echo "$items" | fzf --prompt="Evidence Commander › " --height=90% --border --ansi
  else
    echo "$items"; read -rp "Select (type line exactly): " choice; echo "$choice"
  fi
}

exec_choice(){
  local c="$1"
  case "$c" in
    "Run: OMEGA (all)")        apply_filters; run_env EC_OMEGA ;;
    "Run: FAST scan")          apply_filters; THREADS=24 OCR_MAX_PAGES=0 run_env EC_SCAN ;;
    "Run: ACCURATE (force OCR)") apply_filters; OCR_MAX_PAGES=0 OCR_LANGS="eng+osd" THREADS=8 run_env EC_ROLL ;;
    "Run: STORY only")         apply_filters; run_env EC_STORY ;;
    "Run: DRIVE-ONLY")         cp -f "$HOME/EC_OUT/indices/local_roots.txt"{,.bak} 2>/dev/null || true; :> "$HOME/EC_OUT/indices/local_roots.txt"; apply_filters; run_env EC_OMEGA; mv -f "$HOME/EC_OUT/indices/local_roots.txt"{.bak,} 2>/dev/null || true ;;
    "Run: LOCAL-ONLY")         :> "$HOME/EC_OUT/indices/drive_full_pdf.json"; run_env EC_OMEGA ;;
    "Run: DEEP SWEEP")         DEEP=1 DO_ORGANIZE=${FORCE_ORGANIZE:-0} apply_filters; run_env EC_OMEGA ;;
    "Run: ORGANIZE (PDF) — DRY RUN") DO_ORGANIZE=1 DRY_RUN=1 run_env EC_SCAN ;;
    "Run: ORGANIZE (PDF) — EXECUTE") DO_ORGANIZE=1 DRY_RUN=0 FORCE_ORGANIZE=1 run_env EC_SCAN ;;
    "Run: AUTH REFRESH")       run_env EC_AUTH_ONLY ;;
    "Run: NEO4J EXPORT (nodes/edges)") run_env EC_NEO4J ;;
    "Run: GRAPH PREVIEW (10k from latest ledger)") run_env EC_GRAPH ;;
    "Run: MIFILE ZIP PACKER (prompt)")
        prompt "Case number (e.g., 2025-002760-CZ)" CASE "2025-002760-CZ"
        prompt "Max exhibits to include" EXHIBITS "50"
        CASE="$CASE" EXHIBITS="$EXHIBITS" run_env EC_MIFILE ;;
    "Run: PURGE CACHE")        rm -rf "$HOME/EC_OUT/ocr_tmp" "$HOME/EC_OUT/cache/"* 2>/dev/null || true; echo "Cache cleared."; read -r -p "[Enter]" _ ;;
    "Run: CASE DOCKET MD")     for d in "$HOME/EC_OUT"/case_timeline/*/timeline.csv; do CASE=$(basename "$(dirname "$d")"); mkdir -p "$HOME/EC_OUT/narratives/$CASE"; awk -F, 'NR>1{print "- **"$1"** — *"$2"* — "$5"  \\n  `"$3"`"}' "$d" > "$HOME/EC_OUT/narratives/$CASE/docket.md"; echo "Built docket for $CASE"; done; read -r -p "[Enter]" _ ;;
    "Set: THREADS ("*")")      prompt "Threads" THREADS "$THREADS"; set_cfg THREADS "$THREADS" ;;
    "Set: OCR_MAX_PAGES ("*")") prompt "OCR MAX pages (0=all)" OCR_MAX_PAGES "$OCR_MAX_PAGES"; set_cfg OCR_MAX_PAGES "$OCR_MAX_PAGES" ;;
    "Set: OCR_LANGS ("*")")    prompt "OCR languages (eng or eng+osd)" OCR_LANGS "$OCR_LANGS"; set_cfg OCR_LANGS "$OCR_LANGS" ;;
    "Set: REMOTE ("*")")       prompt "Rclone remote name" REMOTE "$REMOTE"; set_cfg REMOTE "$REMOTE" ;;
    "Set: DRIVE_PATH ("*")")   prompt "Drive base path (/ or MyFolder)" DRIVE_PATH "$DRIVE_PATH"; set_cfg DRIVE_PATH "$DRIVE_PATH" ;;
    "Set: DRIVE_PREFIX filter ("*")") prompt "Drive prefix (e.g., Evidence/CaseX) empty=none" DRIVE_PREFIX "$DRIVE_PREFIX"; set_cfg DRIVE_PREFIX "$DRIVE_PREFIX" ;;
    "Set: CASE_FILTER ("*")")  prompt "Case number" CASE_FILTER "$CASE_FILTER"; set_cfg CASE_FILTER "$CASE_FILTER"; [ -n "$CASE_FILTER" ] && case_subset "$CASE_FILTER"; read -r -p "[Enter]" _ ;;
    "Toggle: DEEP ("*")")      if [ "$DEEP" = "1" ]; then DEEP=0; else DEEP=1; fi; set_cfg DEEP "$DEEP" ;;
    "Toggle: DO_ORGANIZE ("*")") if [ "$DO_ORGANIZE" = "1" ]; then DO_ORGANIZE=0; else DO_ORGANIZE=1; fi; set_cfg DO_ORGANIZE "$DO_ORGANIZE" ;;
    "Toggle: FORCE_ORGANIZE ("*")") if [ "$FORCE_ORGANIZE" = "1" ]; then FORCE_ORGANIZE=0; else FORCE_ORGANIZE=1; fi; set_cfg FORCE_ORGANIZE "$FORCE_ORGANIZE" ;;
    "Toggle: DRY_RUN ("*")")   if [ "$DRY_RUN" = "1" ]; then DRY_RUN=0; else DRY_RUN=1; fi; set_cfg DRY_RUN "$DRY_RUN" ;;
    "Exit") exit 0 ;;
    *) echo "Unknown choice"; sleep 1 ;;
  esac
}

for c in EC_OMEGA EC_SCAN EC_ROLL EC_STORY EC_NEO4J EC_MIFILE EC_GRAPH; do
  if ! command -v "$c" >/dev/null 2>&1; then
    echo "Missing command: $c"
    echo "Run your installers first, then re-run this menu."
    exit 1
  fi
 done

while true; do
  if command -v fzf >/dev/null 2>&1; then choice="$(menu_fzf)"; else echo "fzf not found; install with: pkg install -y fzf jq"; choice="$(menu_fzf)"; fi
  exec_choice "$choice"
done
MENU
  chmod +x "$HOME/EC_MENU_FZF.sh"
fi

echo "✅ Graph preview installed. Use: EC_GRAPH  (returns preview file path)"
echo "Also available in the menu: 'Run: GRAPH PREVIEW (10k from latest ledger)'"
