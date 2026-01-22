# Superset Map Guide

The superset map is a canonical, generated graph of the corpus modules.

## Build
```
python -m scripts.build_superset_graph
```

Outputs:
- graphs/superset_map.json
- graphs/html/superset_map.html
- graphs/neo4j/nodes.csv, edges.csv, import.cypher

## Metadata nodes
The superset map includes metadata nodes for performance and validation:
- `graphs/metadata/hardware.json` (environment snapshot)
- `graphs/metadata/benchmarks.json` (benchmark results)
- `graphs/metadata/golden.json` (golden correctness output)

## HTML viewer
Open `graphs/html/superset_map.html` in your browser (offline).
