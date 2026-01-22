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

## HTML viewer
Open `graphs/html/superset_map.html` in your browser (offline).
