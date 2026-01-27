"""ESD-BLUEPRINT GRAPH (atomized)

Overview:
- Purpose: deterministic tranche flow for LitigationOS Neo4jLegalBrain (MI-first). This blueprint defines tranche atoms, run ledger, graph schema, fusion gates, command grammar, and self-test plans.

Core node types (detailed): File, Page, Tranche, Run, Artifact, ProofObligation, SupersetNode, Case, Entity, Action, BackupLog, DuplicateGroup

Sample edges and semantics:
- (Run)-[:RAN]->(Tranche)
- (Tranche)-[:EMITS]->(Artifact)
- (Artifact)-[:DERIVED_FROM]->(Artifact)
- (File)-[:HAS_PAGE]->(Page)
- (Page)-[:MENTIONS]->(Entity)
- (File)-[:IN_CASE]->(Case)
- (Action)-[:AFFECTS]->(File)

Determinism & IDs:
- Stable ID strategy: ULID for runs + SHA1(path+mtime+len) for file stable IDs + tranche IDs T###_NAME
- Content-addressed artifacts: outputs written to cycles/<cycle_id>/artifacts/<sha256>._ext
- Run ledger: JSONL append-only in runledger/<cycle_id>.jsonl with atomic write via tempfile+rename

Proof obligations & gates:
- Each tranche defines CORE vs NONCORE POs; CORE must be SATISFIED for PCG to PASS
- PO record shape: {po_id, auth_refs[], evid_refs[], test, validator_ver, assurance, ts}

Fusion & performance:
- Fusion only allowed when gates validated; fusion plans emitted as FusionPlan.json and scheduled on Ray or Dask with Temporal orchestration

Import & replay:
- Neo4j CSV import templates provided in neo4j/import_nodes_edges.cypher
- Replay validation: run IMPORT in a clean test DB and compare node/edge deltas against expected hashes

Self-test plan (smoke & convergence):
1. Unit tests: deterministic ID, tranche dry-run does not write files, artifact generation matches expected sha256
2. Integration test: small sample zip -> run full cycle in dry-run -> compare RunLedger and outputs
3. Replay test: export cycle pack -> import to ephemeral Neo4j -> run replay validator (idempotence, node counts, delta epsilon)

Files added:
- neo4j/constraints.cypher
- neo4j/import_nodes_edges.cypher
- schemas/tranche.json
- schemas/tranche_run.json
- graph/nodes.csv
- graph/edges.csv

Next actions:
- Add run harness (FastAPI + Celery) to execute tranche runs and emit runledger
- Add visualization react demo for neon graph
- Iterate until convergence by running the self-test suite in CI and resolving blockers
"""