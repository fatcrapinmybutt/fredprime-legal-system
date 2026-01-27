# Hypervisor Planes, Toroidal Sharding, and Atomization (v2026-01-25)
## Concept: Planes
The Hypervisor runner implements a multi-plane execution model, even though it runs inside one Python process.
Plane definitions:
- Plane 0 — Invocation
  - Parses machine token into `invocation.json`
  - Applies schema gating and strict guards
- Plane 1 — Inventory
  - Deterministic walk of workspace
  - Emits `inventory.json` and `inventory.csv`
- Plane 2 — Atomization
  - Assigns stable IDs
  - Creates node proposals for artifacts, kinds, and terms
- Plane 3 — Graph Proposal
  - Emits `graph.nodes.json`, `graph.edges.json`, and CSV mirrors
  - Emits Neo4j import template (`import.neo4j.cypher`)
- Plane 4 — Window / Bloom / Torus
  - Computes deterministic torus coordinates per node ID
  - Emits `bloom.torus_positions.csv`
- Plane 5 — Distillation
  - Applies adaptive backpressure if edges exceed cap
  - Computes delta keys
- Plane 6 — Governance and Stop
  - Convergence evaluation with EPS and STABLE_N
  - Emits `delta_summary.md`, run ledger rows
- Plane 7 — Packaging
  - Optionally zips the run folder into `run_<uuid>.zip`
## Concept: Toroidal sharding
Toroidal sharding here means mapping each node to an `(x,y)` coordinate on a fixed grid (default 64x64) derived from `crc32(node_id)`.
This provides:
- Deterministic placement across cycles
- Low-cost windowing for UI layers
- A stable sharding key for parallel follow-on processing
## Concept: Atomization
Atomization is implemented as:
- Workspace Root node
- Artifact nodes (each file)
- Kind nodes (document, image, audio_video, data, code, other)
- Term nodes (top N filename terms)
Edges:
- Root `CONTAINS` Artifact
- Artifact `IS_KIND` Kind
- Artifact `HAS_TERM` Term
## HYPERVISOR — Invocation Normalization v1.1 (Plane Table)
| Field  | Plane Table Entry |
| ------ | ----------------- |
| INV-01 | **Normalize `max_cycles=0`** → interpret as `MAX_CYCLES=INF` with `STOP.mode=CONVERGENCE` enforced; convergence uses `eps` + `stable_n` + `delta_keys`; stop requires `VRpt=PASS` streak. |
| INV-02 | **Schema-first hardening** → implement `control.schema.json` using JSON Schema 2020-12 with `unevaluatedProperties=false` on strict objects so unknown keys fail validation. ([JSON Schema][1]) |
| INV-03 | **Determinism contract** → lock `StableIDRule`, `CanonicalPaths`, `SortOrder`, `SeedPolicy` and require each artifact to carry `RebuildCommand` and `InputDigest` in `MANIFEST.json`. |
| INV-04 | **Emit/Out split** → treat `emit=[RunLedger,Manifest,DeltaSummary,StratumMetrics,VRpt]` as required “control artifacts”; treat `out=[ZIP,MD,JSON,CSV,CYPHER]` as format targets; `fail_closed=true` blocks ZIP packaging if any control artifact gate fails. |
| INV-05 | **Backpressure semantics** → `adaptive` means queue budgets per plane + per modality pool; throttle events must be recorded in `RUN_LEDGER.jsonl` and summarized in `StratumMetrics`. |
| INV-06 | **Drive shortcuts safety** → if any Drive traversal exists, default `SKIP_SHORTCUTS=true` (or explicit allow) to prevent recursion and broken shortcut traversal. ([Rclone][2]) |
| INV-07 | **Graph import safety** → enforce `CONSTRAINTS_FIRST=true` before any `LOAD CSV` import cycle to prevent duplicate/colliding entities. ([Graph Database & Analytics][3]) |
| INV-08 | **Viewer portability** → allow Bloom perspective + offline viewer bundles; export perspective JSON as a first-class artifact for replayable visualization. ([Graph Database & Analytics][4]) |
---
## Global Command Grammar v2 (Plane Table)
| Field   | Plane Table Entry |
| ------- | ----------------- |
| GRAM-01 | **Canonical grammar** → `EXPLODE_SUPERPIN:<PLANE>@TAG@TAG?K=V&K=V` (no spaces); tags are order-insensitive but must be canonical-sorted in `control.json` for determinism. |
| GRAM-02 | **Required params (minimum)** → `OUT=ZIP+MD+JSON+CSV` (or subset); `ITER=auto`; `STRICT=true`; `ROOT=F:/CAPSTONE/Litigation_OS`; `APPEND_ONLY=true`; `FAIL_CLOSED=true`. |
| GRAM-03 | **Schema-first switch** → `SCHEMA_FIRST=true` requires `control.schema.json` validation before any plane executes; invalid `control.json` aborts immediately. ([JSON Schema][1]) |
| GRAM-04 | **Convergence stop rule** → `STOP=CONVERGENCE&EPS=0.0005&N=3&DELTA_KEYS=nodes,edges,terms,artifacts` and requires `VRPT_PASS_STREAK=N`. |
| GRAM-05 | **Replay rule** → any run must be reproducible from `control.json` + `RUN_LEDGER.jsonl` + `MANIFEST.json` + recorded toolchain versions. |
---
## Control Artifact Contract (Per Cycle) (Plane Table)
| Field  | Plane Table Entry |
| ------ | ----------------- |
| ART-01 | **RUN_LEDGER.jsonl** → append-only events: plane start/stop, tranche routing, throttle events, validation results, blockers, acquisition tasks. |
| ART-02 | **MANIFEST.json** → file inventory of emitted artifacts with stable IDs, canonical paths, byte sizes, and rebuild command fingerprints. |
| ART-03 | **DELTA_SUMMARY.md** → delta of `{nodes, edges, terms, artifacts}` + promotions + gate changes since prior cycle. |
| ART-04 | **STRATUM_METRICS.csv** → counts by stratum S0–S6 + per-plane throughput + queue depth summary. |
| ART-05 | **VRpt.md** → PASS/FAIL with gate evidence: schema validation, manifest verify, determinism audit, replay audit, packaging gate. |
| ART-06 | **Optional enterprise add-on: SBOM** → `SBOM.cdx.json` (CycloneDX / ECMA-424) when `@SBOM` tag is enabled. ([Ecma International][5]) |
---
## STRATUM Mapping Rules v1.1 (Plane Table)
| Field  | Plane Table Entry |
| ------ | ----------------- |
| STR-01 | **S0_IDENTITY** → people/orgs/roles/anchors (stable IDs). |
| STR-02 | **S1_JURISDICTION** → courts/judges/case IDs/docket constructs. |
| STR-03 | **S2_AUTHORITY** → rules/statutes/caselaw/orders + authority triples. |
| STR-04 | **S3_FACTS** → evidence atoms, quotes, events (bi-temporal). |
| STR-05 | **S4_PROCEDURE** → motions, objections, hearings, service, deadlines. |
| STR-06 | **S5_DECISIONS** → findings, rulings, judgments, conditional orders. |
| STR-07 | **S6_ENFORCEMENT** → show-cause, contempt, sanctions, compliance checkpoints. |
---
## TRIG→TAG Crosswalk (Focused) (Plane Table)
| Field | Plane Table Entry |
| ----- | ----------------- |
| XW-01 | **Plane tables** → `@PLANE_TABLES` (render everything as `Field | Plane Table Entry`). |
| XW-02 | **Append-only** → `@APPEND_ONLY` (immutable prior cycles). |
| XW-03 | **Idempotent / deterministic** → `@DETERMINISM_AUDIT` + locked sort/seed + stable IDs. |
| XW-04 | **Replayable** → `@REPLAYABLE_RUN` (+ recorded versions/toolchain). |
| XW-05 | **Manifest verify** → `@MANIFEST_VERIFY` (filesystem ↔ manifest parity). |
| XW-06 | **Constraints-first** → `@CONSTRAINTS_FIRST` (Neo4j import gating). ([Graph Database & Analytics][3]) |
| XW-07 | **OCR threshold gate** → `@OCR_GATE` then `@OCR_RUN` only for gated pages. |
| XW-08 | **Hybrid retrieval** → `@HYBRID_INDEX` with RRF fusion. ([Qdrant][6]) |
| XW-09 | **Watcher** → `@WATCHER_ON` (watchdog-based) with PollingObserver fallback when needed. ([python-watchdog.readthedocs.io][7]) |
| XW-10 | **Drive shortcut safety** → `@SKIP_SHORTCUTS` (Drive traversal protection). ([Rclone][2]) |
---
## Enterprise & SPEC Grade Usage Pattern Library v2 (P01–P50) as Plane Tables
### CONTROL_PLANE (P01–P08)
| Field | Plane Table Entry |
| ----- | ----------------- |
| P01   | **Policy-locked run envelope** → `EXPLODE_SUPERPIN:CONTROL@STRICT@SCHEMA_FIRST@FAIL_CLOSED?OUT=JSON+MD&ITER=auto&STRICT=true` → validates with JSON Schema 2020-12 + hard-fails unknown fields via `unevaluatedProperties=false`; emits `control.json`, `control.schema.json`, `VRpt.md`. ([JSON Schema][1]) |
| P02   | **Append-only CyclePack** → `EXPLODE_SUPERPIN:PACK@APPEND_ONLY@MANIFEST_VERIFY@FAIL_CLOSED?OUT=ZIP+JSON+MD&ITER=auto&STRICT=true` → emits `CyclePack.zip` only if `MANIFEST_VERIFY=PASS` and `VRpt=PASS`. |
| P03   | **Delta-only diagnostic pass** → `EXPLODE_SUPERPIN:DIAG@DELTA_ONLY@STRICT?OUT=MD+CSV+JSON&ITER=auto&STRICT=true` → emits `DELTA_SUMMARY.md`, `STRATUM_METRICS.csv`, drift flags; no packaging. |
| P04   | **Replay validation** → `EXPLODE_SUPERPIN:REPLAY@REPLAYABLE_RUN@MANIFEST_VERIFY?OUT=MD+JSON&ITER=auto&STRICT=true` → replays from `control.json`; emits mismatch report + PASS/FAIL. |
| P05   | **Determinism audit** → `EXPLODE_SUPERPIN:DIAG@DETERMINISM_AUDIT?OUT=MD+JSON&ITER=auto&STRICT=true` → checks stable ID rules, canonical sort, seed policy; emits `determinism_report.json`. |
| P06   | **Run-ledger compaction** → `EXPLODE_SUPERPIN:CONTROL@LEDGER_COMPACT@APPEND_ONLY?OUT=JSONL+MD&ITER=auto&STRICT=true` → produces compacted ledger without losing provenance. |
| P07   | **Plane table enforcement** → `EXPLODE_SUPERPIN:CONTROL@PLANE_TABLES?OUT=MD+CSV&ITER=auto&STRICT=true` → emits all artifacts as plane-table sections. |
| P08   | **Receipt hardening + SBOM** → `EXPLODE_SUPERPIN:CONTROL@RECEIPTS_STRICT@SBOM?OUT=JSON+MD&ITER=auto&STRICT=true` → emits CycloneDX SBOM + receipt integrity gates. ([Ecma International][5]) |
### HYPERVISOR_PLANE (P09–P15)
| Field | Plane Table Entry |
| ----- | ----------------- |
| P09   | **Full-plane convergence run** → `EXPLODE_SUPERPIN:HYPERVISOR@HYPERVISOR_ON@AUTONOMY_MAX@CONVERGENCE_EPS@VRPT_PASS_STREAK?EPS=0.0005&N=3&OUT=ZIP+MD+CSV+JSON&ITER=auto&STRICT=true` → routes tranches until `Δ<eps` for `N` cycles and `VRpt PASS` streak satisfied; emits tranche routing tables. |
| P10   | **Parallel tranche execution** → `EXPLODE_SUPERPIN:HYPERVISOR@SHARD_BY_DOC@BACKPRESSURE?OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → emits `TRANCHE_RUNS.csv`, `PARALLEL_TRACK_STATUS.json`, throttle events. |
| P11   | **Risk-based PO promotion** → `EXPLODE_SUPERPIN:HYPERVISOR@PROMOTE_NONCORE_TO_CORE?OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → emits promotion list + new CORE blockers + justification. |
| P12   | **Multimodal pool budgets** → `EXPLODE_SUPERPIN:HYPERVISOR@MULTIMODAL_POOLS@BACKPRESSURE?OUT=JSON+MD&ITER=auto&STRICT=true` → emits per-pool budget + queue depth. |
| P13   | **Stop-rule sweep** → `EXPLODE_SUPERPIN:HYPERVISOR@CONVERGENCE_EPS?EPS=0.001&N=5&OUT=MD+CSV+JSON&ITER=auto&STRICT=true` → evaluates stability across stricter EPS and longer streak; emits comparative metrics. |
| P14   | **Tranche-family prioritizer** → `EXPLODE_SUPERPIN:HYPERVISOR@AUTONOMY_MAX?PRIORITY=FORMS,DEADLINES,SERVICE,EXHIBITS,VALIDATION&OUT=MD+JSON&ITER=auto&STRICT=true` → biases routing to packaging-unblock planes first. |
| P15   | **Backlog→plane routing** → `EXPLODE_SUPERPIN:HYPERVISOR@AUTONOMY_MAX?BACKLOG=AUTHORITY_GAPS,FORM_GAPS,ORDER_GAPS&OUT=MD+JSON&ITER=auto&STRICT=true` → emits routing map + tranche queue plan. |
### HARVEST_PLANE (P16–P21)
| Field | Plane Table Entry |
| ----- | ----------------- |
| P16   | **Drive-universe discovery** → `EXPLODE_SUPERPIN:HARVEST@CANON_PATHS?ROOT=F:/CAPSTONE/Litigation_OS&OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → emits `paths.csv`, `file_index.json`, canonical map. |
| P17   | **Mirror-aware intake** → `EXPLODE_SUPERPIN:HARVEST@MIRROR_AWARE?ROOT=F:/CAPSTONE/Litigation_OS&MIRRORS=gdrive:/EDS-USB,gdrive:/Litigation_OS$,gdrive:/LITIGATION_INTAKE/&OUT=JSON+MD&ITER=auto&STRICT=true` → emits mirror equivalence classes + conflicts. |
| P18   | **Archive unpack tranche** → `EXPLODE_SUPERPIN:HARVEST@UNPACK_ARCHIVES@APPEND_ONLY?OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → unpacks to deterministic folders; emits unpack ledger. |
| P19   | **Type census** → `EXPLODE_SUPERPIN:HARVEST@FILETYPE_CENSUS?OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → emits extension/mime distribution and top offenders. |
| P20   | **Dedup working set** → `EXPLODE_SUPERPIN:HARVEST@DEDUPE_WORKING_SET?OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → groups duplicates while preserving originals. |
| P21   | **Change-set ledger** → `EXPLODE_SUPERPIN:HARVEST@CHANGESET_LEDGER@APPEND_ONLY?OUT=JSONL+MD&ITER=auto&STRICT=true` → emits changes since last run (add/modify/move). |
### EXTRACT_PLANE (P22–P28)
| Field | Plane Table Entry |
| ----- | ----------------- |
| P22   | **Atomic text extraction** → `EXPLODE_SUPERPIN:EXTRACT@ATOMIC_PARSING?OUT=JSONL+CSV+MD&ITER=auto&STRICT=true` → shards by page/paragraph/clause; emits `shards.jsonl`. |
| P23   | **Table extraction tranche** → `EXPLODE_SUPERPIN:EXTRACT@TABLES?OUT=CSV+JSONL+MD&ITER=auto&STRICT=true` → emits cell-level table captures + pointers. |
| P24   | **Metadata normalization** → `EXPLODE_SUPERPIN:EXTRACT@METADATA_NORMALIZE?OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → emits normalized doc headers + mime/time fields. |
| P25   | **QuoteRef compiler** → `EXPLODE_SUPERPIN:EXTRACT@QUOTELOCK?OUT=JSONL+MD&ITER=auto&STRICT=true` → enforces pointer+verification for verbatim quotes. |
| P26   | **Directive candidate scan** → `EXPLODE_SUPERPIN:EXTRACT@DIRECTIVE_SCAN?OUT=JSONL+MD&ITER=auto&STRICT=true` → extracts candidate commands/tags/obligations from text. |
| P27   | **Bi-temporal timeline build** → `EXPLODE_SUPERPIN:EXTRACT@BITEMP_TIMELINE?OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → emits event-time vs ingest-time timeline slices. |
| P28   | **Contradiction map seed** → `EXPLODE_SUPERPIN:EXTRACT@CONTRADICTION_SCAN?OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → emits conflict candidates across filings/orders/transcripts. |
### OCR_PLANE (P29–P32)
| Field | Plane Table Entry |
| ----- | ----------------- |
| P29   | **OCR gate evaluation** → `EXPLODE_SUPERPIN:OCR@OCR_GATE?OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → computes text-density/confidence; selects pages for OCR only. |
| P30   | **OCR run on gated pages** → `EXPLODE_SUPERPIN:OCR@OCR_RUN@APPEND_ONLY?OUT=JSONL+MD&ITER=auto&STRICT=true` → emits per-page OCR JSONL with pointers. |
| P31   | **OCR citation harvest** → `EXPLODE_SUPERPIN:OCR@CITATION_HARVEST?OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → extracts cite-like patterns into a reviewable index. |
| P32   | **OCR QA metrics** → `EXPLODE_SUPERPIN:OCR@OCR_QA?OUT=CSV+MD&ITER=auto&STRICT=true` → emits OCR quality measures and regressions by doc class. |
### NLP_PLANE (P33–P38)
| Field | Plane Table Entry |
| ----- | ----------------- |
| P33   | **Keyword glossary build** → `EXPLODE_SUPERPIN:NLP@GLOSSARY_INDEX?OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → emits canonical term IDs + synonym sets. |
| P34   | **Authority term normalizer** → `EXPLODE_SUPERPIN:NLP@TERM_NORMALIZE?DOMAIN=AUTHORITY&OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → normalizes MCR/MCL/MRE citation variants. |
| P35   | **Entity canonicalization** → `EXPLODE_SUPERPIN:NLP@ENTITY_CANON?OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → emits people/org/court dictionaries + stable IDs. |
| P36   | **Distilled proposition index** → `EXPLODE_SUPERPIN:NLP@KNOWLEDGE_DISTILL?OUT=JSONL+CSV+MD&ITER=auto&STRICT=true` → produces proposition tuples (claim→support→pointer). |
| P37   | **Semantic conflict detector** → `EXPLODE_SUPERPIN:NLP@CONFLICT_DETECT?OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → ranks contradictions and flags high-risk deltas. |
| P38   | **Hybrid lakehouse write (optional)** → `EXPLODE_SUPERPIN:NLP@LAKEHOUSE_WRITE@TIMETRAVEL?OUT=JSON+MD&ITER=auto&STRICT=true` → records time-sliced indices; keeps replay consistent. |
### VECTOR_PLANE (P39–P41)
| Field | Plane Table Entry |
| ----- | ----------------- |
| P39   | **Hybrid retrieval index build** → `EXPLODE_SUPERPIN:VECTOR@HYBRID_INDEX@RRF?OUT=JSON+MD&ITER=auto&STRICT=true` → builds lexical+dense with Reciprocal Rank Fusion. ([Qdrant][6]) |
| P40   | **ContextPack compiler** → `EXPLODE_SUPERPIN:VECTOR@CONTEXT_PACK?TOPK=25&OUT=JSON+MD&ITER=auto&STRICT=true` → emits graph-filtered pack + evidence pointers. |
| P41   | **Time-windowed retrieval** → `EXPLODE_SUPERPIN:VECTOR@TIME_SLICE?EVENT_START=2025-01-01&EVENT_END=2025-12-31&OUT=JSON+MD&ITER=auto&STRICT=true` → enforces bitemporal correctness for retrieval windows. |
### GRAPH_PLANE (P42–P46)
| Field | Plane Table Entry |
| ----- | ----------------- |
| P42   | **Schema contract emission** → `EXPLODE_SUPERPIN:GRAPH@SCHEMA_LOCK?OUT=JSON+MD&ITER=auto&STRICT=true` → emits immutable schema snapshot + migration rules. |
| P43   | **Constraints-first import pack** → `EXPLODE_SUPERPIN:GRAPH@CONSTRAINTS_FIRST@DETERMINISTIC_IMPORT?OUT=CYPHER+CSV+JSON+MD&ITER=auto&STRICT=true` → emits constraints + ordered CSV + import Cypher; blocks on constraint failure. ([Graph Database & Analytics][3]) |
| P44   | **Stratum focus build** → `EXPLODE_SUPERPIN:GRAPH@STRATUM_FOCUS?FOCUS=AUTHORITY,DECISIONS,ENFORCEMENT&OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → emits `node_stratum_map.csv` + focused view config. |
| P45   | **Graph enrichment pass** → `EXPLODE_SUPERPIN:GRAPH@ENRICH?OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → adds derived edges (SUPPORTS/CONFLICTS_WITH/DERIVES). |
| P46   | **Tranche flow overlay** → `EXPLODE_SUPERPIN:GRAPH@TRANCHE_FLOW?OUT=CSV+JSON+MD&ITER=auto&STRICT=true` → emits plane-to-plane routing graph for visualization. |
### NUCLEUS_PLANE / VIEWER_PLANE (P47–P50)
| Field | Plane Table Entry |
| ----- | ----------------- |
| P47   | **Nucleus seed generation** → `EXPLODE_SUPERPIN:GRAPH@NEO4J_NUCLEUS@NUCLEUS_SEED?OUT=JSON+MD&ITER=auto&STRICT=true` → emits seed rules + membership constraints; stop when membership stable + VRpt PASS. |
| P48   | **Mindeye2 offline viewer bundle** → `EXPLODE_SUPERPIN:RENDER@MINDEYE2@VIEWER_OFFLINE?OUT=HTML+JSON+MD&ITER=auto&STRICT=true` → emits offline HTML viewer + JSON config. |
| P49   | **Progressive disclosure view** → `EXPLODE_SUPERPIN:RENDER@PROGRESSIVE_DISCLOSURE@EDGE_BUNDLING?OUT=HTML+JSON+MD&ITER=auto&STRICT=true` → reduces hairball with bundling + strata lanes. |
| P50   | **Bloom perspective export/import** → `EXPLODE_SUPERPIN:RENDER@BLOOM_PERSPECTIVE_EXPORT@MANIFEST_VERIFY?OUT=JSON+MD&ITER=auto&STRICT=true` → emits Bloom perspective JSON for portable visual reuse. ([Graph Database & Analytics][4]) |
---
## GDrive-Folders-organizer-main.zip → Upgrade Blueprint (Plane Table)
| Field | Plane Table Entry |
| ----- | ----------------- |
| GD-01 | **Replace Colab mount** → `@DRIVE_BRIDGE_RCLONE` preferred; use rclone Drive backend for listing/sync; treat Drive shortcuts explicitly (`@SKIP_SHORTCUTS` default). ([Rclone][2]) |
| GD-02 | **Shortcuts correctness** → Drive shortcuts are `mimeType=application/vnd.google-apps.shortcut` with `shortcutDetails.targetId`; resolver must dereference before categorization. ([Google for Developers][8]) |
| GD-03 | **Safe categorization** → replace “move in-place” with **view-layer** categories: `copy to categorized view` or `index-only` to avoid destructive operations; all move/copy ops logged. |
| GD-04 | **Collision policy required** → introduce explicit `COLLISION=skip|rename_suffix|content_hash_name|conflicts_folder` and record outcome per file in `MOVE_LEDGER.csv`. |
| GD-05 | **Watcher lane** → if you add live sorting, use `watchdog` observers; include PollingObserver fallback when watching Windows drives from WSL contexts. ([python-watchdog.readthedocs.io][7]) |
| GD-06 | **LitigationOS-grade receipts** → every operation emits `RUN_LEDGER.jsonl`, `MANIFEST.json`, `VRpt.md`; packaging blocked on FAIL. |
---
# Options (Required) — Plane Tables
## Option 1 — Hypervisor: full-plane tranche convergence (Plane Table)
| Field                   | Plane Table Entry |
| ----------------------- | ----------------- |
| Next Best Action        | Execute convergence across all planes; autoprioritize unblockers: FORMS → DEADLINES → SERVICE → EXHIBITS → VALIDATION → GRAPH/VIEWER. |
| Enterprise/SPEC Pattern | `EXPLODE_SUPERPIN:HYPERVISOR@HYPERVISOR_ON@AUTONOMY_MAX@SHARD_BY_DOC@BACKPRESSURE@MULTIMODAL_POOLS@PROMOTE_NONCORE_TO_CORE@REPLAYABLE_RUN@MANIFEST_VERIFY@FAIL_CLOSED?EPS=0.0005&N=3&OUT=ZIP+MD+CSV+JSON&ITER=auto&STRICT=true` |
| Expected Outputs        | `CyclePack.zip`; `RUN_LEDGER.jsonl`; `MANIFEST.json`; `DELTA_SUMMARY.md`; `STRATUM_METRICS.csv`; `TRANCHE_RUNS.csv`; `PARALLEL_TRACK_STATUS.json`; `VRpt.md`. |
## Option 2 — Neo4j nucleus: schema contract + constraints-first + deterministic import + offline viewer (Plane Table)
| Field                   | Plane Table Entry |
| ----------------------- | ----------------- |
| Next Best Action        | Build nucleus slice and iterate until membership stabilizes; focus strata S2 Authority, S5 Decisions, S6 Enforcement; export Bloom perspective JSON for portability. ([Graph Database & Analytics][3]) |
| Enterprise/SPEC Pattern | `EXPLODE_SUPERPIN:GRAPH@NEO4J_NUCLEUS@SCHEMA_LOCK@CONSTRAINTS_FIRST@DETERMINISTIC_IMPORT@NUCLEUS_SEED@STRATUM_FOCUS@VIEWER_OFFLINE@EDGE_BUNDLING@PROGRESSIVE_DISCLOSURE@MANIFEST_VERIFY@FAIL_CLOSED?FOCUS=AUTHORITY,DECISIONS,ENFORCEMENT&OUT=ZIP+CSV+JSON+HTML+MD+CYPHER&ITER=auto&STRICT=true` |
| Expected Outputs        | `neo4j/schema_contract.json`; `neo4j/constraints.cypher`; `neo4j/import.cypher`; `graph/nodes.csv`; `graph/edges.csv`; `nucleus/seeds.json`; `viewer/index.html`; `VRpt.md`; `CyclePack.zip`. |
## Option 3 — Forms-first Vehicle Router: Relief → Form → Standard → Elements → POs → Deadlines → Service → Exhibits (Plane Table)
| Field                   | Plane Table Entry |
| ----------------------- | ----------------- |
| Next Best Action        | Run the forms-first router as the packaging gatekeeper; fail-closed if any CORE PO, deadline, service, or VRpt is uncertain; promote NONCORE→CORE when risk threshold triggers. |
| Enterprise/SPEC Pattern | `EXPLODE_SUPERPIN:FORMS@PCW@FORMSFIRST@VEHICLE_MAP@PO_DB_BUILD@DEADLINE_ENGINE@SERVICE_CHAIN@EXHIBIT_MATRIX@PROMOTE_NONCORE_TO_CORE@QUOTELOCK@REPLAYABLE_RUN@FAIL_CLOSED?PIPE=RELIEF>FORM>STANDARD>ELEMENTS>PO>DEADLINES>SERVICE>EXHIBITS&OUT=ZIP+MD+CSV+JSON&ITER=auto&STRICT=true` |
| Expected Outputs        | `VehicleMap.md`; `PO_DB.csv`; `Deadlines.csv`; `ServicePlan.md`; `ExhibitMatrix.csv`; `VRpt.md`; `CyclePack.zip`. |
[1]: https://json-schema.org/draft/2020-12/json-schema-core?utm_source=chatgpt.com "A Media Type for Describing JSON Documents"
[2]: https://rclone.org/drive/?utm_source=chatgpt.com "Google drive"
[3]: https://neo4j.com/docs/cypher-manual/current/clauses/load-csv/?utm_source=chatgpt.com "LOAD CSV - Cypher Manual"
[4]: https://neo4j.com/docs/bloom-user-guide/current/bloom-perspectives/perspective-creation/?utm_source=chatgpt.com "Creation and use - Neo4j Bloom"
[5]: https://ecma-international.org/publications-and-standards/standards/ecma-424/?utm_source=chatgpt.com "ECMA-424"
[6]: https://qdrant.tech/documentation/concepts/hybrid-queries/?utm_source=chatgpt.com "Hybrid Queries"
[7]: https://python-watchdog.readthedocs.io/en/stable/api.html?utm_source=chatgpt.com "API Reference — watchdog 2.1.5 documentation"
[8]: https://developers.google.com/workspace/drive/api/guides/shortcuts?utm_source=chatgpt.com "Create a shortcut to a Drive file"
**Interpretation rules (for your runner):**
* `MAX_CYCLES=0` = **unbounded** (runs until convergence stop condition is satisfied).
* `LEARN_MODE=carryforward` = each cycle’s outputs become the next cycle’s priors (prompt/context/kernel/state), with `REFINE=expand+compress` enforcing growth + distillation per cycle.
