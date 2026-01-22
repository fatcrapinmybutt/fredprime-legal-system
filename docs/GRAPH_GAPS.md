# Graphic Gap Checklist (LitigationOS Graph) — Full Corpus

## Scope
This checklist enumerates **missing or under‑specified elements** relative to the MI‑only PoDP→ADD→PCG specification. It is organized to mirror the graph spine and end‑to‑end procedural lifecycle.

---

## A) Graph Spine + Semantics
- **SemanticEdge taxonomy** is not modeled as first‑class nodes/edges with explicit `type` and constraints. Required edge types include: `ENFORCES`, `INTERPRETS`, `SUPERSEDES`, `LIMITS`, `IMPLEMENTS`, `GUIDES`, `CONFLICTS`.  
- **Edge provenance** is missing (edge‑level authority + pinpoint and effective date) so derived logic is not auditable.  
- **Edge confidence/assurance** (ADD) is missing at the edge level (edge‑specific confidence + freshness).  
- **Edge conflict resolution** (CONFLICTS) lacks explicit conflict objects with precedence logic and “superseded‑by” trails.  
- **Snapshot anchoring** of all semantic edges is not explicit (graph should support historical snapshots).  

## B) Authority Layer (PoDP‑compliant)
- **Authority identity** fields missing: content hash, official source URI, effective date, and authoritative version ID.  
- **AuthoritySnapshot** is present in the graphic but lacks **AuthoritySource → AuthorityPinpoint → AuthorityAtom** chain requirements.  
- **Pinpoint granularity** is under‑specified (page/line/Bates/time offsets, selector prefix/suffix).  
- **Supersession chains** (SUPERSEDES/LIMITS) are not represented as causal sequences with effective date boundaries.  
- **Controlling‑authority flags** are missing (e.g., controlling MI orders vs persuasive materials).  
- **Authority scope gates** (MI‑only constraint) are not encoded as structural constraints.  

## C) Evidence + Facts (Proof‑Lock)
- **EvidenceAtom** is present but lacks explicit **artifact hash**, **integrity key**, and **origin artifact ID** linkage.  
- **Evidence foundation checks** (authenticity/chain of custody) are under‑modeled as pass/fail + proof obligations.  
- **Statement → Fact** transform edges are missing graded assurance and dispute state.  
- **Fact pinpoints** (path + pg/ln/Bates|time) are not explicitly enforced per Fact node.  
- **Contradiction objects** exist but lack explicit **resolution state**, **assurance effect**, and **blocked/allowed** flags.  
- **Redaction actions** exist but lack **why/authority** linkage and **downstream impact** on pinpoints.  

## D) Vehicles / Forms (Vehicle‑First Core)
- **Relief → Vehicle → Form** chain is not explicit as a dedicated graph path (required to drive all downstream steps).  
- **Vehicle prerequisites** are not modeled as enforceable constraints (e.g., “requires notice”, “requires prior order”).  
- **Vehicle standard/elements** are not structured as machine‑readable requirement sets (element → authority + pinpoint).  
- **Vehicle deadlines** and **service rules** are not explicitly attached to the vehicle path.  
- **Fallback/escalation vehicles** (deny‑aware counters) are not represented as alternate ProceduralPaths.  

## E) Procedural Paths + Tracks
- **ProceduralPath** entity is missing for end‑to‑end steps: relief → vehicle → rule/standard → prereqs → deadlines → service → exhibits → preserve → risks → fallback.  
- **Track** entity is missing for parallel lanes (trial/COA/JTC; federal overlay gated).  
- **Denial‑aware routing** is missing (denial reason → alternative path).  
- **Bi‑temporal timelines** (event/record/service) are not modeled as linked temporal tracks.  

## F) PCG Gates + Proof Obligations
- **ProofObligation** exists but lacks explicit **test definitions**, **required pins**, and **satisfaction conditions**.  
- **PCG gate** metadata is insufficient: missing irreversible action flags, gate dependencies, and escalation on FAIL/HOLD.  
- **GateResult** lacks **fixlist outputs** and explicit acquisition plans for missing proof.  
- **Gate coverage** for required categories (juris, service/notice, deadlines, fees/bonds, orders, preserve, contradictions, red‑team) is not explicitly enumerated.  

## G) Assurance (ADD)
- **Assurance bands** (A/B/C/D) are not modeled per Fact/Edge/Path.  
- **Freshness decay** is absent as a first‑class mechanism (timestamp + decay function).  
- **Economics tuning** (certainty vs cost) is missing as a decision attribute.  
- **Conflict‑surfacing** rules are absent (conflicts should be visible but not auto‑blocking).  

## H) ContextPack + CEA (RAG)
- **ContextPack** artifact is not defined (minimal pack of orders + key facts + controlling authority).  
- **CEA mapping** (claim → evidence → authority) is missing as a traceable structure.  
- **Decision trace** (non‑CoT) for derived conclusions is missing.  
- **Abort conditions** (proof packet cannot be produced) are not wired to gate evaluation.  

## I) Drafting + Packaging
- **DraftDoc** lacks explicit linkage to the ContextPack and CEA map that justified it.  
- **Packet** lacks explicit **record survival** and **appeal‑ready** validation metadata.  
- **Release** lacks explicit **PCG gate pass** requirements and proof packet linkage.  
- **Exhibit matrix** structure is missing (exhibits → evidence atoms → pinpoints).  

## J) Service Layer
- **ServicePlan** lacks explicit **method rules**, **address validity**, and **deadline** constraints.  
- **ServiceAttempt** lacks **notice compliance** checks and **defect remediation** pathing.  
- **ServiceProof** lacks authority‑pinpoint linkage to service rules and exhibits used.  

## K) Case State + Persistence
- **CASE_STATE** (≤25 lines) not represented as a distinct, immutable node.  
- **LEDGERΔ** (SoR, Exhibits, Timeline‑bitemp, AuthorityTriples, Contradictions, Deadlines) missing as structured sub‑ledgers.  
- **REGISTRY** append‑only IDs/pointers for reprints/diffs not modeled.  
- **State transitions** for HARVEST → ANALYSIS → FILING are not explicit.  

## L) Appeals + Denial Readiness
- **Adequate‑remedy analysis** is not represented as an object or gate.  
- **Record sufficiency** checks are not modeled as proof obligations.  
- **SoR lineage** (Statement of Record) is not encoded as a primary artifact.  
- **Denial counters** and “if denied → next steps” tracking are missing.  

## M) Operator Views (UX/Workflow)
- **What to file now** view is not represented (derived from PCG gate pass + vehicle paths).  
- **What’s missing** view is not represented (derived from proof obligations + gate failures).  
- **Appeal‑ready** view is not represented (SoR + record sufficiency + preservation).  
- **Denial‑response** view is not represented (conflict/denial nodes → fallback path).  

---

## Quick Map (What to add to the graphic)
1. **SemanticEdge** nodes/edges with provenance + confidence.  
2. **Authority identity** (hash/source/effective date) + **pinpoint enforcement**.  
3. **ProceduralPath/Track/Denial/Conflict** graph layer.  
4. **PCG + ProofObligation** enrichment (test criteria + required pins + fixlist).  
5. **ADD assurance** (bands + decay) for facts/edges/paths.  
6. **ContextPack + CEA** artifacts, linked to DraftDoc/Packet/Release.  
7. **CASE_STATE + LEDGERΔ + REGISTRY** persistence primitives.  
8. **Appeal‑ready** checks (SoR, adequate‑remedy, record sufficiency) + denial counters.  

---

## N) Field‑Level Gaps (Entity Attribute Checklist)
Use this as a **field‑level checklist** for rendering the graphic and validating schema completeness.

### N1) AuthoritySource / AuthoritySnapshot / AuthorityPinpoint
- **AuthoritySource** needs: `source_uri`, `publisher`, `effective_date`, `version_label`, `content_hash`, `jurisdiction`, `citation_style`, `is_controlling`.  
- **AuthoritySnapshot** needs: `snapshot_id`, `created_at`, `scope`, `source_ref`, `effective_window`, `superseded_by`.  
- **AuthorityPinpoint** needs: `page`, `line_start`, `line_end`, `bates`, `selector_prefix`, `selector_suffix`, `offset`, `media_fragment`.  

### N2) SemanticEdge
- **Edge** needs: `edge_type`, `source_node_id`, `target_node_id`, `authority_pin_id`, `effective_date`, `confidence`, `conflict_group_id`.  
- **Edge** must support **multi‑pin** support (multiple authority pinpoints per edge).  

### N3) EvidenceAtom / EvidenceItem / Statement / Fact
- **EvidenceAtom** needs: `artifact_id`, `content_hash`, `pinpoint`, `confidence`, `source_ref`, `extraction_method`, `integrity_key`.  
- **EvidenceItem** needs: `source_file`, `hash`, `received_at`, `auth_status`, `custody_notes`.  
- **Statement** needs: `speaker_id`, `context`, `time`, `source_ref`, `confidence`.  
- **Fact** needs: `fact_text`, `pinpoint`, `evidence_refs`, `assurance_band`, `dispute_state`.  

### N4) Contradiction / Conflict
- **Contradiction** needs: `type`, `left_fact_id`, `right_fact_id`, `impact_scope`, `resolution_state`, `assurance_delta`.  
- **Conflict** needs: `conflict_id`, `conflict_type`, `precedence_rule`, `authority_refs`, `status`.  

### N5) Vehicle / Form / Element
- **Vehicle** needs: `relief_type`, `form_id`, `rule_refs`, `element_list`, `prereq_list`, `deadline_rules`, `service_rules`.  
- **Form** needs: `form_id`, `form_title`, `revision_date`, `instructions_ref`, `required_fields`.  
- **Element** needs: `element_text`, `authority_pin_id`, `proof_obligations`.  

### N6) ProceduralPath / Track / Denial
- **ProceduralPath** needs: `path_id`, `relief`, `vehicle_id`, `stage_list`, `prereq_edges`, `deadline_edges`, `service_edges`.  
- **Track** needs: `track_type`, `jurisdiction`, `parallel_of`, `status`.  
- **Denial** needs: `denial_reason`, `authority_basis`, `counter_path_id`, `status`.  

### N7) ProofObligation / GateResult / PCG
- **ProofObligation** needs: `test_id`, `required_pin_ids`, `satisfied_by`, `status`, `severity`, `deadline`.  
- **GateResult** needs: `gate_id`, `result`, `reason_list`, `fixlist`, `acquire_plan`, `unsat_score`.  
- **PCG Gate** needs: `gate_category`, `irreversible_action`, `depends_on`, `hold_policy`.  

### N8) ContextPack / CEA / DraftDoc / Packet / Release
- **ContextPack** needs: `fact_ids`, `authority_ids`, `order_ids`, `snapshot_id`, `created_at`.  
- **CEA** needs: `claim_id`, `evidence_ids`, `authority_pin_ids`, `assurance_band`.  
- **DraftDoc** needs: `context_pack_id`, `cea_map_id`, `vehicle_id`, `trace_map_id`.  
- **Packet** needs: `packet_manifest`, `record_survival_checks`, `service_plan_id`.  
- **Release** needs: `pcg_gate_pass_id`, `proof_packet_refs`, `attestation`, `release_ts`.  

### N9) ServicePlan / ServiceAttempt / ServiceProof
- **ServicePlan** needs: `party_id`, `method`, `address_ref`, `deadline`, `rule_refs`.  
- **ServiceAttempt** needs: `attempted_at`, `carrier`, `tracking`, `result`, `defect_notes`.  
- **ServiceProof** needs: `proof_type`, `artifact_id`, `verified_by`, `verified_at`.  

### N10) CASE_STATE / LEDGERΔ / REGISTRY
- **CASE_STATE** needs: `case_id`, `current_stage`, `open_obligations`, `last_update`.  
- **LEDGERΔ** needs: `sor_delta`, `exhibit_delta`, `timeline_delta`, `authority_delta`, `deadline_delta`.  
- **REGISTRY** needs: `entry_id`, `artifact_ptr`, `hash`, `diff_ptr`, `append_only`.  

---

## O) Layers + Phases (End‑to‑End Lifecycle)
Use this as the **layered phase checklist** to ensure the graphic depicts the full pipeline.

### O1) L0 Storage / Eligibility (Roots)
- Canonical storage roots (F:/, D:/, Vault SSOT).  
- Eligibility checks for storage media and retention policy.  
- Artifact hash registry and integrity baseline.  

### O2) L1 Intake + Delta Harvest
- Intake manifest with file list, hashes, and timestamps.  
- Delta detection between intake sets (IntakeΔ).  
- Receipt artifacts (ROA/MiFile receipts) with provenance pins.  

### O3) L2 EvidenceAtom + Provenance
- Atomization rules (statement/event/quote -> atom).  
- Pointer reopen receipt for reproducibility.  
- IntegrityKey assignment and artifact chaining.  

### O4) L3 ChronoDB + QuoteDB
- Bi‑temporal timeline (event time + record time).  
- Quote candidates with verification workflow (QuoteLock).  
- Timeline conflict hooks (contradictions -> Conflict nodes).  

### O5) L4 AuthoritySnapshot + Vehicles
- MI‑only authority snapshot creation.  
- Vehicle map from SCAO forms and instructions.  
- Element/standard grids with authority pins.  

### O6) L5 Contracts (C2→C3)
- Field catalog superset (entity schemas).  
- Relationship superset (typed edges).  
- Neo4j constraints/indices (uniqueness).  

### O7) L6 Scoring + Gates (ADD + PCG)
- ADD assurance scoring and freshness decay.  
- ProofObligation lifecycle and satisfaction checks.  
- PCG gate execution (irreversible action lock).  

### O8) L7 Actions + Automation
- Orchestrator run definitions (Run/RunStep/RunEvent).  
- Self‑audit and metrics capture.  
- Gate‑driven action plan generation.  

### O9) L8 Packaging + Filing Packs
- Exhibit matrix + service chain assembly.  
- Record survival checks + appeal‑ready flags.  
- MiFile‑ready bundles with release attestation.  

---

## P) Extra Layers (Operational Hardening)
### P1) Deterministic Run IDs
- RunID = `case_id + intake_hash + utc_ts` with collision guard.  
- Embedded in filenames, manifests, and log entries.  

### P2) Provenance Chain Enforcement
- Every derived artifact references **source hash + byte offsets**.  
- Machine‑checked “no orphan artifacts” rule.  

### P3) Hashing Strategy
- File‑level hashes + chunk hashes for partial reprocessing.  
- Per‑exhibit integrity report.  

### P4) Intake Diffing
- IntakeΔ.json listing added/changed/deleted files.  
- Diff attached to run manifest.  

### P5) Service Chain Tracker
- Service events as timeline items.  
- Detect gaps that break proof of service.  

### P6) Deadlines Engine v2
- Rule‑driven deadline computation (MCR/MCL/MRE).  
- Trigger event → deadline → service dependency.  

### P7) Red‑Team Validation
- Adversarial checklist in ValidationReport.  
- Surface denial risks and missing pins.  

### P8) Record Survival
- Automatic appellate readiness checks.  
- SoR + adequate remedy + record sufficiency signals.  

---

## Q) Cross‑Cutting Requirements (Fail‑Closed + MI‑Only)
### Q1) MI‑Only Authority Gate
- All authority nodes must declare **jurisdiction=MI** and provenance to official MI sources.  
- Any non‑MI or federal authority must be explicitly flagged and blocked unless an override is recorded.  

### Q2) PINPOINT‑Required Claims
- Every claim/edge/element must include `authority_pin_id` or be marked `PINPOINT_MISSING`.  
- No draft/export/filing without complete pinpoint coverage for all asserted standards.  

### Q3) Fail‑Closed Gates
- Any unmet gate condition yields **FAIL** with a **FixList + Acquire Plan**.  
- HOLD state allowed only for reversible steps; irreversible actions require PASS.  

### Q4) Proof Packet Emission Rules
- Proof packets generated **only on citation/challenge** (PoDP).  
- Store proof packets with hash references, not embedded in primary artifacts.  

---

## R) Output Index Completeness (Master Indices)
These indices must be represented in the graphic as **deliverables** with lineage links to source artifacts.

### R1) Required Master Indices
- `SoR_ledger_master.csv`  
- `QuoteDB_master.jsonl`  
- `ChronoDB_master.csv`  
- `ExhibitMatrix_master.csv`  
- `ContradictionMap_master.csv`  
- `DeadlinesNotice_master.csv`  
- `AuthorityTriples_master.jsonl`  
- `VehicleMap_master.jsonl`  
- `FormRegistry_master.jsonl`  
- `ViolationDB_master.jsonl`  
- `DueProcessDB_master.jsonl`  
- `CanonJTC_master.jsonl`  
- `master_manifest.json`  
- `ValidationReport_master.md`  

### R2) Lineage + Integrity
- Each master index must reference **input CyclePack IDs** and source hashes.  
- Master manifest must include **completeness score** and validation results.  

---

## S) GraphRAG Memory + Retrieval (Context Discipline)
### S1) Indexing Discipline
- Chunking rules tied to authority/evidence types (no mixed source chunks).  
- Index versioning tied to AuthoritySnapshot IDs and run IDs.  
- Recency bias must be explicit and bounded (no silent overwrite).  

### S2) Retrieval Constraints
- Retrieval must enforce MI‑only scope and ignore out‑of‑scope sources.  
- Retrieval output must include source pins for **every** snippet.  
- Any missing pins triggers `PINPOINT_MISSING` and blocks claim creation.  

### S3) ContextPack Assembly
- ContextPack must be minimal: controlling authority + key facts + orders.  
- Explicit “excluded evidence” list for transparency.  
- ContextPack hash stored in DraftDoc and Packet.  

---

## T) Denial‑Aware Escalation Layer
### T1) Denial Reasons Catalog
- Standardized denial categories mapped to rules/standards.  
- Denial reason must link to authority pin(s).  

### T2) Counter‑Paths
- Each denial reason maps to at least one fallback vehicle path.  
- Counter‑path must declare additional proof obligations and deadlines.  

### T3) Appeal Tracks
- COA/JTC tracks must inherit all denial reasons and preserve SoR.  
- Appeal‑ready flags required before release.  

---

## U) Quality Gates + Validation Reports
### U1) ValidationReport Structure
- Gate pass/fail summary with proof obligation list.  
- Pinpoint coverage statistics per document and per claim.  
- Contradiction count + unresolved conflicts list.  

### U2) Fail‑Closed Reporting
- FixList must include **owner**, **source**, **required artifact**, **deadline**.  
- “No release” watermark if any gate is FAIL or HOLD.  

---

## V) Field‑Level Gaps (Extended)
### V1) QuoteDB
- `quote_id`, `source_artifact_id`, `pinpoint`, `verbatim_text`, `variant_type`, `confidence`, `verification_status`.  

### V2) ChronoDB
- `event_id`, `event_time`, `record_time`, `actor`, `location`, `evidence_refs`, `confidence`.  

### V3) ExhibitMatrix
- `exhibit_id`, `artifact_id`, `page_range`, `pinpoint_refs`, `hash`, `label_color`.  

### V4) DeadlinesNotice
- `deadline_id`, `trigger_event_id`, `rule_pin_id`, `computed_deadline`, `service_dependency`, `status`.  

### V5) AuthorityTriples
- `subject_id`, `predicate`, `object_id`, `authority_pin_id`, `effective_date`, `confidence`.  

---

## W) UI/Operator Artifacts + Views
### W1) Operator Dashboard Blocks
- **Now Filing**: list of PASS gates and ready vehicles.  
- **Missing Proof**: outstanding ProofObligations with deadlines.  
- **Denial Radar**: active denial risks and fallback paths.  
- **Appeal‑Ready**: SoR status + record sufficiency checks.  

### W2) Evidence Review Panels
- EvidenceAtom viewer with highlight + pinpoint reference.  
- Contradiction viewer with resolution state + impact.  
- Quote verification panel with variant comparison.  

### W3) Release Pack Preview
- Packet manifest + exhibit matrix preview.  
- Service plan summary + proof checklist.  
- Gate pass stamp + attestation summary.  

---

## X) Security + Audit + Compliance
### X1) Audit Chain Requirements
- Append‑only audit log for all runs and gate decisions.  
- Every run step references a RunID + hash of inputs.  

### X2) Access Control
- Role‑based access to authority ingestion and release steps.  
- Separate “author” vs “publisher” roles to enforce PCG.  

### X3) Redaction & Privacy
- Redaction actions must cite authority and be reversible until release.  
- Redaction log must map to affected pinpoints.  

---

## Y) Deployment & Ops Guardrails
### Y1) Environment Separation
- Distinct dev/stage/prod AuthoritySnapshots and index stores.  
- No cross‑environment artifact mixing.  

### Y2) Versioning Strategy
- Versioned schemas with migration rules per snapshot.  
- Master indices include schema version metadata.  

### Y3) Backups + Recovery
- Snapshot backups of canonical intel and master indices.  
- Recovery drills with hash‑verified restoration.  

---

## Z) CyclePack + Canonical Intel Consolidation
### Z1) CyclePack Minimum Bundle
- Must include all required index deltas + run logs + manifest.  
- Missing deliverables trigger FAIL and block canonical merge.  

### Z2) Canonical Merge Rules
- Merge must be deterministic (sorted by RunID + timestamp).  
- Conflict resolution requires authority pins and explicit reviewer.  

### Z3) Post‑Merge Validation
- Rebuild master indices and compare completeness scores.  
- Emit ValidationReport_master.md with diff summary.  

---

## AA) Data Contracts + Schema Governance
### AA1) Contract Registry
- Central registry for all JSON schemas and CSV headers.  
- Each contract version tagged with effective date and supersession rules.  

### AA2) Schema Validation
- All outputs must validate against their contract before merge.  
- Invalid outputs must fail the run and generate FixList entries.  

### AA3) Backward Compatibility
- Migration rules for schema version bumps.  
- Deprecation window tracking per contract.  

---

## AB) Neo4j / Graph Storage Constraints
### AB1) Uniqueness Constraints
- Unique IDs for Authority, EvidenceAtom, Fact, Vehicle, ProofObligation, Run.  
- Prevent duplicate AuthorityPinpoints within a snapshot.  

### AB2) Relationship Constraints
- Enforce required edge types for claims, standards, and evidence.  
- Disallow dangling nodes without authority or evidence lineage.  

### AB3) Indexing Strategy
- Composite indices for `case_id + run_id`.  
- Full‑text indices for authority and evidence text.  

---

## AC) Evidence Forensics + Authenticity
### AC1) Authenticity Checks
- Hash‑verified provenance for each EvidenceItem.  
- Chain verification before atomization.  

### AC2) Media Forensics
- EXIF/metadata capture for images/audio/video.  
- Timestamp consistency checks across artifacts.  

### AC3) Integrity Exceptions
- Exception log for missing hashes or mismatch.  
- Exception triggers FixList and blocks release.  

---

## AD) Service & Notice Compliance
### AD1) Notice Rules
- Notice requirements mapped to authority pins.  
- Notice deadlines tied to procedural path.  

### AD2) Service Defect Remediation
- Remediation path per service method.  
- Record of attempted cures and outcomes.  

### AD3) Service Proof Packaging
- Proof artifacts linked to ServicePlan and Release.  
- Proof metadata included in Packet manifest.  
