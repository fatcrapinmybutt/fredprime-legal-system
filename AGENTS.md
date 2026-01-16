# agents.md - LitigationOS Build Charter (Windows x64)

## 1) Mission
Build a Windows x64 Litigation Operating System with a polished GUI, a Bloom-style interactive Neo4j graph (open-source implementation), and an offline-first LLM/AI toolchain that produces judicial-grade, Michigan-locked outputs under Proof-Carrying Workflow gates.

## 2) Non-negotiables (hard gates)
### Legal/truth gates
- Michigan-first authority lock: MCR/MCL/MRE, MJI benchbooks, SCAO MC/FOC forms and instructions, controlling orders.
- Truth-Lock: no invented facts or authority. Any proposition without a pinned pointer is DISPUTED; missing items become PINPOINT_MISSING with an acquisition plan.
- Forms-first VehicleMap: choose the mandated form/vehicle before drafting; map relief->vehicle/form->rule/standard->proof obligations (prereqs, deadlines, service, fees/bonds, controlling orders)->exhibits->risks->fallback.
- Proof-Carrying Workflow (PCW): every filing-capable output is blocked until all mandatory Proof Obligations are SATISFIED; final irreversible actions pass PCG FAIL-CLOSED.

### Engineering gates
- No skeletons/stubs/MVP. Deliver runnable, fully populated implementations only.
- Offline-first and open-source: no third-party APIs unless self-hosted by us. Prefer local services and reproducible builds.
- Deterministic paths; exclude C drive by default (auto-detect eligible drives).
- Never ask the operator to patch/merge; deliver complete files.

### Packaging gates
- Bundle root is fixed: LITIGATIONOS__MASTERv1.0.
- VERSION is monotonic (v0001+). CURRENT is runnable. VERSIONS are immutable.
- Provide a FULL release zip when >2 files change or any multi-module feature ships.
- Size cap: FULL zip for the conversation <=700MB. Exclude weights/media/binaries by default; use PATCHES when projected >650MB; emit size budget when growth >50MB.

## 3) System architecture (reference implementation)
### Desktop app
- GUI: Electron + React + TypeScript.
- IPC: desktop app launches and supervises local services; no remote dependencies.
- Settings: theme packs, color palettes, layout presets, keyboard-driven workflows, per-case profiles, safety toggles (Truth-Lock strictness, PCW thresholds), and audit visibility.

### Local services
- Core service: Python 3.11+ (FastAPI or stdio-based IPC) hosting the workflow engines (Harvest, Graph, Proof, Packaging).
- Neo4j: local Neo4j Community managed by the app (Docker optional). Provide migrations, constraints, and seed data.
- Extraction: Apache Tika (local server or embedded) for PDFs/DOCX/HTML/text.
- Storage/mirroring: rclone for Google Drive and local drive synchronization (operator-controlled).
- LLM: Ollama for local inference (GGUF model format for local inference); optional Hugging Face local models (no hosted calls). Weights are external assets and must not be bundled inside FULL zips.

### UI surface areas (must exist)
1) Intake/Harvest dashboard: queue, progress, errors, deterministic output roots.
2) Graph Explorer (Bloom-style): search, filters, perspectives, styling, hover cards, click-through to authority text/exhibit pointers.
3) Proof Console: Proof Obligations by vehicle, status (OPEN/PARTIAL/SATISFIED), and evidence/authority links.
4) Form/Vehicle Library: SCAO forms mapped to governing rules and prerequisites.
5) Output Studio: judicial-grade exports (forms filled, filings assembled) only when gates PASS.
6) Audit/Preservation: logs, manifests, version history, contradiction map, deadlines.

## 4) Graph model (minimum viable canon)
### Nuclei (3-5 perspectives)
- Nucleus A: Courts and jurisdictions (Muskegon small claims -> 14th Circuit -> COA -> MSC -> JTC; federal overlay nodes allowed but gated).
- Nucleus B: Authority and forms (SCAO/FOC/MC forms <-> MCR/MCL/MRE propositions; benchbook references).
- Nucleus C: Case/evidence (sources of record, exhibits, timeline events, service/notice, orders).
- Optional D: Vehicles/Remedies (VehicleMap, standards, burdens, deadlines).
- Optional E: Operations (runs, bundles, manifests, QA outcomes).

### Required node types (labels)
Court, Jurisdiction, Judge, Form, Vehicle, Authority, AuthorityChunk, Proposition, ProofObligation, Deadline, Case, Order, Filing, Event, Source, Exhibit, QuotePin, Contradiction, Run, Bundle.

### Required edge types
GOVERNS(Form/Vehicle->Authority/Proposition), REQUIRES(Vehicle->ProofObligation), SATISFIED_BY(ProofObligation->QuotePin/Exhibit/AuthorityChunk), ISSUED_BY(Order->Court/Judge), FILED_IN(Filing->Court), SUPPORTS(Exhibit/QuotePin->Proposition), CONFLICTS_WITH(QuotePin<->QuotePin), TRIGGERS(Event->Vehicle/Deadline), DERIVES_FROM(Artifact->Source).

### Pinpoint object (must be standardized)
EvidencePinpoint = {source_path: string, page_or_timecode: int|string, bates_or_hash_optional: string|null, captured_at: datetime, note: string}.
LawPinpoint = {authority_id: string, section/subsection: string, effective_date: date, chunk_pointer: string}.

## 5) Agent responsibilities and rules of engagement
### Roles
- Architect: enforces gates, repo layout, contracts, and end-to-end cohesion.
- UI Agent: implements GUI, theming, settings, graph explorer, and operator workflows.
- Graph Agent: Neo4j schema, migrations, constraints, seed packs, traversal APIs.
- Harvest Agent: rclone integration, Tika extraction, deterministic intake, dedupe, manifests.
- LLM Agent: Ollama/HF local adapters, prompt templates, retrieval hooks, hallucination containment.
- Proof/Forms Agent: VehicleMap library, ProofObligation templates, SCAO form overlays, gating logic.
- QA/Release Agent: tests, reproducibility, size policy, installer/zip releases.

### What agents MUST output per change
- Working code, runnable locally.
- Updated manifest + changelog entries.
- Smoke test proof (log excerpt or test output).
- If feature spans multiple modules: FULL zip release via TOOLS/bundle_builder.py.

## 6) Build, test, release (minimum)
- `audit`: lint + static checks + unit tests + integration smoke test (start services, connect Neo4j, run one harvest, render graph, verify PCW blocks unsafe exports).
- `continue--package`: produce Windows x64 installer or portable zip; include versioned release notes; do not bundle model weights.

## 7) Licensing and supply-chain hygiene
- Only use reputable open-source dependencies. Record licenses in a dependency ledger.
- No blog-scraped code without provenance. Prefer GitHub upstream repos and Hugging Face model cards.
- No telemetry by default. All network features are opt-in and operator-controlled.

## 8) Definition of done for each milestone
Milestone PASS requires:
- GUI boots, services start, Neo4j connects, harvest runs, graph renders, proofs display, export is blocked until obligations are satisfied, and release zip builds under size policy.
