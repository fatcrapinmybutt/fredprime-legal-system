# agents.md — LitigationOS Build Charter (Windows x64)

## 1) Mission
Build a Windows x64 Litigation Operating System with a polished GUI, a Bloom-style interactive Neo4j graph (open-source implementation), and an offline-first LLM/AI toolchain that produces judicial-grade, Michigan-locked outputs under Proof-Carrying Workflow gates.

## 2) Non-negotiables (hard gates)
### Legal/truth gates
- Michigan-first authority lock: MCR/MCL/MRE, MJI benchbooks, SCAO MC/FOC forms and instructions, controlling orders.
- Truth-Lock: no invented facts or authority. Any proposition without a pinned pointer is DISPUTED; missing items become PINPOINT_MISSING with an acquisition plan.
- Forms-first VehicleMap: relief->vehicle/form->rule/standard->proof obligations (prereqs, deadlines, service, fees/bonds, controlling orders)->exhibits->risks->fallback.
- Proof-Carrying Workflow (PCW): every filing-capable output is blocked until mandatory Proof Obligations are SATISFIED; final irreversible actions pass PCG FAIL-CLOSED.

### Engineering gates
- No skeletons/stubs/MVP. Deliver runnable, fully populated implementations only.
- Offline-first and open-source: no third-party APIs unless self-hosted by us. Prefer local services and reproducible builds.
- Deterministic paths; exclude C: by default (auto-discover eligible drives).
- Never ask the operator to patch/merge; deliver complete files.

### Packaging gates
- Bundle root is fixed: LITIGATIONOS__MASTERv1.0.
- VERSION is monotonic (v0001+). CURRENT is runnable. VERSIONS are immutable.
- Provide FULL release zip when >2 files change or any multi-module feature ships.
- Size cap: FULL zip <=700MB. Exclude weights/media/binaries by default; use PATCHES when projected >650MB; emit size budget when growth >50MB.

## 3) System architecture (reference implementation)
### Desktop app
- GUI: Electron + React + TypeScript.
- IPC: desktop app launches and supervises local services; no remote dependencies.
- Settings: theme packs, color palettes, layout presets, per-case profiles, Truth-Lock strictness, PCW gates, audit visibility.

### Local services
- Core service: Python 3.11+ (FastAPI or stdio IPC) hosting workflow engines (Harvest, Graph, Proof, Packaging).
- Neo4j: local Neo4j Community managed by the app (Docker optional). Provide migrations, constraints, seed data.
- Extraction: Apache Tika (local server or embedded) for PDFs/DOCX/HTML/text.
- Storage/mirroring: rclone for Google Drive and local drive sync (operator-controlled).
- LLM: Ollama local inference; optional Hugging Face local models (no hosted inference). Weights external only.

### UI surface areas (must exist)
1) Intake/Harvest dashboard
2) Graph Explorer (Bloom-style)
3) Proof Console (POs + evidence/authority links)
4) Form/Vehicle Library
5) Output Studio (gated exports)
6) Audit/Preservation (logs, manifests, contradictions, deadlines)

## 4) Graph model (minimum viable canon)
### Nuclei (3–5 perspectives)
- A: Courts/Jurisdictions
- B: Authority/Forms
- C: Case/Evidence
- D: Vehicles/Proof
- E: Operations/Runs

### Required node types
Court, Jurisdiction, Judge, Form, Vehicle, Authority, AuthorityChunk, Proposition, ProofObligation, Deadline, Case, Order, Filing, Event, Source, Exhibit, QuotePin, Contradiction, Run, Bundle.

### Required edge types
GOVERNS, REQUIRES, SATISFIED_BY, ISSUED_BY, FILED_IN, SUPPORTS, CONFLICTS_WITH, TRIGGERS, DERIVES_FROM.

### Pinpoint object
EvidencePinpoint={source_path,page_or_timecode,bates_or_hash_optional,captured_at,note}
LawPinpoint={authority_id,section/subsection,effective_date,chunk_pointer}

## 5) Agent roles
- Architect: repo layout, contracts, cohesion.
- UI Agent: GUI + theming + workflows.
- Graph Agent: Neo4j schema + migrations + seed packs.
- Harvest Agent: rclone + Tika + intake + manifests.
- LLM Agent: Ollama/HF adapters + prompt templates + safety gates.
- Proof/Forms Agent: VehicleMap + ProofObligations + SCAO overlays.
- QA/Release Agent: tests + reproducibility + size policy + release zips.

## 6) Build/test/release (minimum)
- audit: lint + unit tests + integration smoke test (start services, connect Neo4j, run harvest, render graph, verify PCW blocks unsafe exports).
- continue--package: produce Win x64 installer or portable zip; include versioned release notes; do not bundle model weights.

## 7) Licensing and supply-chain hygiene
- Use reputable open-source dependencies only. Record licenses in a dependency ledger.
- No blog-scraped code without provenance. Prefer GitHub upstream repos and HF model cards.
- No telemetry by default. All network features are opt-in and operator-controlled.

## 8) Definition of done
GUI boots, services start, Neo4j connects, harvest runs, graph renders, proofs display, export blocked until obligations satisfied, release zip builds under size policy.
