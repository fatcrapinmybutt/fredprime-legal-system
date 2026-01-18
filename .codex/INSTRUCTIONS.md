# Codex Instructions (LitigationOS)

## Non-negotiables
- No skeletons/stubs/MVP. Ship fully working code only.
- Truth-Lock: no invented facts/authority; missing proof => PINPOINT_MISSING + acquisition plan.
- Michigan-first authority lock; forms-first VehicleMap; Proof-Carrying Workflow; final gate is FAIL-CLOSED.
- Open-source + offline-first: no third-party APIs unless self-hosted by us.
- Network calls allowed only through our own NetworkBroker; default=deny.

## Output contract (every cycle)
1) Increment VERSION (v0001+), update CURRENT (runnable), snapshot VERSIONS/vNNNN (immutable).
2) Update CHANGELOG + MANIFEST.
3) Run smoke tests and capture logs.
4) If >2 files change or any multi-module feature is added: build FULL release zip.
5) Enforce size policy: exclude large binaries/weights/media; use PATCHES mode if projected >650MB; report size budget when growth >50MB.

## Bundle root
BUNDLE_ROOT is fixed: `LITIGATIONOS__MASTERv1.0`.

## Build targets
- Windows x64 desktop application with state-of-the-art GUI.
- Bloom-style interactive graph explorer backed by Neo4j (open-source graph UI components).
- Local LLM/AI integration: prefer Ollama and/or Hugging Face local models; no hosted inference.
- Document pipeline: Apache Tika for extraction; rclone for storage/mirroring.

## Graph UI / Bloom policy
Default is “Bloom-style” using open-source graph components (Cytoscape.js/Sigma.js).
Official Neo4j Bloom is optional external only if licensed/installed; never hard-depend.

## External model weights policy
Never bundle LLM weights or large binaries in FULL zips. Store pointers in `ASSETS_EXTERNAL/asset_registry.json`.
Required fields: asset_id, kind(model|embedding|binary|dataset), local_paths[], expected_bytes(optional), sha256(optional), source_url(optional), notes(optional).
Builder must validate presence (and hash if provided) before enabling LLM features.

## Drive gate (C: excluded)
Implement one-time eligible drive discovery. Hard exclude: C:\ always.
Eligible letters (priority): F:, D:, Z:, Q:, E: then others except C:. Allow override via config.
Fail-closed with acquisition plan if no eligible root is available.

## Security boundary / Offline-by-default
Outbound network calls are forbidden by default (including package managers, model downloads, update checks).
All network activity requires explicit operator enablement: OnlineUpdateMode=true.
When enabled, log every outbound target + purpose and provide a “network off” kill switch.
