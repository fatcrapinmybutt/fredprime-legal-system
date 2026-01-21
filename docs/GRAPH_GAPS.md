# Graph Gap Checklist (LitigationOS Graph)

## Missing or Under-specified Components (Relative to Full Spec)
- **SemanticEdge layer** for typed graph relationships (ENFORCES, INTERPRETS, SUPERSEDES, LIMITS, IMPLEMENTS, GUIDES, CONFLICTS).  
- **ProceduralPath/Track entities** to represent vehicle-first pathing, parallel tracks, and escalation routes.  
- **Conflict/Denial nodes** to encode contradictions, denials, and counter-paths for denial-aware routing.  
- **Assurance scoring (ADD)** structures for confidence bands, freshness decay, and conflict surfacing.  
- **PoDP identity fields** (authority content hash, official source, effective date) and provenance linkage.  
- **ContextPack + CEA artifacts** (claim → evidence → authority mapping and decision trace).  
- **PCG execution gate metadata** beyond current gate result node (explicit proof obligations, irreversible action gating).  
- **Case-state persistence** primitives (CASE_STATE, LEDGERΔ, REGISTRY append-only IDs/pointers).  
- **Operator views** for “what to file / what’s missing / denial / appeal-ready” status.  
