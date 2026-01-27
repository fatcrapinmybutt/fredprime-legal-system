// BUMPER_QUERY_PACK v1 (Signal Events / Evidence Atoms / MV Mapping)
// Assumes you import NEO4J_IMPORT CSVs as:
//  (:EvidenceAtom {eaid, path, doctype, bucket, locator, snippet, ocr_needed})
//  (:SignalEvent {event_id, category, pattern_id, severity, weight, match_text, actor_tags, ts_utc})
//  (:MisconductVector {mv_id, name})
//  (ea)-[:HAS_EVENT]->(ev)
//  (ev)-[:MAPS_TO {w}]->(mv)

// Q1: Top categories by count
MATCH (ev:SignalEvent)
RETURN ev.category AS category, count(*) AS n
ORDER BY n DESC;

// Q2: Top HIGH severity events by weight
MATCH (ev:SignalEvent)
WHERE ev.severity = 'HIGH'
RETURN ev.category, ev.pattern_id, ev.weight, ev.match_text, ev.ts_utc
ORDER BY ev.weight DESC, ev.ts_utc DESC
LIMIT 50;

// Q3: Find evidence that mentions ex parte or no notice
MATCH (ea:EvidenceAtom)-[:HAS_EVENT]->(ev:SignalEvent)
WHERE ev.category IN ['EX_PARTE_OVERREACH','NOTICE_DEFECT']
RETURN ea.path, ea.locator, ea.snippet, ev.category, ev.pattern_id, ev.match_text
ORDER BY ea.path;

// Q4: MV heatmap (counts by MV)
MATCH (ev:SignalEvent)-[r:MAPS_TO]->(mv:MisconductVector)
RETURN mv.mv_id, mv.name, count(*) AS n, avg(r.w) AS avg_w
ORDER BY n DESC;

// Q5: Actor tag cross-tab (what roles show up in segments)
MATCH (ev:SignalEvent)
WITH split(coalesce(ev.actor_tags,''),'|') AS roles
UNWIND roles AS role
WITH role WHERE role <> ''
RETURN role, count(*) AS n
ORDER BY n DESC;
