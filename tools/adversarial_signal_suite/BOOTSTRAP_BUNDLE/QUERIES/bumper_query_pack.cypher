// BUMPER_QUERY_PACK v1
// Top categories by count
MATCH (ev:SignalEvent)
RETURN ev.category AS category, count(*) AS n
ORDER BY n DESC;

// HIGH severity events
MATCH (ev:SignalEvent)
WHERE ev.severity = 'HIGH'
RETURN ev.category, ev.pattern_id, ev.weight, ev.match_text, ev.ts_utc
ORDER BY ev.weight DESC, ev.ts_utc DESC
LIMIT 50;

// Evidence with ex parte or notice defects
MATCH (ea:EvidenceAtom)-[:EVIDENCE_HAS_EVENT]->(ev:SignalEvent)
WHERE ev.category IN ['EX_PARTE_OVERREACH', 'NOTICE_DEFECT']
RETURN ea.path, ea.locator, ea.snippet, ev.category, ev.pattern_id, ev.match_text
ORDER BY ea.path;

// MV heatmap
MATCH (ev:SignalEvent)-[r:MAPS_TO]->(mv:MisconductVector)
RETURN mv.mv_id, mv.name, count(*) AS n, avg(r.w) AS avg_w
ORDER BY n DESC;

// Actor tag cross-tab
MATCH (ev:SignalEvent)
WITH split(coalesce(ev.actor_tags,''),'|') AS roles
UNWIND roles AS role
WITH role WHERE role <> ''
RETURN role, count(*) AS n
ORDER BY n DESC;
