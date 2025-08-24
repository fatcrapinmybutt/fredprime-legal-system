-- Top evidence by signal
SELECT filename, filepath, relevance_score
FROM evidence ORDER BY relevance_score DESC LIMIT 100;

-- All timeline events (chronological)
SELECT event_dt, actor, action, details
FROM timelines WHERE event_dt <> '' ORDER BY event_dt ASC;

-- Items that reference IIED or Abuse of Process
SELECT filename, claims_json
FROM evidence
WHERE claims_json LIKE '%IIED%' OR claims_json LIKE '%abuse of process%';

-- Statutes/rules used most
SELECT statutes_json, COUNT(*)
FROM evidence GROUP BY statutes_json ORDER BY COUNT(*) DESC LIMIT 20;
