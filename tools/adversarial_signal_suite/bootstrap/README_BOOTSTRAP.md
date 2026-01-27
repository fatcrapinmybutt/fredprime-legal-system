# BOOTSTRAP_BUNDLE (v2_2)

This folder is designed to be dropped into your LitigationOS schema and queries areas.

## Contents
- SCHEMA/doctype_registry.json
- SCHEMA/bucket_rules.json
- SCHEMA/event_to_mv_map.json
- SCHEMA/ADVERSARIAL_CONFIG_DEFAULT.json
- QUERIES/bumper_query_pack.cypher
- QUERIES/bumper_query_pack.meta.json

## Override patterns
1) Copy SCHEMA/ADVERSARIAL_CONFIG_DEFAULT.json to OUT_ROOT/ADVERSARIAL_CONFIG.json
2) Edit patterns/actors/synonyms
3) Re-run scan or watch
