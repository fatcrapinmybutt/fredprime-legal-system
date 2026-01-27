# Adversarial Signal Suite

This tool scans document sets for adversarial, rights-violation, and negative-statement patterns.
It is append-only and does not modify source files. Use the bootstrap bundle to seed schema and
query packs in downstream systems.

## Quick start

```bash
python tools/adversarial_signal_suite/adversarial_signal_suite.py bootstrap --out ./BOOTSTRAP_BUNDLE
python tools/adversarial_signal_suite/adversarial_signal_suite.py scan --roots /data --out /data/harvest/ADV
```

## Outputs
- OUT/adversarial_events.jsonl: append-only match events
- OUT/adversarial_summary.json: summary snapshot
- RUN/run_ledger.jsonl: run progress ledger
- RUN/provenance_index.json: run metadata
- RUN/convergence_report.json: convergence history
- NEO4J_IMPORT/*.csv: optional graph import files

## Bootstrap bundle
The repo includes a prebuilt bundle at `tools/adversarial_signal_suite/bootstrap` with schema
and query packs. You can also emit a fresh bundle with the `bootstrap` command.
