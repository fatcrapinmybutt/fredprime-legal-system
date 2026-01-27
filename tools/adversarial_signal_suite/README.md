# Adversarial Signal Suite v2_2

This bundle provides a deterministic, append-only scanner for adversarial signals in legal document sets.

## CLI

```bash
python tools/adversarial_signal_suite/LITIGATIONOS_ADVERSARIAL_SIGNAL_SUITE_v2_2.py bootstrap --out ./BOOTSTRAP_BUNDLE
python tools/adversarial_signal_suite/LITIGATIONOS_ADVERSARIAL_SIGNAL_SUITE_v2_2.py scan --roots ./cases --out ./ADV --max-cycles 10 --stable-n 2 --eps 0.0 --neo4j-csv
python tools/adversarial_signal_suite/LITIGATIONOS_ADVERSARIAL_SIGNAL_SUITE_v2_2.py watch --roots ./cases --out ./ADV --poll-seconds 5
python tools/adversarial_signal_suite/LITIGATIONOS_ADVERSARIAL_SIGNAL_SUITE_v2_2.py merge-graphs --graph-dir ./graphs --out ./ADV
python tools/adversarial_signal_suite/LITIGATIONOS_ADVERSARIAL_SIGNAL_SUITE_v2_2.py package --bundle-root ./BOOTSTRAP_BUNDLE --zip ./BOOTSTRAP_BUNDLE_v2_2.zip
```

## Outputs
- `OUT/adversarial_events.jsonl` for append-only event records.
- `OUT/file_status.jsonl` for per-file scan status.
- `OUT/adversarial_summary.json` for per-run summaries.
- `RUN/convergence_report.json` for convergence cycle history.
- `NEO4J_IMPORT/*.csv` when `--neo4j-csv` is enabled.
