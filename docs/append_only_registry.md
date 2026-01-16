# Append-Only Registry Template

This registry is designed to be append-only. Every new entry is written as a new
record and never overwrites or deletes prior entries. Use it to track evidence,
authority, and outputs without mutating history.

## Record format (JSONL)

Each line is a standalone JSON object. Required fields are shown below.

```json
{"registry_id":"REG-20240101-0001","kind":"Evidence","pointer":"evidence/Exhibit_A.pdf","hash_sha256":"<sha256>","bates_id":"LIT-000001","captured_at":"2024-01-01T12:00:00Z","note":"Original intake copy"}
```

### Required fields

* `registry_id` — Unique immutable ID for the record.
* `kind` — One of: `Evidence`, `Authority`, `Filing`, `Output`, `Contradiction`, `Deadline`.
* `pointer` — Stable path or locator (file path, URL, or internal object ID).
* `hash_sha256` — SHA-256 hash for file-based artifacts; if not applicable, set to `""`.
* `bates_id` — Bates label if assigned; otherwise `""`.
* `captured_at` — ISO 8601 UTC timestamp.
* `note` — Short, factual note describing the record.

### Optional fields

* `source_pinpoint` — Evidence pinpoint (page/line/timecode) for facts.
* `law_pinpoint` — Authority pinpoint (section/subsection/effective date).
* `related_ids` — Array of related registry IDs.

## Append-only rules

1. Never edit prior lines. Add a new record for updates or corrections.
2. Use a new `registry_id` for each update; link to prior items in `related_ids`.
3. Keep facts and authority separate; link them through `related_ids`.
4. Do not insert placeholders. If data is missing, leave the field empty and add a clear `note`.
