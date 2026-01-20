# Windows Drive Organizer Runbook

## Output-root policy

- The organizer writes **only** under the configured `--output-root`.
- The default output root is on `Z:` to avoid accidental writes to `C:`.
- Any `output_root` on `C:` is rejected unless you explicitly pass `--allow-c-drive`.
- `TEMP` staging and `LOGS` directories are always created under the chosen output root.

### Default output root

If `--output-root` is not provided, the organizer uses:

```
Z:/LitigationOS/Runs/<RUN_ID>
```

`<RUN_ID>` is generated from the current UTC timestamp unless `--run-id` is provided.

### Examples

Use the safe default (creates `TEMP` and `LOGS` under Z:):

```
python cli/windows_drive_organizer.py
```

Specify a safe output root on `Z:`:

```
python cli/windows_drive_organizer.py --output-root "Z:/LitigationOS/Logs"
```

Override the C: drive guard (only when explicitly required):

```
python cli/windows_drive_organizer.py --output-root "C:/LitigationOS/Logs" --allow-c-drive
```
