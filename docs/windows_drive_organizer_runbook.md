# Windows Drive Organizer Runbook

## Default Drive Policy

The Windows drive organizer scans only the Q, D, and Z drive roots by default. The default drive list is:

- `Q:/`
- `D:/`
- `Z:/`

These defaults are enforced in `cli/windows_drive_organizer.py`.

## C: Drive Exclusion

Drive roots that resolve to `C:` are explicitly rejected. This includes values like `C:/` or `C:\`, or any path that resolves to the C drive. The organizer fails closed if a `C:` root is detected.

## Preflight Validation Behavior

Before discovery begins, the organizer performs a preflight check:

- All three required drives (`Q:`, `D:`, `Z:`) must be present.
- Any missing required drive causes the run to stop immediately.
- Any drive root that resolves to `C:` causes the run to stop immediately.

## Example Usage

```bash
python cli/windows_drive_organizer.py --drives Q:/ D:/ Z:/
```
