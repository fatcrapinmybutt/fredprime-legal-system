# Operations Runbook

## Windows Drive Organizer

`cli/windows_drive_organizer.py` scans a Windows drive while skipping high-risk/system
folders by default. The denylist is matched against directory names (case-insensitive)
and excludes the following directories and their children from discovery:

- `Windows`
- `Program Files`
- `Program Files (x86)`
- `$Recycle.Bin`
- `System Volume Information`
- `node_modules`
- `.git`
- `__pycache__`
- `venv`
- `dist`
- `build`

### Override or Extend the Denylist

Use `--deny-dirs` to append additional directory names. Provide a comma-separated list
or repeat the flag as needed. To replace the default denylist entirely, include `none`
in the list.

```bash
python cli/windows_drive_organizer.py F:/ --deny-dirs "Temp,Cache"
python cli/windows_drive_organizer.py F:/ --deny-dirs "none,CustomSafe"
```
