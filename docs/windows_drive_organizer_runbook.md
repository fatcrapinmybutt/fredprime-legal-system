# Windows Drive Organizer Runbook

This runbook explains how to install dependencies, configure branch expansions, and execute `cli/windows_drive_organizer.py` on Windows hosts.

## 1. Install dependencies

```powershell
# PowerShell (Windows)
python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements.txt
```

```bash
# Git Bash / WSL / macOS
python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements.txt
```

The organizer uses only standard-library modules by default, but these commands ensure the rest of Litigation OS requirements are present in the environment.

## 2. Prepare branches and drives

* Required drive roots: `Q:/`, `D:/`, and `Z:/` on Windows (the organizer fails closed if any are missing).
* C: is disallowed by default for both scan roots and output/temp paths (use `--allow-c-drive` only if you must override policy).
* To add dedicated branch folders (for case-specific review trees), use either `--branch LABEL=PATH` or a JSON file via `--branches-file`.
* Example JSON file:

```json
[
  {"label": "CASE_ALPHA", "path": "R:/CaseAlpha"},
  {"label": "CASE_BRAVO", "path": "\\\\fileserver\\Evidence\\Bravo"}
]
```

## 3. Run commands

```powershell
# Dry run with default drives plus a custom branch
py -3 cli\windows_drive_organizer.py \
    --dry-run \
    --branch CASE_ALPHA=R:/CaseAlpha \
    --output-root Z:/LitigationOS/Runs

# Live run with SQLite indexing, MiFILE packaging, and branch config file
py -3 cli\windows_drive_organizer.py \
    --sqlite-index \
    --mifile-ready \
    --branches-file C:\cases\branches.json \
    --output-root Z:/LitigationOS/Runs
```

```bash
# WSL/Linux
python cli/windows_drive_organizer.py \
    --drives /mnt/f /mnt/d \
    --branch CASE_MERIDIAN=/data/meridian \
    --output-root /evidence/OUTPUT
```

## 4. Outputs

Artifacts land under the output root you supply (default base `Z:/LitigationOS/Runs/<RUN_ID>`):

* `COLLECTED/<extension>/...` — evidence copies, including `BRANCHES/<label>/...` folders when branch expansions are used.
* `LOGS/drive_organizer*.log` and `LOGS/drive_organizer.jsonl` — rotating UTC text logs and structured JSONL entries.
* `manifest_<timestamp>_<token>.json` and `.csv` — manifests capturing branch labels, Bates IDs, hashes, and statuses.
* `checksums_<timestamp>_<token>.sha256` — sorted SHA-256 list.
* `bundle_<timestamp>_<token>.zip` — deterministic archive of collected files.
* Optional: `mifile_<timestamp>_<token>.zip`, SQLite index, and secret findings reports.

Review the manifests for summary counts (scanned/copied/errors/branches) and confirm exit codes (0 for success, >0 for warnings/errors).

## 5. Output and temp root policy

* The default output root base is `Z:/LitigationOS/Runs`, with a run-specific subfolder created automatically.
* Temp staging defaults to `Z:/LitigationOS/_TMP` and is cleaned after each run.
* To override temp staging, use `--temp-root` (avoid C: unless `--allow-c-drive` is set).

## 6. Denylisted directories

* The scanner skips common system directories by default (e.g., `Windows`, `Program Files`, `.git`, `node_modules`).
* To override or extend the denylist, use `--deny-dirs`.
