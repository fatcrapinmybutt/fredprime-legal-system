# FRED PRIME Litigation Deployment System

This repository demonstrates how to automate litigation tasks offline using PrivateGPT. A helper PowerShell script is provided for Windows users to set up the environment and launch the app.

## Features
- Auto-label exhibits
- Link motions to matching exhibits
- Validate MCR 1.109(D)(3) signature block compliance
- Build parenting time violation matrices
- Track false reports and PPO misuse
- Log judicial irregularities

## Quick Start
Run the setup script from a PowerShell terminal:

```powershell
./privategpt_setup.ps1
```

Install Python requirements if you plan to use the Google Drive sync tool:

```bash
pip install -r requirements.txt
```

### Offline vs Online Use

Most utilities in this repo work entirely offline. The Google Drive upload feature in `storage_sync.py` is optional and will gracefully degrade if the required packages are not installed or network access is unavailable.

By default this installs the application into `C:\privategpt` and launches it with the `settings-local.yaml` configuration. You can override the install path or model name:

```powershell
./privategpt_setup.ps1 -InstallPath "D:\custom_dir" -Model "phi3"
```


## Form Database Example
A basic script is provided to demonstrate how court forms can be stored and queried locally.

Each form entry can include references to relevant MCR sections, statutes, and benchbook guidance. The example manifest now contains several common family court motions.

Run the importer:

```bash
python src/form_db.py --db forms.db --manifest data/forms_manifest.json --forms-dir forms
```

To look up a form by ID:

```bash
python src/form_db.py --db forms.db --get MC-12
```

List all stored forms:

```bash
python src/form_db.py --db forms.db --list
```

Search for forms containing a keyword across titles, IDs, and rule references:

```bash
python src/form_db.py --db forms.db --search custody
```

Export all forms to a standalone JSON file:

```bash
python src/form_db.py --db forms.db --export all_forms.json
```

### Automatically scrape forms
If internet access is available, run `scao_scraper.py` to download the latest SCAO
form list and save it locally. Any network errors are caught so the command can
be executed offline.

```bash
python src/scao_scraper.py --out data/scao_forms.json
```

The resulting JSON can be imported with `form_db.py` just like the provided
manifest.

This loads form metadata from `data/forms_manifest.json` and saves it into a SQLite database.

## Storage Sync
The `storage_sync.py` utility can scan a directory and optionally upload files to Google Drive.

If the Google API libraries are missing, uploads are skipped and a message is shown so the tool can still be used offline for hashing.

Install dependencies:

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

Scan a folder and output file hashes:

```bash
python src/storage_sync.py --scan F:\
```

Upload a single file to Google Drive (requires OAuth credentials in `credentials.json`):

```bash
python src/storage_sync.py --upload example.txt
```

## Knowledge Store and Evidence Analysis

`knowledge_store.py` maintains a local SQLite database linking evidence to forms. `evidence_analysis.py` scans this store and suggests motions using the cross‑references.

Add evidence and link it to a form:

```bash
python src/knowledge_store.py --add-evidence path/to/file.txt --desc "Custody dispute text"
python src/knowledge_store.py --link 1:FOC-65
```

List links and generate suggestions:

```bash
python src/knowledge_store.py --list
python src/evidence_analysis.py --knowledge knowledge.db
```
