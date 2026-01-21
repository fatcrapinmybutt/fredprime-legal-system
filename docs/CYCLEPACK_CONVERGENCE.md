# CyclePack Convergence Runbook (Local‑Only)

## Constraints (Current Job)
- **No Drive connector available** in this environment.  
- **No local CyclePack inputs** are present in this repo.  
- **PINPOINT_MISSING:** consolidation/harvest scripts referenced in user guidance are **not** present in this repository.  

---

## Required Local Steps (Fail‑Closed)
Use these steps on your local environment where Drive + tools are available.

### A) Discover New Intake (Drive)
```
rclone lsl gdrive:/LITIGATION_INTAKE/ --max-depth 5 | sort
rclone lsf gdrive:/LITIGATION_INTAKE/ --recursive --files-only ^
  --include "*CYCLEPACK*.zip" --include "*HARVEST*.zip" --include "*cyclepack*.zip" --include "*harvest*.zip"
```

### B) Pull Intake + CyclePacks (Append‑Only)
```
$env:LITIGATIONOS_HOME="F:\LitigationOS"
New-Item -ItemType Directory -Force -Path "$env:LITIGATIONOS_HOME\_intake" | Out-Null
rclone copy gdrive:/LITIGATION_INTAKE/ "$env:LITIGATIONOS_HOME\_intake\" --checksum --create-empty-src-dirs

New-Item -ItemType Directory -Force -Path "$env:LITIGATIONOS_HOME\_cyclepacks_in" | Out-Null
rclone copy gdrive:/LITIGATION_INTAKE/ "$env:LITIGATIONOS_HOME\_cyclepacks_in\" --checksum `
  --include "*CYCLEPACK*.zip" --include "*HARVEST*.zip" --include "*cyclepack*.zip" --include "*harvest*.zip" --exclude "*"
```

### C) Consolidate to CANONICAL_INTEL
```
$env:TIKA_SERVER_PORT="9999"
python "F:\LitigationOS\setup_kit\scripts\verify_env_and_ports.py" --emit-json --emit-md --fail-closed

python "F:\LitigationOS\setup_kit\tools\consolidate_canonical_intel.py" `
  --cyclepack-root "F:\LitigationOS\_cyclepacks_in" `
  --out-root "F:\LitigationOS\CANONICAL_INTEL"

rclone sync "F:\LitigationOS\CANONICAL_INTEL\" "gdrive:/Litigation_OS$/CANONICAL_INTEL/" --checksum
```

### D) Required Outputs (Per CyclePack)
- SoR ledger delta (CSV)  
- QuoteDB (JSONL)  
- ChronoDB bitemporal timeline (CSV)  
- ExhibitMatrix (CSV)  
- ContradictionMap (CSV/JSONL)  
- AuthorityTriples (JSONL)  
- DeadlinesNotice (CSV)  
- ValidationReport (MD)  
- `master_manifest.json` + run logs + hashes  

---

## Missing From This Repo (PINPOINT_MISSING)
The following **script entrypoints** are not present here and must be located locally:
- `verify_env_and_ports.py`  
- `consolidate_canonical_intel.py`  
- Harvester/OCR entrypoints  
- CyclePack packager entrypoints  

Use local discovery to locate your real entrypoints:
```
Get-ChildItem "F:\LitigationOS" -Recurse -File -Include "*harvest*.py","*ingest*.py","*ocr*.py","*parse*.py" | Select-Object FullName
Get-ChildItem "F:\LitigationOS" -Recurse -File -Include "*ollama*.py","*analy*.py","*graph*.py" | Select-Object FullName
Get-ChildItem "F:\LitigationOS" -Recurse -File -Include "*cyclepack*.py","*pack*.py","*bundle*.py" | Select-Object FullName
```
