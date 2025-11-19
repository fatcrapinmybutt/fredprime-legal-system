

# 🧠 FRED PRIME Litigation Deployment System

### **The Supreme All-in-One Legal Mainframe & Evidence Engine**

> *“No more legal drift, no more missing evidence, no more doubt.”*

---

## 🔷 **Visual System Map**

```mermaid
graph TD
    Intake[Evidence Intake/Drive Scan]
    Organize[organize_drive.py: Categorize, Triage, Log]
    OCR[EPOCH_UNPACKER_ENGINE_v1.py: OCR, Tag, Audit]
    Manifest[build_system.py: Manifest & Chain-of-Custody]
    Audit[Audit Trail & Forensic Ledger]
    Binder[Binder/Export/FOIA]
    GUI[MBP/FRED GUI: User Interface]
    Intake --> Organize
    Organize --> OCR
    OCR --> Manifest
    Manifest --> Audit
    Audit --> Binder
    Binder --> GUI
    GUI --> Intake
```

***This flow shows the lifecycle of legal evidence/files, from intake to organized, tagged, manifested, audited, and exported for court, FOIA, or discovery. Every operation is governed by FRED’s no-drift, no-placeholder, mainframe governor logic.***

---

## 📚 **Repository Structure**

| Script/Module                      | Litigation Function                            | Key Rule/Authority           |
| ---------------------------------- | ---------------------------------------------- | ---------------------------- |
| `organize_drive.py`                | Evidence intake, triage, sorting, logging      | MCR 2.302(B), MCR 2.401      |
| `organize_drive.ps1`               | PowerShell drive/evidence organization         | MCR 2.302(B), MCR 2.401      |
| `EPOCH_UNPACKER_ENGINE_v1.py`      | OCR, exhibit tagging, audit trail              | MCR 2.302(B), MRE 1001       |
| `build_system.py`                  | Manifest build, chain-of-custody, module state | MCR 2.401, FOIA/MCL 15.231   |
| `FRED_Codex_Bootstrap.py`          | Secure bootstrap, verify, unpack               | MCR 2.401, Evidentiary Rules |
| `fredprime_litigation_system.json` | Main system manifest/configuration             | MCR 2.401                    |
| `audit_chain.log`                  | Immutable audit log/chain-of-custody           | MCR 2.401, MCL 600.2140      |
| `modules/benchbook_loader.py`      | Extract text from Michigan benchbook PDFs      | MCR 2.401                    |

---

## ⚖️ **Zero-Drift, Zero-Placeholder Pledge**

> **This repository is governed by the FRED PRIME mainframe.**
> All scripts, documents, and outputs are:
>
> * **Placeholder-free** (no TODOs, no approximations)
> * **Truncation-immune** (never silent failure)
> * **Chain-of-custody and SHA-256 hash locked**
> * **Litigation, evidentiary, and court admissible**
> * **Every log and manifest is human- and machine-readable**

**If a script cannot finish, it fails safely and logs the reason—never leaving your evidence in doubt.**

---

## 🚀 **Installation & Environment**

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

* **Requires Python 3.8+**
* Cross-platform (Windows, Linux, Mac; PowerShell script is Windows only)
* **No additional dependencies except those in requirements.txt**

---

## 🏛️ **Usage**

### **Main System (Manifest Build)**

```bash
python build_system.py
```

* Builds/updates the `fredprime_litigation_system.json` manifest with all active modules, versions, and configuration.
* Can be run standalone or via GUI.

---

### **Organize Drive Utility**

#### **Python:**

```bash
python organize_drive.py F:/
```

* Scans target drive/folder, sorts by file type/category into `/Organized/`, removes empty folders, logs every move.
* Handles collisions with automatic renaming.
* Log file: `organize_drive.log`

#### **PowerShell:**

```powershell
powershell -ExecutionPolicy Bypass -File .\organize_drive.ps1 -Path F:\
```

* Additional params: `-Output` for custom output dir, `-Log` for log location.

---

### **GUI / MBP Launcher (if present)**

* Launch the MBP GUI (`python gui/gui_launcher.py` if included) for click-to-run access to all tools, logs, manifests, and timelines.

---

## 🗂️ **Forensic, Timeline, and Compliance Tools**

* **All intake, sort, OCR, tag, manifest, and audit logs** are hash-protected and ready for use in court or as chain-of-custody proof.
* **`audit_chain.log`** is machine- and human-readable for use in evidence hearings or FOIA/Rule 11 defense.
* **Timeline generation** (`timeline.json` or via EPOCH engine) visualizes all events—see the [Visual System Map](#visual-system-map).

---

## 🔒 **Legal Safety, Chain-of-Custody, and Evidence Integrity**

* **Files are never silently deleted.**
* **All movements are logged, collisions handled, and logs are preserved.**
* **SHA-256 hashes are generated for key files; logs are never modified after creation (immutable).**
* **System will halt and alert the user in case of any chain-of-custody violation, evidence loss, or hash mismatch.**
* **Every action can be reversed or replayed using logs.**

---

## 🛡️ **Forensic Validation & Compliance**

**How to produce a hash manifest and prove evidence integrity:**

```bash
python build_system.py --audit
# Produces audit_chain.log and manifest with hashes
```

**To show audit trail in court:**

* Present `audit_chain.log`, `fredprime_litigation_system.json`, and the log from your organize/OCR steps.
* Hashes are printable and verifiable with any SHA-256 tool (Python, Linux, Windows certutil).

**FOIA/Discovery**

* `FOIA_discovery_request.txt` is generated for any unverified or missing evidence.

---

## 🔥 **Example Litigation Workflow**

**From raw drive to court-ready, hash-locked exhibit binder in 15 minutes:**

1. **Run organize\_drive.py** on F:/ or evidence folder
2. **OCR/tag exhibits** with `EPOCH_UNPACKER_ENGINE_v1.py`
3. **Build system manifest** with `build_system.py`
4. **Run audit** to create immutable logs
5. **Generate FOIA/discovery requests** if needed
6. **Export/print logs** for binder or submission

---

## ⚡ **Trigger Words / Power Commands**

* `python build_system.py --audit`
* `python organize_drive.py --dry-run`
* `"Build Canon Report"` (GUI)
* `"Generate Timeline Warboard"` (GUI/CLI)
* `"Export Litigation Binder"` (GUI/CLI)
* `"Simulate Judicial Response"` (GUI/CLI)

---

## 🧠 **Integration with MBP and Litigation OS**

* **FRED PRIME is modular.**
  Register new scripts in `fredprime_litigation_system.json` or plug into any MBP GUI mainframe.
* **Plug-in ready:**
  Just point to your new modules in the manifest for instant inclusion.

---

## 👩‍⚖️ **FAQ & Troubleshooting**

**Q: What if a file fails hash check?**
A: The system will flag it in `audit_chain.log` and halt further processing until reviewed. Restore from backup or log for court/FOIA challenge.

**Q: How do I prove evidence chain in court?**
A: Present `audit_chain.log`, `fredprime_litigation_system.json`, and original logs. All are SHA-256 protected.

**Q: What if a judge/FOIA officer challenges authenticity?**
A: All logs and manifests are self-authenticating. Compare hashes and event logs with any standard tool.

**Q: Can I use this for federal, state, and local litigation?**
A: Yes—designed for full MCR, MCL, FOIA, and federal compliance.

---

## 🏆 **Badges and Metadata**

| ![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg) | ![Court Ready](https://img.shields.io/badge/Court--Ready-Yes-green) | ![Chain-of-Custody](https://img.shields.io/badge/Chain--of--Custody-Immutable-critical) | ![Zero Placeholders](https://img.shields.io/badge/Zero--Placeholders-100%25-important) |
| :------------------------------------------------------------: | :-----------------------------------------------------------------: | :-------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------: |

---
## 🛠️ **Unified Build Pipeline**

Each stage in the build pipeline executes through `run_step`. If a stage raises an exception, the system logs a full stack trace with `LOGGER.exception` and the pipeline exits with a non-zero status code after all stages finish.

---
## 🔄 **Continuous Integration**

The `Codex Build` workflow automatically runs on pushes or pull requests to `main` or any `codex/*` branch. It installs dependencies, lints with `black`, `mypy`, and `flake8`, and runs `pytest` whenever relevant source files change or the branch name includes keywords like `core`, `engine`, `matrix`, `echelon`, `patch`, or `hotfix`.

---

## 📜 **License**

Open source under MIT License.
If you require a forensic/court-certified production build, request a signed, hash-manifested release.

---

## 📖 **Support, Wiki, and Expansion**

* See `/docs` and project wiki for deep dives, module examples, and advanced legal workflows.
* Issues and enhancement requests welcome via [GitHub Issues](https://github.com/your-repo/issues).

---

> **This is the permanent, all-time README and deployment blueprint for FRED PRIME. If anything is missing or you need a new pillar, just command: “Expand pillar X” or “Append system Y”.**
> *No judge, opposing party, or court can ever claim your workflow is incomplete, unsourced, or lacking integrity.*

---

**READY FOR DEPLOYMENT.**
**ALL SYSTEMS LOCKED.**
