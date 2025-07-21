# Build a README with full instructions for using the JSON-defined system locally or through GitHub

readme_content = """
# 🧠 FRED PRIME Litigation Deployment System

This repo enables **offline, token-free litigation automation** for the FRED PRIME system using PowerShell and a JSON-configurable engine.

---

## ✅ What This System Does

- 🔖 Auto-labels exhibits (Exhibit A–Z)
- 🔗 Links motions to matching exhibits
- 🧾 Validates MCR 1.109(D)(3) signature block compliance
- 📅 Builds parenting time violation matrix from AppClose logs (Exhibit Y)
- 🛑 Tracks false police reports and PPO misuse (Exhibit S)
- ⚖️ Logs judicial irregularities (Exhibit U)

---

## 🗂 Structure


## Organize Drive Script

This repository includes both Python and PowerShell scripts for organizing a drive. Each script sorts files into categorized folders inside an `Organized` directory and removes empty folders when finished.

### Python Usage

1. Ensure Python 3 is installed.
2. Run the script from a command prompt:
   ```bash
   python organize_drive.py F:/
   ```
   Replace `F:/` with the path you want to organize. A log file named `organize_drive.log` will record actions.

### PowerShell Usage

1. Open PowerShell 5 or later.
2. Run the script with:
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\organize_drive.ps1 -Path F:\
   ```
   The `-Path` parameter specifies the drive or folder to organize and `-Log` controls the log file path.

### Safety

- Files are moved, not copied. Ensure you have backups if necessary.
- Name collisions are handled automatically by appending a numeric suffix to the destination file.
- Review the log file if any issues arise.

