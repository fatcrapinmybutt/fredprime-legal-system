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

By default this installs the application into `C:\privategpt` and launches it with the `settings-local.yaml` configuration. You can override the install path or model name:

```powershell
./privategpt_setup.ps1 -InstallPath "D:\custom_dir" -Model "phi3"
```


To generate the system configuration JSON, run the Python script:

```bash
python firstimport.py
```
