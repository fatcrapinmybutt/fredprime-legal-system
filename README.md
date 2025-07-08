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


## Generating the Litigation System JSON
Use the helper script to create a JSON definition of the entire FRED PRIME litigation system:

```bash
python generate_litigation_system.py --output fredprime_litigation_system.json
```

This produces `fredprime_litigation_system.json` describing all modules and default paths used by the deployment engine.
