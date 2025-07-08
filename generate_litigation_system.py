import json
from pathlib import Path
import argparse

SYSTEM_CONFIG = {
    "system": "FRED PRIME Litigation Deployment Engine",
    "base_path": "F:/GitRepo/fredprime",
    "output_path": "F:/GitRepo/fredprime/output",
    "log_path": "F:/GitRepo/fredprime/logs",
    "config": {
        "exhibit_labeling": True,
        "motion_linking": True,
        "signature_validation": True,
        "judicial_audit": True,
        "parenting_time_matrix": True,
        "conspiracy_tracker": True
    },
    "modules": {
        "exhibit_labeler": "Renames evidence files A–Z and builds Exhibit_Index.md",
        "motion_exhibit_linker": "Scans motions, finds exhibit references, builds Motion_to_Exhibit_Map.md",
        "signature_validator": "Checks for MCR 1.109(D)(3) compliance",
        "judicial_conduct_tracker": "Builds Exhibit U with judge behavior patterns",
        "appclose_matrix": "Parses AppClose logs to generate Exhibit Y (violations matrix)",
        "conspiracy_log": "Parses police reports and logs false allegations into Exhibit S"
    },
    "execution_command": "powershell -ExecutionPolicy Bypass -File fred_deploy.ps1",
    "offline_capable": True,
    "token_usage": "Zero (local execution only)",
    "dependencies": ["PowerShell 5+", "Git (if pushing back)", "Windows OS"]
}

parser = argparse.ArgumentParser(description="Generate litigation system definition")
parser.add_argument("--output", default="fredprime_litigation_system.json", help="Path to output JSON file")
args = parser.parse_args()

Path(args.output).write_text(json.dumps(SYSTEM_CONFIG, indent=4))
print(f"Wrote {args.output}")
