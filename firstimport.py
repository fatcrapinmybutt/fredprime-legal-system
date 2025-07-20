import json
import os
from pathlib import Path
from datetime import datetime
import hashlib
import platform
import getpass
import logging

# === CONFIGURATION ===
SYSTEM_NAME = "FRED PRIME Litigation Deployment Engine"
BASE_PATH = Path("F:/GitRepo/fredprime")
OUTPUT_PATH = BASE_PATH / "output"
LOG_PATH = BASE_PATH / "logs"
JSON_PATH = Path("/mnt/data/fredprime_litigation_system.json")
VERSION = "v2025.07.20"

CONFIG = {
    "exhibit_labeling": True,
    "motion_linking": True,
    "signature_validation": True,
    "judicial_audit": True,
    "parenting_time_matrix": True,
    "conspiracy_tracker": True,
}

MODULES = {
    "exhibit_labeler": "Renames evidence files A–Z and builds Exhibit_Index.md",
    "motion_exhibit_linker": "Scans motions, finds exhibit references, builds Motion_to_Exhibit_Map.md",
    "signature_validator": "Checks for MCR 1.109(D)(3) compliance",
    "judicial_conduct_tracker": "Builds Exhibit U with judge behavior patterns",
    "appclose_matrix": "Parses AppClose logs to generate Exhibit Y (violations matrix)",
    "conspiracy_log": "Parses police reports and logs false allegations into Exhibit S",
}

DEPENDENCIES = ["PowerShell 5+", "Git (if pushing back)", "Windows OS"]
EXEC_COMMAND = "powershell -ExecutionPolicy Bypass -File fred_deploy.ps1"

# === FUNCTIONS ===

def safe_mkdir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)

def sha256_obj(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def validate_paths(paths):
    missing = []
    for p in paths:
        if not p.exists():
            missing.append(str(p))
    return missing

def build_systemdef():
    # Ensure output/log directories exist
    safe_mkdir(OUTPUT_PATH)
    safe_mkdir(LOG_PATH)

    # Core object
    litigation_system_definition = {
        "system": SYSTEM_NAME,
        "version": VERSION,
        "generated": datetime.now().isoformat(),
        "os": platform.system(),
        "user": getpass.getuser(),
        "base_path": str(BASE_PATH),
        "output_path": str(OUTPUT_PATH),
        "log_path": str(LOG_PATH),
        "config": CONFIG,
        "modules": MODULES,
        "execution_command": EXEC_COMMAND,
        "offline_capable": True,
        "token_usage": "Zero (local execution only)",
        "dependencies": DEPENDENCIES,
    }

    # Validation
    critical_paths = [BASE_PATH, OUTPUT_PATH, LOG_PATH]
    missing_paths = validate_paths(critical_paths)
    validation = {
        "missing_paths": missing_paths,
        "all_paths_exist": not bool(missing_paths),
    }
    litigation_system_definition["validation"] = validation

    # Hash for evidence/provenance
    litigation_system_definition["config_hash"] = sha256_obj(litigation_system_definition)

    # Metadata/audit
    litigation_system_definition["audit"] = {
        "generator": "systemdef_builder.py",
        "timestamp": datetime.now().isoformat(),
    }

    return litigation_system_definition

def write_systemdef_file(systemdef: dict, path: Path):
    try:
        with open(path, "w") as f:
            json.dump(systemdef, f, indent=4)
        print(f"✅ System definition written to: {path}")
        print(f"SHA-256: {systemdef['config_hash']}")
    except Exception as e:
        logging.basicConfig(filename="systemdef_build_errors.log", level=logging.ERROR)
        logging.error(f"Failed to write system definition: {e}")
        print(f"❌ Failed to write system definition: {e}")

if __name__ == "__main__":
    systemdef = build_systemdef()
    write_systemdef_file(systemdef, JSON_PATH)
