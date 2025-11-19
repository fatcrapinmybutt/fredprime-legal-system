"""Build a JSON system definition for FRED PRIME.

This script is intentionally small and cross-platform. It:
- builds a JSON object describing the system
- writes the JSON to a configurable output path
- provides a CLI to override defaults

Usage: `python firstimport.py --out /path/to/out.json` or set `FREDPRIME_BASE` env var.
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import hashlib
import platform
import getpass
import logging

# === CONFIGURATION (defaults) ===
SYSTEM_NAME = "FRED PRIME Litigation Deployment Engine"
VERSION = "v2025.07.20"

DEFAULT_BASE = Path(os.environ.get("FREDPRIME_BASE", Path.cwd()))
DEFAULT_OUTPUT = DEFAULT_BASE / "output"
DEFAULT_LOG = DEFAULT_BASE / "logs"
DEFAULT_JSON = Path(os.environ.get("FREDPRIME_JSON", DEFAULT_OUTPUT / "fredprime_litigation_system.json"))

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


def safe_mkdir(path: Path):
    """Create a directory if it doesn't exist."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If path is a file or permission problem, let caller handle it
        pass


def sha256_obj(obj):
    """Hash a Python object for provenance."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def validate_paths(paths):
    """Return list of missing paths (strings)."""
    missing = []
    for p in paths:
        try:
            if not Path(p).exists():
                missing.append(str(p))
        except Exception:
            missing.append(str(p))
    return missing


def build_systemdef(base_path: Path, output_path: Path, log_path: Path):
    # Ensure output/log directories exist
    safe_mkdir(output_path)
    safe_mkdir(log_path)

    litigation_system_definition = {
        "system": SYSTEM_NAME,
        "version": VERSION,
        "generated": datetime.now().isoformat(),
        "os": platform.system(),
        "user": getpass.getuser(),
        "base_path": str(base_path),
        "output_path": str(output_path),
        "log_path": str(log_path),
        "config": CONFIG,
        "modules": MODULES,
        "execution_command": EXEC_COMMAND,
        "offline_capable": True,
        "token_usage": "Zero (local execution only)",
        "dependencies": DEPENDENCIES,
    }

    # Validation block
    critical_paths = [base_path, output_path, log_path]
    missing_paths = validate_paths(critical_paths)
    validation = {
        "missing_paths": missing_paths,
        "all_paths_exist": not bool(missing_paths),
    }
    litigation_system_definition["validation"] = validation

    # Add a hash for provenance
    litigation_system_definition["config_hash"] = sha256_obj(litigation_system_definition)

    # Audit metadata
    litigation_system_definition["audit"] = {
        "generator": "firstimport.py",
        "timestamp": datetime.now().isoformat(),
    }

    return litigation_system_definition


def write_systemdef_file(systemdef: dict, path: Path):
    try:
        safe_mkdir(path.parent)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(systemdef, f, indent=4)
        print(f"✅ System definition written to: {path}")
        print(f"SHA-256: {systemdef['config_hash']}")
    except Exception as e:
        LOGFILE = Path(systemdef.get("log_path", Path.cwd() / "logs")) / "systemdef_build_errors.log"
        logging.basicConfig(filename=str(LOGFILE), level=logging.ERROR)
        logging.error(f"Failed to write system definition: {e}")
        print(f"❌ Failed to write system definition: {e}")


def _parse_args():
    p = argparse.ArgumentParser(description="Build FRED PRIME system definition JSON")
    p.add_argument("--base", type=Path, default=DEFAULT_BASE, help="Base path for the project")
    p.add_argument("--out", type=Path, default=DEFAULT_JSON, help="Output JSON file path")
    p.add_argument("--version", action="store_true", help="Print version and exit")
    return p.parse_args()


def main():
    args = _parse_args()
    if getattr(args, "version", False):
        print(VERSION)
        return

    base = Path(args.base)
    output = Path(args.out)
    log = base / "logs"

    systemdef = build_systemdef(base, output.parent, log)
    write_systemdef_file(systemdef, output)


if __name__ == "__main__":
    main()
