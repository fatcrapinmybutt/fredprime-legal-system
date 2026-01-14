import argparse
import json
import os
from pathlib import Path
from datetime import datetime
import hashlib
import platform
import getpass
import logging

# Optional: try to import jsonschema for validation (not required)
try:
    import jsonschema
    from jsonschema import validate as _jsonschema_validate
except Exception:
    jsonschema = None

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


def build_systemdef(base_path: Path = None, output_path: Path = None, log_path: Path = None):
    base = Path(base_path) if base_path is not None else DEFAULT_BASE
    output = Path(output_path) if output_path is not None else DEFAULT_OUTPUT
    log = Path(log_path) if log_path is not None else DEFAULT_LOG

    safe_mkdir(output)
    safe_mkdir(log)

    litigation_system_definition = {
        "system": SYSTEM_NAME,
        "version": VERSION,
        "generated": datetime.now().isoformat(),
        "os": platform.system(),
        "user": getpass.getuser(),
        "base_path": str(base),
        "output_path": str(output),
        "log_path": str(log),
        "config": CONFIG,
        "modules": MODULES,
        "execution_command": EXEC_COMMAND,
        "offline_capable": True,
        "token_usage": "Zero (local execution only)",
        "dependencies": DEPENDENCIES,
    }

    critical_paths = [base, output, log]
    missing_paths = validate_paths(critical_paths)
    validation = {
        "missing_paths": missing_paths,
        "all_paths_exist": not bool(missing_paths),
    }
    litigation_system_definition["validation"] = validation

    litigation_system_definition["config_hash"] = sha256_obj(litigation_system_definition)

    litigation_system_definition["audit"] = {
        "generator": "firstimport.py",
        "timestamp": datetime.now().isoformat(),
    }

    return litigation_system_definition


def validate_systemdef(systemdef: dict, schema_path: Path):
    """Validate the system definition against a JSON Schema file.

    Raises jsonschema.ValidationError on failure.
    """
    if jsonschema is None:
        raise RuntimeError("jsonschema is not installed; cannot validate system definition")
    with open(schema_path, "r", encoding="utf-8") as fh:
        schema = json.load(fh)
    _jsonschema_validate(instance=systemdef, schema=schema)


def write_systemdef_file(systemdef: dict, path: Path, validate: bool = True):
    try:
        safe_mkdir(path.parent)

        if validate:
            schema_path = Path(os.environ.get("FREDPRIME_SCHEMA", Path(systemdef.get("base_path", ".")) / "schema" / "systemdef.schema.json"))
            if schema_path.exists() and jsonschema is not None:
                try:
                    validate_systemdef(systemdef, schema_path)
                except Exception as ve:
                    logging.basicConfig(filename=str(path.parent / "systemdef_build_errors.log"), level=logging.ERROR)
                    logging.error(f"Schema validation failed: {ve}")
                    raise

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
    p.add_argument("--no-validate", action="store_true", help="Skip JSON Schema validation before writing output")
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
    write_systemdef_file(systemdef, output, validate=not getattr(args, "no_validate", False))


if __name__ == "__main__":
    main()
