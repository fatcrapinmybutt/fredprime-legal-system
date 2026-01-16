"""Build the FRED PRIME litigation system definition as JSON."""

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

litigation_system_definition = {
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
        "conspiracy_tracker": True,
    },
    "modules": {
        "exhibit_labeler": "Renames evidence files A–Z and builds Exhibit_Index.md",
        "motion_exhibit_linker": "Scans motions, finds exhibit references, builds Motion_to_Exhibit_Map.md",
        "signature_validator": "Checks for MCR 1.109(D)(3) compliance",
        "judicial_conduct_tracker": "Builds Exhibit U with judge behavior patterns",
        "appclose_matrix": "Parses AppClose logs to generate Exhibit Y (violations matrix)",
        "conspiracy_log": "Parses police reports and logs false allegations into Exhibit S",
    },
    "execution_command": "powershell -ExecutionPolicy Bypass -File fred_deploy.ps1",
    "offline_capable": True,
    "token_usage": "Zero (local execution only)",
    "dependencies": ["PowerShell 5+", "Git (if pushing back)", "Windows OS"],
}


def build_json(path: Path) -> Path:
    """Write the system definition to *path* and return the path."""
    logger.info("Writing system definition to %s", path)
    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(litigation_system_definition, indent=4))
    except Exception as exc:
        logger.error("Failed to write JSON: %s", exc)
        raise
    else:
        logger.info("Successfully wrote %s", path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("./output/fredprime_litigation_system.json"),
        help="Path of the generated JSON file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase output verbosity (use -v or -vv)",
    )
    return parser.parse_args()


def configure_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity > 1:
        level = logging.DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    path = build_json(args.output)
    print(path)


if __name__ == "__main__":
    main()
