"""Windows drive organizer with guarded output roots."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path, PureWindowsPath

DEFAULT_OUTPUT_ROOT_TEMPLATE = "Z:/LitigationOS/Runs/{run_id}"


class EvidenceOrganizer:
    def __init__(self, output_root: Path, *, allow_c_drive: bool = False) -> None:
        self.output_root = Path(output_root)
        self.allow_c_drive = allow_c_drive
        self._validate_output_root()
        self.temp_dir = self.output_root / "TEMP"
        self.logs_dir = self.output_root / "LOGS"
        self._ensure_directories()

    def _validate_output_root(self) -> None:
        pure_windows_path = PureWindowsPath(str(self.output_root))
        if pure_windows_path.drive.lower() == "c:" and not self.allow_c_drive:
            raise ValueError(
                "Output root on C: is blocked by default. Use a Z: path or pass --allow-c-drive to override."
            )

    def _ensure_directories(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def build_default_output_root(run_id: str) -> Path:
    return Path(DEFAULT_OUTPUT_ROOT_TEMPLATE.format(run_id=run_id))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Windows drive organizer")
    parser.add_argument(
        "--output-root",
        default=None,
        help=f"Output root directory. Defaults to {DEFAULT_OUTPUT_ROOT_TEMPLATE}.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier used in the default output path.",
    )
    parser.add_argument(
        "--allow-c-drive",
        action="store_true",
        help="Allow output-root on C: (blocked by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    output_root = Path(args.output_root) if args.output_root else build_default_output_root(run_id)

    organizer = EvidenceOrganizer(output_root, allow_c_drive=args.allow_c_drive)

    logging.basicConfig(
        filename=str(organizer.logs_dir / "windows_drive_organizer.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Initialized EvidenceOrganizer at %s", organizer.output_root)


if __name__ == "__main__":
    main()
