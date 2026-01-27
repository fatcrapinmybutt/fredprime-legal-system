"""Windows drive organizer with guarded output roots."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path, PureWindowsPath


DEFAULT_OUTPUT_ROOT_TEMPLATE = "Z:/LitigationOS/Runs/{run_id}"


class EvidenceOrganizer:
    def __init__(
        self,
        output_root: Path,
        *,
        allow_c_drive: bool = False,
        allow_relative_output: bool = False,
    ) -> None:
        self.output_root = Path(output_root)
        self.allow_c_drive = allow_c_drive
        self.allow_relative_output = allow_relative_output
        self._validate_output_root()
        self.temp_dir = self.output_root / "TEMP"
        self.logs_dir = self.output_root / "LOGS"
        self._ensure_directories()

    def _validate_output_root(self) -> None:
        original_windows_path = PureWindowsPath(str(self.output_root))
        resolved_path = self.output_root.expanduser().resolve()
        resolved_windows_path = PureWindowsPath(str(resolved_path))

        if not original_windows_path.is_absolute() and not (
            self.allow_c_drive or self.allow_relative_output
        ):
            raise ValueError(
                "Output root must be an absolute Windows path with a drive letter or UNC path. "
                f"Resolved path: {resolved_windows_path}. "
                "Use --allow-relative-output or --allow-c-drive to override."
            )
        if not resolved_windows_path.drive:
            raise ValueError(
                "Output root must include a drive letter or UNC path. "
                f"Resolved path: {resolved_windows_path}."
            )
        if resolved_windows_path.drive.lower() == "c:" and not self.allow_c_drive:
            raise ValueError(
                "Output root on C: is blocked by default. "
                f"Resolved path: {resolved_windows_path}. "
                "Use a Z: path or pass --allow-c-drive to override."
            )

        self.output_root = resolved_path

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
        help=(
            "Output root directory. Defaults to "
            f"{DEFAULT_OUTPUT_ROOT_TEMPLATE}."
        ),
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
    parser.add_argument(
        "--allow-relative-output",
        action="store_true",
        help="Allow relative output paths (default requires absolute Windows path).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    output_root = Path(args.output_root) if args.output_root else build_default_output_root(run_id)

    organizer = EvidenceOrganizer(
        output_root,
        allow_c_drive=args.allow_c_drive,
        allow_relative_output=args.allow_relative_output,
    )

    logging.basicConfig(
        filename=str(organizer.logs_dir / "windows_drive_organizer.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info("Initialized EvidenceOrganizer at %s", organizer.output_root)


if __name__ == "__main__":
    main()
