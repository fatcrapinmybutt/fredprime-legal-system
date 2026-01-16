import argparse
import os
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Iterable, List

DEFAULT_DRIVES = [Path(r"Q:/"), Path(r"D:/"), Path(r"Z:/")]
REQUIRED_DRIVES = {"Q:", "D:", "Z:"}


def _drive_letter(path: Path) -> str:
    return PureWindowsPath(str(path)).drive.upper()


def _resolve_path(path: Path) -> Path:
    try:
        return path.resolve(strict=False)
    except TypeError:
        return path.resolve()


@dataclass
class EvidenceOrganizer:
    drives: List[Path]

    def __init__(self, drives: Iterable[Path] | None = None) -> None:
        self.drives = list(drives) if drives is not None else list(DEFAULT_DRIVES)

    def _validate_drives(self) -> None:
        drive_letters = {_drive_letter(drive) for drive in self.drives}
        missing_required = REQUIRED_DRIVES - drive_letters
        if missing_required:
            missing = ", ".join(sorted(missing_required))
            raise ValueError(f"Missing required drives: {missing}.")

        missing_paths = [drive for drive in self.drives if not drive.exists()]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"Required drives are not available: {missing}.")

        invalid_c = []
        for drive in self.drives:
            for candidate in (drive, _resolve_path(drive)):
                if _drive_letter(candidate) == "C:":
                    invalid_c.append(str(drive))
                    break

        if invalid_c:
            drives = ", ".join(invalid_c)
            raise ValueError(f"C: drive roots are not permitted: {drives}.")

    def _discover_files(self) -> List[Path]:
        discovered = []
        for drive in self.drives:
            for root, _, files in os.walk(drive):
                for filename in files:
                    discovered.append(Path(root) / filename)
        return discovered

    def run(self) -> List[Path]:
        self._validate_drives()
        return self._discover_files()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evidence organizer drive scan")
    parser.add_argument(
        "--drives",
        nargs="*",
        default=[str(drive) for drive in DEFAULT_DRIVES],
        help="Drive roots to scan (default: Q:/ D:/ Z:/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    organizer = EvidenceOrganizer(drives=[Path(drive) for drive in args.drives])
    organizer.run()


if __name__ == "__main__":
    main()
