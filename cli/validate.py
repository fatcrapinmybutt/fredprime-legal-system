"""Validation operations for litigator CLI."""

from pathlib import Path

from typing import Callable, cast

from ZIP_VALIDATOR import validate_zip_folder as _validate_zip_folder

validate_zip_folder = cast(Callable[[str], None], _validate_zip_folder)


def run(base_path: Path) -> None:
    """Validate required files before creating a ZIP archive.

    Args:
        base_path: Directory containing litigation files.
    """
    validate_zip_folder(str(base_path))
