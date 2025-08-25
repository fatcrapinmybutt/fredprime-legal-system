"""Ingestion operations for litigator CLI."""

from pathlib import Path
from typing import Optional

from typing import Callable, cast

from EPOCH_UNPACKER_ENGINE_v1 import (
    run_headless as _run_headless,
    set_base_dir as _set_base_dir,
)

run_headless = cast(Callable[[str], None], _run_headless)
set_base_dir = cast(Callable[[Path], None], _set_base_dir)


def run(zip_path: Path, base_dir: Optional[Path] = None) -> None:
    """Process an evidentiary ZIP archive.

    Args:
        zip_path: Path to the ZIP file to process.
        base_dir: Directory where extracted data and logs will reside.
    """
    if base_dir is not None:
        set_base_dir(base_dir)
    run_headless(str(zip_path))
