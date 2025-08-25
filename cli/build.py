"""Build operations for litigator CLI."""

from pathlib import Path

from typing import cast

from build_system import build_json as _build_json


def run(output: Path) -> Path:
    """Generate the litigation system definition JSON.

    Args:
        output: Destination path for the generated JSON file.

    Returns:
        The path to the generated file.
    """
    return cast(Path, _build_json(output))
