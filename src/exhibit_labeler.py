"""Utility to rename exhibit files sequentially and record an index."""

from pathlib import Path
from typing import List


def label_exhibits(exhibit_dir: str, output_index: str = "Exhibit_Index.md") -> List[str]:
    """Rename files in *exhibit_dir* and create a markdown index.

    Files are renamed in alphabetical order as ``Exhibit_A`` etc. The index file
    lists each renamed exhibit.
    """
    path = Path(exhibit_dir)
    if not path.is_dir():
        raise ValueError(f"{exhibit_dir} is not a directory")

    files = sorted([p for p in path.iterdir() if p.is_file()])
    labeled = []
    for i, file in enumerate(files):
        label = chr(ord('A') + i)
        new_name = f"Exhibit_{label}{file.suffix}"
        target = file.with_name(new_name)
        file.rename(target)
        labeled.append(target.name)

    index_lines = [f"- {name}" for name in labeled]
    Path(output_index).write_text("\n".join(index_lines) + "\n")
    return labeled
