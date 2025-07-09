import os
from pathlib import Path
from typing import List


def label_exhibits(exhibit_dir: str, output_index: str = "Exhibit_Index.md") -> List[str]:
    """Rename files in exhibit_dir to Exhibit_A, Exhibit_B, ... and produce an index.

    Returns a list of new filenames.
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
    Path(output_index).write_text("\n".join(index_lines))
    return labeled
