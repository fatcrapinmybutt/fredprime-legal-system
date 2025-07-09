from pathlib import Path
import re
from typing import Dict, List


def link_motions(motion_dir: str, output_map: str = "Motion_to_Exhibit_Map.md") -> Dict[str, List[str]]:
    """Scan motion files for exhibit references and produce a map."""
    motion_path = Path(motion_dir)
    if not motion_path.is_dir():
        raise ValueError(f"{motion_dir} is not a directory")

    mapping = {}
    pattern = re.compile(r"Exhibit\s+([A-Z])")

    for motion_file in motion_path.glob("*.txt"):
        text = motion_file.read_text(errors="ignore")
        exhibits = pattern.findall(text)
        mapping[motion_file.name] = exhibits

    lines = []
    for motion, exhibits in mapping.items():
        line = f"- {motion}: {', '.join(exhibits) if exhibits else 'None'}"
        lines.append(line)

    Path(output_map).write_text("\n".join(lines))
    return mapping
