from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

MODULES_DOC = Path(__file__).resolve().parent.parent / "docs" / "high_impact_modules.md"


@dataclass
class HighImpactModule:
    name: str
    level: int = 9999


def _load_module_names(doc_path: Path = MODULES_DOC) -> List[str]:
    names: List[str] = []
    with open(doc_path, encoding="utf-8") as fh:
        for line in fh:
            match = re.match(r"\d+\.\s+(.*)", line.strip())
            if match:
                names.append(match.group(1))
    return names


def build_modules(level: int = 9999) -> List[HighImpactModule]:
    return [HighImpactModule(name=name, level=level) for name in _load_module_names()]
