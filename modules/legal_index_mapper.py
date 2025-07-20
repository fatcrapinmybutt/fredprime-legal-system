from typing import Dict, Iterable


def legal_index_mapper(modules: Iterable[str]) -> Dict[str, str]:
    """Map module names to citations of governing authority."""
    citation_map = {}
    for m in modules:
        lower = m.lower()
        if "canon" in lower:
            citation_map[m] = "Canon 2A, 3B, 3C"
        elif "custody" in lower:
            citation_map[m] = "MCL 722.23, MCL 722.27"
        elif "motion" in lower:
            citation_map[m] = "MCR 2.119"
        elif "timeline" in lower:
            citation_map[m] = "Benchbook, Ch. 4"
    return citation_map
