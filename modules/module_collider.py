from typing import Dict, Iterable, List


def module_collider(modules: Iterable[str]) -> Dict[str, List[str]]:
    """Deduplicate modules by canonical key."""
    canonical: Dict[str, List[str]] = {}
    for m in modules:
        key = m.lower().replace("engine", "").replace("module", "").strip()
        canonical.setdefault(key, []).append(m)
    return canonical
