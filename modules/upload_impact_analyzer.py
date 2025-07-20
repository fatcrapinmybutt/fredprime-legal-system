from typing import Iterable, List, Tuple, Dict


def upload_impact_analyzer(
    upload_index: Iterable[Dict[str, str]], module_history: Iterable[str]
) -> List[Tuple[str, str]]:
    """Link uploaded files to modules named in their history."""
    upload_links = []
    for u in upload_index:
        for m in module_history:
            if u["filename"].split(".")[0].lower() in m.lower():
                upload_links.append((u["filename"], m))
    return upload_links
