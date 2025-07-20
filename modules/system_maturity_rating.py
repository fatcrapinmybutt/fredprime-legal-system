from typing import Dict


def system_maturity_rating(module_notes: Dict[str, str]) -> Dict[str, str]:
    """Classify modules by maturity based on notes."""
    rating = {}
    for name, data in module_notes.items():
        lowered = data.lower()
        if "generated .py" in lowered and "gui integration" in lowered:
            rating[name] = "\U0001f7e2 PRODUCTION"
        elif "drafted logic" in lowered:
            rating[name] = "\U0001f7e1 IN DEVELOPMENT"
        else:
            rating[name] = "\U0001f534 SKETCH ONLY"
    return rating
