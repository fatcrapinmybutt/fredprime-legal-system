from pathlib import Path


def validate_file(path: str) -> bool:
    """Mock validation for file path."""
    return Path(path).is_file()


def classify_legal_function(path: str) -> str:
    """Return a simple classification based on filename."""
    name = Path(path).name.lower()
    if "canon" in name:
        return "canon_scanner"
    if "motion" in name:
        return "motion_module"
    return "generic"
