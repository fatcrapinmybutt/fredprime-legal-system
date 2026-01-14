import json
from pathlib import Path


def generate_manifest(output_path: str = "manifest.json") -> str:
    """Generate a simple manifest of files in the current directory."""
    manifest = {"files": [str(p) for p in Path(".").iterdir()]}
    with open(output_path, "w") as f:
        json.dump(manifest, f)
    return output_path
