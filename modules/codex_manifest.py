import json
import hashlib
from pathlib import Path
from typing import Iterable, Dict, Any

from core.local_llm import analyze_content


def generate_manifest(modules: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Generate a manifest mapping module paths to metadata."""
    manifest: Dict[str, Dict[str, Any]] = {}
    for item in modules:
        path = Path(item["path"])
        data = path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        text = data.decode("utf-8", errors="ignore")
        analysis = analyze_content(text)
        manifest[str(path)] = {
            "sha256": sha256,
            "legal_function": item.get("legal_function"),
            "dependencies": item.get("dependencies", []),
            "word_count": analysis["word_count"],
        }
    return manifest


def save_manifest(manifest: Dict[str, Dict[str, Any]], file_path: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def verify_all_modules(manifest: Dict[str, Dict[str, Any]]) -> None:
    """Validate manifest entries and hashes."""
    for path, info in manifest.items():
        if not info.get("legal_function"):
            raise ValueError(f"Manifest entry for {path} missing 'legal_function'")
        if "dependencies" not in info:
            raise ValueError(f"Manifest entry for {path} missing 'dependencies'")
        if not isinstance(info["dependencies"], list):
            raise ValueError(f"Dependencies for {path} must be a list")
        if "word_count" not in info:
            raise ValueError(f"Manifest entry for {path} missing 'word_count'")
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        with open(p, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        if sha256 != info["sha256"]:
            raise ValueError(f"Hash mismatch for {path}")
