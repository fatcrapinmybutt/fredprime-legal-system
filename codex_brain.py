import ast
import hashlib
import json
from pathlib import Path
from typing import List, Dict

from modules.codex_guardian import run_guardian  # Supreme policy: audit every manifest build
from modules.codex_supreme import self_diagnostic  # Full system health check and reporting

MANIFEST = "codex_manifest.json"

def legal_function_from_name(path: Path) -> str:
    """Returns the most likely legal function of a file, by filename."""
    name = path.name.lower()
    if "motion" in name:
        return "motion (MCR 2.119)"
    if "affidavit" in name:
        return "affidavit (MCR 2.119(B))"
    if "order" in name:
        return "court order"
    if "scanner" in name or "ocr" in name:
        return "evidence intake (MCR 2.302)"
    if "binder" in name:
        return "exhibit binder generator"
    if "timeline" in name:
        return "timeline engine"
    if "warboard" in name:
        return "litigation warboard"
    if "backup" in name:
        return "disaster recovery/backup"
    if "api" in name or "efiling" in name:
        return "API/e-filing automation"
    if "sim" in name or "mock_trial" in name:
        return "litigation simulation"
    if "ai" in name or "nlp" in name:
        return "AI/ML engine"
    return "module"

def hash_file(path: Path) -> str:
    """Returns the SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()

def parse_dependencies(path: Path) -> List[str]:
    """Parses a Python file for imported module names."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        deps: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    deps.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                deps.append(node.module)
        return sorted(set(deps))
    except Exception as e:
        return [f"ERROR: {e}"]

def should_skip_path(p: Path) -> bool:
    """Skip hidden folders and output, venv, .git, or test artifacts."""
    parts = [part.lower() for part in p.parts]
    skip_dirs = {".git", ".venv", "venv", "__pycache__", "output", "dist", "build", "installer_scripts"}
    return any(part in skip_dirs or part.startswith(".") for part in parts)

def update_manifest() -> None:
    """
    Scans the repo for all .py files (excluding hidden/system dirs),
    computes their hashes, legal function, and dependencies,
    and writes the manifest to codex_manifest.json.
    """
    manifest: List[Dict] = []
    for p in Path(".").rglob("*.py"):
        if should_skip_path(p):
            continue
        manifest.append(
            {
                "module": p.stem,
                "path": str(p),
                "hash": hash_file(p),
                "legal_function": legal_function_from_name(p),
                "dependencies": parse_dependencies(p),
            }
        )
    Path(MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

def main() -> None:
    """
    Runs security guardian, updates the manifest,
    then runs a system-wide self-diagnostic.
    """
    print("Running Codex Guardian compliance scan...")
    run_guardian()
    print("Building and updating codex manifest...")
    update_manifest()
    print("Running Codex Supreme diagnostics...")
    self_diagnostic()
    print("Codex manifest updated (eternity+ compliance enforced)")

if __name__ == "__main__":
    main()
