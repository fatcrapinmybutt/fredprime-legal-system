import hashlib
import json
from pathlib import Path
from typing import Tuple

from modules.codex_guardian import run_guardian
from modules.codex_supreme import self_diagnostic
from modules.codex_manifest import generate_manifest, verify_all_modules

MANIFEST = "codex_manifest.json"


def hash_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def extract_metadata(path: Path) -> Tuple[str, list[str]]:
    """Extract legal metadata from a module.

    The function looks for either a sibling ``*.meta.json`` file or
    header comments in the source. Comments should take the form::

        # legal_function: short description
        # dependencies: pkg1, pkg2

    Args:
        path: Module path.

    Returns:
        A tuple containing the legal function description and a list of
        dependency names.
    """

    legal_function = "unknown"
    dependencies: list[str] = []
    meta_path = path.with_suffix(".meta.json")

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            legal_function = meta.get("legal_function", legal_function)
            dependencies = meta.get("dependencies", dependencies)
        except json.JSONDecodeError:
            pass

    if legal_function == "unknown" or not dependencies:
        try:
            with path.open("r", encoding="utf-8") as f:
                for _ in range(10):
                    line = f.readline()
                    if not line:
                        break
                    lower = line.lower()
                    if lower.startswith("# legal_function:"):
                        legal_function = line.split(":", 1)[1].strip()
                    elif lower.startswith("# dependencies:"):
                        deps = line.split(":", 1)[1]
                        dependencies = [d.strip() for d in deps.split(",") if d.strip()]
                    if legal_function != "unknown" and dependencies:
                        break
        except OSError:
            pass

    return legal_function, dependencies


def update_manifest() -> None:
    modules: list[dict[str, object]] = []
    for p in Path(".").rglob("*.py"):
        if p.parts[0].startswith("."):
            continue
        legal_function, dependencies = extract_metadata(p)
        modules.append(
            {
                "module": p.stem,
                "path": str(p),
                "hash": hash_file(p),
                "legal_function": legal_function,
                "dependencies": dependencies,
            }
        )

    manifest_map = generate_manifest(modules)
    verify_all_modules(manifest_map)
    Path(MANIFEST).write_text(json.dumps(modules, indent=2))


def main() -> None:
    run_guardian()
    update_manifest()
    self_diagnostic()
    print("codex manifest updated")


if __name__ == "__main__":
    main()
