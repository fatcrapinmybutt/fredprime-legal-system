"""Core coordination script for manifest generation and diagnostics."""

import ast
import importlib
import hashlib
import json
from pathlib import Path
from typing import Callable

run_guardian: Callable[[], None] = getattr(
    importlib.import_module("modules.codex_guardian"), "run_guardian"
)
self_diagnostic: Callable[[], None] = getattr(
    importlib.import_module("modules.codex_supreme"), "self_diagnostic"
)

MANIFEST = "codex_manifest.json"
CONFIG_PATH = Path(".codex_config.yaml")


def hash_file(path: Path) -> str:
    """Return the SHA-256 hash of a file."""
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def parse_metadata(path: Path) -> tuple[list[str], str]:
    """Extract dependency and purpose metadata from a Python source file."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return [], "unspecified"
    deps: set[str] = set()
    legal_function = "unspecified"
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, ast.Str)
    ):
        legal_function = tree.body[0].value.s.strip().splitlines()[0]
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                deps.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            deps.add(node.module.split(".")[0])
    return sorted(deps), legal_function


def verify_codex_config() -> None:
    """Ensure the codex configuration file exists and has required keys."""
    data = json.loads(CONFIG_PATH.read_text())
    required = ["tests", "linting", "branch_rules"]
    missing = [k for k in required if k not in data]
    if missing:
        raise RuntimeError(f"Missing codex config keys: {', '.join(missing)}")


def update_manifest() -> None:
    manifest = []
    for p in Path(".").rglob("*.py"):
        if p.parts[0].startswith("."):
            continue
        deps, legal_fn = parse_metadata(p)
        manifest.append(
            {
                "module": p.stem,
                "path": str(p),
                "hash": hash_file(p),
                "legal_function": legal_fn,
                "dependencies": deps,
            }
        )
    Path(MANIFEST).write_text(json.dumps(manifest, indent=2))


def main() -> None:
    verify_codex_config()
    run_guardian()
    update_manifest()
    self_diagnostic()
    print("codex manifest updated")


if __name__ == "__main__":
    main()
