import ast
import datetime
import json
from pathlib import Path
from typing import Any, cast

import yaml

from modules.hash_utils import hash_file

from modules.codex_guardian import run_guardian
from modules.codex_supreme import self_diagnostic

MANIFEST = "codex_manifest.json"

CONFIG_FILE = ".codex_config.yaml"


def enforce_source_hygiene(
    path: Path, tokens: tuple[str, ...], calls: tuple[str, ...]
) -> None:
    text = path.read_text()
    for token in tokens:
        if token in text:
            raise ValueError(f"{path} contains forbidden token '{token}'")
    for call in calls:
        if call in text:
            raise ValueError(f"{path} contains forbidden call '{call}'")


def collect_dependencies(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return []
    deps: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                deps.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                deps.append(node.module)
    return deps


def update_manifest(tokens: tuple[str, ...], calls: tuple[str, ...]) -> None:
    manifest: list[dict[str, Any]] = []
    for p in Path(".").rglob("*.py"):
        if p.parts[0].startswith("."):
            continue
        if p.name != "codex_guardian.py" and "tests" not in p.parts:
            enforce_source_hygiene(p, tokens, calls)
        manifest.append(
            {
                "module": p.stem,
                "path": str(p),
                "hash": hash_file(p),
                "legal_function": p.stem.replace("_", " ").title(),
                "dependencies": collect_dependencies(p),
                "timestamp": datetime.datetime.fromtimestamp(
                    p.stat().st_mtime
                ).isoformat(),
                "validated": False,
            }
        )
    Path(MANIFEST).write_text(json.dumps(manifest, indent=2))


def load_config() -> dict[str, Any]:
    if not Path(CONFIG_FILE).exists():
        raise FileNotFoundError(CONFIG_FILE)
    data = yaml.safe_load(Path(CONFIG_FILE).read_text())
    return cast(dict[str, Any], data)


def main() -> None:
    run_guardian()
    config = load_config()
    tokens = tuple(config.get("forbidden_tokens", []))
    calls = tuple(config.get("forbidden_calls", []))
    update_manifest(tokens, calls)
    self_diagnostic()
    print("codex manifest updated")


if __name__ == "__main__":
    main()
