import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, cast

from modules.codex_guardian import run_guardian
from modules.codex_supreme import self_diagnostic

MANIFEST = "codex_manifest.json"
CONFIG_PATH = Path(".codex_config.yaml")


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError("Missing .codex_config.yaml configuration file")
    try:
        return cast(Dict[str, Any], json.loads(CONFIG_PATH.read_text()))
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON in .codex_config.yaml") from exc


def validate_required_structure(config: Dict[str, Any]) -> None:
    enforcement = config.get("enforcement", {})
    raw_dirs: Iterable[Any]
    if isinstance(enforcement, dict):
        raw_dirs = enforcement.get("required_directories", [])
    else:
        raw_dirs = []
    for rel_path in [str(item) for item in raw_dirs]:
        path = Path(rel_path)
        if not path.exists():
            raise FileNotFoundError(f"Required directory missing: {path}")


def should_track(path: Path, config: Dict[str, Any]) -> bool:
    manifest_settings = config.get("manifest", {})
    raw_extensions: Iterable[Any] = [".py"]
    if isinstance(manifest_settings, dict):
        raw_extensions = manifest_settings.get("track_extensions", [".py"])
    extensions = {str(ext) for ext in raw_extensions}
    return path.suffix in extensions


def determine_legal_function(path: Path, config: Dict[str, Any]) -> str:
    manifest_settings = config.get("manifest", {})
    legal_map: Dict[str, str] = {}
    if isinstance(manifest_settings, dict):
        raw_map = manifest_settings.get("legal_map", {})
        if isinstance(raw_map, dict):
            legal_map = {str(key): str(value) for key, value in raw_map.items()}
    root = path.parts[0] if path.parts else "root"
    return legal_map.get(root, "general-litigation-logic")


def extract_dependencies(path: Path) -> List[str]:
    dependencies: Set[str] = set()
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("import "):
            modules = stripped.partition(" ")[2].split(",")
            for module in modules:
                dependencies.add(module.strip().split(" ")[0].split(".")[0])
        elif stripped.startswith("from ") and " import " in stripped:
            module = stripped.split()[1]
            dependencies.add(module.split(".")[0])
    return sorted(dep for dep in dependencies if dep)


def load_existing_manifest() -> Dict[str, Dict[str, Any]]:
    if Path(MANIFEST).exists():
        try:
            data = cast(List[Dict[str, Any]], json.loads(Path(MANIFEST).read_text()))
            return {entry["path"]: entry for entry in data}
        except json.JSONDecodeError:
            return {}
    return {}


def build_manifest_entry(
    path: Path, config: Dict[str, Any], existing: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    entry = {
        "module": path.stem,
        "path": str(path),
        "hash": hash_file(path),
        "legal_function": determine_legal_function(path, config),
        "dependencies": extract_dependencies(path),
    }
    if path_entry := existing.get(str(path)):
        if path_entry.get("legal_function"):
            entry["legal_function"] = path_entry["legal_function"]
        if path_entry.get("dependencies"):
            entry["dependencies"] = sorted(set(path_entry["dependencies"]))
    return entry


def update_manifest(config: Dict[str, Any]) -> None:
    existing_entries = load_existing_manifest()
    manifest: List[Dict[str, Any]] = []
    for file_path in Path(".").rglob("*.py"):
        if not file_path.parts:
            continue
        if file_path.parts[0].startswith("."):
            continue
        if "__pycache__" in file_path.parts:
            continue
        if should_track(file_path, config):
            manifest.append(build_manifest_entry(file_path, config, existing_entries))
    Path(MANIFEST).write_text(json.dumps(manifest, indent=2))


def run_high_priority_checks(config: Dict[str, Any]) -> None:
    test_config = config.get("tests", {})
    commands: Iterable[str] = []
    validator_path: str | None = None
    if isinstance(test_config, dict):
        commands = [str(command) for command in test_config.get("commands", [])]
        validator_value = test_config.get("zip_validator_path")
        if isinstance(validator_value, str):
            validator_path = validator_value
    for command in commands:
        subprocess.run(shlex.split(command), check=True)
    if validator_path and Path(validator_path).exists():
        subprocess.run(["python", validator_path], check=True)


def main() -> None:
    config = load_config()
    validate_required_structure(config)
    high_priority_branch = run_guardian()
    update_manifest(config)
    diagnostics = self_diagnostic()
    if high_priority_branch:
        run_high_priority_checks(config)
    for line in diagnostics:
        print(line)
    print("codex manifest updated")


if __name__ == "__main__":
    main()
