"""Simple agent discovery and loader for the `agents/` folder.

Usage:
    from agents import loader
    agents = loader.discover_agents()

This loader reads each agent's `config.yaml` and exposes a simple dict
of available agents. It's intentionally lightweight and avoids external
dependencies.
"""
import os
from pathlib import Path

# Prefer PyYAML for config parsing if available; otherwise fall back to a
# small builtin parser used earlier. This lets callers use richer YAML when
# desired without hard dependency.
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


def read_simple_yaml(path: Path):
    """Very small YAML-like reader for our simple `config.yaml` files.

    It supports key: value pairs and ignores complexities. For richer
    configs, use `pyyaml`.
    """
    data = {}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return data
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip().strip('"')
    return data


def discover_agents(root: str = None):
    """Discover agent folders under `agents/` and return metadata dict.

    Returns a dict mapping agent id -> {path, config}
    """
    base = Path(root) if root else Path(__file__).parent
    results = {}
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        cfg = entry / "config.yaml"
        if cfg.exists():
            if _HAS_YAML:
                try:
                    meta = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
                except Exception:
                    meta = read_simple_yaml(cfg)
            else:
                meta = read_simple_yaml(cfg)
            agent_id = meta.get("id") or entry.name
            results[agent_id] = {"path": str(entry.resolve()), "config": meta}
    return results
