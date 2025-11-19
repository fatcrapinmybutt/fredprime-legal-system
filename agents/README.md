# Agents scaffolding for Michigan court processes

This folder contains lightweight agent scaffolds intended to represent
specialist agents for Michigan court rules, procedures and filing
practices. Each agent includes:

- `agent_core.py` — runnable placeholder exposing a small `Agent` class.
- `config.yaml` — basic metadata used by the loader.

Shared modules:

- `michigan_reference.py` — a minimal local index of commonly-cited
  Michigan Court Rules and Michigan Rules of Evidence for offline use.
- `loader.py` — small discovery helper that reads `config.yaml` files.

## Usage examples

Discover agents:

```python
from agents import loader
agents = loader.discover_agents()
print(list(agents.keys()))
```

Run an agent (simple):

```bash
python -m agents.agent_001.agent_core
```

## Notes

These scaffolds are intentionally lightweight. For production use,
replace the simple YAML reader with `pyyaml`, add robust logging and
error handling, and expand the `michigan_reference` dataset.
