# Development setup

Quick steps to prepare a development environment and enforce formatting/hooks.

1. Create a virtualenv and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt || true
pip install pre-commit
```

2. Install pre-commit hooks locally:

```bash
pre-commit install
pre-commit run --all-files
```

3. Run tests:

```bash
pytest
```

Notes:

- The repo includes Black, isort and Flake8 hooks via `.pre-commit-config.yaml`.
- Use `pre-commit run --all-files` to autoformat and detect issues before commits.
