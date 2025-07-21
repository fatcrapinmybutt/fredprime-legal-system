# AGENTS Instructions

This repository is governed by the "MBP Codex Supreme" ruleset. The key points are:

- Run `black`, `mypy --strict`, `flake8`, `pytest`, and any selftests when code logic changes or when branches include `core`, `engine`, `matrix`, `protocol`, `epoch`, `echelon`, `patch`, or `hotfix`.
- Skip tests for pure documentation changes.
- Branches should follow `codex/{feature-name}`. Branch names containing the above keywords trigger full builds.
- All modules must have a corresponding file under `tests/`.
- Manifest enforcement: every logic file is hashed and recorded in `codex_manifest.json` with module name, path, hash, legal function, and dependencies.
- Prohibit `eval` and `exec`.
- Reject commits containing the keywords `TODO`, `WIP`, `temp_var`, or `placeholder`.
- Required project folders: `core`, `modules`, `gui`, `cli`, `config`, `docs`, `tests`.
- Output files are written to `output/ZIP/` and `output/docx/`.
- Commit messages follow `[type] message`, where `type` includes `core`, `hotfix`, `docs`, `merge`, `patch`, `engine`, `matrix`, or `echelon`.

These rules apply to all future contributions and code changes.
