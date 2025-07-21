# Agents and Modules Overview

This repository is the FRED PRIME / MBP Supreme Litigation OS. It contains a large collection of
modules and utilities for Michigan focused litigation. The list below summarises the key
components and their roles.

## Repository Layout

| Path | Purpose |
| ---- | ------- |
| `core/` | Base engines and system level utilities. |
| `modules/` | Stand‑alone features such as evidence processing, timeline builders and automation helpers. |
| `gui/` | Graphical user interface files. |
| `cli/` | Command line entry points. |
| `docs/` | Supplemental documentation. |
| `tests/` | Unit tests for each module. |
| `output/` | Generated reports and artefacts. |

The root contains scripts like `build_system.py`, `codex_brain.py` and the
`MBP_Omnia_Engine.py` which combine all modules into the full litigation OS.

## Development Rules

* **Commit messages** must follow `[type] message` format. Example: `[docs] Update help`. Valid
  types include `core`, `hotfix`, `docs`, `merge` and similar short descriptors.
* **Code changes** trigger the test suite. Documentation‑only edits skip tests.
* **Manifest enforcement:** every logic file is hashed and recorded in `codex_manifest.json`.
  Commits that modify code should update this manifest using the provided tools.
* **No placeholders** such as `TODO` or `WIP` may be left in committed code.

## Using the System

Each major script or module logs actions to the immutable ledger. The main workflows are:

1. **Evidence intake** using `organize_drive.py` and `EPOCH_UNPACKER_ENGINE_v1.py`.
2. **Manifest build** via `build_system.py` which records modules and file hashes.
3. **Patch management** handled by `codex_patch_manager.py`.
4. **GUI launch** with files under `gui/` providing access to scanning, audits and notarisation.
5. **Supreme notarisation** is integrated via `quantum_blockchain_ai_extension.py` which can
   anchor hashes to public chains and run adversarial analysis. Use the GUI option or run
   `python core/quantum_blockchain_ai_extension.py <file>`.

The system focuses on Michigan law, including all MCR and MCL references. Benchbook guidance and
SCAO/FOC forms are embedded in several modules. See `README.md` for a more detailed walkthrough.

