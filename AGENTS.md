# Agents and Modules Overview

This repository is the FRED PRIME / MBP Supreme Litigation OS. It contains many utilities for Michigan-focused litigation. Below is a detailed summary of key folders, development rules, and how to run notarisation.

## Repository Layout

| Path | Purpose |
| ---- | ------- |
| `binder/` | Tools to generate exhibit binders and tab sheets. |
| `cli/` | Command line entry points such as `generate_manifest.py`. |
| `contradictions/` | Evidence contradiction scanners and helpers. |
| `docs/` | Additional documentation and references. |
| `entity_trace/` | Scripts that map relationships between defendants. |
| `foia/` | FOIA helpers and discovery templates. |
| `gui/` | Tkinter based user interface files. |
| `modules/` | Utilities like `codex_manifest.py` that build manifest data. |
| `mifile/` | Michigan specific legal resources. |
| `motions/` | Model motions and language snippets. |
| `notices/` | Form notices and court filings. |
| `scanner/` | Evidence scanning utilities and OCR. |
| `scheduling/` | Scripts for court scheduling calculations. |
| `src/` | Installable package source for CLI commands. |
| `tests/` | Unit tests for modules. |
| `timeline/` | Timeline builders and exhibit ordering tools. |
| `warboard/` | Visual case-mapping utilities. |
| root scripts | `build_system.py`, `codex_brain.py`, `codex_patch_manager.py`, `MBP_Omnia_Engine.py`. |

## Development Rules

* Commit messages must follow `[type] message` format (`core`, `docs`, `hotfix`, etc.).
* Code changes trigger lint and tests. Documentation‑only commits skip testing.
* Every logic file is hashed and recorded in `codex_manifest.json`.
* Do not commit placeholders such as `TODO` or `WIP`.

## Using the System

1. **Evidence intake**: run `organize_drive.py` and `EPOCH_UNPACKER_ENGINE_v1.py` to sort and OCR evidence.
2. **Manifest build**: execute `build_system.py` to register modules and hashes.
3. **Patch management**: apply updates via `codex_patch_manager.py`.
4. **GUI launch**: start the interface under `gui/` for scanning, audits, and binder export.
5. **Notarisation**: use `core/quantum_blockchain_ai_extension.py <file>` to anchor hashes to public chains and run adversarial analysis.

Michigan MCR, MCL and Benchbook guidance are embedded across the modules. See `README.md` for a walkthrough of common workflows and additional commands.
