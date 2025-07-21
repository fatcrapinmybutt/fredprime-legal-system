# FRED PRIME Legal System

This repository contains an early prototype of the **FRED PRIME Litigation Deployment Engine**. The engine is intended to assist with automating various tasks involved in the litigation process (exhibit labeling, motion linking, signature validation, etc.).

## Repository Contents

- `build_system.py` – Python helper that generates `fredprime_litigation_system.json` describing the FRED PRIME configuration. The resulting JSON file is written to `/mnt/data` by default.
- `FRED_Codex_Bootstrap.py` – Downloads the stage two deployment archive from `FRED_STAGE2_URL`, verifies its SHA256 checksum using `FRED_STAGE2_SHA256`, and extracts the archive.
- `EPOCH_UNPACKER_ENGINE_v1.py` – Extracts ZIP archives, performs OCR, and tags exhibits. Can run with a GUI or in headless mode.

## Usage

1. Ensure you have Python 3 installed.
2. Clone this repository and change into the project directory.
3. Run the script:
   ```bash
   python3 build_system.py
