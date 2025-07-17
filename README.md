# FRED PRIME Legal System

This repository contains an early prototype of the **FRED PRIME Litigation Deployment Engine**. The engine is intended to assist with automating various tasks involved in the litigation process (exhibit labeling, motion linking, signature validation, etc.). The current repository includes a single Python script that generates the engine configuration in JSON format.

## Repository Contents

- `firstimport.json` – A short Python script that creates `fredprime_litigation_system.json` describing the FRED PRIME configuration. The resulting JSON file is written to `/mnt/data` when the script is executed.
- `FRED_Codex_Bootstrap.py` – Downloads the stage two deployment archive from `FRED_STAGE2_URL`, verifies its SHA256 checksum using `FRED_STAGE2_SHA256`, and extracts the archive.
- `EPOCH_UNPACKER_ENGINE_v1.py` – Extracts ZIP archives, performs OCR, and tags exhibits. Can run with a GUI or in headless mode.

## Configuration

The script expects an environment variable named `FREDPRIME_REPO_PATH`. Set this
variable to the path or URL of the repository you want to reference before
running the script. The default value `/path/to/repo` is only a placeholder and
should be replaced with the intended source.

## Usage

1. Ensure you have Python 3 installed.
2. Clone this repository and change into the project directory.
3. Run the script:
   ```bash
   python3 firstimport.json
   ```
   The script writes `fredprime_litigation_system.json` to `/mnt/data`.
4. Use the generated JSON with the accompanying PowerShell scripts (not included in this repository) to deploy the litigation engine.
5. (Optional) Download additional components:
   ```bash
   python3 FRED_Codex_Bootstrap.py
   ```
   This fetches and extracts the stage two deployment archive referenced by `FRED_STAGE2_URL`.
6. (Optional) Process evidence ZIPs:
   ```bash
   python3 EPOCH_UNPACKER_ENGINE_v1.py gui
   ```
   Use `process` instead of `gui` to run without the graphical interface.

## Build Instructions

There is no build process for this repository. The Python script may be run directly. If you wish to package or modify the deployment engine, edit `firstimport.json` and regenerate the JSON file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
