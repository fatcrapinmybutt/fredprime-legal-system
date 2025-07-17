# FRED PRIME Legal System

This repository contains an early prototype of the **FRED PRIME Litigation Deployment Engine**. The engine is intended to assist with automating various tasks involved in the litigation process (exhibit labeling, motion linking, signature validation, etc.). The current repository includes a single Python script that generates the engine configuration in JSON format.

## Repository Contents

- `firstimport.json` – A short Python script that creates `fredprime_litigation_system.json` describing the FRED PRIME configuration. The resulting JSON file is written to `/mnt/data` when the script is executed.

## Usage

1. Ensure you have Python 3 installed.
2. Clone this repository and change into the project directory.
3. Run the script:
   ```bash
   python3 firstimport.json
   ```
   The script writes `fredprime_litigation_system.json` to `/mnt/data`.
4. Use the generated JSON with the accompanying PowerShell scripts (not included in this repository) to deploy the litigation engine.

## Build Instructions

There is no build process for this repository. The Python script may be run directly. If you wish to package or modify the deployment engine, edit `firstimport.json` and regenerate the JSON file.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
