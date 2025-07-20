# FRED Prime Litigation System

Note: this project is a simplified prototype. It does not provide a full litigation operating system. Use it only as a starting point and verify all outputs manually.
This repository contains scripts for building a local litigation toolkit. The latest additions include a **Warboard Visualizer** that assembles a timeline of events and contradictions into DOCX and SVG maps. The generator can upload results to Google Drive when a `token.json` credential file is present.

## Components
- `gui/frontend.py` – Tkinter GUI with tabs for multiple warboards
- `warboard/warboard_engine.py` – builds `SHADY_OAKS_WARBOARD` exports
- `warboard/ppo_warboard.py` – constructs the PPO timeline
- `warboard/custody_interference_engine.py` – maps custody interference events
- `warboard/svg_builder.py` – SVG helper used by all warboards
- `gdrive_sync.py` – optional helper for uploading generated files to Google Drive (requires `token.json`)
- `warboard/warboard_matrix_export.py` – bundle all warboard outputs into a ZIP archive
- `requirements.txt` – minimal Python dependencies

Run the GUI with:
```bash
python gui/frontend.py
```

To generate the warboard without the GUI run:

```bash
python -m warboard.warboard_engine
```

Additional warboards can be built with:

```bash
python -m warboard.ppo_warboard
python -m warboard.custody_interference_engine
```

To bundle all warboard outputs into a single ZIP archive run:

```bash
python -m warboard.warboard_matrix_export
```

## Scanning local drives

The repository now includes a simple scanner that can index `.docx`, `.txt`, and `.pdf` files from local drives. By default it scans the Windows-style drives `F:/` and `D:/` if they exist. You can set the environment variable `SCAN_DRIVES` to a list of paths separated by your OS path separator to override the defaults. The results are saved to `data/scan_index.json`. A timeline can then be built from this index:

```bash
# example overriding drives on Linux
SCAN_DRIVES="/mnt/data1:/mnt/data2" python -m scanner.scan_engine
python -m timeline.builder
```

You can also pass one or more paths as arguments to `scanner.scan_engine`.

The warboard engine reads `data/timeline.json`, so running the scanner and timeline builder before generating the warboard will populate the SVG and DOCX with real events.

