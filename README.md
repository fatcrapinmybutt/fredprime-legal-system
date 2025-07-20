# FRED Prime Litigation System

This repository contains scripts for building a local litigation toolkit. The latest addition is a **Warboard Visualizer** that assembles a timeline of events and contradictions into a DOCX and SVG map.

## Components
- `gui/frontend.py` – launches a Tkinter GUI with a Warboard Visualizer tab
- `warboard/warboard_engine.py` – builds `SHADY_OAKS_WARBOARD.docx` and `SHADY_OAKS_WARBOARD.svg`
- `warboard/svg_builder.py` – generates the SVG timeline from `data/timeline.json`
- `gdrive_sync.py` – optional helper for uploading generated files to Google Drive (requires `token.json`)
- `requirements.txt` – minimal Python dependencies

Run the GUI with:
```bash
python gui/frontend.py
```
