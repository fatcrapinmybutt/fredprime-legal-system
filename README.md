# FRED Prime Litigation System

This repository contains scripts for building a local litigation toolkit. The latest additions include a **Warboard Visualizer** that assembles a timeline of events and contradictions into DOCX and SVG maps. The generator can upload results to Google Drive when a `token.json` credential file is present.

## Components
- `gui/frontend.py` – Tkinter GUI with tabs for multiple warboards
- `warboard/warboard_engine.py` – builds `SHADY_OAKS_WARBOARD` exports
- `warboard/ppo_warboard.py` – constructs the PPO timeline
- `warboard/custody_interference_engine.py` – maps custody interference events
- `warboard/svg_builder.py` – SVG helper used by all warboards
- `gdrive_sync.py` – optional helper for uploading generated files to Google Drive (requires `token.json`)
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
