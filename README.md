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
- `foia/autopacker.py` – create basic FOIA request documents and ZIP them
- `press/press_draft_engine.py` – generate a press-summary document
- `motions/protective_order.py` – build a sample protective order motion
- `gui/modules/entity_suppression_feed.py` – track filings that were rejected
- `judge_sim_ladas_hoopes_v1.py` – example outcome predictor
- `contradictions/contradiction_matrix.py` – create a simple contradiction log
- `requirements.txt` – minimal Python dependencies
- `entity_trace/ai_entity_review.py` – generate a sample entity overlap report
- `violations/misconduct_letter.py` – create a judicial misconduct letter
- `federal/complaint_generator.py` – draft a placeholder federal complaint
- `motions/emergency_injunction.py` – build an emergency injunction motion
- `scheduling/scheduler.py` – export a court calendar and ICS file

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

## Other helpers

Generate the entity overlap report:

```bash
python -m entity_trace.ai_entity_review
```

Create the judicial misconduct letter:

```bash
python -m violations.misconduct_letter
```

Draft the federal complaint and emergency injunction motion:

```bash
python -m federal.complaint_generator
python -m motions.emergency_injunction
```

Build a simple court calendar from the timeline:

```bash
python -m scheduling.scheduler
```

Create FOIA request packet and press summary:

```bash
python -m foia.autopacker
python -m press.press_draft_engine
```

Generate a protective order motion:

```bash
python -m motions.protective_order
```

Run the contradiction detector:

```bash
python -m contradictions.contradiction_matrix
```


Additional helper scripts added in this release:

- `timeline/fusion_engine.py` – merge manual post-writ events with the scan-based timeline and create a simple SVG overview.
- `mifile/stack_dispatcher.py` – package key motions into a MiFile-ready ZIP bundle.
- `foia/video_request_builder.py` – generate a FOIA request for sheriff bodycam footage.
- `notices/notice_of_claim.py` – create a basic notice of claim under 42 USC §1983.
- `binder/tab_forger.py` – build a Binder Tab C document listing post-writ exhibits.

Example usage:

```bash
python -m timeline.fusion_engine
python -m mifile.stack_dispatcher
python -m foia.video_request_builder
python -m notices.notice_of_claim
python -m binder.tab_forger
```
