# FRED PRIME Litigation Deployment System

This repository contains tools for the FRED PRIME litigation automation workflow.
The main component is the **EPOCH Unpacker**, which extracts case documents from
ZIP archives, runs OCR, flags potential canon issues, and classifies exhibits.

## Running the EPOCH Unpacker

```bash
python EPOCH_UNPACKER_ENGINE_v1.py gui
```
Launches the graphical interface where you can choose a ZIP archive. The files
are extracted to `unzipped_epoch` under the base directory.

```bash
python EPOCH_UNPACKER_ENGINE_v1.py process /path/to/archive.zip
```
Processes the archive in headless mode and prints progress for each file.

Use `--dir <directory>` to override the extraction directory and
`--reset` to clear cached logs and the processing queue.

The base directory defaults to the current folder, but you can set the
`LITIGATION_DATA_DIR` environment variable to store data and logs elsewhere.
