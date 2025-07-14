# EPOCH_UNPACKER_ENGINE_v1.py
# Now integrated with Litigation OS Control Panel + Exhibit Linker + Canon Detection + Live Progress GUI

import zipfile
import os
import json
from pathlib import Path
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import hashlib
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import argparse

# === CONFIGURATION === #
# Allow overriding the base working directory so this tool can
# integrate with other parts of the litigation OS. Defaults to the
# `LITIGATION_DATA_DIR` environment variable if present or falls back
# to the current directory.
BASE_DIR = Path(os.environ.get("LITIGATION_DATA_DIR", "."))

def set_base_dir(path: Path):
    """Update all file paths to use a new base directory."""
    global BASE_DIR, EXTRACT_DIR, QUEUE_FILE, OCR_LOG, CANON_LOG, EXHIBIT_LOG, PROGRESS_FILE
    BASE_DIR = Path(path)
    EXTRACT_DIR = BASE_DIR / "unzipped_epoch"
    QUEUE_FILE = BASE_DIR / "epoch_queue.json"
    OCR_LOG = BASE_DIR / "ocr_output.json"
    CANON_LOG = BASE_DIR / "canon_flags.json"
    EXHIBIT_LOG = BASE_DIR / "exhibit_log.json"
    PROGRESS_FILE = BASE_DIR / "progress_status.json"

set_base_dir(BASE_DIR)

# === UTILITIES === #
def sha256_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def load_queue():
    if QUEUE_FILE.exists():
        with open(QUEUE_FILE, 'r') as f:
            return json.load(f)
    return {"index": []}

def save_queue(queue):
    with open(QUEUE_FILE, 'w') as f:
        json.dump(queue, f, indent=2)

def log_ocr(filename, content):
    logs = {}
    if OCR_LOG.exists():
        with open(OCR_LOG, 'r') as f:
            logs = json.load(f)
    logs[filename] = content
    with open(OCR_LOG, 'w') as f:
        json.dump(logs, f, indent=2)

def log_progress(status):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(status, f, indent=2)

def run_canon_validator(text):
    flags = []
    canon_triggers = ["bias", "impartiality", "canon", "due process", "appearance of impropriety", "judicial misconduct"]
    for term in canon_triggers:
        if term in text.lower():
            flags.append(f"\u26a0\ufe0f Canon Flag: '{term}'")
    return flags

def log_canon(filename, flags):
    logs = {}
    if CANON_LOG.exists():
        with open(CANON_LOG, 'r') as f:
            logs = json.load(f)
    logs[filename] = flags
    with open(CANON_LOG, 'w') as f:
        json.dump(logs, f, indent=2)

def run_exhibit_classifier(text):
    classifications = []
    keywords = {
        "Rent Ledger": ["rent", "balance", "amount due", "ledger"],
        "Utility Bill": ["electric", "water", "sewer", "usage", "trash"],
        "Eviction Notice": ["notice to quit", "7-day", "termination", "possession"],
        "Judicial Order": ["order", "signed by judge", "court order"],
        "Custody Record": ["parenting time", "custody", "visitation"]
    }
    for exhibit_type, triggers in keywords.items():
        for word in triggers:
            if word in text.lower():
                classifications.append(exhibit_type)
                break
    return list(set(classifications))

def log_exhibit(filename, types):
    logs = {}
    if EXHIBIT_LOG.exists():
        with open(EXHIBIT_LOG, 'r') as f:
            logs = json.load(f)
    logs[filename] = types
    with open(EXHIBIT_LOG, 'w') as f:
        json.dump(logs, f, indent=2)

# === ZIP UNPACKER === #
def unpack_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        queue = load_queue()
        for file in file_list:
            file_hash = sha256_hash(file)
            if not any(q["hash"] == file_hash for q in queue["index"]):
                zip_ref.extract(file, EXTRACT_DIR)
                queue["index"].append({"filename": file, "hash": file_hash, "status": "pending"})
        save_queue(queue)

# === OCR + Canon + Exhibit Processor === #
def process_next_file():
    queue = load_queue()
    for item in queue["index"]:
        if item["status"] == "pending":
            full_path = EXTRACT_DIR / item["filename"]
            try:
                if str(full_path).lower().endswith(".pdf"):
                    reader = PdfReader(full_path)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() or ""
                elif str(full_path).lower().endswith(('.png', '.jpg', '.jpeg')):
                    image = Image.open(full_path)
                    text = pytesseract.image_to_string(image)
                else:
                    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()

                log_ocr(item["filename"], text.strip())
                canon_flags = run_canon_validator(text)
                log_canon(item["filename"], canon_flags)
                exhibit_tags = run_exhibit_classifier(text)
                log_exhibit(item["filename"], exhibit_tags)

                item["status"] = "done"
                save_queue(queue)
                log_progress({"current": item["filename"], "status": "done"})
                return item["filename"]
            except Exception as e:
                item["status"] = "error"
                item["error"] = str(e)
                save_queue(queue)
                log_progress({"current": item["filename"], "status": f"error: {e}"})
                return None
    return None

# === GUI === #
def run_gui():
    def select_file():
        file_path = filedialog.askopenfilename(title="Select ZIP File", filetypes=[("ZIP files", "*.zip")])
        if file_path:
            zip_entry.delete(0, tk.END)
            zip_entry.insert(0, file_path)

    def start_processing():
        def run():
            zip_path = zip_entry.get()
            if not EXTRACT_DIR.exists():
                EXTRACT_DIR.mkdir(parents=True)
            unpack_zip(zip_path)
            processed = 0
            while True:
                result = process_next_file()
                if not result:
                    break
                processed += 1
                progress_var.set(f"Processed {processed} files...")
            progress_var.set("All files processed.")
            messagebox.showinfo("Done", "All files processed.")
        threading.Thread(target=run, daemon=True).start()

    window = tk.Tk()
    window.title("EPOCH UNPACKER | Litigation OS Panel")
    progress_var = tk.StringVar(value="Idle")
    tk.Label(window, text="ZIP File Path:").pack()
    zip_entry = tk.Entry(window, width=60)
    zip_entry.pack()
    tk.Button(window, text="Browse", command=select_file).pack()
    tk.Button(window, text="Start Scan", command=start_processing).pack()
    tk.Label(window, textvariable=progress_var).pack()
    window.mainloop()

# === HEADLESS MODE === #
def run_headless(zip_path):
    if not EXTRACT_DIR.exists():
        EXTRACT_DIR.mkdir(parents=True)
    unpack_zip(zip_path)
    while True:
        result = process_next_file()
        if not result:
            break
        print(f"Processed {result}")
    print("All files processed.")

# === ENTRY POINT === #
def parse_args():
    parser = argparse.ArgumentParser(description="EPOCH Unpacker")
    sub = parser.add_subparsers(dest="command")

    gui_p = sub.add_parser("gui", help="Launch graphical interface")
    gui_p.add_argument('--dir', default=str(EXTRACT_DIR), help='Base directory for extracted files and logs')

    proc_p = sub.add_parser("process", help="Process ZIP without GUI")
    proc_p.add_argument('zip', help='Path to ZIP archive to process')
    proc_p.add_argument('--dir', default=str(EXTRACT_DIR), help='Base directory for extracted files and logs')

    parser.add_argument('--reset', action='store_true', help='Clear cached logs and queue')
    return parser.parse_args()


def reset_logs():
    for path in [QUEUE_FILE, OCR_LOG, CANON_LOG, EXHIBIT_LOG, PROGRESS_FILE]:
        if path.exists():
            path.unlink()


if __name__ == '__main__':
    args = parse_args()

    if args.reset:
        reset_logs()

    if args.command == 'process':
        set_base_dir(Path(args.dir))
        run_headless(args.zip)
    else:
        if args.command == 'gui' and hasattr(args, 'dir'):
            set_base_dir(Path(args.dir))
        run_gui()
