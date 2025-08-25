import os
import shutil
import json
import zipfile
import hashlib
import magic
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from rich import print
from transformers import pipeline
import spacy

# Install necessary dependencies (if running the script for the first time)
try:
    import transformers
    import datasets
    import torch
except ImportError:
    os.system("pip install transformers datasets torch tqdm rich python-docx spacy magic")

# Initialize NLP Model - BART model for classification
nlp_model = pipeline('zero-shot-classification', model="facebook/bart-large-mnli")

# File Paths (Adjust for your drives)
DRIVES = ["E:/", "D:/", "F:/"]
LOG_DIR = os.path.join("F:/", "LegalResults/_SCAN_LOGS")
SORTED_DIR = os.path.join("F:/", "LegalResults/SortedFiles")
EXTRACTED_ZIP_DIR = os.path.join(SORTED_DIR, "_ExtractedZIPs")
EXHIBIT_BUNDLE_DIR = os.path.join(SORTED_DIR, "ExhibitBundles")

ALL_FILE_DATA = []

SUPPORTED_TYPES = {
    "txt": "TEXT DOCUMENT",
    "pdf": "PDF DOCUMENT",
    "png": "IMAGE (PNG)",
    "jpg": "IMAGE (JPG)",
    "jpeg": "IMAGE (JPEG)",
    "gif": "IMAGE (GIF)",
    "docx": "WORD DOCUMENT (DOCX)",
    "odt": "OPEN DOCUMENT (ODT)",
    "xlsx": "EXCEL SPREADSHEET (XLSX)",
    "pptx": "POWERPOINT DOCUMENT (PPTX)",
    "avi": "VIDEO (AVI)",
    "mp4": "VIDEO (MP4)",
    "mov": "VIDEO (MOV)",
    "mp3": "AUDIO (MP3)",
    "wav": "AUDIO (WAV)",
    "zip": "ZIP ARCHIVE",
    "rar": "RAR ARCHIVE",
    "html": "HTML DOCUMENT",
    "csv": "CSV FILE",
    "json": "JSON FILE",
    "xml": "XML FILE",
    "md": "MARKDOWN FILE",
    "eml": "EMAIL FILE",
    "exe": "EXECUTABLE FILE",
    "apk": "ANDROID PACKAGE",
    "iso": "ISO IMAGE",
}

# Create directories if not exists
def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

# Calculate the SHA-256 hash of a file (for integrity verification)
def hash_file(filepath):
    sha = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()
    except Exception as e:
        return f"ERROR: {str(e)}"

# Retrieve file information (type, size, timestamp, hash)
def get_file_info(filepath):
    try:
        filetype = magic.from_file(filepath, mime=True)
        size = os.path.getsize(filepath)
        modified = datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()
        sha256 = hash_file(filepath)
        return filetype, size, modified, sha256
    except Exception as e:
        return "unknown", 0, "error", f"ERROR: {str(e)}"

# Classify the file using NLP model (AI-based classification)
def classify_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as file:
            text = file.read(5000)  # Read first 5000 characters for classification
            labels = ["Complaint", "Motion", "Affidavit", "Legal Brief", "Evidence", "Other"]
            result = nlp_model(text, candidate_labels=labels)
            return result['labels'][0]  # Classify based on highest probability
    except Exception as e:
        return SUPPORTED_TYPES.get(filepath.split('.')[-1], "OTHER")

# Extract ZIP file to a specific destination
def extract_zip(filepath, dest):
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            inner_dir = os.path.join(dest, os.path.basename(filepath).replace(".zip", ""))
            safe_mkdir(inner_dir)
            zip_ref.extractall(inner_dir)
            return inner_dir
    except Exception as e:
        return f"EXTRACTION_ERROR: {str(e)}"

# Scan the directory recursively and process each file
def scan_directory(drives):
    exhibit_counter = 1
    for drive in drives:
        for root, _, files in os.walk(drive):
            for file in files:
                try:
                    fullpath = os.path.join(root, file)
                    ftype, fsize, mtime, fhash = get_file_info(fullpath)
                    category = classify_file(fullpath)

                    file_record = {
                        "file_path": fullpath,
                        "category": category,
                        "type": ftype,
                        "size_bytes": fsize,
                        "modified_time": mtime,
                        "sha256": fhash,
                        "exhibit_label": f"Exhibit {exhibit_counter}",
                        "file_description": f"Legal document related to {category}",
                    }

                    if category == "ZIP ARCHIVE" or category == "RAR ARCHIVE":
                        extracted = extract_zip(fullpath, EXTRACTED_ZIP_DIR)
                        file_record["extracted_to"] = extracted

                    # Organize files into exhibit bundles
                    exhibit_folder = os.path.join(EXHIBIT_BUNDLE_DIR, f"Exhibit_{exhibit_counter}")
                    safe_mkdir(exhibit_folder)
                    shutil.copy(fullpath, exhibit_folder)

                    ALL_FILE_DATA.append(file_record)
                    exhibit_counter += 1

                except Exception as e:
                    print(f"[red]Failed on: {file}[/red] — {str(e)}")

# Write the scanned files data to CSV, JSON, and TXT logs
def write_outputs():
    df = pd.DataFrame(ALL_FILE_DATA)
    safe_mkdir(LOG_DIR)
    df.to_csv(os.path.join(LOG_DIR, "file_inventory.csv"), index=False)
    with open(os.path.join(LOG_DIR, "file_inventory.json"), "w") as jf:
        json.dump(ALL_FILE_DATA, jf, indent=2)
    with open(os.path.join(LOG_DIR, "file_inventory.txt"), "w") as tf:
        for row in ALL_FILE_DATA:
            tf.write(json.dumps(row) + "\n")

# Generate Motion Template based on file data
def generate_motion_from_evidence():
    motions = []
    for record in ALL_FILE_DATA:
        exhibit_label = record["exhibit_label"]
        file_description = record["file_description"]
        motions.append(f"Motion to include {exhibit_label}: {file_description}")

    motion_file = os.path.join(SORTED_DIR, "GeneratedMotion.txt")
    with open(motion_file, "w") as motion:
        motion.write("\n".join(motions))

# Chain the evidence into an affidavit builder template
def chain_to_affidavit_builder():
    affidavit = []
    for record in ALL_FILE_DATA:
        exhibit_label = record["exhibit_label"]
        file_description = record["file_description"]
        affidavit.append(f"Affidavit: Include {exhibit_label} for {file_description}")

    affidavit_file = os.path.join(SORTED_DIR, "GeneratedAffidavit.txt")
    with open(affidavit_file, "w") as affidavit_out:
        affidavit_out.write("\n".join(affidavit))

# Main function to run the file scanning and categorization process
def run_mainframe_sorter():
    print(f"[bold blue]🔍 Scanning drives {', '.join(DRIVES)}... This will recursively parse all files.[/bold blue]")
    scan_directory(DRIVES)
    print(f"[green]✅ Scan complete. Writing outputs...[/green]")
    write_outputs()
    generate_motion_from_evidence()  # Trigger Motion Generation
    chain_to_affidavit_builder()  # Trigger Affidavit Builder
    print(f"[bold green]✔️ All done. Logs saved to {LOG_DIR}[/bold green]")

if __name__ == "__main__":
    safe_mkdir(SORTED_DIR)
    safe_mkdir(EXTRACTED_ZIP_DIR)
    safe_mkdir(EXHIBIT_BUNDLE_DIR)
    run_mainframe_sorter()
