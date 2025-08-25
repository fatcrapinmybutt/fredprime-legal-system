import os
import json
import re
import csv
from datetime import datetime

# === CONFIG ===
TARGET_DRIVE = "F:/"
OUTPUT_DIR = os.path.join(TARGET_DRIVE, "FRED_LOGS")
TEXT_LOG = os.path.join(OUTPUT_DIR, "housing_master.txt")
JSON_LOG = os.path.join(OUTPUT_DIR, "housing_master.json")
CSV_LOG = os.path.join(OUTPUT_DIR, "keyword_matches.csv")
FILE_EXTENSIONS = [".txt", ".docx"]

KEYWORDS = [
    "shady oaks", "homes of america", "eviction", "trailer", "lot rent", "sewage",
    "EGLE", "Karen Rettig", "True North", "Partridge", "lease", "ledger",
    "retaliation", "title", "water shutoff", "rent increase", "destruction",
    "property damage", "coercion", "Alden Global", "HOA", "writ", "court officer",
    "notice to quit", "witness statement", "EGLE report", "sewage leak", "forced to sign",
    "no lease", "utility shutoff", "unjust rent", "threat to evict", "illegal eviction",
    "fraudulent ledger", "housing retaliation", "loss of home", "rent demand", "mobile home title",
    "forced move", "True North legal aid", "coerced into sale", "offered $750", "public sewer violation",
    "LLC fraud", "shell company", "Partridge & Assoc", "Karen Rettig email", "witnessed eviction",
    "child present", "property destroyed", "EGLE sewer", "Friedman Management", "HUD complaint"
]

# === Ensure output directory exists ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Helper: Check file type ===
def is_target_file(filename):
    return any(filename.lower().endswith(ext) for ext in FILE_EXTENSIONS)

# === Helper: Extract .txt or .docx content ===
def read_file_content(filepath):
    try:
        if filepath.lower().endswith(".txt"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif filepath.lower().endswith(".docx"):
            import docx
            doc = docx.Document(filepath)
            return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        return ""
    return ""

# === MAIN SCANNER ===
def scan_drive():
    master_text = []
    master_json = []
    csv_rows = []

    for root, dirs, files in os.walk(TARGET_DRIVE):
        for file in files:
            if is_target_file(file):
                path = os.path.join(root, file)
                content = read_file_content(path).lower()
                for keyword in KEYWORDS:
                    if keyword.lower() in content:
                        snippet = extract_snippet(content, keyword)
                        record = {
                            "file": path,
                            "keyword": keyword,
                            "snippet": snippet,
                            "timestamp": datetime.now().isoformat()
                        }
                        master_text.append(f"[{keyword}] {path}\n{snippet}\n{'-'*80}")
                        master_json.append(record)
                        csv_rows.append([path, keyword, snippet[:100]])

    # === Append to master logs ===
    with open(TEXT_LOG, "a", encoding="utf-8") as f:
        f.write("\n\n".join(master_text) + "\n")

    with open(JSON_LOG, "a", encoding="utf-8") as f:
        for entry in master_json:
            f.write(json.dumps(entry) + "\n")

    with open(CSV_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in csv_rows:
            writer.writerow(row)

    print(f"✅ Scan complete. Matches saved to: {TEXT_LOG}, {JSON_LOG}, {CSV_LOG}")

# === Snippet Generator ===
def extract_snippet(text, keyword, window=100):
    idx = text.find(keyword.lower())
    if idx == -1:
        return ""
    start = max(0, idx - window)
    end = min(len(text), idx + len(keyword) + window)
    return text[start:end].replace("\n", " ")

# === Run ===
if __name__ == "__main__":
    scan_drive()
