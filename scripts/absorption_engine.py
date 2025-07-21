import os
import json
import hashlib
import time
import logging
import tkinter as tk
import datetime
import importlib.util
import shutil
from tkinter import messagebox, filedialog

TARGET_DIRS = ["F:/", "D:/"]
EXTENSIONS = [".py", ".json", ".txt", ".docx"]
MANIFEST_FILE = "codex_manifest.json"
PATCH_DIR = "patches/"
PATCH_MANIFEST = "patch_manifest.json"
PATCH_HISTORY = "patch_history.json"
ERROR_LOG = "logs/codex_errors.log"

os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename=ERROR_LOG, level=logging.ERROR)

HELP_TOPICS = {
    "motion": (
        "MCR 2.119 governs motions. Benchbook, Motions section details formal "
        "requirements. Ensure all mandatory elements (caption, relief, signature) are present."
    ),
    "affidavit": (
        "MCR 2.119(B) details affidavits: must be signed, state facts, and have a notary if required."
    ),
    "canon": (
        "Canon violations should be documented with factual detail and referenced to the Michigan Judicial Conduct Canon text."
    ),
}


def classify_legal_function(filepath):
    lower = filepath.lower()
    if "motion" in lower:
        return "motion (MCR 2.119)"
    if "affidavit" in lower:
        return "affidavit (MCR 2.119(B))"
    if "order" in lower:
        return "court order"
    return "uncategorized"


def validate_file(filepath):
    try:
        with open(filepath, "r", errors="ignore") as f:
            content = f.read()
            return any(x in content for x in ["MCR", "Benchbook", "MCL"])
    except Exception:
        return False


def get_metadata(filepath):
    try:
        stat = os.stat(filepath)
        sha256 = hashlib.sha256(open(filepath, "rb").read()).hexdigest()
        legal_function = classify_legal_function(filepath)
        validated = validate_file(filepath)
        return {
            "sha256": sha256,
            "timestamp": time.ctime(stat.st_mtime),
            "source": "absorption_engine",
            "legal_function": legal_function,
            "validated": validated,
        }
    except Exception as e:
        logging.error(f"Metadata error for {filepath}: {e}")
        return {}


def scan_drives():
    manifest = {}
    for root_dir in TARGET_DIRS:
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if any(file.endswith(ext) for ext in EXTENSIONS):
                    filepath = os.path.join(subdir, file)
                    manifest[filepath] = get_metadata(filepath)
    try:
        with open(MANIFEST_FILE, "w") as out:
            json.dump(manifest, out, indent=2)
        messagebox.showinfo(
            "Scan Complete", f"Absorption complete. {len(manifest)} files processed."
        )
    except Exception as e:
        logging.error(f"Failed to write manifest: {e}")
        messagebox.showerror("Scan Failed", f"Failed to write manifest: {e}")


def backup_file(filepath):
    bak_name = f"{filepath}.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.bak"
    shutil.copy2(filepath, bak_name)
    return bak_name


def apply_patch(patch_path, target_file):
    bak = backup_file(target_file)
    try:
        spec = importlib.util.spec_from_file_location("patch_module", patch_path)
        patch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch_module)
        log_patch(patch_path, target_file, bak, "success")
    except Exception as e:
        logging.error(f"Patch failed for {patch_path} on {target_file}: {e}")
        shutil.copy2(bak, target_file)
        log_patch(patch_path, target_file, bak, "rollback")
        print(f"Rolled back patch {patch_path}")


def log_patch(patch, target, backup, status):
    entry = {
        "patch": patch,
        "target": target,
        "backup": backup,
        "status": status,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if os.path.exists(PATCH_HISTORY):
        with open(PATCH_HISTORY, "r") as f:
            history = json.load(f)
    else:
        history = []
    history.append(entry)
    with open(PATCH_HISTORY, "w") as f:
        json.dump(history, f, indent=2)


def run_patch_manager():
    if not os.path.exists(PATCH_DIR):
        messagebox.showwarning("Patch Manager", "No patches found.")
        return
    if not os.path.exists(PATCH_MANIFEST):
        messagebox.showwarning("Patch Manager", "Patch manifest not found.")
        return
    with open(PATCH_MANIFEST) as f:
        manifest = json.load(f)
    for patch_file in os.listdir(PATCH_DIR):
        if patch_file.endswith(".py"):
            patch_path = os.path.join(PATCH_DIR, patch_file)
            target = manifest.get(patch_file, {}).get("target", None)
            if target and os.path.exists(target):
                apply_patch(patch_path, target)
            else:
                logging.error(f"No valid target for patch: {patch_file}")


RED_FLAGS = [
    "no citation",
    "missing Benchbook",
    "incomplete motion",
    "unsigned",
    "date missing",
    "no legal_function",
]


def scan_manifest_for_red_flags():
    flagged = {}
    if not os.path.exists(MANIFEST_FILE):
        return flagged
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)
    for path, meta in manifest.items():
        meta_str = json.dumps(meta).lower()
        for flag in RED_FLAGS:
            if flag in meta_str or not meta.get("legal_function"):
                flagged[path] = flag
    if flagged:
        with open("red_flag_report.json", "w") as out:
            json.dump(flagged, out, indent=2)
    return flagged


def generate_foia_request():
    if not os.path.exists(MANIFEST_FILE):
        messagebox.showerror("FOIA Generator", "Manifest not found!")
        return
    with open(MANIFEST_FILE) as f:
        manifest = json.load(f)
    missing = [k for k, v in manifest.items() if not v.get("validated")]
    body = (
        f"Date: {datetime.date.today()}\n\n"
        "To Whom It May Concern:\n\n"
        "This is a formal FOIA/Discovery request for production of documents or evidence not yet produced "
        "as required by law. The following files or evidence appear absent or unverified in your records:\n\n"
    )
    body += "\n".join(missing)
    body += (
        "\n\nPlease provide these documents or a statement of nonexistence under penalty of perjury within "
        "the statutory response window. Thank you.\n\nSincerely,\n[Your Name]"
    )
    with open("FOIA_discovery_request.txt", "w") as f:
        f.write(body)
    messagebox.showinfo("FOIA Generator", "FOIA/discovery request generated.")


def export_codex_data():
    files = {
        "manifest": MANIFEST_FILE,
        "timeline": "timeline.json",
        "config": "codex_config.yaml",
    }
    export_path = filedialog.asksaveasfilename(defaultextension=".zip")
    if export_path:
        import zipfile

        with zipfile.ZipFile(export_path, "w") as z:
            for label, f in files.items():
                if os.path.exists(f):
                    z.write(f)
        messagebox.showinfo("Export Complete", f"Exported: {export_path}")


def import_codex_data():
    import zipfile

    file_path = filedialog.askopenfilename(filetypes=[("ZIP Files", "*.zip")])
    if file_path:
        with zipfile.ZipFile(file_path, "r") as z:
            z.extractall(".")
        messagebox.showinfo("Import Complete", f"Imported: {file_path}")


def show_help(context):
    msg = HELP_TOPICS.get(context.lower(), "No help available for this context.")
    messagebox.showinfo("Contextual Help", msg)


def simulate_judicial_response():
    action = tk.simpledialog.askstring(
        "Simulate Judicial Response", "Describe action/context:"
    )
    if not action:
        return
    context = action.lower()
    if "ex parte" in context:
        msg = "Warning: Ex parte relief often triggers heightened judicial scrutiny. If opposed, expect a prompt hearing."
    elif "canon violation" in context:
        msg = "A well-documented Canon violation may trigger recusal or misconduct review. Be sure to cite all supporting facts."
    elif "motion to dismiss" in context:
        msg = "Court may grant only if no genuine issue of material fact exists. Ensure record completeness."
    else:
        msg = "No simulation logic for this action/context."
    messagebox.showinfo("Judicial Logic Simulation", msg)


def get_status_log():
    try:
        with open(ERROR_LOG, "r") as f:
            return f.read()[-5000:]
    except Exception:
        return "No system errors logged."


def launch_gui():
    window = tk.Tk()
    window.title("MBP Litigation OS - Codex Supreme (All-in-One)")
    window.geometry("1000x750")
    status_log = tk.Text(window, height=18, width=120)
    status_log.pack(pady=8)

    def update_status():
        status_log.delete("1.0", tk.END)
        status_log.insert(tk.END, get_status_log())
        window.after(4000, update_status)

    update_status()
    tk.Label(window, text="MBP Litigation OS", font=("Helvetica", 18, "bold")).pack(
        pady=8
    )
    tk.Button(window, text="Scan F:/ + D:/", command=scan_drives, width=35).pack(pady=3)
    tk.Button(
        window, text="Run Patch Manager", command=run_patch_manager, width=35
    ).pack(pady=3)
    tk.Button(
        window,
        text="Run Compliance Scan",
        command=scan_manifest_for_red_flags,
        width=35,
    ).pack(pady=3)
    tk.Button(
        window,
        text="Generate FOIA/Discovery Request",
        command=generate_foia_request,
        width=35,
    ).pack(pady=3)
    tk.Button(window, text="Export Data", command=export_codex_data, width=25).pack(
        pady=2
    )
    tk.Button(window, text="Import Data", command=import_codex_data, width=25).pack(
        pady=2
    )
    tk.Button(
        window, text="Help: Motions", command=lambda: show_help("motion"), width=20
    ).pack(pady=1)
    tk.Button(
        window,
        text="Help: Affidavits",
        command=lambda: show_help("affidavit"),
        width=20,
    ).pack(pady=1)
    tk.Button(
        window, text="Help: Canons", command=lambda: show_help("canon"), width=20
    ).pack(pady=1)
    tk.Button(
        window,
        text="Simulate Judicial Response",
        command=simulate_judicial_response,
        width=35,
    ).pack(pady=3)
    tk.Label(window, text="Version: FINAL", font=("Helvetica", 10)).pack(
        side="bottom", pady=10
    )
    window.mainloop()


if __name__ == "__main__":
    launch_gui()
