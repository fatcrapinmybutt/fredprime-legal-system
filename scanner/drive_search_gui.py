from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List

import tkinter as tk
from tkinter import ttk, messagebox
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

LEGAL_TERMS = [
    "motion",
    "affidavit",
    "order",
    "complaint",
    "ledger",
]

keyword_entry: ttk.Entry
filetype_var: tk.StringVar
legal_term_var: tk.StringVar
output_box: tk.Text

drive: GoogleDrive | None = None


def authenticate_drive() -> GoogleDrive:
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)


def scan_local_drives() -> List[str]:
    hits: List[str] = []
    for root_drive in ("F:/", "D:/"):
        if os.path.exists(root_drive):
            for root_dir, _, files in os.walk(root_drive):
                for name in files:
                    lower = name.lower()
                    if any(term in lower for term in LEGAL_TERMS):
                        hits.append(os.path.join(root_dir, name))
    return hits


def inject_results_to_organized_data(results: List[Dict[str, str]]) -> None:
    target = Path("F:/LAWFORGE_SERVER/organized_litigation_data.py")
    try:
        with target.open("a", encoding="utf-8") as handle:
            handle.write("\n# Auto-imported from Google Drive scan\n")
            for result in results:
                handle.write(f"drive_data.append({json.dumps(result)})\n")
    except Exception as exc:
        print(f"Failed to inject: {exc}")


def search_drive() -> List[Dict[str, str]]:
    assert drive is not None
    keyword = keyword_entry.get()
    filetype = filetype_var.get()
    legal_term = legal_term_var.get()
    full_query: List[str] = []
    if keyword:
        full_query.append(f"title contains '{keyword}'")
    if filetype:
        full_query.append(f"title contains '{filetype}'")
    if legal_term:
        full_query.append(f"title contains '{legal_term}'")
    query = " and ".join(full_query) if full_query else "trashed=false"
    try:
        file_list = drive.ListFile({"q": query}).GetList()
        results: List[Dict[str, str]] = []
        output_box.delete("1.0", tk.END)
        for file in file_list:
            output_box.insert(tk.END, f"{file['title']} ({file['mimeType']})\n")
            results.append({"title": file["title"], "mime": file["mimeType"]})
        inject_results_to_organized_data(results)
        local_hits = scan_local_drives()
        for path in local_hits:
            output_box.insert(tk.END, f"LOCAL: {path}\n")
        return results
    except Exception as exc:
        messagebox.showerror("Error", str(exc))
        return []


def periodic_rescan(interval: int = 21600) -> None:
    while True:
        print("Auto-rescanning Google Drive...")
        if drive is not None:
            search_drive()
        time.sleep(interval)


def register_task_scheduler() -> None:
    if os.name == "nt":
        exe_path = Path.cwd() / "LAWFORGE_DRIVE_SEARCH.exe"
        cmd = [
            "schtasks",
            "/Create",
            "/SC",
            "HOURLY",
            "/MO",
            "6",
            "/TN",
            "LawForgeDriveScan",
            "/TR",
            str(exe_path),
        ]
        try:
            subprocess.run(cmd, check=True)
        except Exception as exc:
            print(f"Task schedule register failed: {exc}")


def launch_drive_search_gui() -> None:
    global drive, keyword_entry, filetype_var, legal_term_var, output_box
    drive = authenticate_drive()
    root = tk.Tk()
    root.title("Google Drive Search Tool")
    root.geometry("600x500")

    frame = ttk.Frame(root, padding=10)
    frame.pack(fill=tk.BOTH, expand=True)

    ttk.Label(frame, text="Keyword:").grid(column=0, row=0, sticky=tk.W)
    keyword_entry = ttk.Entry(frame, width=40)
    keyword_entry.grid(column=1, row=0, columnspan=2, sticky=tk.W)

    ttk.Label(frame, text="File Type:").grid(column=0, row=1, sticky=tk.W)
    filetype_var = tk.StringVar()
    filetype_dropdown = ttk.Combobox(frame, textvariable=filetype_var)
    filetype_dropdown["values"] = (
        ".pdf",
        ".docx",
        ".txt",
        ".xlsx",
        ".py",
        ".json",
        ".csv",
        ".ps1",
        "",
    )
    filetype_dropdown.grid(column=1, row=1, sticky=tk.W)

    ttk.Label(frame, text="Legal Term:").grid(column=0, row=2, sticky=tk.W)
    legal_term_var = tk.StringVar()
    legal_term_dropdown = ttk.Combobox(frame, textvariable=legal_term_var)
    legal_term_dropdown["values"] = (
        "motion",
        "affidavit",
        "proposed order",
        "timeline",
        "MCR",
        "MCL",
        "Benchbook",
        "exhibit",
        "PPO",
        "contempt",
        "canon",
        "violation",
        "binder",
        "",
    )
    legal_term_dropdown.grid(column=1, row=2, sticky=tk.W)

    search_button = ttk.Button(frame, text="Search Drive", command=search_drive)
    search_button.grid(column=1, row=3, pady=10, sticky=tk.W)

    output_box = tk.Text(frame, height=20)
    output_box.grid(column=0, row=4, columnspan=3, sticky="nsew")
    frame.rowconfigure(4, weight=1)
    frame.columnconfigure(2, weight=1)

    rescan_thread = threading.Thread(target=periodic_rescan, daemon=True)
    rescan_thread.start()
    register_task_scheduler()
    root.mainloop()


if __name__ == "__main__":
    launch_drive_search_gui()
