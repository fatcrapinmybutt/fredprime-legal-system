import os
import tkinter as tk
from tkinter import ttk
from tkinterweb import HtmlFrame
from warboard.warboard_engine import deploy_supra_warboard
from warboard.ppo_warboard import build_ppo_warboard
from warboard.custody_interference_engine import build_custody_warboard
from scheduling.scheduler import build_schedule
from gui.modules.entity_suppression_feed import load_events
from scanner.drive_search_gui import launch_drive_search_gui


def launch_dashboard() -> None:
    root = tk.Tk()
    root.title("Litigation OS")
    root.geometry("900x600")

    menu = tk.Menu(root)
    scanner_menu = tk.Menu(menu, tearoff=0)
    scanner_menu.add_command(
        label="Google Drive Search", command=launch_drive_search_gui
    )
    menu.add_cascade(label="Scanner", menu=scanner_menu)
    root.config(menu=menu)

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    warboard_tab = ttk.Frame(notebook)
    ppo_tab = ttk.Frame(notebook)
    custody_tab = ttk.Frame(notebook)
    schedule_tab = ttk.Frame(notebook)
    suppression_tab = ttk.Frame(notebook)

    notebook.add(warboard_tab, text="🗺️ Shady Oaks Warboard")
    notebook.add(ppo_tab, text="🛡️ PPO Timeline")
    notebook.add(custody_tab, text="👨‍👩‍👧 Custody Map")
    notebook.add(schedule_tab, text="📆 Timeline Tracker")
    notebook.add(suppression_tab, text="🧱 Entity Suppression")

    frame = HtmlFrame(warboard_tab)
    frame.pack(fill="both", expand=True)
    ppo_frame = HtmlFrame(ppo_tab)
    ppo_frame.pack(fill="both", expand=True)
    cust_frame = HtmlFrame(custody_tab)
    cust_frame.pack(fill="both", expand=True)
    schedule_text = tk.Text(schedule_tab)
    schedule_text.pack(fill="both", expand=True)
    suppression_text = tk.Text(suppression_tab)
    suppression_text.pack(fill="both", expand=True)

    def refresh_warboard() -> None:
        deploy_supra_warboard()
        with open("warboard/exports/SHADY_OAKS_WARBOARD.svg") as f:
            frame.set_content(f.read())

    def refresh_ppo() -> None:
        build_ppo_warboard()
        with open("warboard/exports/PPO_WARBOARD.svg") as f:
            ppo_frame.set_content(f.read())

    def refresh_custody() -> None:
        build_custody_warboard()
        with open("warboard/exports/CUSTODY_INTERFERENCE_MAP.svg") as f:
            cust_frame.set_content(f.read())

    def refresh_schedule() -> None:
        build_schedule()
        if os.path.exists("data/timeline.json"):
            import json

            with open("data/timeline.json") as f:
                events = json.load(f)
            schedule_text.delete("1.0", tk.END)
            for e in events:
                schedule_text.insert(tk.END, f"{e['date']} - {e['description']}\n")

    def refresh_suppression() -> None:
        events = load_events()
        suppression_text.delete("1.0", tk.END)
        for ev in events:
            suppression_text.insert(tk.END, f"{ev['entity']}: {ev['action']}\n")

    ttk.Button(warboard_tab, text="Build Warboard", command=refresh_warboard).pack(
        pady=5
    )
    ttk.Button(ppo_tab, text="Build PPO Warboard", command=refresh_ppo).pack(pady=5)
    ttk.Button(custody_tab, text="Build Custody Map", command=refresh_custody).pack(
        pady=5
    )
    ttk.Button(schedule_tab, text="Sync From File", command=refresh_schedule).pack(
        pady=5
    )
    ttk.Button(suppression_tab, text="Refresh", command=refresh_suppression).pack(
        pady=5
    )

    refresh_warboard()
    refresh_suppression()

    root.mainloop()


if __name__ == "__main__":
    launch_dashboard()
