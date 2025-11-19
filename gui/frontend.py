import os
import tkinter as tk
from tkinter import ttk
from tkinterweb import HtmlFrame
from warboard.warboard_engine import deploy_supra_warboard
from warboard.ppo_warboard import build_ppo_warboard
from warboard.custody_interference_engine import build_custody_warboard
from scheduling.scheduler import build_schedule
from gui.modules.entity_suppression_feed import load_events
from agents import menu as agents_menu
import threading
import importlib
import subprocess
import sys
from pathlib import Path


def launch_dashboard():
    root = tk.Tk()
    root.title('Litigation OS')
    root.geometry('900x600')

    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)

    warboard_tab = ttk.Frame(notebook)
    ppo_tab = ttk.Frame(notebook)
    custody_tab = ttk.Frame(notebook)
    schedule_tab = ttk.Frame(notebook)
    suppression_tab = ttk.Frame(notebook)

    notebook.add(warboard_tab, text='🗺️ Shady Oaks Warboard')
    notebook.add(ppo_tab, text='🛡️ PPO Timeline')
    notebook.add(custody_tab, text='👨‍👩‍👧 Custody Map')
    notebook.add(schedule_tab, text='📆 Timeline Tracker')
    notebook.add(suppression_tab, text='🧱 Entity Suppression')

    frame = HtmlFrame(warboard_tab)
    frame.pack(fill='both', expand=True)
    ppo_frame = HtmlFrame(ppo_tab)
    ppo_frame.pack(fill='both', expand=True)
    cust_frame = HtmlFrame(custody_tab)
    cust_frame.pack(fill='both', expand=True)
    schedule_text = tk.Text(schedule_tab)
    schedule_text.pack(fill='both', expand=True)
    suppression_text = tk.Text(suppression_tab)
    suppression_text.pack(fill='both', expand=True)
    # Agents tab: list available agents and allow running their sample main()
    agents_tab = ttk.Frame(notebook)
    notebook.add(agents_tab, text='🤖 Agents')

    agents_listbox = tk.Listbox(agents_tab, height=12)
    agents_listbox.pack(fill='both', expand=False, padx=8, pady=8)
    agent_desc = tk.Text(agents_tab, height=6)
    agent_desc.pack(fill='both', expand=False, padx=8, pady=4)

    def load_agents_into_list():
        items = agents_menu.get_menu_items(root=None)
        agents_listbox.delete(0, tk.END)
        agents_listbox._items = items
        for it in items:
            agents_listbox.insert(tk.END, f"{it['id']} - {it['name']}")

    def on_agent_select(evt=None):
        sel = agents_listbox.curselection()
        if not sel:
            return
        it = agents_listbox._items[sel[0]]
        agent_desc.delete('1.0', tk.END)
        agent_desc.insert(tk.END, f"{it.get('description','')}\nTopics: {it.get('topics','')}\nPath: {it.get('path','')}")

    def run_selected_agent():
        sel = agents_listbox.curselection()
        if not sel:
            return
        it = agents_listbox._items[sel[0]]
        # run the agent script in a subprocess for isolation
        def _run_subproc():
            try:
                mod_path = Path(it.get('path')) / 'agent_core.py'
                if not mod_path.exists():
                    print(f"Agent file not found: {mod_path}")
                    return
                # Use the same Python interpreter used by the GUI
                proc = subprocess.run([sys.executable, str(mod_path)], capture_output=True, text=True, timeout=60)
                if proc.stdout:
                    print(f"[agent stdout] {proc.stdout}")
                if proc.stderr:
                    print(f"[agent stderr] {proc.stderr}")
            except Exception as e:
                print('Agent subprocess error:', e)

        threading.Thread(target=_run_subproc, daemon=True).start()

    agents_listbox.bind('<<ListboxSelect>>', on_agent_select)
    ttk.Button(agents_tab, text='Refresh Agents', command=load_agents_into_list).pack(pady=4)
    ttk.Button(agents_tab, text='Run Selected Agent', command=run_selected_agent).pack(pady=4)
    load_agents_into_list()

    def refresh_warboard():
        deploy_supra_warboard()
        with open('warboard/exports/SHADY_OAKS_WARBOARD.svg') as f:
            frame.set_content(f.read())

    def refresh_ppo():
        build_ppo_warboard()
        with open('warboard/exports/PPO_WARBOARD.svg') as f:
            ppo_frame.set_content(f.read())

    def refresh_custody():
        build_custody_warboard()
        with open('warboard/exports/CUSTODY_INTERFERENCE_MAP.svg') as f:
            cust_frame.set_content(f.read())

    def refresh_schedule():
        build_schedule()
        if os.path.exists('data/timeline.json'):
            import json
            with open('data/timeline.json') as f:
                events = json.load(f)
            schedule_text.delete('1.0', tk.END)
            for e in events:
                schedule_text.insert(tk.END, f"{e['date']} - {e['description']}\n")

    def refresh_suppression():
        events = load_events()
        suppression_text.delete('1.0', tk.END)
        for ev in events:
            suppression_text.insert(tk.END, f"{ev['entity']}: {ev['action']}\n")

    ttk.Button(warboard_tab, text='Build Warboard', command=refresh_warboard).pack(pady=5)
    ttk.Button(ppo_tab, text='Build PPO Warboard', command=refresh_ppo).pack(pady=5)
    ttk.Button(custody_tab, text='Build Custody Map', command=refresh_custody).pack(pady=5)
    ttk.Button(schedule_tab, text='Sync From File', command=refresh_schedule).pack(pady=5)
    ttk.Button(suppression_tab, text='Refresh', command=refresh_suppression).pack(pady=5)

    refresh_warboard()
    refresh_suppression()

    root.mainloop()


if __name__ == '__main__':
    launch_dashboard()
