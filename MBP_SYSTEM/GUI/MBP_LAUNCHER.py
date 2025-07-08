import tkinter as tk
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run_genesis():
    subprocess.Popen(['python', str(ROOT / 'GENESIS_CORE_TITAN_ENGINE.py')])


def run_threads():
    subprocess.Popen(['python', str(ROOT / 'ENGINE' / 'litigation_overdrive_engine.py')])


def gui():
    win = tk.Tk()
    win.title('MBP LAUNCHER')
    win.geometry('400x200')
    tk.Button(win, text='Start Core Engine', command=run_genesis).pack(pady=10)
    tk.Button(win, text='Run Threadstorm', command=run_threads).pack(pady=10)
    tk.Button(win, text='Exit', command=win.quit).pack(pady=10)
    win.mainloop()

if __name__ == '__main__':
    gui()
