import os
from pathlib import Path

BASE = Path('MBP_SYSTEM')

FOLDERS = [
    'VFS/TEMP',
    'VFS/MODELS',
    'VFS/TRANSCRIPTS',
    'VFS/FORM_OVERLAYS',
    'VFS/GENERATED_ZIPS',
    'ENGINE',
    'UPGRADES',
    'GUI',
    'FORMS',
    'EXHIBITS',
    'CHAINLOGS'
]


def create_folders():
    for folder in FOLDERS:
        path = BASE / folder
        path.mkdir(parents=True, exist_ok=True)
    print('Created directory structure under', BASE)


def write_file(path: Path, content: str):
    path.write_text(content.strip() + '\n', encoding='utf-8')
    print('Wrote', path)


def generate_files():
    # GENESIS CORE ENGINE
    genesis_code = '''
import os
import sys
import importlib
import shutil
from pathlib import Path

MBP_ROOT = Path(__file__).resolve().parent.parent
UPGRADE_PATH = MBP_ROOT / "UPGRADES"


def setup_dirs():
    for d in ["ENGINE", "UPGRADES", "GUI", "FORMS", "EXHIBITS", "CHAINLOGS", "VFS"]:
        (MBP_ROOT / d).mkdir(exist_ok=True)


def apply_upgrades():
    for f in UPGRADE_PATH.glob("*.py"):
        dst = MBP_ROOT / "ENGINE" / f.name
        shutil.copy(f, dst)
        print("Installed", f.name)


def load_modules():
    engine_dir = MBP_ROOT / "ENGINE"
    sys.path.append(str(engine_dir))
    for f in engine_dir.glob("*.py"):
        name = f.stem
        try:
            importlib.import_module(name)
            print("Loaded", name)
        except Exception as e:
            print("Failed", name, e)


def main():
    print("MBP GENESIS CORE INIT")
    setup_dirs()
    apply_upgrades()
    load_modules()
    print("SYSTEM ONLINE")

if __name__ == '__main__':
    main()
    '''
    write_file(BASE / 'GENESIS_CORE_TITAN_ENGINE.py', genesis_code)

    # THREAD OVERDRIVE
    overdrive_code = '''
import os
import time
import concurrent.futures

THREAD_COUNT = os.cpu_count() * 10
TEMP_PATH = 'MBP_SYSTEM/VFS/TEMP'


def task(i):
    with open(os.path.join(TEMP_PATH, f"thread_{i}.log"), 'w') as f:
        f.write(f"Task {i} OK")
    time.sleep(0.1)
    return f"Done {i}"


def run():
    print(f'Threadstorm using {THREAD_COUNT} workers')
    with concurrent.futures.ThreadPoolExecutor(max_workers=THREAD_COUNT) as exe:
        list(exe.map(task, range(100)))
    print('Completed Thread Tasks')

if __name__ == '__main__':
    run()
    '''
    write_file(BASE / 'ENGINE' / 'litigation_overdrive_engine.py', overdrive_code)

    # UPGRADE CHAINLINK
    chainlink_code = '''
import os
import importlib.util

UPGRADE_DIR = 'MBP_SYSTEM/UPGRADES'


def load(path):
    name = os.path.basename(path).replace('.py', '')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print('Linked', name)
    return mod


def activate_all():
    for f in os.listdir(UPGRADE_DIR):
        if f.endswith('.py'):
            load(os.path.join(UPGRADE_DIR, f))

if __name__ == '__main__':
    activate_all()
    '''
    write_file(BASE / 'ENGINE' / 'upgrade_chainlink.py', chainlink_code)

    # GUI LAUNCHER
    gui_launcher = '''
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
    '''
    write_file(BASE / 'GUI' / 'MBP_LAUNCHER.py', gui_launcher)

    # Example upgrade module
    upgrade_module = '''
def auto_build_motion(data=None):
    print('AUTO-MOTION triggered')
    if not data:
        data = {'type': 'Contempt Defense', 'facts': 'Test facts'}
    print(f"Motion for: {data['type']} with facts: {data['facts']}")
    '''
    write_file(BASE / 'UPGRADES' / 'upgrade_motion_autobuilder.py', upgrade_module)


def main():
    create_folders()
    generate_files()
    print('MBP_SYSTEM scaffold is installed.')


if __name__ == '__main__':
    main()
