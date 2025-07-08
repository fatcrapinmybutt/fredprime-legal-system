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
