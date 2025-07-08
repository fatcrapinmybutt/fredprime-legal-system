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
