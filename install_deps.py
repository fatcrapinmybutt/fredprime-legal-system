import subprocess
import sys
from pathlib import Path

REQUIREMENTS = Path('requirements.txt')


def install_missing():
    if not REQUIREMENTS.exists():
        print('requirements.txt not found')
        return
    with REQUIREMENTS.open() as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    for pkg in packages:
        try:
            __import__(pkg.split('==')[0].replace('-', '_'))
            continue
        except Exception:
            print(f'Installing missing package: {pkg}')
            subprocess.run([sys.executable, '-m', 'pip', 'install', pkg], check=False)


if __name__ == '__main__':
    install_missing()
