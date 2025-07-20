import os
import json
from datetime import datetime

DEFAULT_OUTPUT = os.path.join('data', 'scan_index.json')


def scan_directory(root_dir: str, index: dict) -> None:
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(('.docx', '.pdf', '.txt')):
                path = os.path.join(dirpath, name)
                try:
                    created = datetime.fromtimestamp(os.path.getctime(path)).isoformat()
                    index[path] = {'created': created}
                except OSError:
                    pass


def run_scan(drives=None, output: str = DEFAULT_OUTPUT) -> None:
    if drives is None:
        drives = ['F:/', 'D:/']
    index = {}
    for drive in drives:
        if os.path.exists(drive):
            scan_directory(drive, index)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w') as f:
        json.dump(index, f, indent=2)
    print(f'Scan complete. Indexed {len(index)} files to {output}')


if __name__ == '__main__':
    run_scan()
