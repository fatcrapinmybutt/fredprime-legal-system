import json
import os
from datetime import datetime

SCAN_INDEX = os.path.join('data', 'scan_index.json')
TIMELINE_OUTPUT = os.path.join('data', 'timeline.json')


def build_timeline(scan_index: str = SCAN_INDEX, output: str = TIMELINE_OUTPUT) -> None:
    if not os.path.exists(scan_index):
        print('Scan index not found; run the scan engine first.')
        return
    with open(scan_index, 'r') as f:
        data = json.load(f)

    events = []
    for path, meta in data.items():
        date = meta.get('created')
        if date:
            events.append({'date': date, 'description': os.path.basename(path)})

    events.sort(key=lambda e: datetime.fromisoformat(e['date']))
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, 'w') as f:
        json.dump(events, f, indent=2)
    print(f'Timeline written to {output} with {len(events)} events')


if __name__ == '__main__':
    build_timeline()
