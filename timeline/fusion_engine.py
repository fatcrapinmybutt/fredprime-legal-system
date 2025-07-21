import json
import os
from datetime import datetime

BASE_DIR = os.getenv('LEGAL_RESULTS_DIR', os.path.join('F:/', 'LegalResults'))
OUTPUT_DIR = os.path.join(BASE_DIR, 'TIMELINES')
FUSION_JSON = os.path.join(OUTPUT_DIR, 'timeline_fusion_shady_oaks.json')
FUSION_SVG = os.path.join(OUTPUT_DIR, 'shady_oaks_timeline_fusion.svg')

DEFAULT_EVENTS = [
    {'date': '2025-02-01', 'description': 'EGLE sewer contamination begins', 'type': 'Environmental', 'entity': 'HOA'},
    {'date': '2025-03-01', 'description': 'Rent raised from $395 to $695', 'type': 'Economic Coercion', 'entity': 'Homes of America'},
    {'date': '2025-05-20', 'description': 'Water shut off with child present', 'type': 'Suppression Harm', 'entity': 'HOA'},
    {'date': '2025-05-24', 'description': 'Writ executed; property destroyed', 'type': 'Physical Suppression', 'entity': 'Unknown Deputies'},
]

TIMELINE_FILE = os.path.join('data', 'timeline.json')


def merge_events(existing, new):
    merged = existing + new
    merged.sort(key=lambda x: datetime.fromisoformat(x['date']))
    return merged


def write_svg(events, path):
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="600">',
             '<style>text{font-size:12px;}</style>']
    for i, e in enumerate(events):
        x = 100 + i * 100
        y = 100 + (i % 5) * 70
        lines.append(f'<circle cx="{x}" cy="{y}" r="20" fill="#4f46e5" />')
        lines.append(f'<text x="{x-40}" y="{y+35}">{e["date"]}</text>')
        lines.append(f'<text x="{x-40}" y="{y+50}">{e["description"][:40]}</text>')
    lines.append('</svg>')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def build_fusion_timeline():
    existing = []
    if os.path.exists(TIMELINE_FILE):
        with open(TIMELINE_FILE, 'r') as f:
            existing = json.load(f)
    events = merge_events(existing, DEFAULT_EVENTS)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(FUSION_JSON, 'w') as f:
        json.dump(events, f, indent=2)
    write_svg(events, FUSION_SVG)
    print(f'Fusion timeline saved to {FUSION_JSON} and {FUSION_SVG}')


if __name__ == '__main__':
    build_fusion_timeline()
