import json
import os

SVG_EXPORT = os.path.join('warboard', 'exports', 'SHADY_OAKS_WARBOARD.svg')
TIMELINE_FILE = os.path.join('data', 'timeline.json')


def generate_svg_warboard():
    if not os.path.exists(TIMELINE_FILE):
        print('Timeline file not found; cannot build SVG warboard.')
        return

    with open(TIMELINE_FILE, 'r') as f:
        events = json.load(f)

    width = 2000
    height = 600
    spacing = max(width // max(len(events), 1), 100)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-size:12px;}</style>'
    ]

    y_base = 100
    for i, event in enumerate(events):
        x = 50 + i * spacing
        y = y_base + (i % 4) * 80
        label = event.get('description', '')
        date = event.get('date', '')[:10]
        svg_lines.append(f'<circle cx="{x}" cy="{y}" r="20" fill="#4f46e5" />')
        svg_lines.append(f'<text x="{x - 40}" y="{y + 35}">{date}</text>')
        svg_lines.append(f'<text x="{x - 40}" y="{y + 50}">{label}</text>')

    svg_lines.append('</svg>')

    os.makedirs(os.path.dirname(SVG_EXPORT), exist_ok=True)
    with open(SVG_EXPORT, 'w') as f:
        f.write('\n'.join(svg_lines))
    print(f'SVG warboard saved to {SVG_EXPORT}')


if __name__ == '__main__':
    generate_svg_warboard()
