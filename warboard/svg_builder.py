import json
import os

DEFAULT_SVG_EXPORT = os.path.join('warboard', 'exports', 'SHADY_OAKS_WARBOARD.svg')
TIMELINE_FILE = os.path.join('data', 'timeline.json')


def generate_svg_warboard(events=None, svg_path=DEFAULT_SVG_EXPORT):
    """Create an SVG timeline from events.

    Parameters
    ----------
    events : list[dict] | None
        List of events with ``date`` and ``description`` keys. When ``None``,
        events are loaded from ``TIMELINE_FILE``.
    svg_path : str
        Destination path for the SVG file.
    """
    if events is None:
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

    os.makedirs(os.path.dirname(svg_path), exist_ok=True)
    with open(svg_path, 'w') as f:
        f.write('\n'.join(svg_lines))
    print(f'SVG warboard saved to {svg_path}')


if __name__ == '__main__':
    generate_svg_warboard()
