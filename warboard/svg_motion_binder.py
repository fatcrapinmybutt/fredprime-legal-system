import json
import os

DEFAULT_SVG_EXPORT = os.path.join("warboard", "exports", "SHADY_OAKS_WARBOARD.svg")
TIMELINE_FILE = os.path.join("data", "timeline.json")
BASE_RESULTS_DIR = os.getenv("LEGAL_RESULTS_DIR", os.path.join("F:/", "LegalResults"))
MOTION_DIR = os.path.join(BASE_RESULTS_DIR, "motions")


def bind_motion_links(svg_path=DEFAULT_SVG_EXPORT):
    """Embed motion file links into an SVG generated from the timeline.

    Parameters
    ----------
    svg_path : str
        Path to the SVG file to augment.
    """
    if not os.path.exists(TIMELINE_FILE):
        print("Timeline not found; cannot bind motion links.")
        return

    if not os.path.exists(svg_path):
        print("SVG export not found; cannot bind motion links.")
        return

    with open(TIMELINE_FILE, "r") as f:
        events = json.load(f)

    spacing = max(100, 2000 // max(len(events), 1))
    svg_lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="2200" height="1000">',
        "<style>text{font-size:12px;}</style>",
    ]

    for i, event in enumerate(events):
        x = 100 + i * spacing
        y = 100 + (i % 4) * 120
        date = event.get("date", "")[:10]
        desc = event.get("description", "")
        motions = event.get("linked_motions", [])
        if motions:
            doc_name = motions[0].replace(" ", "_") + ".docx"
            href = os.path.join(MOTION_DIR, doc_name)
            href = href.replace("\\", "/")
            svg_lines.append(f'<a xlink:href="file:///{href}">')
        svg_lines.append(f'<circle cx="{x}" cy="{y}" r="20" fill="#4f46e5" />')
        if motions:
            svg_lines.append("</a>")
        svg_lines.append(f'<text x="{x - 40}" y="{y + 35}">{date}</text>')
        svg_lines.append(f'<text x="{x - 40}" y="{y + 55}">{desc[:45]}</text>')

    svg_lines.append("</svg>")

    with open(svg_path, "w") as f:
        f.write("\n".join(svg_lines))
    print(f"Linked SVG written to {svg_path}")
