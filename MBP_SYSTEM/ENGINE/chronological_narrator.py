import os
import re
import datetime

SOURCE_DIRS = [
    'MBP_SYSTEM/EXHIBITS',
    'MBP_SYSTEM/VFS/TRANSCRIPTS',
    'MBP_SYSTEM/CHAINLOGS',
]

OUTFILE = 'MBP_SYSTEM/VFS/GENERATED_ZIPS/Chronological_Narrative.txt'

DATE_PATTERNS = [
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}, \d{4}\b",
]


def extract_events():
    timeline = []
    for directory in SOURCE_DIRS:
        for filename in os.listdir(directory):
            fullpath = os.path.join(directory, filename)
            if not os.path.isfile(fullpath):
                continue
            with open(fullpath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for line in lines:
                    for pattern in DATE_PATTERNS:
                        match = re.search(pattern, line)
                        if match:
                            timeline.append({'date': match.group(), 'event': line.strip(), 'source': filename})
    return timeline


def normalize_date(raw):
    for fmt in ('%m/%d/%Y', '%Y-%m-%d', '%B %d, %Y'):
        try:
            return datetime.datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def run():
    raw = extract_events()
    parsed = []
    for e in raw:
        norm = normalize_date(e['date'])
        if norm:
            e['norm_date'] = norm
            parsed.append(e)
    sorted_events = sorted(parsed, key=lambda x: x['norm_date'])
    with open(OUTFILE, 'w', encoding='utf-8') as f:
        for event in sorted_events:
            f.write(f"[{event['date']}] - {event['event']} (Source: {event['source']})\n")
    print(f"Narrative saved to: {OUTFILE}")


if __name__ == '__main__':
    run()
