import os
from datetime import datetime

SOURCE_DIR = 'MBP_SYSTEM/EXHIBITS'
OUTFILE = 'MBP_SYSTEM/VFS/GENERATED_ZIPS/Habitability_Claims.txt'

HABITABILITY_TRIGGERS = [
    'sewage', 'leak', 'water shutoff', 'no heat', 'toxic', 'unsafe', 'egle', 'mold',
    'infestation', 'electrical', 'collapse',
]


def scan_exhibits():
    claims = []
    for fname in os.listdir(SOURCE_DIR):
        path = os.path.join(SOURCE_DIR, fname)
        if not fname.endswith('.txt'):
            continue
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if any(term in line.lower() for term in HABITABILITY_TRIGGERS):
                    claims.append(f"{fname} [Line {i+1}]: {line.strip()}")
    return claims


def save_claims(claims):
    with open(OUTFILE, 'w', encoding='utf-8') as f:
        f.write('Habitability Violations under MCL 554.139:\n\n')
        for c in claims:
            f.write(f"{c}\n")
    print(f'Claims saved to {OUTFILE}')


if __name__ == '__main__':
    results = scan_exhibits()
    save_claims(results)
