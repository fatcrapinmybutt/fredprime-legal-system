import os
import re

EXHIBIT_DIR = 'MBP_SYSTEM/EXHIBITS'
OUTFILE = 'MBP_SYSTEM/VFS/GENERATED_ZIPS/Billing_Violations.txt'

PATTERNS = {
    'rent increase': r'\$\d{3,4}',
    'admin fee': r'(admin(istrative)? fee|late fee)',
    'water|sewer': r'(water|sewer|utility charge)',
    'trash': r'(trash|refuse)',
    'double billing': r'(charged twice|duplicate|again)',
}


def analyze():
    findings = []
    for fname in os.listdir(EXHIBIT_DIR):
        path = os.path.join(EXHIBIT_DIR, fname)
        if not fname.endswith('.txt'):
            continue
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().lower()
            for label, pat in PATTERNS.items():
                if re.search(pat, content):
                    findings.append(f"{fname}: Possible {label} violation")
    return findings


def output(findings):
    with open(OUTFILE, 'w') as f:
        f.write('Billing Violation Findings (MCL 445.903):\n\n')
        for line in findings:
            f.write(line + '\n')
    print(f'Billing violations logged to {OUTFILE}')


if __name__ == '__main__':
    result = analyze()
    output(result)
