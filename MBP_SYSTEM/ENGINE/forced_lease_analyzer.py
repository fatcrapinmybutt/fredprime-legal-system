import os

OUTFILE = 'MBP_SYSTEM/VFS/GENERATED_ZIPS/Lease_Coercion_Report.txt'
KEY_TRIGGERS = [
    'required to sign', 'non-renewed', 'forced', 'moved under threat', 'pay to stay',
    'no other option', 'eviction if not signed', 'Cricklewood', 'Kim Davis', 'new lease',
    'increase to 695', 'had to drive',
]

EXHIBIT_DIR = 'MBP_SYSTEM/EXHIBITS'


def detect_coercion():
    findings = []
    for fname in os.listdir(EXHIBIT_DIR):
        path = os.path.join(EXHIBIT_DIR, fname)
        if not fname.endswith('.txt'):
            continue
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if any(key in line.lower() for key in KEY_TRIGGERS):
                    findings.append(f"{fname} [Line {i+1}]: {line.strip()}")
    return findings


def report(findings):
    with open(OUTFILE, 'w', encoding='utf-8') as f:
        f.write('Forced Lease / Coercion Evidence:\n\n')
        for entry in findings:
            f.write(f"{entry}\n")
    print(f'Coercion report saved to: {OUTFILE}')


if __name__ == '__main__':
    findings = detect_coercion()
    report(findings)
