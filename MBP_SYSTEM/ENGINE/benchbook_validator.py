import os

RULES = {
    'MCR 2.114': 'Signature on documents',
    'MCR 3.207': 'Motions in domestic cases',
    'MCL 722.27': 'Custody modification standards',
    'Canon 3': 'Judicial bias/disqualification',
}


def validate_text(filepath):
    print(f'Scanning: {filepath}')
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    for rule, desc in RULES.items():
        if rule in text:
            print(f'Found {rule}: {desc}')
        else:
            print(f'Missing {rule}: {desc}')


if __name__ == '__main__':
    validate_text('MBP_SYSTEM/CHAINLOGS/sample.txt')
