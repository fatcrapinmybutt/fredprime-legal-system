import os
from ENGINE import motion_generator, affidavit_synthesizer

TRIGGER_LOG = 'MBP_SYSTEM/CHAINLOGS/trigger_events.log'

TRIGGERS = {
    'parenting time denied': 'Parenting Time Enforcement',
    'false police report': 'Contempt Defense',
    'welfare check': 'Protective Order Removal',
    'threat': 'Custody Modification',
}


def scan_exhibits():
    exhibit_dir = 'MBP_SYSTEM/EXHIBITS'
    events = []
    for file in os.listdir(exhibit_dir):
        path = os.path.join(exhibit_dir, file)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().lower()
            for key, motion_type in TRIGGERS.items():
                if key in content:
                    events.append((motion_type, file))
    return events


def trigger(events):
    with open(TRIGGER_LOG, 'a') as log:
        for motion_type, file in events:
            motion_generator.build_motion(motion_type, f'Trigger from {file}', 'Requested relief')
            log.write(f"Triggered {motion_type} from {file}\n")


if __name__ == '__main__':
    events = scan_exhibits()
    trigger(events)
