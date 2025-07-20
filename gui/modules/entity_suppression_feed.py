import json
import os

LOG_PATH = os.path.join('data', 'entity_suppression_log.json')

def log_event(entity: str, action: str):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    try:
        with open(LOG_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []
    data.append({'entity': entity, 'action': action})
    with open(LOG_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def load_events():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            return json.load(f)
    return []
