def auto_build_motion(data=None):
    print('AUTO-MOTION triggered')
    if not data:
        data = {'type': 'Contempt Defense', 'facts': 'Test facts'}
    print(f"Motion for: {data['type']} with facts: {data['facts']}")
