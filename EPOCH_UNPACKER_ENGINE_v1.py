"""EPOCH Unpacker Engine - Extract and process epoch-based timestamps."""


def unpack_epoch(epoch_value):
    """
    Convert epoch timestamp to human-readable format.
    
    Args:
        epoch_value: Unix epoch timestamp (seconds since 1970-01-01)
    
    Returns:
        Formatted datetime string
    """
    from datetime import datetime
    return datetime.fromtimestamp(epoch_value).strftime('%Y-%m-%d %H:%M:%S')


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        try:
            epoch = int(sys.argv[1])
            print(unpack_epoch(epoch))
        except ValueError:
            print("Error: Please provide a valid epoch timestamp")
    else:
        print("Usage: python EPOCH_UNPACKER_ENGINE_v1.py <epoch_timestamp>")
