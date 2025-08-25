import os
import json
import logging
import threading
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Optional

DEFAULT_OUTPUT = os.path.join("data", "scan_index.json")


logging.basicConfig(level=logging.INFO, format="%(threadName)s: %(message)s")
_index_lock = threading.Lock()


def scan_drive(root_dir: str) -> Dict[str, Dict[str, str]]:
    """Scan ``root_dir`` using an explicit stack and return an index of files."""
    index: Dict[str, Dict[str, str]] = {}
    stack = [root_dir]
    while stack:
        current = stack.pop()
        try:
            for entry in os.scandir(current):
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif entry.is_file() and entry.name.lower().endswith(
                    (".docx", ".pdf", ".txt")
                ):
                    try:
                        created = datetime.fromtimestamp(
                            os.path.getctime(entry.path)
                        ).isoformat()
                        index[entry.path] = {"created": created}
                        logging.info("Match: %s", entry.path)
                    except OSError:
                        pass
        except OSError:
            pass
    return index


def run_scan(drives: Optional[List[str]] = None, output: str = DEFAULT_OUTPUT) -> None:
    """Scan the provided drives and write an index of legal files."""

    if drives is None:
        env_drives = os.getenv("SCAN_DRIVES")
        if env_drives:
            drives = env_drives.split(os.pathsep)
        else:
            drives = ["F:/", "D:/"]

    index: dict[str, dict[str, str]] = {}

    existing = [d for d in drives if os.path.exists(d)]
    missing = [d for d in drives if not os.path.exists(d)]
    for drive in missing:
        logging.info("Skip missing drive: %s", drive)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_drive = {
            executor.submit(scan_drive, drive): drive for drive in existing
        }
        for future in concurrent.futures.as_completed(future_to_drive):
            drive = future_to_drive[future]
            try:
                result = future.result()
                with _index_lock:
                    index.update(result)
                logging.info("Completed scan: %s (%d files)", drive, len(result))
            except Exception as exc:  # pragma: no cover - log unexpected errors
                logging.error("%s generated an exception: %s", drive, exc)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(index, f, indent=2)
    logging.info("Scan complete. Indexed %d files to %s", len(index), output)


if __name__ == "__main__":
    import sys

    drives = sys.argv[1:] or None
    run_scan(drives)
