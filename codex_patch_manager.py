import os
import json
import shutil
import logging
import datetime
import hashlib
import builtins
from typing import Any, List, Optional, Sequence

PATCH_DIR: str = "patches/"
MANIFEST_FILE: str = "patch_manifest.json"
ERROR_LOG: str = "logs/codex_errors.log"
PATCH_HISTORY: str = "patch_history.json"
RESTRICTED_MODULES = {"os", "sys", "subprocess"}

os.makedirs(os.path.dirname(ERROR_LOG), exist_ok=True)
logging.basicConfig(filename=ERROR_LOG, level=logging.INFO)


def backup_file(filepath: str) -> str:
    bak_name = f"{filepath}.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.bak"
    shutil.copy2(filepath, bak_name)
    return bak_name


def verify_patch_hash(patch_path: str, expected_hash: str) -> bool:
    hasher = hashlib.sha256()
    with open(patch_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest() == expected_hash


def apply_patch(
    patch_path: str, target_file: str, allowed_modules: Optional[List[str]] = None
) -> None:
    bak = backup_file(target_file)
    allowed = set(allowed_modules or [])

    def restricted_import(
        name: str,
        globals: Optional[dict[str, Any]] = None,
        locals: Optional[dict[str, Any]] = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if name in RESTRICTED_MODULES and name not in allowed:
            raise ImportError(f"Import of {name} is restricted")
        return __import__(name, globals, locals, fromlist, level)

    safe_builtins = {k: getattr(builtins, k) for k in dir(builtins)}
    safe_builtins["__import__"] = restricted_import

    try:
        with open(patch_path, "r", encoding="utf-8") as f:
            code = f.read()
        exec(code, {"__builtins__": safe_builtins})
        log_patch(patch_path, target_file, bak, "success")
        logging.info("Patch applied: %s -> %s", patch_path, target_file)
    except Exception:
        logging.exception("Patch failed for %s on %s", patch_path, target_file)
        shutil.copy2(bak, target_file)
        log_patch(patch_path, target_file, bak, "rollback")
        print(f"Rolled back patch {patch_path}")


def log_patch(patch: str, target: str, backup: str, status: str) -> None:
    entry = {
        "patch": patch,
        "target": target,
        "backup": backup,
        "status": status,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if os.path.exists(PATCH_HISTORY):
        with open(PATCH_HISTORY, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []
    history.append(entry)
    with open(PATCH_HISTORY, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def main() -> None:
    if not os.path.exists(PATCH_DIR):
        print("⚠️ No patches found.")
        return

    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except FileNotFoundError:
        logging.error("Manifest file not found.")
        return

    for patch_file in os.listdir(PATCH_DIR):
        if patch_file.endswith(".py"):
            patch_path = os.path.join(PATCH_DIR, patch_file)
            meta = manifest.get(patch_file, {})
            target = meta.get("target")
            expected_hash = meta.get("hash")
            allowed = meta.get("allow", [])
            if not target or not os.path.exists(target):
                logging.error("No valid target for patch: %s", patch_file)
                continue
            if not expected_hash or not verify_patch_hash(patch_path, expected_hash):
                logging.error("Hash mismatch for patch: %s", patch_file)
                continue
            print(f"🔧 Applying patch: {patch_file} to {target}")
            apply_patch(patch_path, target, allowed)

    print("✅ All patches processed.")


if __name__ == "__main__":
    main()
