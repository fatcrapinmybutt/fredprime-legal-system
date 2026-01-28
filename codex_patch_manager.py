import datetime
import importlib.util
import json
import logging
import os
import shutil

PATCH_DIR = "patches/"
MANIFEST_FILE = "patch_manifest.json"
ERROR_LOG = "logs/codex_errors.log"
PATCH_HISTORY = "patch_history.json"
logging.basicConfig(filename=ERROR_LOG, level=logging.ERROR)


def backup_file(filepath):
    bak_name = f"{filepath}.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.bak"
    shutil.copy2(filepath, bak_name)
    return bak_name


def apply_patch(patch_path, target_file):
    bak = backup_file(target_file)
    try:
        spec = importlib.util.spec_from_file_location("patch_module", patch_path)
        patch_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch_module)
        log_patch(patch_path, target_file, bak, "success")
    except Exception as e:
        logging.error(f"Patch failed for {patch_path} on {target_file}: {e}")
        shutil.copy2(bak, target_file)
        log_patch(patch_path, target_file, bak, "rollback")
        print(f"Rolled back patch {patch_path}")


def log_patch(patch, target, backup, status):
    entry = {
        "patch": patch,
        "target": target,
        "backup": backup,
        "status": status,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if os.path.exists(PATCH_HISTORY):
        with open(PATCH_HISTORY, "r") as f:
            history = json.load(f)
    else:
        history = []
    history.append(entry)
    with open(PATCH_HISTORY, "w") as f:
        json.dump(history, f, indent=2)


if not os.path.exists(PATCH_DIR):
    print("⚠️ No patches found.")
    exit(0)

with open(MANIFEST_FILE) as f:
    manifest = json.load(f)

for patch_file in os.listdir(PATCH_DIR):
    if patch_file.endswith(".py"):
        patch_path = os.path.join(PATCH_DIR, patch_file)
        target = manifest.get(patch_file, {}).get("target", None)
        if target and os.path.exists(target):
            print(f"🔧 Applying patch: {patch_file} to {target}")
            apply_patch(patch_path, target)
        else:
            logging.error(f"No valid target for patch: {patch_file}")

print("✅ All patches applied successfully.")
