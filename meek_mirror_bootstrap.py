# -*- coding: utf-8 -*-
"""
MEEK Mirror Bootstrap — Windows, copy/paste ready, no admin required.

What it does (default = one-way local → Google Drive):
  1) Ensures local rclone in ./bin (downloads from official site if missing)
  2) Ensures rclone config dir and a 'gdrive' remote (prompts OAuth once)
  3) Creates Drive folders /MIRRORS/MEEK1 and /MIRRORS/MEEK2 if missing
  4) Mirrors F:\MEEK1 → gdrive:/MIRRORS/MEEK1 and F:\MEEK2 → gdrive:/MIRRORS/MEEK2
  5) Logs to .\logs\meek1_sync.log and .\logs\meek2_sync.log
  6) Writes a short status receipt to .\logs\last_run.txt
  7) Optional: installs an hourly Windows Scheduled Task that re-runs this script

Usage:
  - Run once, interactive OAuth if needed:
      python meek_mirror_bootstrap.py
  - Install hourly task (then it runs in background every hour):
      python meek_mirror_bootstrap.py --install-task
  - Remove the scheduled task:
      python meek_mirror_bootstrap.py --remove-task
  - Use bidirectional sync instead of one-way:
      python meek_mirror_bootstrap.py --bisync
  - Override paths or remote via env or CLI:
      MEEK1=E:\Housing  MEEK2=E:\Custody  GDRIVE_REMOTE=gdrive  python meek_mirror_bootstrap.py
      python meek_mirror_bootstrap.py --meek1 "E:\Housing" --meek2 "E:\Custody" --remote "gdrive"

Notes:
  - First run may open a browser for Google OAuth. Approve once. Token persists.
  - If you prefer one-time daily task at 02:30 local: add  --daily 02:30  to --install-task.
  - To switch to bidirectional later, re-run with --bisync (it uses rclone bisync).
"""
import os
import sys
import shutil
import subprocess
import zipfile
import io
import time
import argparse
import json
from urllib.request import urlopen

HERE = os.path.abspath(os.path.dirname(__file__) or ".")
BIN_DIR = os.path.join(HERE, "bin")
LOG_DIR = os.path.join(HERE, "logs")
os.makedirs(BIN_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

DEFAULT_MEEK1 = os.environ.get("MEEK1", r"F:\MEEK1")
DEFAULT_MEEK2 = os.environ.get("MEEK2", r"F:\MEEK2")
DEFAULT_REMOTE = os.environ.get("GDRIVE_REMOTE", "gdrive")

RCLONE_EXE = os.path.join(BIN_DIR, "rclone.exe")
RCLONE_DOWNLOAD = "https://downloads.rclone.org/rclone-current-windows-amd64.zip"  # official rolling URL


def run(cmd, check=True, capture=False):
    """Run a command. Returns (rc, out)."""
    shell = not isinstance(cmd, list)
    try:
        if capture:
            out = subprocess.check_output(cmd, shell=shell, stderr=subprocess.STDOUT)
            return 0, out.decode("utf-8", errors="ignore")
        rc = subprocess.call(cmd, shell=shell)
        if check and rc != 0:
            raise RuntimeError(f"Command failed: {cmd} (rc={rc})")
        return rc, ""
    except subprocess.CalledProcessError as exc:
        if check:
            raise
        return exc.returncode, exc.output.decode("utf-8", errors="ignore")


def windows_quote(path):
    return f'"{path}"'


def is_windows():
    return os.name == "nt"


def ensure_rclone():
    if os.path.isfile(RCLONE_EXE):
        return
    print("[*] rclone not found. Downloading…")
    with urlopen(RCLONE_DOWNLOAD) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        exe_member = None
        for name in zf.namelist():
            if name.lower().endswith("/rclone.exe"):
                exe_member = name
                break
        if exe_member is None:
            raise RuntimeError("rclone.exe not found in zip")
        with zf.open(exe_member) as src, open(RCLONE_EXE, "wb") as dst:
            shutil.copyfileobj(src, dst)
    print(f"[+] rclone installed → {RCLONE_EXE}")


def rclone_env():
    env = os.environ.copy()
    cfg_dir = os.path.join(HERE, "rclone_config")
    os.makedirs(cfg_dir, exist_ok=True)
    env["RCLONE_CONFIG"] = os.path.join(cfg_dir, "rclone.conf")
    return env


def list_remotes():
    rc, out = run(f"{windows_quote(RCLONE_EXE)} listremotes", check=False, capture=True)
    if rc != 0:
        return []
    return [line.strip().rstrip(":") for line in out.splitlines() if line.strip()]


def ensure_remote(remote):
    remotes = list_remotes()
    if remote in remotes:
        return
    print(f"[*] rclone remote '{remote}' not found. Creating it now.")
    create_cmd = f"{windows_quote(RCLONE_EXE)} config create {remote} drive scope=drive"
    rc, out = run(create_cmd, check=False, capture=True)
    if rc != 0 or "Successfully created" not in out:
        print("[*] Launching OAuth in your browser. Approve access for rclone.")
        reconnect_cmd = f"{windows_quote(RCLONE_EXE)} config reconnect {remote}:"
        run(reconnect_cmd, check=True)
    print(f"[+] Remote '{remote}' ready.")


def ensure_drive_folders(remote):
    for sub in ("MIRRORS/MEEK1", "MIRRORS/MEEK2"):
        run(f"{windows_quote(RCLONE_EXE)} mkdir {remote}:/{sub}", check=False)


def sync_once(remote, meek_path, drive_sub, mode_bisync=False, log_name="sync.log"):
    log_file = os.path.join(LOG_DIR, log_name)
    if not os.path.isdir(meek_path):
        print(f"[!] Local path missing, creating: {meek_path}")
        os.makedirs(meek_path, exist_ok=True)
    dst = f"{remote}:/MIRRORS/{drive_sub}"
    if mode_bisync:
        marker = os.path.join(
            HERE, f".bisync_initialized_{drive_sub.replace('/', '_')}"
        )
        base_flags = (
            "--verbose --check-access --remove-empty-dirs "
            f"--log-file {windows_quote(log_file)} --log-level INFO"
        )
        if not os.path.isfile(marker):
            cmd = (
                f"{windows_quote(RCLONE_EXE)} bisync {windows_quote(meek_path)} {dst} "
                f"--resync {base_flags}"
            )
            rc, _ = run(cmd, check=False, capture=True)
            if rc == 0:
                with open(marker, "w", encoding="utf-8") as file:
                    file.write(str(time.time()))
        cmd = (
            f"{windows_quote(RCLONE_EXE)} bisync {windows_quote(meek_path)} {dst} "
            f"{base_flags}"
        )
    else:
        flags = [
            "--create-empty-src-dirs",
            "--copy-links",
            "--log-level INFO",
            f"--log-file {windows_quote(log_file)}",
            "--drive-stop-on-upload-limit",
            "--stats 30s",
        ]
        cmd = (
            f"{windows_quote(RCLONE_EXE)} sync {windows_quote(meek_path)} {dst} "
            + " ".join(flags)
        )
    print(f"[*] {drive_sub}: running {'bisync' if mode_bisync else 'sync'} …")
    rc, out = run(cmd, check=False, capture=True)
    receipt = {
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "bisync" if mode_bisync else "sync",
        "local": meek_path,
        "remote": dst,
        "rc": rc,
        "tail": "\n".join(out.splitlines()[-20:]),
        "log": log_file,
    }
    with open(os.path.join(LOG_DIR, "last_run.txt"), "a", encoding="utf-8") as file:
        file.write(json.dumps(receipt, ensure_ascii=False) + "\n")
    if rc != 0:
        print(f"[!] {drive_sub}: non-zero return ({rc}). Check {log_file}.")
    else:
        print(f"[+] {drive_sub}: complete. Log → {log_file}")


def install_task(py_path, args, daily_at=None):
    if not is_windows():
        print("[!] Scheduler install only supported on Windows.")
        return
    task_name = r"LAWFORGE\MEEK_MIRROR"
    run(
        f'schtasks /Create /TN "{task_name}" /SC HOURLY /TR "{py_path} {args}" /F',
        check=False,
    )
    if daily_at:
        run(f'schtasks /Delete /TN "{task_name}" /F', check=False)
        run(
            f'schtasks /Create /TN "{task_name}" /SC DAILY /ST {daily_at} '
            f'/TR "{py_path} {args}" /F',
            check=True,
        )
    print(f"[+] Scheduled Task installed: {task_name}")


def remove_task():
    task_name = r"LAWFORGE\MEEK_MIRROR"
    run(f'schtasks /Delete /TN "{task_name}" /F', check=False)
    print(f"[+] Scheduled Task removed: {task_name}")


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--meek1", default=DEFAULT_MEEK1, help="Local path for MEEK1 (Housing)"
    )
    parser.add_argument(
        "--meek2", default=DEFAULT_MEEK2, help="Local path for MEEK2 (Custody)"
    )
    parser.add_argument(
        "--remote", default=DEFAULT_REMOTE, help="rclone remote name (default: gdrive)"
    )
    parser.add_argument(
        "--bisync", action="store_true", help="Use bidirectional sync (rclone bisync)"
    )
    parser.add_argument(
        "--install-task",
        action="store_true",
        help="Install hourly Windows Scheduled Task",
    )
    parser.add_argument(
        "--daily", default=None, help="Use with --install-task to run daily at HH:MM"
    )
    parser.add_argument(
        "--remove-task", action="store_true", help="Remove the scheduled task"
    )
    args = parser.parse_args()

    if args.remove_task:
        remove_task()
        return

    ensure_rclone()
    ensure_remote(args.remote)
    ensure_drive_folders(args.remote)

    sync_once(
        args.remote,
        args.meek1,
        "MEEK1",
        mode_bisync=args.bisync,
        log_name="meek1_sync.log",
    )
    sync_once(
        args.remote,
        args.meek2,
        "MEEK2",
        mode_bisync=args.bisync,
        log_name="meek2_sync.log",
    )

    if args.install_task:
        py_exe = windows_quote(sys.executable)
        script = windows_quote(os.path.abspath(sys.argv[0]))
        sched_args = f'{script} --remote "{args.remote}" --meek1 "{args.meek1}" --meek2 "{args.meek2}"'
        if args.bisync:
            sched_args += " --bisync"
        install_task(py_exe, sched_args, daily_at=args.daily)

    print("[✓] MEEK mirror finished.")


if __name__ == "__main__":
    if not is_windows():
        print("[!] This bootstrap targets Windows. Run on Windows 10/11.")
        sys.exit(1)
    try:
        main()
    except Exception as err:
        err_file = os.path.join(LOG_DIR, "fatal_error.txt")
        with open(err_file, "a", encoding="utf-8") as file:
            file.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {repr(err)}\n")
        print(f"[!] Fatal error. See {err_file}")
        sys.exit(2)
