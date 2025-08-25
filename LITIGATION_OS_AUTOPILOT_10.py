
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LITIGATION_OS_AUTOPILOT_10.py
# Chained, streaming autopilot for 10 cycles driving LITIGATION_OS_CROSSWIRE_OMEGA_PLUS.py.
#
# Cycle steps:
#   1) Crawl code inventory
#   2) Bundle code and zip
#   3) LLM code audit (integration plan)
#   4) Evidence scan once (F:/, Z:/, C:/Litigation)
#   5) Attempt MiFILE uploads for files in READY_TO_FILE (heuristics for case/doc type)
#   6) Rebuild dashboard
#
# Logs stream to console and also to logs/LITOS_autopilot_<timestamp>.log.

import os, sys, subprocess, time, re
from pathlib import Path
from datetime import datetime

ENGINE = Path(__file__).parent / "LITIGATION_OS_CROSSWIRE_OMEGA_PLUS.py"
PYTHON = sys.executable

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def run_engine(args, timeout=None):
    cmd = [PYTHON, str(ENGINE)] + args
    log(f"$ {' '.join(a if ' ' not in a else repr(a) for a in cmd)}")
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True) as p:
        start = time.time()
        for line in p.stdout:
            print(line.rstrip())
            try:
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(line)
            except Exception:
                pass
            if timeout and (time.time() - start > timeout):
                p.kill()
                log("Process timed out and was terminated.")
                break
        rc = p.wait()
    log(f"Return code: {rc}")
    return rc

def infer_case_and_doctype(path: Path):
    name = path.name.lower()
    case = "housing"
    if "custody" in name or re.search(r"\\bdc\\b", name): case = "custody"
    if re.search(r"\\blt\\b", name) or "district" in name: case = "lt"
    doctype = "Filing"
    if "motion" in name: doctype = "Motion"
    elif "brief" in name: doctype = "Brief"
    elif "pos" in name or "proof of service" in name: doctype = "Proof of Service"
    elif "exhibit" in name: doctype = "Exhibit"
    elif "affidavit" in name: doctype = "Affidavit"
    return case, doctype

def mifile_push_ready():
    # Read engine config for READY_TO_FILE path
    cfg_path = Path(os.getenv("APPDATA") or Path.home()/".config") / "LitOS" / "config.json"
    if not cfg_path.exists():
        log("Config not found; running engine once to create it.")
        run_engine(["--rebuild"])
    rtf = None
    try:
        import json
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        rtf = Path(cfg.get("ready_to_file") or "F:/READY_TO_FILE")
    except Exception:
        rtf = Path("F:/READY_TO_FILE")
    if not rtf.exists():
        log(f"READY_TO_FILE not found: {rtf}")
        return 0
    count = 0
    for p in sorted(rtf.glob("*.*")):
        try:
            if (time.time() - p.stat().st_mtime) < 3:
                continue
        except Exception:
            pass
        case, doctype = infer_case_and_doctype(p)
        log(f"Auto-upload via MiFILE: {p}  (case={case}, doc_type={doctype})")
        rc = run_engine(["--mifile-upload", str(p), "--case", case, "--doc-type", doctype], timeout=300)
        if rc == 0:
            count += 1
    return count

def main():
    if not ENGINE.exists():
        print(f"ERROR: Engine not found at {ENGINE}. Place this script next to LITIGATION_OS_CROSSWIRE_OMEGA_PLUS.py")
        sys.exit(2)
    cycles = 10
    log(f"Starting autonomous autopilot for {cycles} cycles.")
    for i in range(1, cycles+1):
        log("="*40)
        log(f"BEGIN CYCLE {i}/{cycles}")
        run_engine(["--crawl-code"])
        run_engine(["--bundle-code"])
        run_engine(["--audit-code"])
        run_engine(["--once"])
        pushed = mifile_push_ready()
        log(f"MiFILE uploads attempted: {pushed}")
        run_engine(["--rebuild"])
        log(f"END CYCLE {i}/{cycles}")
        if i < cycles:
            log("Sleeping 30 seconds before next cycle...")
            time.sleep(30)
    log("Autopilot complete.")

if __name__ == "__main__":
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR = Path.cwd() / "logs"
    LOG_FILE = LOG_DIR / f"LITOS_autopilot_{ts}.log"
    main()
