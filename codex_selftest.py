# codex_selftest.py
# 🔧 LAWFORGE / FRED System Validator + Auto-Fix Installer

from __future__ import annotations

import hashlib
import importlib
import os
import subprocess
import sys
from typing import List, Optional, TypedDict


class ModuleEntry(TypedDict):
    name: str
    path: str
    hash: Optional[str]


MODULES: List[ModuleEntry] = [
    {
        "name": "benchbook_loader",
        "path": "modules/benchbook_loader.py",
        "hash": "659ea627d4c1f2b1b93118ce5f16d52e13fb23e79cbb44a6ade1b609c14ece18",
    },
    {
        "name": "meek_ingest_cli",
        "path": "modules/meek_ingest_cli.py",
        "hash": "b34897f35bb45f98f13fc37f0eacce21d3c44cb410469a43bc52e4cc81a23a60",
    },
    {
        "name": "fts_cli",
        "path": "modules/fts_cli.py",
        "hash": "503f84e22f31b76426d56bb8d154fe6e97f1734844d48f0971b06743f9ad6a94",
    },
    {
        "name": "meek_pipeline_launcher",
        "path": "scripts/meek_pipeline_launcher.py",
        "hash": "44c2d8693f71df0a1de0049e5d6900116c6ba3cb3fcfd544376b0323c07e642f",
    },
    {
        "name": "sentinel_code_scanner",
        "path": "scripts/sentinel_code_scanner.py",
        "hash": "dc516b16751466af63629a6018cc8d0f0dc07e5385f4166dc74dba1562fa9f5c",
    },
]

DEPENDENCIES: List[str] = [
    "PyPDF2",
    "pytesseract",
    "pdf2image",
    "PIL",
    "docx",
    "pandas",
    "regex",
    "unidecode",
    "tqdm",
]

OUTPUT_FILE = "F:/LegalResults/MEEK_DB.jsonl"


def sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_modules() -> None:
    print("\n🔍 [MODULE CHECK]")
    for mod in MODULES:
        path = mod["path"]
        name = mod["name"]
        expected_hash = mod.get("hash")
        print(f"🔧 {name} — ", end="")
        if not os.path.exists(path):
            print(f"❌ MISSING: {path}")
            continue
        if expected_hash:
            actual = sha256(path)
            if actual != expected_hash:
                print(
                    f"⚠️ HASH MISMATCH (Expected {expected_hash[:8]}..., Got {actual[:8]}...)"
                )
            else:
                print("✅ OK (hash match)")
        else:
            print("✅ OK")


def verify_dependencies(auto_fix: bool = True) -> None:
    print("\n📦 [DEPENDENCY CHECK]")
    missing: List[str] = []
    for dep in DEPENDENCIES:
        try:
            importlib.import_module(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} — missing")
            missing.append(dep)
    if auto_fix and missing:
        print("\n⚙️ Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        print("✅ Dependencies installed.")
        print("🔁 Re-running dependency check...\n")
        verify_dependencies(auto_fix=False)


def verify_output() -> None:
    print("\n📁 [OUTPUT PATH CHECK]")
    if os.path.exists(OUTPUT_FILE):
        size = os.path.getsize(OUTPUT_FILE)
        print(f"✅ Output found: {OUTPUT_FILE} ({size / 1024:.2f} KB)")
    else:
        print(f"⚠️ Output missing: {OUTPUT_FILE}")


def run_selftest() -> None:
    print("=" * 60)
    print("🧠 LAWFORGE / FRED SELFTEST + AUTO-FIX")
    print("=" * 60)
    verify_modules()
    verify_dependencies(auto_fix=True)
    verify_output()
    print("\n✅ Self-test complete.\n")


if __name__ == "__main__":
    run_selftest()
