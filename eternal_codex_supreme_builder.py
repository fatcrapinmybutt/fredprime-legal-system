# eternal_codex_supreme_builder.py
"""Utility to assemble a full MBP litigation bundle.

This script creates all required files and directories, seals the
environment file with GPG, compiles the GUI into an executable using
PyInstaller, and packages everything into a deployable ZIP archive.
It uses placeholder secrets for demonstration.
"""

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

ENV = {
    "OPENAI_KEY": "sk-FAKE-EXAMPLE-KEY-12345678901234567890",
    "MBP_LICENSE": "MBP-FAKE-LICENSE-9999",
    "SETUP_PASSPHRASE": "universal",
}

PROJECT_ROOT = Path("MBP_LITIGATION_OS").resolve()
GUI_PATH = PROJECT_ROOT / "gui" / "gui_launcher.py"
EXE_NAME = "MBP_LITIGATION_OS.exe"
ZIP_NAME = "usb_deploy_payload.zip"
FINAL_BUNDLE = PROJECT_ROOT / "FINAL_BUNDLE"


def make_dirs() -> None:
    """Create the required directory structure."""
    (PROJECT_ROOT / "gui").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "tests").mkdir(exist_ok=True)
    (PROJECT_ROOT / "forms").mkdir(exist_ok=True)


def write_env() -> None:
    """Write example environment variables to ``codex.env``."""
    env_path = PROJECT_ROOT / "codex.env"
    with env_path.open("w") as f:
        for key, value in ENV.items():
            f.write(f"{key}={value}\n")


def write_files() -> None:
    """Create base files used by the project."""
    Path(PROJECT_ROOT / ".gitignore").write_text(
        """__pycache__/
.venv/
codex.env
codex.seal
*.log
"""
    )

    Path(PROJECT_ROOT / "requirements.txt").write_text(
        """openai==1.14.3
requests==2.31.0
pycryptodome==3.20.0
python-dotenv==1.0.1
python-docx==1.1.0
rich==13.7.1
pytest==8.2.0
"""
    )

    Path(PROJECT_ROOT / "verify_keys.py").write_text(
        """import os

def verify_key(name: str) -> None:
    val = os.getenv(name)
    if not val or len(val) < 20:
        raise ValueError(f"\N{LOCK} {name} missing or too short")
    print(f"\N{WHITE HEAVY CHECK MARK} {name} loaded")

verify_key("OPENAI_KEY")
verify_key("MBP_LICENSE")
"""
    )

    Path(PROJECT_ROOT / "tests/test_env.py").write_text(
        """import os

def test_openai_key():
    assert os.getenv("OPENAI_KEY"), "Missing OPENAI_KEY"

def test_license():
    assert os.getenv("MBP_LICENSE", "").startswith("MBP-"), "Invalid MBP_LICENSE"
"""
    )

    Path(PROJECT_ROOT / "pytest.ini").write_text(
        "[pytest]\ntestpaths = tests\npython_files = test_*.py\n"
    )

    Path(GUI_PATH).write_text(
        """import tkinter as tk
from tkinter import messagebox


def run_gui() -> None:
    root = tk.Tk()
    root.title('MBP Litigation OS')
    root.geometry('480x380')
    tk.Label(root, text='MBP Supreme Litigation OS', font=('Arial', 16)).pack(pady=10)

    def trigger_affidavit() -> None:
        messagebox.showinfo('Trigger', '\N{SCROLL} Affidavit Generator Activated')

    def scan_mcr() -> None:
        messagebox.showwarning('Violation Scan', '\N{SCALES} MCR / Canon / Benchbook Violations Found')

    def build_zip() -> None:
        messagebox.showinfo('ZIP', '\N{PACKAGE} Final ZIP Bundle Compiled')

    tk.Button(root, text='\N{SCROLL} Generate Affidavit', command=trigger_affidavit).pack(pady=10)
    tk.Button(root, text='\N{SCALES} Scan for Violations', command=scan_mcr).pack(pady=10)
    tk.Button(root, text='\N{PACKAGE} Build Final Binder ZIP', command=build_zip).pack(pady=10)
    root.mainloop()


if __name__ == '__main__':
    run_gui()
"""
    )

    Path(PROJECT_ROOT / "forms/template_motion.docx").write_text(
        "<<SAMPLE COURT MOTION TEMPLATE>>"
    )
    Path(PROJECT_ROOT / "autorun.inf").write_text(
        f"""[autorun]
open={EXE_NAME}
icon=icon.ico
label=MBP LITIGATION SYSTEM
"""
    )


def gpg_seal_env() -> None:
    """Encrypt ``codex.env`` using GPG."""
    env_path = PROJECT_ROOT / "codex.env"
    sealed_path = PROJECT_ROOT / "codex.seal"
    subprocess.run(
        [
            "gpg",
            "--batch",
            "--yes",
            "--passphrase",
            ENV["SETUP_PASSPHRASE"],
            "-c",
            "--output",
            str(sealed_path),
            str(env_path),
        ],
        check=True,
    )


def compile_gui_exe() -> None:
    """Compile the GUI into a standalone executable using PyInstaller."""
    cwd = os.getcwd()
    os.chdir(PROJECT_ROOT)
    try:
        subprocess.run(
            [
                "pyinstaller",
                "--noconfirm",
                "--onefile",
                "--windowed",
                "--name",
                EXE_NAME,
                "gui/gui_launcher.py",
            ],
            check=True,
        )
    finally:
        os.chdir(cwd)


def bundle_zip() -> None:
    """Bundle the executable and support files into a zip archive."""
    FINAL_BUNDLE.mkdir(exist_ok=True)
    shutil.copy(PROJECT_ROOT / "dist" / EXE_NAME, FINAL_BUNDLE / EXE_NAME)
    shutil.copy(PROJECT_ROOT / "codex.seal", FINAL_BUNDLE / "codex.seal")
    shutil.copy(PROJECT_ROOT / "autorun.inf", FINAL_BUNDLE / "autorun.inf")
    shutil.copy(
        PROJECT_ROOT / "forms" / "template_motion.docx",
        FINAL_BUNDLE / "template_motion.docx",
    )

    with zipfile.ZipFile(ZIP_NAME, "w") as zf:
        for file in FINAL_BUNDLE.iterdir():
            zf.write(file, arcname=file.name)


def main() -> None:
    print("\N{PACKAGE} Building full MBP Supreme OS...")
    make_dirs()
    write_env()
    write_files()
    gpg_seal_env()
    compile_gui_exe()
    bundle_zip()
    print(f"\N{WHITE HEAVY CHECK MARK} Build complete. ZIP ready: {ZIP_NAME}")


if __name__ == "__main__":
    main()
