import os, subprocess
from pathlib import Path

def write(path, content):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

print("🔧 Generating Litigation OS: Eternal Absolution Build...")

# 1. requirements.txt
write("requirements.txt", """openai==1.14.3
requests==2.31.0
pycryptodome==3.20.0
python-dotenv==1.0.1
python-docx==1.1.0
rich==13.7.1
pytest==8.2.0
""")

# 2. codex.env example
write("codex.env", """# Secrets go here (never commit)
OPENAI_KEY=sk-REPLACE_ME
MBP_LICENSE=MBP-FRED-SUPREME-999999
""")

# 3. .gitignore
write(".gitignore", "__pycache__/\n.venv/\ncodex.env\ncodex.seal\n*.log\n")

# 4. verify_keys.py
write("verify_keys.py", """import os
def verify_key(name):
    val = os.getenv(name)
    if not val or len(val) < 20:
        raise ValueError(f\"🔒 {name} missing or too short\")
    print(f\"✅ {name} loaded\")
verify_key(\"OPENAI_KEY\")
verify_key(\"MBP_LICENSE\")
""")

# 5. SovereignSetup.sh
write("SovereignSetup.sh", """#!/bin/bash
echo \"🔐 Initializing Litigation OS Environment...\"
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt || { echo '❌ Install failed'; exit 1; }
if [ -f codex.env ]; then
  echo \"🔐 Encrypting codex.env...\"
  gpg --quiet --batch --yes --symmetric --cipher-algo AES256 --passphrase \"$SETUP_PASSPHRASE\" codex.env
  mv codex.env codex.seal && rm codex.env
fi
echo \"🎯 Litigation OS setup complete.\"
""")

# 6. Windows BAT builder
write("build_setup_windows.bat", """@echo off
echo 🔧 Creating venv...
python -m venv .venv
call .venv\\Scripts\\activate
echo 📦 Installing packages...
pip install --upgrade pip
pip install -r requirements.txt
echo 🔐 Decrypting codex.seal...
gpg --quiet --batch --yes --passphrase %SETUP_PASSPHRASE% --output codex.env --decrypt codex.seal
echo 💪 Verifying...
python verify_keys.py
echo 🏗️ Building .exe...
pyinstaller FRED.spec --clean
echo ✅ Done. Run from /dist/FRED_LITIGATION_OS/
""")

# 7. PyInstaller spec
write("FRED.spec", """# -*- mode: python ; coding: utf-8 -*-
block_cipher = None
a = Analysis(['gui/gui_launcher.py'],
             pathex=['.'],
             binaries=[],
             datas=[('forms/*.docx', 'forms')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='FRED_LITIGATION_OS',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='FRED_LITIGATION_OS')
""")

# 8. GUI Launcher placeholder
write("gui/gui_launcher.py", """import os
print(\"🧠 Litigation GUI Loaded (placeholder)\")
# TODO: Replace with full MBP Strategist GUI logic
""")

# 9. Forms and affidavit template
write("forms/template_motion.docx", "")  # placeholder file

# 10. Sample test
write("tests/test_env.py", """import os
def test_openai_key():
    assert os.getenv('OPENAI_KEY'), 'Missing OPENAI_KEY'
def test_license():
    assert os.getenv('MBP_LICENSE', '').startswith('MBP-')
""")

# 11. pytest.ini
write("pytest.ini", "[pytest]\ntestpaths = tests\npython_files = test_*.py\n")

# 12. Summary
print("✅ All Eternal Codex Build files created.")
print("⚡ To run setup (Linux/Mac): bash SovereignSetup.sh")
print("💻 To build on Windows: run build_setup_windows.bat")
