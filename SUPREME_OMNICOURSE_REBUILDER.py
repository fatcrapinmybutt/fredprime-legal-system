# SUPREME_OMNICOURSE_REBUILDER.py
# This file recreates all essential files for the litigation automation system.

import os
from pathlib import Path

def write(path, content):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# 1. requirements.txt
write("requirements.txt", """openai==1.14.3
requests==2.31.0
pycryptodome==3.20.0
python-dotenv==1.0.1
python-docx==1.1.0
rich==13.7.1
pytest==8.2.0
""")

# 2. SovereignSetup.sh
write("SovereignSetup.sh", """#!/bin/bash
# FRED SUPREME OMNICOURSE – Secure Setup Script
echo "🔐 Initializing Litigation OS Environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  echo "❌ ERROR: requirements.txt missing"
  exit 1
fi
if [ -f codex.seal ]; then
  echo "🧾 Decrypting codex.seal..."
  if [ -z "$SETUP_PASSPHRASE" ]; then
    echo "❌ SETUP_PASSPHRASE not set"
    exit 1
  fi
  gpg --quiet --batch --yes --decrypt --passphrase="$SETUP_PASSPHRASE" --output codex.env codex.seal
fi
if [ -f codex.env ]; then
  echo "✅ Loading codex.env..."
  export $(grep -v '^#' codex.env | xargs)
fi
python3 verify_keys.py || echo "⚠️ Key check failed"
echo "🎯 Setup complete. Run GUI with: python3 gui/gui_launcher.py"
""")

# 3. verify_keys.py
write("verify_keys.py", """import os
def verify_key(name):
    val = os.getenv(name)
    if not val or len(val) < 20:
        raise ValueError(f"🔒 {name} missing or too short")
    print(f"✅ {name} loaded")
verify_key("OPENAI_KEY")
verify_key("MBP_LICENSE")
""")

# 4. pytest.ini
write("pytest.ini", "[pytest]\ntestpaths = tests\npython_files = test_*.py\n")

# 5. .gitignore
write(".gitignore", "__pycache__/\n.venv/\ncodex.env\ncodex.seal\n*.log\n")

# 6. Minimal test
write("tests/test_env.py", """import os
def test_openai_key():
    assert os.getenv("OPENAI_KEY"), "Missing OPENAI_KEY"
def test_license():
    assert os.getenv("MBP_LICENSE", "").startswith("MBP-"), "Invalid MBP_LICENSE"
""")

# 7. Confirm directory and structure created
print("✅ All core files created. Now run: bash SovereignSetup.sh")
