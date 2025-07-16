#!/bin/bash
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
