#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script to ensure CONTEXT7_API_KEY is available for local development
# It will merge/update the key into an env file (default: .mcp_env) rather than
# clobbering any existing variables.
# Usage: ./tools/bootstrap_context7_key.sh [--out PATH]

OUT_FILE=".mcp_env"

usage() {
  cat <<-USAGE
Usage: $0 [--out PATH] [--help]

Writes or updates CONTEXT7_API_KEY in PATH (default: $OUT_FILE) with secure
permissions (600). If CONTEXT7_API_KEY is set in the environment already, that
value will be used; otherwise you'll be prompted interactively.
USAGE
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --out) OUT_FILE="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

# Determine key: prefer env, then prompt
if [[ -n "${CONTEXT7_API_KEY:-}" ]]; then
  KEY="$CONTEXT7_API_KEY"
else
  read -r -p "Enter CONTEXT7_API_KEY: " -s KEY
  echo
fi

if [[ -z "$KEY" ]]; then
  echo "No key provided; aborting." >&2
  exit 2
fi

# Read existing file (if any) into an associative array of KEY=VALUE pairs
declare -A kv
if [[ -f "$OUT_FILE" ]]; then
  while IFS='=' read -r k v; do
    # skip empty lines and comments
    [[ -z "$k" || "$k" =~ ^# ]] && continue
    kv["$k"]="$v"
  done < <(grep -E '^[A-Za-z_][A-Za-z0-9_]*=' "$OUT_FILE" || true)
fi

# Update or set the CONTEXT7_API_KEY
kv[CONTEXT7_API_KEY]="$KEY"

# Write back to a temporary file then atomically move
TMPFILE="${OUT_FILE}.tmp"
{
  for k in "${!kv[@]}"; do
    printf '%s=%s\n' "$k" "${kv[$k]}"
  done
} > "$TMPFILE"

chmod 600 "$TMPFILE"
mv "$TMPFILE" "$OUT_FILE"

echo "Wrote/updated CONTEXT7_API_KEY in $OUT_FILE (mode 600)."
echo
echo "To load the key in this shell session, run:"
echo "  set -a; source $OUT_FILE; set +a"
echo
echo "Devcontainer: add to .devcontainer/devcontainer.json under 'containerEnv':"
cat <<'JSON'
"containerEnv": {
  "CONTEXT7_API_KEY": "${localEnv:CONTEXT7_API_KEY}"
}
JSON

echo
echo "CI: add a repo secret named CONTEXT7_API_KEY and expose it in your workflow envs."

exit 0
