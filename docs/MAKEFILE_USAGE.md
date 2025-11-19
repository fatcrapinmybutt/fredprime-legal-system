Makefile helper: CONTEXT7 key bootstrap
=====================================

This short document explains the `tools/context7-setup` Makefile target and the `tools/bootstrap_context7_key.sh` helper.

Usage

From the repository root:

```bash
# Make the helper executable (once)
chmod +x tools/bootstrap_context7_key.sh

# Provide the key via env and run the helper via make
CONTEXT7_API_KEY=mysecretkey make -C tools context7-setup

# Or run interactively (script will prompt if env not set):
make -C tools context7-setup
```

What it does

- Ensures `tools/bootstrap_context7_key.sh` is executable.
- Runs the helper which merges/updates `CONTEXT7_API_KEY` into `tools/.mcp_env`.
- The helper writes the file atomically and sets file permissions to `600`.

Loading the env into your shell

```bash
set -a; source tools/.mcp_env; set +a
```

Devcontainer

Add to `.devcontainer/devcontainer.json`:

```json
"containerEnv": {
  "CONTEXT7_API_KEY": "${localEnv:CONTEXT7_API_KEY}"
}
```

CI

Store `CONTEXT7_API_KEY` as a repository secret and expose it in workflow `env`.
