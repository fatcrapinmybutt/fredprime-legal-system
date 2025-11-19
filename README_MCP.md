# MCP / CONTEXT7 key setup

This document explains how the MCP entry in `mcp.json` expects the `CONTEXT7_API_KEY` to be provided, and documents the provided `tools/context7-setup` Makefile helper.

## How the MCP reads the key

- The MCP configuration prefers the environment variable `CONTEXT7_API_KEY`.
- It will not prompt interactively; the MCP reads `${env:CONTEXT7_API_KEY}`.

## Local development options

- Create or update `tools/.mcp_env` and add `CONTEXT7_API_KEY=...`, then load it into your shell:

```bash
set -a; source tools/.mcp_env; set +a
```

- Or use the helper Makefile target (from repo root):

```bash
# make the helper executable and run it
chmod +x tools/bootstrap_context7_key.sh
CONTEXT7_API_KEY=mysecretkey make -C tools context7-setup

# or interactively (script will prompt if env not set):
make -C tools context7-setup
```

This target writes/merges the key into `tools/.mcp_env` with secure file permissions (mode `600`).

## Devcontainer

- To provide the key to the devcontainer, add it to `.devcontainer/devcontainer.json` under `containerEnv`:

```json
"containerEnv": {
  "CONTEXT7_API_KEY": "${localEnv:CONTEXT7_API_KEY}"
}
```

## CI

- In CI (GitHub Actions), store the key as a repository secret named `CONTEXT7_API_KEY` and expose it in workflow `env`.

## Notes

- The repository now contains a `tools/bootstrap_context7_key.sh` helper and a Makefile target `tools/context7-setup` to make local setup easier.
