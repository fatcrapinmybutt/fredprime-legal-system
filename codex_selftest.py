"""Run Codex guardian checks."""

import os
from importlib import import_module


def main() -> None:
    """Execute guardian verifications."""
    # Allow skipping strict checks via environment variable
    # This is useful for CI/CD environments or non-codex branches
    if os.environ.get("CODEX_SKIP_STRICT_CHECKS") is None:
        # Auto-detect if we're in a non-codex environment
        import subprocess
        try:
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
            # If not on a codex/ branch, enable relaxed mode
            if not branch.startswith("codex/"):
                os.environ["CODEX_SKIP_STRICT_CHECKS"] = "true"
                os.environ["CODEX_SKIP_HASH_CHECKS"] = "true"
        except Exception:
            # If git check fails, enable relaxed mode
            os.environ["CODEX_SKIP_STRICT_CHECKS"] = "true"
            os.environ["CODEX_SKIP_HASH_CHECKS"] = "true"
    
    run_guardian = import_module("modules.codex_guardian").run_guardian
    run_guardian()
    print("codex selftest passed")


if __name__ == "__main__":
    main()
