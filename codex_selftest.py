"""Run Codex guardian checks."""

from importlib import import_module


def main() -> None:
    """Execute guardian verifications."""
    run_guardian = import_module("modules.codex_guardian").run_guardian
    run_guardian()
    print("codex selftest passed")


if __name__ == "__main__":
    main()
