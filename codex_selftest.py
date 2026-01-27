"""Run Codex guardian checks."""

from importlib import import_module


def main() -> None:
    """Execute guardian verifications."""
    guardian = import_module("modules.codex_guardian")
    strict = guardian.parse_env_flag("CODEX_GUARDIAN_STRICT", default=False)
    guardian.run_guardian(strict=strict)
    print("codex selftest passed")


if __name__ == "__main__":
    main()
