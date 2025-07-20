"""Standalone self-test execution for pre-commit."""

from test_codex_selftest import test_self  # type: ignore[import-not-found]


def main() -> None:
    test_self()
    print("Selftest passed.")


if __name__ == "__main__":
    main()
