from __future__ import annotations
from pathlib import Path


def main() -> None:
    manifest = Path("codex_manifest.json")
    if not manifest.exists():
        raise SystemExit("codex_manifest.json missing")
    data = manifest.read_text(encoding="utf-8")
    if not data.strip():
        raise SystemExit("codex_manifest.json empty")
    print("codex manifest present")


if __name__ == "__main__":
    main()
