from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path


def build() -> None:
    subprocess.run(["pyinstaller", "drive_search_gui.spec"], check=True)
    dist = Path("dist") / "LAWFORGE_DRIVE_SEARCH"
    zip_path = Path("LAWFORGE_DRIVE_SEARCH.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for file in dist.rglob("*"):
            zf.write(file, file.relative_to(dist))


if __name__ == "__main__":
    build()
