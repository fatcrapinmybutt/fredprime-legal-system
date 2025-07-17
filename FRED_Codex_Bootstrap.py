import hashlib
import os
import requests
import zipfile
from pathlib import Path


URL = os.environ.get("FRED_STAGE2_URL", "https://example.com/FRED_STAGE2_FULL_DEPLOY.zip")
EXPECTED_SHA256 = os.environ.get(
    "FRED_STAGE2_SHA256",
    "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
)
DOWNLOAD_NAME = "FRED_STAGE2_FULL_DEPLOY.zip"


def sha256sum(path: Path) -> str:
    """Return the SHA256 hex digest for a given file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, destination: Path) -> None:
    """Stream download a file from ``url`` to ``destination``."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with destination.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def main() -> None:
    dest = Path(DOWNLOAD_NAME)
    print(f"Downloading {URL} -> {dest}")
    download_file(URL, dest)
    print("Calculating SHA256 checksum...")
    checksum = sha256sum(dest)
    print(f"Checksum: {checksum}")
    if checksum != EXPECTED_SHA256:
        raise RuntimeError(
            f"Checksum mismatch! expected {EXPECTED_SHA256} but got {checksum}"
        )
    print("Checksum verified. Extracting archive...")
    with zipfile.ZipFile(dest, "r") as zf:
        zf.extractall()
    print("Extraction complete.")


if __name__ == "__main__":
    main()
