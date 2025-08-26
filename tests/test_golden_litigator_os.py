from pathlib import Path
from golden_litigator_os import sha256_file


def test_sha256_file() -> None:
    assert len(sha256_file(Path(__file__))) == 64
