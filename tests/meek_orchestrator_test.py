from pathlib import Path

from scripts import meek_orchestrator as mo


def test_sha256_file(tmp_path: Path) -> None:
    test_file = tmp_path / "sample.txt"
    test_file.write_text("hello", encoding="utf-8")
    digest = mo.sha256_file(test_file)
    assert len(digest) == 64


def test_evidence_score_basic() -> None:
    score = mo.evidence_score("sample text", False, False)

    assert isinstance(score, int)
