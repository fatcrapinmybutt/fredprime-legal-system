from core.local_llm import analyze_content
from modules import codex_guardian
import pytest


def test_analyze_content_counts_tokens() -> None:
    text = "Hello world from LLM"
    result = analyze_content(text)
    assert result["word_count"] == 4
    assert result["tokens"] == ["hello", "world", "from", "llm"]


def test_verify_commit_message_uses_analyze_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_analyze(text: str) -> dict[str, object]:
        captured["text"] = text
        return {"tokens": ["valid"], "word_count": 1}

    monkeypatch.setattr(codex_guardian, "analyze_content", fake_analyze)
    codex_guardian.verify_commit_message("[core] valid")
    assert captured["text"] == "[core] valid"
