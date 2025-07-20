from pathlib import Path

from modules.legacy_chat_token_scanner import legacy_chat_token_scanner


def test_legacy_chat_token_scanner(tmp_path: Path) -> None:
    sample = tmp_path / "chat.txt"
    sample.write_text("Module engine.py and directive")
    tokens = legacy_chat_token_scanner(str(tmp_path))
    assert "module" in tokens
    assert "engine" in tokens
