from pathlib import Path

from modules import codex_supreme


def test_save_and_load_state(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    codex_supreme.save_state({"x": 1}, state_file=str(state_file))
    state = codex_supreme.load_state(state_file=str(state_file))
    assert state == {"x": 1}


def test_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "file.txt"
    p.write_text("data")
    h = codex_supreme.sha256_file(str(p))
    assert len(h) == 64
