from pathlib import Path
import sqlite3

from golden_litigator_os import heuristic_analysis, init_db, index_to_db, rename_done


def test_heuristic_analysis_extracts_parties_and_quotes() -> None:
    text = 'John Doe said "hello" to Jane Smith.'
    data = heuristic_analysis(text)
    assert "John Doe" in data["parties"]
    assert "Jane Smith" in data["parties"]
    assert "hello" in data["quotes"]


def test_index_and_rename(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.db"
    init_db(str(db_path))
    file_path = tmp_path / "note.txt"
    file_path.write_text("Sample content")
    analysis = {
        "parties": "",
        "claims": "",
        "statutes": "",
        "quotes": "",
        "tags": "",
        "court_relevance": "",
        "exhibit_id": "",
    }
    index_to_db(
        str(file_path), file_path.name, "Sample content", analysis, str(db_path)
    )
    rename_done(str(file_path))
    assert (tmp_path / "note__DONE__.txt").exists()
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT filename FROM evidence")
    row = cur.fetchone()
    conn.close()
    assert row[0] == "note.txt"
