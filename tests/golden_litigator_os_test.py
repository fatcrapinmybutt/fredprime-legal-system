from pathlib import Path
import sqlite3
import pytest

from golden_litigator_os import (
    Document,
    build_filing_package,
    generate_certificate_of_service,
    heuristic_analysis,
    init_db,
    index_to_db,
    rename_done,
    zip_filing_package,
)


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


@pytest.mark.skipif(Document is None, reason="python-docx not installed")
def test_certificate_and_zip(tmp_path: Path) -> None:
    out_dir = tmp_path / "package"
    generate_certificate_of_service(["John Doe"], str(out_dir))
    assert (out_dir / "Certificate_of_Service.docx").exists()
    zip_path = zip_filing_package(str(out_dir))
    assert zip_path.exists()
    db_path = str(tmp_path / "ledger.db")
    init_db(db_path)
    built = build_filing_package(["John Doe"], str(out_dir / "full"), db_path)
    assert built.exists()
