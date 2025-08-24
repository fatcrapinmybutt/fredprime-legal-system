import sqlite3
from pathlib import Path

import case_meta_seed
import pytest
from utils import init_db


def test_upsert_writes_case_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_file = tmp_path / "case.db"
    init_db(str(db_file))
    monkeypatch.setattr(case_meta_seed, "db_conn", lambda _p: sqlite3.connect(db_file))
    meta = {
        "court_name": "Court",
        "case_number": "1",
        "caption_plaintiff": "P",
        "caption_defendant": "D",
        "jurisdiction": "J",
    }
    case_meta_seed.upsert(meta)
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("SELECT court_name FROM case_meta")
    assert cur.fetchone()[0] == "Court"
    conn.close()
