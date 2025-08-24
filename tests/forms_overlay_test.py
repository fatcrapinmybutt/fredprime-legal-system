import sqlite3
from pathlib import Path

import forms_overlay
import pytest
from utils import init_db


def test_forms_created(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_file = tmp_path / "case.db"
    init_db(str(db_file))
    conn = sqlite3.connect(db_file)
    conn.execute(
        (
            "INSERT INTO case_meta (court_name, case_number, caption_plaintiff, "
            "caption_defendant, judge, jurisdiction, division) VALUES (?,?,?,?,?,?,?)"
        ),
        ("Court", "1", "P", "D", "", "J", ""),
    )
    conn.commit()
    conn.close()
    monkeypatch.setattr(forms_overlay, "db_conn", lambda _p: sqlite3.connect(db_file))
    forms_dir = tmp_path / "Forms"
    monkeypatch.setattr(forms_overlay, "FORMS_DIR", forms_dir)
    forms_dir.mkdir(parents=True, exist_ok=True)
    forms_overlay.mc20_fee_waiver()
    forms_overlay.mc12_proof_of_service("Doc")
    assert (forms_dir / "MC20_FeeWaiver_Draft.docx").exists()
    assert (forms_dir / "MC12_ProofOfService_Draft.docx").exists()
