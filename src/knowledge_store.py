"""Simple knowledge store linking evidence to forms and rules."""

import json
import sqlite3
from pathlib import Path
from typing import List, Optional


class KnowledgeStore:
    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT,
                description TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS links (
                evidence_id INTEGER,
                form_id TEXT,
                note TEXT,
                FOREIGN KEY(evidence_id) REFERENCES evidence(id)
            )
            """
        )
        self.conn.commit()

    def add_evidence(self, path: Path, description: str = "") -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO evidence (path, description) VALUES (?, ?)",
            (str(path), description),
        )
        self.conn.commit()
        return cur.lastrowid

    def link_form(self, evidence_id: int, form_id: str, note: str = "") -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO links (evidence_id, form_id, note) VALUES (?, ?, ?)",
            (evidence_id, form_id, note),
        )
        self.conn.commit()

    def get_links(self) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT evidence.path, evidence.description, links.form_id, links.note
            FROM links
            JOIN evidence ON links.evidence_id = evidence.id
            """
        )
        rows = cur.fetchall()
        return [
            {
                "path": row[0],
                "description": row[1],
                "form_id": row[2],
                "note": row[3],
            }
            for row in rows
        ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Manage litigation knowledge store")
    parser.add_argument("--db", default="knowledge.db")
    parser.add_argument("--add-evidence", help="File path to add as evidence")
    parser.add_argument("--desc", help="Optional description for evidence")
    parser.add_argument("--link", help="Link evidence ID to form ID (format: id:FORM)")
    parser.add_argument("--list", action="store_true", help="List evidence links")
    args = parser.parse_args()

    ks = KnowledgeStore(Path(args.db))

    if args.add_evidence:
        eid = ks.add_evidence(Path(args.add_evidence), args.desc or "")
        print(f"Added evidence {eid}")
    elif args.link:
        evid, form = args.link.split(":", 1)
        ks.link_form(int(evid), form)
        print("Link stored")
    elif args.list:
        for item in ks.get_links():
            print(json.dumps(item, indent=2))
    else:
        parser.print_help()
