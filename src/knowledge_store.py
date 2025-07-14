"""Persistent store linking evidence, transcripts, and forms."""

import json
import sqlite3
from pathlib import Path
from typing import List

from constants import KEYWORDS_TO_FORMS


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
            CREATE TABLE IF NOT EXISTS transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT,
                description TEXT,
                date TEXT
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
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS transcript_links (
                transcript_id INTEGER,
                form_id TEXT,
                note TEXT,
                FOREIGN KEY(transcript_id) REFERENCES transcripts(id)
            )
            """
        )
        self.conn.commit()

    def add_evidence(self, path: Path, description: str = "", auto_link: bool = False) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO evidence (path, description) VALUES (?, ?)",
            (str(path), description),
        )
        self.conn.commit()
        evid_id = cur.lastrowid
        if auto_link and description:
            lower = description.lower()
            for word, form_id in KEYWORDS_TO_FORMS.items():
                if word in lower:
                    self.link_form(evid_id, form_id)
        return evid_id

    def add_transcript(self, path: Path, description: str = "", date: str = "") -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO transcripts (path, description, date) VALUES (?, ?, ?)",
            (str(path), description, date),
        )
        self.conn.commit()
        return cur.lastrowid

    def link_transcript(self, transcript_id: int, form_id: str, note: str = "") -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO transcript_links (transcript_id, form_id, note) VALUES (?, ?, ?)",
            (transcript_id, form_id, note),
        )
        self.conn.commit()

    def remove_evidence(self, evid_id: int) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM links WHERE evidence_id=?", (evid_id,))
        cur.execute("DELETE FROM evidence WHERE id=?", (evid_id,))
        self.conn.commit()

    def remove_transcript(self, trans_id: int) -> None:
        cur = self.conn.cursor()
        cur.execute("DELETE FROM transcript_links WHERE transcript_id=?", (trans_id,))
        cur.execute("DELETE FROM transcripts WHERE id=?", (trans_id,))
        self.conn.commit()

    def search_evidence(self, keyword: str) -> List[dict]:
        cur = self.conn.cursor()
        like = f"%{keyword.lower()}%"
        cur.execute(
            "SELECT id, path, description FROM evidence WHERE LOWER(description) LIKE ?",
            (like,),
        )
        rows = cur.fetchall()
        return [
            {"id": row[0], "path": row[1], "description": row[2]} for row in rows
        ]

    def link_form(
        self, evidence_id: int, form_id: str, note: str = ""
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO links (evidence_id, form_id, note) VALUES (?, ?, ?)",
            (evidence_id, form_id, note),
        )
        self.conn.commit()

    def get_links(self) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute(
            (
                "SELECT evidence.path, evidence.description, "
                "links.form_id, links.note FROM links "
                "JOIN evidence ON links.evidence_id = evidence.id"
            )
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

    def get_transcript_links(self) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute(
            (
                "SELECT transcripts.path, transcripts.description, "
                "transcript_links.form_id, transcript_links.note FROM transcript_links "
                "JOIN transcripts ON transcript_links.transcript_id = transcripts.id"
            )
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

    parser = argparse.ArgumentParser(
        description="Manage litigation knowledge store"
    )
    parser.add_argument("--db", default="knowledge.db")
    parser.add_argument(
        "--add-evidence",
        help="File path to add as evidence",
    )
    parser.add_argument(
        "--desc",
        help="Optional description for evidence",
    )
    parser.add_argument(
        "--auto-link",
        action="store_true",
        help="Automatically link evidence to forms based on keywords",
    )
    parser.add_argument(
        "--link",
        help="Link evidence ID to form ID (format: id:FORM)",
    )
    parser.add_argument(
        "--add-transcript",
        help="File path to add as transcript",
    )
    parser.add_argument(
        "--date",
        help="Date of transcript (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--link-transcript",
        help="Link transcript ID to form ID (format: id:FORM)",
    )
    parser.add_argument(
        "--remove",
        type=int,
        help="Remove evidence by ID",
    )
    parser.add_argument(
        "--remove-transcript",
        type=int,
        help="Remove transcript by ID",
    )
    parser.add_argument(
        "--search",
        help="Search evidence descriptions",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List evidence links",
    )
    parser.add_argument(
        "--list-transcripts",
        action="store_true",
        help="List transcript links",
    )
    args = parser.parse_args()

    ks = KnowledgeStore(Path(args.db))

    if args.add_evidence:
        eid = ks.add_evidence(Path(args.add_evidence), args.desc or "", args.auto_link)
        print(f"Added evidence {eid}")
    elif args.add_transcript:
        tid = ks.add_transcript(Path(args.add_transcript), args.desc or "", args.date or "")
        print(f"Added transcript {tid}")
    elif args.link:
        evid, form = args.link.split(":", 1)
        ks.link_form(int(evid), form)
        print("Link stored")
    elif args.link_transcript:
        tid, form = args.link_transcript.split(":", 1)
        ks.link_transcript(int(tid), form)
        print("Transcript link stored")
    elif args.remove is not None:
        ks.remove_evidence(args.remove)
        print(f"Removed evidence {args.remove}")
    elif args.remove_transcript is not None:
        ks.remove_transcript(args.remove_transcript)
        print(f"Removed transcript {args.remove_transcript}")
    elif args.search:
        results = ks.search_evidence(args.search)
        for item in results:
            print(json.dumps(item, indent=2))
    elif args.list:
        for item in ks.get_links():
            print(json.dumps(item, indent=2))
    elif args.list_transcripts:
        for item in ks.get_transcript_links():
            print(json.dumps(item, indent=2))
    else:
        parser.print_help()
