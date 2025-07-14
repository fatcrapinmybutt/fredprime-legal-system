import argparse
import json
import sqlite3
from pathlib import Path
from typing import List, Optional


class FormDatabase:
    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS forms (
                id TEXT PRIMARY KEY,
                title TEXT,
                filename TEXT,
                rules TEXT,
                statutes TEXT,
                benchbook TEXT,
                constitution TEXT,
                federal TEXT
            )
            """
        )
        self.conn.commit()

    def add_form(
        self,
        form_id: str,
        title: str,
        filename: str,
        rules: Optional[List[str]] = None,
        statutes: Optional[List[str]] = None,
        benchbook: Optional[List[str]] = None,
        constitution: Optional[List[str]] = None,
        federal: Optional[List[str]] = None,
    ) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO forms (
                id, title, filename, rules, statutes,
                benchbook, constitution, federal
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                form_id,
                title,
                filename,
                json.dumps(rules or []),
                json.dumps(statutes or []),
                json.dumps(benchbook or []),
                json.dumps(constitution or []),
                json.dumps(federal or []),
            ),
        )
        self.conn.commit()

    def get_form(self, form_id: str) -> Optional[dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM forms WHERE id = ?", (form_id,))
        row = cur.fetchone()
        if row:
            keys = [column[0] for column in cur.description]
            record = dict(zip(keys, row))
            for k in [
                "rules",
                "statutes",
                "benchbook",
                "constitution",
                "federal",
            ]:
                record[k] = json.loads(record.get(k, "[]"))
            return record
        return None

    def list_forms(self) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM forms ORDER BY id")
        rows = cur.fetchall()
        forms = []
        for row in rows:
            keys = [column[0] for column in cur.description]
            record = dict(zip(keys, row))
            for k in [
                "rules",
                "statutes",
                "benchbook",
                "constitution",
                "federal",
            ]:
                record[k] = json.loads(record.get(k, "[]"))
            forms.append(record)
        return forms

    def search_forms(self, keyword: str) -> List[dict]:
        cur = self.conn.cursor()
        like = f"%{keyword.lower()}%"
        cur.execute(
            (
                "SELECT * FROM forms WHERE LOWER(id) LIKE ? "
                "OR LOWER(title) LIKE ? ORDER BY id"
            ),
            (like, like),
        )
        rows = cur.fetchall()
        results = []
        for row in rows:
            keys = [column[0] for column in cur.description]
            record = dict(zip(keys, row))
            for k in [
                "rules",
                "statutes",
                "benchbook",
                "constitution",
                "federal",
            ]:
                record[k] = json.loads(record.get(k, "[]"))
            results.append(record)
        return results


def load_manifest(
    manifest_path: Path, db: FormDatabase, forms_dir: Path
) -> None:
    data = json.loads(manifest_path.read_text())
    for entry in data:
        file_path = forms_dir / entry.get("filename", "")
        db.add_form(
            form_id=entry.get("id"),
            title=entry.get("title"),
            filename=str(file_path),
            rules=entry.get("rules"),
            statutes=entry.get("statutes"),
            benchbook=entry.get("benchbook"),
            constitution=entry.get("constitution"),
            federal=entry.get("federal"),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import court forms into a database"
    )
    parser.add_argument(
        "--db",
        default="forms.db",
        help="SQLite database path",
    )
    parser.add_argument(
        "--manifest",
        default="data/forms_manifest.json",
        help="JSON manifest describing forms",
    )
    parser.add_argument(
        "--forms-dir",
        default="forms",
        help="Directory containing form files",
    )
    parser.add_argument("--get", help="Lookup a form by ID")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all stored forms",
    )
    parser.add_argument("--search", help="Find forms matching a keyword")
    args = parser.parse_args()

    db_path = Path(args.db)
    manifest_path = Path(args.manifest)
    forms_dir = Path(args.forms_dir)

    db = FormDatabase(db_path)
    if args.get:
        form = db.get_form(args.get)
        if form:
            print(json.dumps(form, indent=2))
        else:
            print(f"Form {args.get} not found")
        return
    if args.list:
        for form in db.list_forms():
            print(f"{form['id']}: {form['title']}")
        return
    if args.search:
        results = db.search_forms(args.search)
        for form in results:
            print(f"{form['id']}: {form['title']}")
        return

    load_manifest(manifest_path, db, forms_dir)
    print(f"Imported forms from {manifest_path} into {db_path}")


if __name__ == "__main__":
    main()
