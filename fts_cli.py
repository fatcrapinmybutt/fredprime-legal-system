import argparse
import sqlite3
from pathlib import Path

DB_PATH = Path("data.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS records (id INTEGER PRIMARY KEY, description TEXT)")
    cur.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(description, content='records', content_rowid='id')"
    )
    cur.execute(
        "INSERT INTO records_fts(rowid, description) SELECT id, description FROM records WHERE id NOT IN (SELECT rowid FROM records_fts)"
    )
    conn.commit()
    conn.close()
    print("Database initialized.")


def add_record(description: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO records(description) VALUES (?)", (description,))
    rowid = cur.lastrowid
    cur.execute(
        "INSERT INTO records_fts(rowid, description) VALUES (?, ?)",
        (rowid, description),
    )
    conn.commit()
    conn.close()
    print(f"Record {rowid} added.")


def search_records(query: str):
    conn = get_connection()
    cur = conn.cursor()
    for rowid, desc in cur.execute("SELECT rowid, description FROM records_fts WHERE records_fts MATCH ?", (query,)):
        print(f"{rowid}: {desc}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Simple FTS CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Initialize the database")
    add_p = subparsers.add_parser("add", help="Add a new record")
    add_p.add_argument("description", help="Description text")
    search_p = subparsers.add_parser("search", help="Full-text search")
    search_p.add_argument("query", help="Search query")

    args = parser.parse_args()

    if args.command == "init":
        init_db()
    elif args.command == "add":
        add_record(args.description)
    elif args.command == "search":
        search_records(args.query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
