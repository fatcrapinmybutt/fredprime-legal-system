"""Analyze evidence descriptions and suggest motions."""

import argparse
from pathlib import Path
from typing import List

from form_db import FormDatabase
from knowledge_store import KnowledgeStore


KEYWORDS_TO_FORMS = {
    "custody": "FOC-65",
    "parenting": "FOC-87",
    "adjourn": "MC-12",
    "injunctive": "MC-97",
}


def suggest_forms(descriptions: List[str]) -> List[str]:
    found = set()
    for desc in descriptions:
        lower = desc.lower()
        for word, form_id in KEYWORDS_TO_FORMS.items():
            if word in lower:
                found.add(form_id)
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description="Suggest motions based on evidence")
    parser.add_argument("--db", default="forms.db", help="Form database path")
    parser.add_argument("--knowledge", default="knowledge.db", help="Knowledge store db")
    args = parser.parse_args()

    db = FormDatabase(Path(args.db))
    ks = KnowledgeStore(Path(args.knowledge))
    descriptions = [item["description"] for item in ks.get_links()]
    forms = suggest_forms(descriptions)
    for fid in forms:
        record = db.get_form(fid)
        if record:
            print(f"Suggested form {fid}: {record['title']}")
            print(f"  Rules: {', '.join(record['rules'])}")
            if record.get('statutes'):
                print(f"  Statutes: {', '.join(record['statutes'])}")
            if record.get('benchbook'):
                print(f"  Benchbook: {', '.join(record['benchbook'])}")
            print()


if __name__ == "__main__":
    main()
