"""Utility script to seed case metadata for state or federal matters."""

from __future__ import annotations
import argparse
from typing import Dict

from utils import init_db, db_conn


def upsert(meta: Dict[str, str]) -> None:
    conn = db_conn("golden_litigator.db")
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO case_meta (
            court_name, case_number, caption_plaintiff, caption_defendant, judge, jurisdiction, division
        ) VALUES (?,?,?,?,?,?,?)""",
        (
            meta["court_name"],
            meta["case_number"],
            meta["caption_plaintiff"],
            meta["caption_defendant"],
            meta.get("judge", ""),
            meta["jurisdiction"],
            meta.get("division", ""),
        ),
    )
    conn.commit()
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", action="store_true")
    parser.add_argument("--federal", action="store_true")
    args = parser.parse_args()
    init_db("golden_litigator.db")
    if args.state:
        upsert(
            {
                "court_name": "IN THE 60TH DISTRICT COURT FOR MUSKEGON COUNTY, MICHIGAN",
                "case_number": "",
                "caption_plaintiff": "Shady Oaks Park MHP LLC",
                "caption_defendant": "Andrew J. Pigors",
                "judge": "",
                "jurisdiction": "State of Michigan",
                "division": "Landlord–Tenant",
            }
        )
        print("Seeded state case_meta.")
    if args.federal:
        upsert(
            {
                "court_name": "UNITED STATES DISTRICT COURT, WESTERN DISTRICT OF MICHIGAN",
                "case_number": "",
                "caption_plaintiff": "Andrew J. Pigors",
                "caption_defendant": "Emily Watson, et al.",
                "judge": "",
                "jurisdiction": "Federal",
                "division": "Southern Division",
            }
        )
        print("Seeded WDMI case_meta.")


if __name__ == "__main__":
    main()
