import json
import re
from pathlib import Path
from typing import List
from urllib.parse import urljoin

import requests  # type: ignore
from bs4 import BeautifulSoup

BASE_URL = "https://courts.michigan.gov/Administration/SCAO/Forms/Pages/default.aspx"


def fetch_forms() -> List[dict]:
    """Attempt to scrape form metadata from the Michigan courts website."""
    try:
        resp = requests.get(BASE_URL, timeout=15)
        resp.raise_for_status()
    except Exception as exc:  # pragma: no cover - network may fail
        print(f"Unable to retrieve forms: {exc}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    forms = []
    for link in soup.find_all("a", href=True):
        href = str(link["href"])
        if not href.lower().endswith(".pdf"):
            continue
        text = link.get_text(strip=True)
        m = re.search(r"([A-Z]+[- ]?\d+)", text) or re.search(r"([A-Z]+[- ]?\d+)", href)
        if not m:
            continue
        form_id = m.group(1).replace(" ", "-")
        forms.append({"id": form_id, "title": text, "url": urljoin(BASE_URL, href)})
    return forms


def save_forms(forms: List[dict], output: Path) -> None:
    output.write_text(json.dumps(forms, indent=2))
    print(f"Saved {len(forms)} forms to {output}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch SCAO form metadata")
    parser.add_argument("--out", default="data/scao_forms.json")
    args = parser.parse_args()

    forms = fetch_forms()
    if forms:
        save_forms(forms, Path(args.out))


if __name__ == "__main__":
    main()
