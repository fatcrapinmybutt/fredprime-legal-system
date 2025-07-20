import os
import re
from typing import List


def legacy_chat_token_scanner(chat_log_dir: str) -> List[str]:
    """Scan chat logs for tokenized references to modules or directives."""
    tokens = []
    for file in os.listdir(chat_log_dir):
        path = os.path.join(chat_log_dir, file)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
            terms = [
                "module",
                "system",
                "script",
                "trigger",
                "engine",
                "directive",
                "upload",
                r"\.py",
            ]
            pattern = "(" + "|".join(terms) + ")"
            found = re.findall(pattern, text, re.IGNORECASE)
            tokens.extend(found)
    return list({t.lower() for t in tokens})
