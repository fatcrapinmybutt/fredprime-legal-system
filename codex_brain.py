import hashlib
import json
from pathlib import Path


def legal_function_from_name(path: Path) -> str:
    name = path.name.lower()
    if "motion" in name:
        return "motion (MCR 2.119)"
    if "affidavit" in name:
        return "affidavit (MCR 2.119(B))"
    if "order" in name:
        return "court order"
    return "module"

MANIFEST = 'codex_manifest.json'


def hash_file(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def update_manifest() -> None:
    manifest = {}
    for p in Path('.').rglob('*.py'):
        if p.parts[0].startswith('.'):  # skip hidden dirs
            continue
        manifest[str(p)] = {
            'sha256': hash_file(p),
            'legal_function': legal_function_from_name(p),
            'dependencies': [],
        }
    Path(MANIFEST).write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    update_manifest()
    print('codex manifest updated')
