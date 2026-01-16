from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


class AssetsRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    kind: str
    local_paths: List[Path]
    expected_bytes: Optional[int] = None
    sha256: Optional[str] = None
    source_url: Optional[str] = None
    notes: Optional[str] = None


class AssetsRegistry:
    def __init__(self, path: Path, records: List[AssetRecord]) -> None:
        self.path = path
        self.records = records

    @classmethod
    def from_path(cls, path: Path) -> "AssetsRegistry":
        if not path.exists():
            raise AssetsRegistryError(f"Assets registry not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise AssetsRegistryError(f"Invalid JSON in assets registry: {exc}") from exc
        if not isinstance(payload, dict) or "assets" not in payload:
            raise AssetsRegistryError("Assets registry must be an object with an 'assets' list.")
        records: List[AssetRecord] = []
        for item in payload["assets"]:
            if not isinstance(item, dict):
                raise AssetsRegistryError("Each asset entry must be an object.")
            asset_id = str(item.get("asset_id", "")).strip()
            kind = str(item.get("kind", "")).strip()
            paths = item.get("local_paths") or []
            if not asset_id or not kind or not paths:
                raise AssetsRegistryError(f"Invalid asset entry: {item}")
            local_paths = [Path(p) for p in paths]
            records.append(
                AssetRecord(
                    asset_id=asset_id,
                    kind=kind,
                    local_paths=local_paths,
                    expected_bytes=item.get("expected_bytes"),
                    sha256=item.get("sha256"),
                    source_url=item.get("source_url"),
                    notes=item.get("notes"),
                )
            )
        return cls(path, records)

    def validate(self) -> Dict[str, List[str]]:
        report = {"missing": [], "hash_mismatch": [], "size_mismatch": []}
        for record in self.records:
            found = False
            for path in record.local_paths:
                if path.exists():
                    found = True
                    if record.expected_bytes is not None and path.stat().st_size != record.expected_bytes:
                        report["size_mismatch"].append(f"{record.asset_id}:{path}")
                    if record.sha256:
                        if self._sha256(path) != record.sha256:
                            report["hash_mismatch"].append(f"{record.asset_id}:{path}")
                    break
            if not found:
                report["missing"].append(record.asset_id)
        return report

    @staticmethod
    def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()
