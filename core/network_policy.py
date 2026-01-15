from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


class NetworkPolicyError(RuntimeError):
    pass


@dataclass(frozen=True)
class AllowListEntry:
    entry_id: str
    target: str
    ports: Iterable[int]
    purpose: str
    module: str
    enabled: bool


class NetworkPolicy:
    def __init__(
        self,
        path: Path,
        online_update_mode: bool,
        default: str,
        allowlist: Dict[str, AllowListEntry],
    ) -> None:
        self.path = path
        self.online_update_mode = online_update_mode
        self.default = default
        self.allowlist = allowlist

    @classmethod
    def from_path(cls, path: Path) -> "NetworkPolicy":
        if not path.exists():
            raise NetworkPolicyError(f"Network policy not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise NetworkPolicyError(f"Invalid JSON in network policy: {exc}") from exc
        if not isinstance(payload, dict):
            raise NetworkPolicyError("Network policy must be a JSON object.")
        online_update_mode = bool(payload.get("OnlineUpdateMode", False))
        default = str(payload.get("default", "deny")).lower()
        if default not in {"deny", "allow"}:
            raise NetworkPolicyError("Network policy 'default' must be 'deny' or 'allow'.")
        allowlist_entries = {}
        for item in payload.get("allowlist", []):
            if not isinstance(item, dict):
                raise NetworkPolicyError("Allowlist entries must be JSON objects.")
            entry_id = str(item.get("id", "")).strip()
            target = str(item.get("target", "")).strip()
            ports = item.get("ports") or []
            purpose = str(item.get("purpose", "")).strip()
            module = str(item.get("module", "")).strip()
            enabled = bool(item.get("enabled", False))
            if not entry_id or not target or not ports or not purpose or not module:
                raise NetworkPolicyError(f"Invalid allowlist entry: {item}")
            allowlist_entries[entry_id] = AllowListEntry(
                entry_id=entry_id,
                target=target,
                ports=ports,
                purpose=purpose,
                module=module,
                enabled=enabled,
            )
        return cls(path, online_update_mode, default, allowlist_entries)

    def is_offline(self) -> bool:
        return not self.online_update_mode and self.default == "deny"

    def allow_request(self, target: str, port: int, module: str) -> bool:
        for entry in self.allowlist.values():
            if not entry.enabled:
                continue
            if entry.target == target and port in entry.ports and entry.module == module:
                return True
        return self.default == "allow"
