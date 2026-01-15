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


@dataclass(frozen=True)
class NetworkLogging:
    enabled: bool
    path: Optional[Path]
    max_mb: int


@dataclass(frozen=True)
class KillSwitch:
    enabled: bool
    flag_file: Optional[Path]


class NetworkPolicy:
    def __init__(
        self,
        path: Path,
        online_update_mode: bool,
        default: str,
        allowlist: Dict[str, AllowListEntry],
        logging: NetworkLogging,
        kill_switch: KillSwitch,
    ) -> None:
        self.path = path
        self.online_update_mode = online_update_mode
        self.default = default
        self.allowlist = allowlist
        self.logging = logging
        self.kill_switch = kill_switch

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
        logging_payload = payload.get("logging") or {}
        logging_path = logging_payload.get("path")
        logging_cfg = NetworkLogging(
            enabled=bool(logging_payload.get("enabled", True)),
            path=Path(logging_path) if logging_path else None,
            max_mb=int(logging_payload.get("max_mb", 50)),
        )
        kill_payload = payload.get("killSwitch") or payload.get("kill_switch") or {}
        kill_flag = kill_payload.get("flagFile") or kill_payload.get("flag_file")
        kill_switch = KillSwitch(
            enabled=bool(kill_payload.get("enabled", True)),
            flag_file=Path(kill_flag) if kill_flag else None,
        )
        return cls(path, online_update_mode, default, allowlist_entries, logging_cfg, kill_switch)

    def is_offline(self) -> bool:
        return not self.online_update_mode and self.default == "deny"

    def kill_switch_active(self) -> bool:
        if not self.kill_switch.enabled or not self.kill_switch.flag_file:
            return False
        return self.kill_switch.flag_file.exists()

    def allow_request(self, target: str, port: int, module: str) -> bool:
        for entry in self.allowlist.values():
            if not entry.enabled:
                continue
            if entry.target == target and port in entry.ports and entry.module == module:
                return True
        return self.default == "allow"
