from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.network_policy import NetworkPolicy


class NetworkBlockedError(RuntimeError):
    pass


@dataclass(frozen=True)
class NetworkEvent:
    timestamp: float
    target: str
    port: int
    module: str
    purpose: str
    allowed: bool


class NetworkBroker:
    def __init__(self, policy: NetworkPolicy) -> None:
        self.policy = policy

    def authorize(self, target: str, port: int, module: str, purpose: str) -> None:
        if self.policy.kill_switch_active():
            self._log_event(target, port, module, purpose, allowed=False)
            raise NetworkBlockedError("Network kill switch is active.")
        allowed = self.policy.allow_request(target, port, module)
        self._log_event(target, port, module, purpose, allowed=allowed)
        if not allowed:
            raise NetworkBlockedError(
                f"Network call blocked: target={target} port={port} module={module}"
            )

    def _log_event(self, target: str, port: int, module: str, purpose: str, allowed: bool) -> None:
        logging_cfg = self.policy.logging
        if not logging_cfg.enabled or not logging_cfg.path:
            return
        payload = NetworkEvent(
            timestamp=time.time(),
            target=target,
            port=port,
            module=module,
            purpose=purpose,
            allowed=allowed,
        )
        line = json.dumps(payload.__dict__, ensure_ascii=False)
        log_path = logging_cfg.path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
