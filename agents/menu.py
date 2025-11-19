"""Expose simple menu items for GUI integration.

`get_menu_items()` returns a list of dicts suitable for building a menu or
graph preview that lists available agents and their short descriptions.
"""
from typing import List, Dict

from . import loader


def get_menu_items(root: str = None) -> List[Dict]:
    agents = loader.discover_agents(root=root)
    items = []
    for aid, info in agents.items():
        cfg = info.get("config") or {}
        items.append(
            {
                "id": aid,
                "name": cfg.get("name") or aid,
                "description": cfg.get("description") or "",
                "topics": cfg.get("topics") or "",
                "path": info.get("path"),
            }
        )
    return items
