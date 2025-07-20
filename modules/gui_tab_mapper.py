from typing import Dict, Iterable


def gui_tab_mapper(modules: Iterable[str]) -> Dict[str, str]:
    """Map module names to GUI tab identifiers."""
    gui_tabs = {}
    for m in modules:
        name = m.lower()
        if "warboard" in name:
            gui_tabs[m] = "\U0001f5fa WARBOARD_VISUALIZER"
        elif "timeline" in name:
            gui_tabs[m] = "\U0001f4c5 Timeline Tracker"
        elif "motion" in name:
            gui_tabs[m] = "\U0001f9fe Motion Forge"
        elif "code" in name:
            gui_tabs[m] = "\U0001f9e0 CODE_KEEPER_LOOKUP"
    return gui_tabs
