from modules.gui_tab_mapper import gui_tab_mapper


def test_gui_tab_mapper() -> None:
    modules = [
        "Warboard Engine",
        "Timeline Fusion",
        "Motion Generator",
        "Code Keeper",
    ]
    tabs = gui_tab_mapper(modules)
    assert tabs["Warboard Engine"].startswith("\U0001f5fa")
    assert tabs["Timeline Fusion"].startswith("\U0001f4c5")
    assert tabs["Motion Generator"].startswith("\U0001f9fe")
    assert tabs["Code Keeper"].startswith("\U0001f9e0")
