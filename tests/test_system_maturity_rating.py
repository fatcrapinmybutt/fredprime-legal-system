from modules.system_maturity_rating import system_maturity_rating


def test_system_maturity_rating() -> None:
    notes = {
        "mod1": "Generated .py with GUI integration",
        "mod2": "Drafted logic only",
        "mod3": "sketch idea",
    }
    rating = system_maturity_rating(notes)
    assert rating["mod1"].startswith("\U0001f7e2")
    assert rating["mod2"].startswith("\U0001f7e1")
    assert rating["mod3"].startswith("\U0001f534")
