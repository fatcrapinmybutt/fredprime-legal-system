from court_engine import requirements_met


def test_requirements_met_false() -> None:
    assert not requirements_met(
        "Motion to Set Aside / Stay Enforcement", {"materials": []}, {}
    )
