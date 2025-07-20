from modules.legal_index_mapper import legal_index_mapper


def test_legal_index_mapper() -> None:
    mods = [
        "Canon Enforcer",
        "Custody Engine",
        "Motion Generator",
        "Timeline Module",
    ]
    mapping = legal_index_mapper(mods)
    assert mapping["Canon Enforcer"].startswith("Canon")
    assert mapping["Custody Engine"].startswith("MCL")
    assert mapping["Motion Generator"].startswith("MCR")
    assert mapping["Timeline Module"].startswith("Benchbook")
