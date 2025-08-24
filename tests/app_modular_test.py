import app_modular


def test_app_modular_has_main() -> None:
    assert callable(app_modular.main)
