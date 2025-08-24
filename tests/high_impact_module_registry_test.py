from modules.high_impact_module_registry import build_modules


def test_build_modules_level_and_count() -> None:
    modules = build_modules()
    assert len(modules) == 150
    assert all(module.level == 9999 for module in modules)
