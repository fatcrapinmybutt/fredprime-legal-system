from modules.module_collider import module_collider


def test_module_collider() -> None:
    mods = ["Timeline Engine", "timeline engine", "Motion Module"]
    result = module_collider(mods)
    key = "timeline"
    assert key in result
    assert len(result[key]) == 2
