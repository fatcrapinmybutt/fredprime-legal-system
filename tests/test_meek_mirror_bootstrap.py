import meek_mirror_bootstrap as mm


def test_is_windows_returns_bool() -> None:
    assert isinstance(mm.is_windows(), bool)
