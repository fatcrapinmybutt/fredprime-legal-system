from utils import excerpt


def test_excerpt_truncates() -> None:
    assert excerpt("a" * 2000, 10) == "a" * 10
