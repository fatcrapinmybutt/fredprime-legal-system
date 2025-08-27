import sys

import LAWFORGE_SENTINEL_GUI as sentinel


def test_build_args_defaults(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog"])
    args = sentinel.build_args()
    assert hasattr(args, "out")
