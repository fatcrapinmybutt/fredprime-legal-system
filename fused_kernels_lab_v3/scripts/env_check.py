from __future__ import annotations

import platform
import sys

import torch


def main() -> int:
    print("Python:", sys.version.replace("\n", " "))
    print("Platform:", platform.platform())
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA devices:", torch.cuda.device_count())
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        print("Current device:", idx, props.name)
        print("Compute capability:", f"{props.major}.{props.minor}")
        print("Total memory (GB):", round(props.total_memory / (1024**3), 2))
        print("CUDA runtime:", torch.version.cuda)
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401

        print("Triton: import OK")
        # probe erf
        fn = getattr(tl, "erf", None)
        if fn is None:
            m = getattr(tl, "math", None)
            fn = getattr(m, "erf", None) if m is not None else None
        print("Triton erf available:", bool(fn))
    except Exception as e:
        print("Triton: import FAIL:", type(e).__name__, str(e))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
