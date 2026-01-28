from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


GeluApprox = Literal["tanh", "none"]


@dataclass(frozen=True)
class KernelRunResult:
    y: torch.Tensor
    used_triton: bool
    reason: str


def has_cuda() -> bool:
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def triton_available() -> bool:
    return triton is not None and tl is not None


def ensure_contiguous(t: torch.Tensor) -> torch.Tensor:
    return t if t.is_contiguous() else t.contiguous()


def check_2d(x: torch.Tensor, name: str) -> None:
    if x.dim() != 2:
        raise ValueError(f"{name} must be 2D [M, N], got {tuple(x.shape)}")


def check_1d(x: torch.Tensor, name: str) -> None:
    if x.dim() != 1:
        raise ValueError(f"{name} must be 1D [N], got {tuple(x.shape)}")


def dtype_supported(x: torch.Tensor) -> bool:
    return x.dtype in (torch.float16, torch.bfloat16, torch.float32)
