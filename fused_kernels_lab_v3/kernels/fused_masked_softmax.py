from __future__ import annotations

from typing import Optional, Tuple

import torch
from kernels._common import KernelRunResult, check_2d, dtype_supported, ensure_contiguous, has_cuda, triton_available

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


def _check_inputs(x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[bool, str]:
    check_2d(x, "x")
    if mask is not None:
        if mask.shape != x.shape:
            return False, "mask must match x shape"
        if mask.device != x.device:
            return False, "mask device must match x device"
        if mask.dtype != torch.bool:
            return False, "mask must be boolean (torch.bool)"
    return True, "ok"


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_warps=8),
        triton.Config({"BLOCK_N": 2048}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _masked_softmax_kernel(
    X_ptr,
    M_ptr,
    Y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_nb = tl.program_id(axis=1)

    offs = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
    base = pid_m * N + offs
    mask = (pid_m < M) & (offs < N)

    x = tl.load(X_ptr + base, mask=mask, other=-float("inf")).to(tl.float32)

    if HAS_MASK:
        m = tl.load(M_ptr + base, mask=mask, other=0).to(tl.int1)
        # If mask is False, set to -inf so exp becomes 0
        x = tl.where(m, x, -float("inf"))

    # softmax over the full row requires max/sum across all blocks.
    # We do a two-pass blockwise approach: compute row-max and row-sum by scanning blocks.
    # First compute max for this block, then reduce across blocks using atomic max? not available.
    # Instead: implement row-wise softmax per-row in a single program by requiring N <= BLOCK_N (not acceptable),
    # OR use a scan over blocks inside a single program (works, but then axis=1 grid not needed).
    #
    # For correctness for all N: we use a single program per row and scan blocks inside the program.
    # Therefore: this kernel is specialized to pid_nb==0 only; we keep 2D grid for autotune keying but guard.
    if pid_nb != 0:
        return

    # Row-wise scan over blocks
    row_max = tl.full([1], -float("inf"), tl.float32)
    for nb in tl.static_range(0, tl.cdiv(N, BLOCK_N)):
        o = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        b = pid_m * N + o
        mk = o < N
        xi = tl.load(X_ptr + b, mask=mk, other=-float("inf")).to(tl.float32)
        if HAS_MASK:
            mi = tl.load(M_ptr + b, mask=mk, other=0).to(tl.int1)
            xi = tl.where(mi, xi, -float("inf"))
        row_max = tl.maximum(row_max, tl.max(xi, axis=0))

    row_sum = tl.zeros([1], tl.float32)
    for nb in tl.static_range(0, tl.cdiv(N, BLOCK_N)):
        o = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        b = pid_m * N + o
        mk = o < N
        xi = tl.load(X_ptr + b, mask=mk, other=-float("inf")).to(tl.float32)
        if HAS_MASK:
            mi = tl.load(M_ptr + b, mask=mk, other=0).to(tl.int1)
            xi = tl.where(mi, xi, -float("inf"))
        ex = tl.exp(xi - row_max)
        row_sum += tl.sum(ex, axis=0)

    inv = 1.0 / row_sum

    for nb in tl.static_range(0, tl.cdiv(N, BLOCK_N)):
        o = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        b = pid_m * N + o
        mk = o < N
        xi = tl.load(X_ptr + b, mask=mk, other=-float("inf")).to(tl.float32)
        if HAS_MASK:
            mi = tl.load(M_ptr + b, mask=mk, other=0).to(tl.int1)
            xi = tl.where(mi, xi, -float("inf"))
        ex = tl.exp(xi - row_max)
        yi = ex * inv
        tl.store(Y_ptr + b, yi, mask=mk)


def fused_masked_softmax(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    force_triton: bool = False,
) -> KernelRunResult:
    """
    Softmax over last dimension of x (2D), with optional boolean mask.
    If mask is provided: positions with mask=False are excluded (probability 0).
    """
    ok, reason = _check_inputs(x, mask)
    if not ok:
        raise ValueError(reason)

    if not has_cuda():
        if mask is None:
            y = torch.softmax(x, dim=-1)
        else:
            y = torch.softmax(x.masked_fill(~mask, float("-inf")), dim=-1)
        return KernelRunResult(y=y, used_triton=False, reason="no CUDA device available")

    if not triton_available():
        if mask is None:
            y = torch.softmax(x, dim=-1)
        else:
            y = torch.softmax(x.masked_fill(~mask, float("-inf")), dim=-1)
        return KernelRunResult(y=y, used_triton=False, reason="triton not importable")

    if not dtype_supported(x):
        if mask is None:
            y = torch.softmax(x, dim=-1)
        else:
            y = torch.softmax(x.masked_fill(~mask, float("-inf")), dim=-1)
        return KernelRunResult(y=y, used_triton=False, reason="dtype not supported for triton path")

    x = ensure_contiguous(x)
    if mask is not None:
        mask = ensure_contiguous(mask)

    try:
        y = torch.empty((x.shape[0], x.shape[1]), device=x.device, dtype=torch.float32).to(x.dtype)
        M, N = x.shape

        def grid(meta):
            # 2D grid for autotune keying; kernel uses pid_nb==0 and scans blocks.
            return (M, 1)

        if mask is None:
            _masked_softmax_kernel[grid](x, x, y, M=M, N=N, HAS_MASK=False)
        else:
            _masked_softmax_kernel[grid](x, mask, y, M=M, N=N, HAS_MASK=True)

        return KernelRunResult(y=y, used_triton=True, reason="ok")
    except Exception as ex:
        if force_triton:
            raise
        if mask is None:
            y = torch.softmax(x, dim=-1)
        else:
            y = torch.softmax(x.masked_fill(~mask, float("-inf")), dim=-1)
        return KernelRunResult(y=y, used_triton=False, reason=f"triton path failed: {type(ex).__name__}: {ex}")
