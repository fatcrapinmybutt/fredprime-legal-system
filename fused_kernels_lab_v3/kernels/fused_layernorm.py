from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from kernels._common import KernelRunResult, has_cuda, triton_available, ensure_contiguous, check_2d, check_1d, dtype_supported

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


def _check_inputs(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> Tuple[bool, str]:
    check_2d(x, "x")
    check_1d(weight, "weight")
    check_1d(bias, "bias")
    if x.shape[1] != weight.shape[0] or x.shape[1] != bias.shape[0]:
        return False, "weight and bias must have shape [N] matching x.shape[1]"
    return True, "ok"


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _layernorm_kernel(
    X_ptr,
    W_ptr,
    B_ptr,
    Y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)

    # Pass 1: compute mean and variance across N in BLOCK_N chunks
    sum_x = tl.zeros([1], dtype=tl.float32)
    sum_x2 = tl.zeros([1], dtype=tl.float32)

    # loop over column blocks
    for nb in tl.static_range(0, tl.cdiv(N, BLOCK_N)):
        offs = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        x = tl.load(X_ptr + pid_m * N + offs, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / N
    var = sum_x2 / N - mean * mean
    rstd = tl.rsqrt(var + EPS)

    # Pass 2: normalize + affine, write output
    for nb in tl.static_range(0, tl.cdiv(N, BLOCK_N)):
        offs = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N

        x = tl.load(X_ptr + pid_m * N + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        y = (x - mean) * rstd
        y = y * w + b

        tl.store(Y_ptr + pid_m * N + offs, y, mask=mask)


def fused_layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    eps: float = 1e-5,
    force_triton: bool = False,
) -> KernelRunResult:
    """
    Row-wise LayerNorm over last dimension:
      y = (x - mean) / sqrt(var + eps) * weight + bias
    """
    ok, reason = _check_inputs(x, weight, bias)
    if not ok:
        raise ValueError(reason)

    if not has_cuda():
        y = torch.nn.functional.layer_norm(x, normalized_shape=(x.shape[1],), weight=weight, bias=bias, eps=eps)
        return KernelRunResult(y=y, used_triton=False, reason="no CUDA device available")

    if not triton_available():
        y = torch.nn.functional.layer_norm(x, normalized_shape=(x.shape[1],), weight=weight, bias=bias, eps=eps)
        return KernelRunResult(y=y, used_triton=False, reason="triton not importable")

    if not dtype_supported(x):
        y = torch.nn.functional.layer_norm(x, normalized_shape=(x.shape[1],), weight=weight, bias=bias, eps=eps)
        return KernelRunResult(y=y, used_triton=False, reason="dtype not supported for triton path")

    x = ensure_contiguous(x)
    weight = ensure_contiguous(weight)
    bias = ensure_contiguous(bias)

    try:
        y = torch.empty_like(x)
        M, N = x.shape

        def grid(_meta):
            return (M,)

        _layernorm_kernel[grid](x, weight, bias, y, M=M, N=N, EPS=eps)
        return KernelRunResult(y=y, used_triton=True, reason="ok")
    except Exception as ex:
        if force_triton:
            raise
        y = torch.nn.functional.layer_norm(x, normalized_shape=(x.shape[1],), weight=weight, bias=bias, eps=eps)
        return KernelRunResult(y=y, used_triton=False, reason=f"triton path failed: {type(ex).__name__}: {ex}")
