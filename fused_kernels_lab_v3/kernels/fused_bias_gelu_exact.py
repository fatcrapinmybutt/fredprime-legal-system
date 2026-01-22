from __future__ import annotations

from typing import Optional, Tuple

import torch

from kernels._common import KernelRunResult, has_cuda, triton_available, ensure_contiguous, check_2d, check_1d, dtype_supported

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


def _get_erf_fn():
    if tl is None:
        return None
    # Triton versions vary: erf may be on tl or tl.math
    fn = getattr(tl, "erf", None)
    if fn is not None:
        return fn
    m = getattr(tl, "math", None)
    if m is not None:
        return getattr(m, "erf", None)
    return None


_ERF = _get_erf_fn()


def triton_supports_erf() -> bool:
    return _ERF is not None


def _check_inputs(x: torch.Tensor, bias: torch.Tensor, residual: Optional[torch.Tensor]) -> Tuple[bool, str]:
    check_2d(x, "x")
    check_1d(bias, "bias")
    if x.shape[1] != bias.shape[0]:
        return False, "x.shape[1] must equal bias.shape[0]"
    if residual is not None:
        if residual.shape != x.shape:
            return False, "residual must match x shape"
        if residual.dtype != x.dtype:
            return False, "residual dtype must match x dtype"
        if residual.device != x.device:
            return False, "residual device must match x device"
    return True, "ok"


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 256}, num_warps=4),
        triton.Config({"BLOCK_N": 512}, num_warps=4),
        triton.Config({"BLOCK_N": 1024}, num_warps=4),
        triton.Config({"BLOCK_N": 2048}, num_warps=8),
    ],
    key=["N"],
)
@triton.jit
def _bias_gelu_exact_kernel(
    X_ptr,
    B_ptr,
    R_ptr,
    Y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Exact GELU:
    # gelu(x) = 0.5*x*(1 + erf(x / sqrt(2)))
    pid_m = tl.program_id(axis=0)
    pid_nb = tl.program_id(axis=1)

    offs = pid_nb * BLOCK_N + tl.arange(0, BLOCK_N)
    base = pid_m * N + offs
    mask = (pid_m < M) & (offs < N)

    x = tl.load(X_ptr + base, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=(offs < N), other=0.0).to(tl.float32)
    z = x + b

    inv_sqrt2 = 0.7071067811865475
    e = _ERF(z * inv_sqrt2)
    g = 0.5 * z * (1.0 + e)

    if HAS_RESIDUAL:
        r = tl.load(R_ptr + base, mask=mask, other=0.0).to(tl.float32)
        g = g + r

    tl.store(Y_ptr + base, g, mask=mask)


def fused_bias_gelu_exact(
    x: torch.Tensor,
    bias: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    *,
    force_triton: bool = False,
) -> KernelRunResult:
    """
    y = gelu(x + bias, approximate='none') [+ residual]
    Triton path is gated by erf capability.
    """
    ok, reason = _check_inputs(x, bias, residual)
    if not ok:
        raise ValueError(reason)

    if not has_cuda():
        y = torch.nn.functional.gelu(x + bias, approximate="none")
        if residual is not None:
            y = y + residual
        return KernelRunResult(y=y, used_triton=False, reason="no CUDA device available")

    if not triton_available():
        y = torch.nn.functional.gelu(x + bias, approximate="none")
        if residual is not None:
            y = y + residual
        return KernelRunResult(y=y, used_triton=False, reason="triton not importable")

    if _ERF is None:
        y = torch.nn.functional.gelu(x + bias, approximate="none")
        if residual is not None:
            y = y + residual
        return KernelRunResult(y=y, used_triton=False, reason="triton erf not available; exact GELU fused path disabled")

    if not dtype_supported(x):
        y = torch.nn.functional.gelu(x + bias, approximate="none")
        if residual is not None:
            y = y + residual
        return KernelRunResult(y=y, used_triton=False, reason="dtype not supported for triton path")

    x = ensure_contiguous(x)
    bias = ensure_contiguous(bias)
    if residual is not None:
        residual = ensure_contiguous(residual)

    try:
        y = torch.empty_like(x)
        M, N = x.shape

        def grid(meta):
            return (M, __import__("triton").cdiv(N, meta["BLOCK_N"]))

        if residual is None:
            _bias_gelu_exact_kernel[grid](x, bias, x, y, M=M, N=N, HAS_RESIDUAL=False)
        else:
            _bias_gelu_exact_kernel[grid](x, bias, residual, y, M=M, N=N, HAS_RESIDUAL=True)

        return KernelRunResult(y=y, used_triton=True, reason="ok")
    except Exception as ex:
        if force_triton:
            raise
        y = torch.nn.functional.gelu(x + bias, approximate="none")
        if residual is not None:
            y = y + residual
        return KernelRunResult(y=y, used_triton=False, reason=f"triton path failed: {type(ex).__name__}: {ex}")
