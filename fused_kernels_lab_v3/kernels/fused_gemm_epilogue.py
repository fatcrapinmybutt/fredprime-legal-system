from __future__ import annotations

from typing import Optional, Tuple

import torch

from kernels._common import KernelRunResult, has_cuda, triton_available, ensure_contiguous, dtype_supported

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


def _check_inputs(a: torch.Tensor, b: torch.Tensor, bias: Optional[torch.Tensor], residual: Optional[torch.Tensor]) -> Tuple[bool, str]:
    if a.dim() != 2 or b.dim() != 2:
        return False, "a and b must be 2D"
    if a.shape[1] != b.shape[0]:
        return False, "a.shape[1] must equal b.shape[0]"
    m, k = a.shape
    _, n = b.shape
    if bias is not None:
        if bias.dim() != 1 or bias.shape[0] != n:
            return False, "bias must be [N]"
        if bias.device != a.device:
            return False, "bias device must match a"
        if bias.dtype != a.dtype:
            return False, "bias dtype must match a"
    if residual is not None:
        if residual.shape != (m, n):
            return False, "residual must be [M, N]"
        if residual.device != a.device or residual.dtype != a.dtype:
            return False, "residual device/dtype must match a"
    return True, "ok"


@triton.autotune(
    configs=[
        triton.Config({"BM": 64, "BN": 64, "BK": 32}, num_warps=4),
        triton.Config({"BM": 128, "BN": 64, "BK": 32}, num_warps=8),
        triton.Config({"BM": 64, "BN": 128, "BK": 32}, num_warps=8),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _gemm_bias_gelu_kernel(
    A_ptr,
    B_ptr,
    Bias_ptr,
    R_ptr,
    C_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    # pointers for A and B blocks
    a_ptrs = A_ptr + (offs_m[:, None] * K + offs_k[None, :])
    b_ptrs = B_ptr + (offs_k[:, None] * N + offs_n[None, :])

    acc = tl.zeros([BM, BN], dtype=tl.float32)

    # K loop
    for k0 in tl.static_range(0, tl.cdiv(K, BK)):
        k_mask = (k0 * BK + offs_k) < K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0).to(tl.float32)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0).to(tl.float32)
        acc += tl.dot(a, b)
        a_ptrs += BK
        b_ptrs += BK * N

    if HAS_BIAS:
        bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    # GELU tanh approximation on the matmul result
    c0 = 0.7978845608028654
    c1 = 0.044715
    z3 = acc * acc * acc
    inner = c0 * (acc + c1 * z3)
    out = 0.5 * acc * (1.0 + tl.tanh(inner))

    if HAS_RESIDUAL:
        r_ptrs = R_ptr + (offs_m[:, None] * N + offs_n[None, :])
        r = tl.load(r_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
        out = out + r

    c_ptrs = C_ptr + (offs_m[:, None] * N + offs_n[None, :])
    tl.store(c_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def fused_gemm_bias_gelu(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    residual: Optional[torch.Tensor] = None,
    *,
    force_triton: bool = False,
) -> KernelRunResult:
    """
    C = GELU(A @ B [+ bias]) [+ residual]
    Implements GEMM + epilogue fusion in Triton as a working track.
    """
    ok, reason = _check_inputs(a, b, bias, residual)
    if not ok:
        raise ValueError(reason)

    if not has_cuda():
        c = a @ b
        if bias is not None:
            c = c + bias
        c = torch.nn.functional.gelu(c, approximate="tanh")
        if residual is not None:
            c = c + residual
        return KernelRunResult(y=c, used_triton=False, reason="no CUDA device available")

    if not triton_available():
        c = a @ b
        if bias is not None:
            c = c + bias
        c = torch.nn.functional.gelu(c, approximate="tanh")
        if residual is not None:
            c = c + residual
        return KernelRunResult(y=c, used_triton=False, reason="triton not importable")

    if not (dtype_supported(a) and dtype_supported(b)):
        c = a @ b
        if bias is not None:
            c = c + bias
        c = torch.nn.functional.gelu(c, approximate="tanh")
        if residual is not None:
            c = c + residual
        return KernelRunResult(y=c, used_triton=False, reason="dtype not supported for triton path")

    a = ensure_contiguous(a)
    b = ensure_contiguous(b)
    if bias is not None:
        bias = ensure_contiguous(bias)
    if residual is not None:
        residual = ensure_contiguous(residual)

    try:
        m, k = a.shape
        _, n = b.shape
        c = torch.empty((m, n), device=a.device, dtype=a.dtype)

        def grid(meta):
            return (triton.cdiv(m, meta["BM"]), triton.cdiv(n, meta["BN"]))

        _gemm_bias_gelu_kernel[grid](
            a, b,
            bias if bias is not None else a,  # placeholder pointer
            residual if residual is not None else a,  # placeholder pointer
            c,
            M=m, N=n, K=k,
            HAS_BIAS=(bias is not None),
            HAS_RESIDUAL=(residual is not None),
        )
        return KernelRunResult(y=c, used_triton=True, reason="ok")
    except Exception as ex:
        if force_triton:
            raise
        c = a @ b
        if bias is not None:
            c = c + bias
        c = torch.nn.functional.gelu(c, approximate="tanh")
        if residual is not None:
            c = c + residual
        return KernelRunResult(y=c, used_triton=False, reason=f"triton path failed: {type(ex).__name__}: {ex}")
