from __future__ import annotations

import torch

from kernels.fused_gemm_epilogue import fused_gemm_bias_gelu


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _rand(shape, dtype):
    return torch.randn(shape, device=_device(), dtype=dtype)


def _assert_close(a: torch.Tensor, b: torch.Tensor):
    torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-2)


def test_gemm_epilogue_matches_torch():
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        a = _rand((256, 128), dtype)
        b = _rand((128, 192), dtype)
        bias = _rand((192,), dtype)
        residual = _rand((256, 192), dtype)
        out = fused_gemm_bias_gelu(a, b, bias=bias, residual=residual).y
        ref = torch.nn.functional.gelu(a @ b + bias, approximate="tanh") + residual
        _assert_close(out, ref)
