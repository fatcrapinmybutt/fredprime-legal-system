from __future__ import annotations

import torch

from kernels.fused_bias_gelu_tanh import fused_bias_gelu_tanh
from kernels.fused_bias_gelu_exact import fused_bias_gelu_exact, triton_supports_erf


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _rand(shape, dtype):
    return torch.randn(shape, device=_device(), dtype=dtype)


def _bias(n, dtype):
    return torch.randn((n,), device=_device(), dtype=dtype)


def _residual(x):
    return torch.randn_like(x)


def _assert_close(a: torch.Tensor, b: torch.Tensor):
    torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-2)


def test_bias_gelu_tanh_matches_torch():
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = _rand((512, 256), dtype)
        bias = _bias(256, dtype)
        residual = _residual(x)
        out = fused_bias_gelu_tanh(x, bias, residual).y
        ref = torch.nn.functional.gelu(x + bias, approximate="tanh") + residual
        _assert_close(out, ref)


def test_bias_gelu_exact_matches_torch():
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = _rand((256, 128), dtype)
        bias = _bias(128, dtype)
        out = fused_bias_gelu_exact(x, bias).y
        ref = torch.nn.functional.gelu(x + bias, approximate="none")
        _assert_close(out, ref)

    if torch.cuda.is_available():
        # If Triton doesn't support erf, the fallback path is still correct
        assert isinstance(triton_supports_erf(), bool)
