from __future__ import annotations

import torch

from kernels.fused_layernorm import fused_layernorm


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _rand(shape, dtype):
    return torch.randn(shape, device=_device(), dtype=dtype)


def _params(n, dtype):
    w = torch.randn((n,), device=_device(), dtype=dtype)
    b = torch.randn((n,), device=_device(), dtype=dtype)
    return w, b


def _assert_close(a: torch.Tensor, b: torch.Tensor):
    torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-2)


def test_layernorm_matches_torch():
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = _rand((1024, 512), dtype)
        w, b = _params(512, dtype)
        out = fused_layernorm(x, w, b).y
        ref = torch.nn.functional.layer_norm(x, normalized_shape=(512,), weight=w, bias=b, eps=1e-5)
        _assert_close(out, ref)
