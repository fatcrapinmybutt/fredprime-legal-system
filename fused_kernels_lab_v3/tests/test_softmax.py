from __future__ import annotations

import torch

from kernels.fused_masked_softmax import fused_masked_softmax


def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _rand(shape, dtype):
    return torch.randn(shape, device=_device(), dtype=dtype)


def _mask(shape):
    return (torch.rand(shape, device=_device()) > 0.4)


def _assert_close(a: torch.Tensor, b: torch.Tensor):
    torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-2)


def test_softmax_matches_torch_no_mask():
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = _rand((512, 256), dtype)
        out = fused_masked_softmax(x).y
        ref = torch.softmax(x, dim=-1)
        _assert_close(out, ref)


def test_softmax_matches_torch_with_mask():
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        x = _rand((128, 128), dtype)
        mask = _mask((128, 128))
        out = fused_masked_softmax(x, mask).y
        ref = torch.softmax(x.masked_fill(~mask, float("-inf")), dim=-1)
        _assert_close(out, ref)
