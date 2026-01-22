from __future__ import annotations

import argparse

import torch

from scripts.env_check import main as env_check_main
from kernels.fused_bias_gelu_tanh import fused_bias_gelu_tanh
from kernels.fused_layernorm import fused_layernorm
from kernels.fused_masked_softmax import fused_masked_softmax
from kernels.fused_gemm_epilogue import fused_gemm_bias_gelu


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _dtype_list(device: torch.device) -> list[torch.dtype]:
    if device.type == "cuda":
        return [torch.float16, torch.bfloat16, torch.float32]
    return [torch.float32]


def _assert_close(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(a, b, rtol=1e-2, atol=1e-2)


def _run_checks() -> None:
    device = _device()
    for dtype in _dtype_list(device):
        x = torch.randn((128, 64), device=device, dtype=dtype)
        bias = torch.randn((64,), device=device, dtype=dtype)
        residual = torch.randn_like(x)
        out = fused_bias_gelu_tanh(x, bias, residual).y
        ref = torch.nn.functional.gelu(x + bias, approximate="tanh") + residual
        _assert_close(out, ref)

        ln_w = torch.randn((64,), device=device, dtype=dtype)
        ln_b = torch.randn((64,), device=device, dtype=dtype)
        ln_out = fused_layernorm(x, ln_w, ln_b).y
        ln_ref = torch.nn.functional.layer_norm(x, normalized_shape=(64,), weight=ln_w, bias=ln_b, eps=1e-5)
        _assert_close(ln_out, ln_ref)

        mask = torch.rand((128, 64), device=device) > 0.2
        sm_out = fused_masked_softmax(x, mask).y
        sm_ref = torch.softmax(x.masked_fill(~mask, float("-inf")), dim=-1)
        _assert_close(sm_out, sm_ref)

        a = torch.randn((64, 32), device=device, dtype=dtype)
        b = torch.randn((32, 48), device=device, dtype=dtype)
        gemm_bias = torch.randn((48,), device=device, dtype=dtype)
        gemm_residual = torch.randn((64, 48), device=device, dtype=dtype)
        gemm_out = fused_gemm_bias_gelu(a, b, bias=gemm_bias, residual=gemm_residual).y
        gemm_ref = torch.nn.functional.gelu(a @ b + gemm_bias, approximate="tanh") + gemm_residual
        _assert_close(gemm_out, gemm_ref)


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick smoke test for fused kernels.")
    parser.add_argument("--skip-env", action="store_true", help="Skip environment check.")
    args = parser.parse_args()

    if not args.skip_env:
        env_check_main()

    _run_checks()
    print("Smoke test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
