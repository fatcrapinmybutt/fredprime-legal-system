from __future__ import annotations

import argparse
import time

import torch
from kernels.fused_bias_gelu_exact import fused_bias_gelu_exact, triton_supports_erf
from kernels.fused_bias_gelu_tanh import fused_bias_gelu_tanh


def _parse_dtype(s: str) -> torch.dtype:
    s2 = s.strip().lower()
    if s2 in ("fp16", "float16"):
        return torch.float16
    if s2 in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s2 in ("fp32", "float32"):
        return torch.float32
    raise ValueError("dtype must be one of: fp16, bf16, fp32")


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=65536)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--gelu", type=str, default="tanh", choices=["tanh", "none"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    torch.manual_seed(0)
    dtype = _parse_dtype(args.dtype)
    device = torch.device("cuda")

    x = torch.randn((args.m, args.n), device=device, dtype=dtype)
    bias = torch.randn((args.n,), device=device, dtype=dtype)
    residual = torch.randn_like(x) if args.residual else None

    def eager_ref():
        y = torch.nn.functional.gelu(x + bias, approximate=args.gelu)
        if residual is not None:
            y = y + residual
        return y

    # warmup
    for _ in range(args.warmup):
        eager_ref()
    _sync()

    # eager
    t0 = time.perf_counter()
    for _ in range(args.iters):
        eager_ref()
    _sync()
    t1 = time.perf_counter()
    eager_ms = (t1 - t0) * 1000.0 / args.iters

    compiled = torch.compile(eager_ref, backend="inductor")
    for _ in range(args.warmup):
        compiled()
    _sync()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        compiled()
    _sync()
    t1 = time.perf_counter()
    comp_ms = (t1 - t0) * 1000.0 / args.iters

    # fused
    if args.gelu == "tanh":
        fused_fn = lambda: fused_bias_gelu_tanh(x, bias, residual).y
        fused_label = "Triton fused tanh"
    else:
        fused_fn = lambda: fused_bias_gelu_exact(x, bias, residual).y
        fused_label = "Triton fused exact" if triton_supports_erf() else "Exact fallback"

    for _ in range(args.warmup):
        fused_fn()
    _sync()

    t0 = time.perf_counter()
    used = 0
    for _ in range(args.iters):
        if args.gelu == "tanh":
            r = fused_bias_gelu_tanh(x, bias, residual)
        else:
            r = fused_bias_gelu_exact(x, bias, residual)
        used += int(r.used_triton)
    _sync()
    t1 = time.perf_counter()
    fused_ms = (t1 - t0) * 1000.0 / args.iters

    print("Benchmark: y = gelu(x + bias) [+ residual]")
    print(f"M={args.m} N={args.n} dtype={dtype} residual={args.residual} gelu={args.gelu}")
    print(f"Eager:         {eager_ms:.4f} ms")
    print(f"torch.compile:  {comp_ms:.4f} ms")
    print(f"{fused_label}:  {fused_ms:.4f} ms  (used_triton {used}/{args.iters})")
    if args.gelu == "none":
        print("Triton erf available:", triton_supports_erf())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
