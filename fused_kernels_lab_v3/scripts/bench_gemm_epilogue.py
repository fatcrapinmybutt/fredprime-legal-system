from __future__ import annotations

import argparse
import time

import torch
from kernels.fused_gemm_epilogue import fused_gemm_bias_gelu


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
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--dtype", type=str, default="bf16")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--residual", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    torch.manual_seed(0)
    dtype = _parse_dtype(args.dtype)
    device = torch.device("cuda")

    a = torch.randn((args.m, args.k), device=device, dtype=dtype)
    b = torch.randn((args.k, args.n), device=device, dtype=dtype)
    bias = torch.randn((args.n,), device=device, dtype=dtype) if args.bias else None
    residual = torch.randn((args.m, args.n), device=device, dtype=dtype) if args.residual else None

    def eager_ref():
        c = a @ b
        if bias is not None:
            c = c + bias
        c = torch.nn.functional.gelu(c, approximate="tanh")
        if residual is not None:
            c = c + residual
        return c

    for _ in range(args.warmup):
        eager_ref()
    _sync()
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

    for _ in range(args.warmup):
        fused_gemm_bias_gelu(a, b, bias=bias, residual=residual).y
    _sync()
    t0 = time.perf_counter()
    used = 0
    for _ in range(args.iters):
        r = fused_gemm_bias_gelu(a, b, bias=bias, residual=residual)
        used += int(r.used_triton)
    _sync()
    t1 = time.perf_counter()
    fused_ms = (t1 - t0) * 1000.0 / args.iters

    print("Benchmark: GEMM + bias + GELU [+ residual] (Triton epilogue track)")
    print(f"M={args.m} N={args.n} K={args.k} dtype={dtype} bias={args.bias} residual={args.residual}")
    print(f"Eager:         {eager_ms:.4f} ms")
    print(f"torch.compile:  {comp_ms:.4f} ms")
    print(f"Triton fused:   {fused_ms:.4f} ms  (used_triton {used}/{args.iters})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
