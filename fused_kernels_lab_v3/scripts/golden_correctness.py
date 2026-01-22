from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import torch

from kernels.fused_bias_gelu_tanh import fused_bias_gelu_tanh
from kernels.fused_bias_gelu_exact import fused_bias_gelu_exact
from kernels.fused_layernorm import fused_layernorm
from kernels.fused_masked_softmax import fused_masked_softmax
from kernels.fused_gemm_epilogue import fused_gemm_bias_gelu


@dataclass(frozen=True)
class ErrorStats:
    max_abs: float
    max_rel: float
    normalized: float
    confidence: float
    passed: bool


def _device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _dtype_list(device: torch.device) -> list[torch.dtype]:
    if device.type == "cuda":
        return [torch.float16, torch.bfloat16, torch.float32]
    return [torch.float32]


def _error_stats(out: torch.Tensor, ref: torch.Tensor, rtol: float, atol: float) -> ErrorStats:
    diff = (out - ref).abs()
    max_abs = diff.max().item()
    ref_abs_max = ref.abs().max().item()
    denom = atol + rtol * ref_abs_max
    max_rel = (max_abs / ref_abs_max) if ref_abs_max > 0 else 0.0
    normalized = max_abs / denom if denom > 0 else float("inf")
    confidence = max(0.0, 1.0 - normalized)
    passed = normalized <= 1.0
    return ErrorStats(
        max_abs=float(max_abs),
        max_rel=float(max_rel),
        normalized=float(normalized),
        confidence=float(confidence),
        passed=passed,
    )


def _record(stats: dict[str, Any], name: str, out: torch.Tensor, ref: torch.Tensor, rtol: float, atol: float) -> None:
    err = _error_stats(out, ref, rtol=rtol, atol=atol)
    stats[name] = {
        "max_abs": err.max_abs,
        "max_rel": err.max_rel,
        "normalized": err.normalized,
        "confidence": err.confidence,
        "passed": err.passed,
    }


def _hardware_snapshot(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {
        "torch_version": torch.__version__,
        "device_type": device.type,
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info.update(
            {
                "device_index": idx,
                "device_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(props.total_memory / (1024**3), 2),
                "cuda_runtime": torch.version.cuda,
            }
        )
    return info


def run_golden(seed: int, rtol: float, atol: float) -> dict[str, Any]:
    torch.manual_seed(seed)
    device = _device()
    stats: dict[str, Any] = {}

    for dtype in _dtype_list(device):
        x = torch.randn((256, 128), device=device, dtype=dtype)
        bias = torch.randn((128,), device=device, dtype=dtype)
        residual = torch.randn_like(x)
        out = fused_bias_gelu_tanh(x, bias, residual).y
        ref = torch.nn.functional.gelu(x + bias, approximate="tanh") + residual
        _record(stats, f"bias_gelu_tanh/{dtype}", out, ref, rtol, atol)

        out = fused_bias_gelu_exact(x, bias).y
        ref = torch.nn.functional.gelu(x + bias, approximate="none")
        _record(stats, f"bias_gelu_exact/{dtype}", out, ref, rtol, atol)

        w = torch.randn((128,), device=device, dtype=dtype)
        b = torch.randn((128,), device=device, dtype=dtype)
        ln_out = fused_layernorm(x, w, b).y
        ln_ref = torch.nn.functional.layer_norm(x, normalized_shape=(128,), weight=w, bias=b, eps=1e-5)
        _record(stats, f"layernorm/{dtype}", ln_out, ln_ref, rtol, atol)

        mask = torch.rand((256, 128), device=device) > 0.3
        sm_out = fused_masked_softmax(x, mask).y
        sm_ref = torch.softmax(x.masked_fill(~mask, float("-inf")), dim=-1)
        _record(stats, f"masked_softmax/{dtype}", sm_out, sm_ref, rtol, atol)

        a = torch.randn((128, 64), device=device, dtype=dtype)
        b = torch.randn((64, 96), device=device, dtype=dtype)
        gemm_bias = torch.randn((96,), device=device, dtype=dtype)
        gemm_residual = torch.randn((128, 96), device=device, dtype=dtype)
        gemm_out = fused_gemm_bias_gelu(a, b, bias=gemm_bias, residual=gemm_residual).y
        gemm_ref = torch.nn.functional.gelu(a @ b + gemm_bias, approximate="tanh") + gemm_residual
        _record(stats, f"gemm_epilogue/{dtype}", gemm_out, gemm_ref, rtol, atol)

    total = len(stats)
    passed = sum(1 for v in stats.values() if v["passed"])
    confidence = sum(v["confidence"] for v in stats.values()) / total if total else 0.0

    return {
        "seed": seed,
        "rtol": rtol,
        "atol": atol,
        "hardware": _hardware_snapshot(device),
        "results": stats,
        "summary": {
            "total": total,
            "passed": passed,
            "confidence": confidence,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Golden correctness harness with confidence scoring.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    report = run_golden(seed=args.seed, rtol=args.rtol, atol=args.atol)
    summary = report["summary"]
    print(
        "Golden correctness:",
        f"passed {summary['passed']}/{summary['total']},",
        f"confidence={summary['confidence']:.4f}",
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("Wrote:", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
