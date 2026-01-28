# SPEC — Fused Kernels Lab v3 (Corpus Grade)

## 1. Scope
This repository defines a reproducible *kernel-corpus* for learning and iterating on fused kernels.

It includes:
- Bias+GELU fusion (tanh and exact/erf)
- LayerNorm fusion (row-wise)
- Masked softmax fusion (row-wise)
- GEMM + epilogue fusion (Triton matmul + bias + GELU + residual)

## 2. Contracts (invariants)
### 2.1 Device
- Triton paths require CUDA-capable GPU and CUDA-enabled PyTorch.

### 2.2 DTypes
- Triton paths support: fp16, bf16, fp32.
- Accumulation/stats are computed in fp32 inside kernels.

### 2.3 Contiguity
- Kernels assume row-major contiguous inputs. Wrappers will `.contiguous()` if needed.

## 3. Kernel APIs

### 3.1 Bias+GELU (tanh)

- `kernels.fused_bias_gelu_tanh.fused_bias_gelu_tanh(x, bias, residual=None) -> KernelRunResult`

### 3.2 Bias+GELU (exact)

- `kernels.fused_bias_gelu_exact.fused_bias_gelu_exact(x, bias, residual=None) -> KernelRunResult`
- Gated by Triton `erf` capability detection.

### 3.3 LayerNorm

- `kernels.fused_layernorm.fused_layernorm(x, weight, bias, eps=1e-5) -> KernelRunResult`

### 3.4 Masked softmax

- `kernels.fused_masked_softmax.fused_masked_softmax(x, mask=None) -> KernelRunResult`
- `mask` must be boolean with shape [M, N]; False positions receive probability 0.

### 3.5 GEMM epilogue fusion

- `kernels.fused_gemm_epilogue.fused_gemm_bias_gelu(A, B, bias=None, residual=None) -> KernelRunResult`

## 4. Validation policy

- Each kernel has a corresponding pytest module comparing against PyTorch reference.
- Benchmarks compare eager vs torch.compile(inductor) vs fused kernel path.

## 5. Superset map outputs

- JSON: `graphs/superset_map.json`
- Offline HTML interactive graph: `graphs/html/superset_map.html`
- Neo4j import: `graphs/neo4j/*`

## 6. Roadmap

- Exact-GELU coverage across Triton versions via richer capability probing
- Attention macro-fusion starter (FlashAttention-style tiling)
- Vendor epilogues: cuBLASLt/CUTLASS integration (see docs/VENDOR_EPILOGUE_TRACK.md)
