# Fused Kernels Specification

## 1. Scope

This specification describes the APIs and invariants for fusedkernels
(<https://github.com/NVIDIA/Fused-Kernels>).

- Bias+GELU fusion (tanh and exact)
- LayerNorm fusion
- Masked softmax
- GEMM epilogue fusion

## 2. Contracts (invariants)

### 2.1 Device

- Triton paths require CUDA-capable GPU with SM >= 70
- cuBLASLt paths require CUDA >= 11.8 and sm_80+ GPU

### 2.2 DTypes

- Triton paths support: fp16, bfloat16, fp32
- cuBLASLt paths support: int8 only (with scaling)

### 2.3 Contiguity

- Kernels assume row-major contiguity for all tensors

## 3. Kernel APIs

### 3.1 Bias+GELU (tanh)

- `kernels.fused_bias_gelu_tanh(input, bias)`
- Returns: output (same shape/dtype as input)

### 3.2 Bias+GELU (exact)

- `kernels.fused_bias_gelu_exact(input, bias)`
- Returns: output (same shape/dtype as input)

### 3.3 LayerNorm

- `kernels.fused_layernorm.fused_ln(input, weight, bias, eps)`
- Returns: (normalized_output, mean, var)

### 3.4 Masked softmax

- `kernels.fused_masked_softmax.fused_softmax_mask(input, mask, scale)`
- Returns: output (same shape as input)

## 3.5 GEMM epilogue fusion

- `kernels.fused_gemm_epilogue.fused_gemm(A, B, bias, epilogue_fn)`
- Returns: output from (A @ B + bias) |> epilogue_fn

## 4. Validation policy

- Each kernel has a corresponding validation test in `tests/`
- Validation includes: numerical accuracy, performance, and numerical stability

## 5. Superset map outputs

- JSON: `graphs/superset_map.json` (all kernel variants and configs)
- HTML: `graphs/html/superset_map.html` (interactive map)

## 6. Roadmap

- Exact-GELU coverage across Triton SM versions
- Additional epilogue operations
