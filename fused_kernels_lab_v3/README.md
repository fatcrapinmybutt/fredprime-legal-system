# Fused Kernels Lab — Spec/Corpus Grade v3 (Triton + torch.compile + Superset Map)

This repo is a **corpus-grade**, reproducible starter kit for building and validating **fused GPU kernels**.

It contains:
- **Bias + GELU** fused (tanh-approx) and **Exact GELU** fused (erf-based) with **capability-gated Triton detection**.
- **LayerNorm fusion starter** (row-wise LN) with fp32 stats and affine.
- **Masked softmax fusion starter** (2D softmax over last dim) with optional boolean mask.
- **GEMM + epilogue fusion track** implemented in Triton (matmul + bias + GELU + optional residual). This is the practical “bridge” toward vendor epilogues.

It also emits a **canonical superset map graph** in two formats:
- Offline HTML interactive map: `graphs/html/superset_map.html`
- Neo4j import pack: `graphs/neo4j/` (nodes.csv, edges.csv, import.cypher)

## Install
pip install -r requirements.txt

## Quick run
### Env check
python -m scripts.env_check

### Correctness
pytest -q

### Benchmarks
python -m scripts.bench_bias_gelu --dtype bf16 --m 65536 --n 4096 --gelu tanh
python -m scripts.bench_layernorm --dtype bf16 --m 32768 --n 4096
python -m scripts.bench_softmax --dtype bf16 --m 8192 --n 4096 --mask_rate 0.25
python -m scripts.bench_gemm_epilogue --dtype bf16 --m 4096 --n 4096 --k 4096

### Build/refresh the superset map
python -m scripts.build_superset_graph

Open:
- graphs/html/superset_map.html (offline, no network required)

## Notes
- Exact GELU fused kernel is gated by Triton availability of `erf`. If not available, the code falls back and explains why.
- The GEMM epilogue path in this repo is implemented in Triton to keep the track fully working and open-source.
  The docs include an interface contract for swapping in cuBLASLt/CUTLASS epilogue paths in a future step.
