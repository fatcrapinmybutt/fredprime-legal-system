# Vendor Epilogue Track — cuBLASLt / CUTLASS (Interface Contract)

This repository currently implements **GEMM + epilogue fusion** in Triton as a working, open-source baseline.

To move into vendor-grade epilogues:
- **cuBLASLt**: use cuBLASLt matmul with epilogue options (bias/activation) where available.
- **CUTLASS**: use CUTLASS GEMM + epilogue visitor trees for custom fusions.

## Interface contract (drop-in replacement)
Target function signature:
