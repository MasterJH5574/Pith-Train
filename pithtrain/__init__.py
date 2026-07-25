import os

# Select TransformerEngine's CUTLASS grouped-GEMM backend for the MoE experts.
os.environ.setdefault("NVTE_USE_CUTLASS_GROUPED_GEMM", "1")
