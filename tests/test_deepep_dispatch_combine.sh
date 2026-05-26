#!/usr/bin/env bash
# 4-rank DeepEP dispatch/combine fwd+bwd unit tests. Requires an IBGDA-enabled
# node with at least 4 GPUs visible.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export OMP_NUM_THREADS=8
EP_SUPPRESS_NCCL_CHECK=1 \
    .venv/bin/torchrun --standalone --nproc-per-node=4 \
        -m pytest tests/test_deepep_dispatch_combine.py -v "$@"
