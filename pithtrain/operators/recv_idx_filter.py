"""Compact DeepEP's dense ``recv_topk_idx`` ([n_recv, K] with -1 sentinels) into
the flat list of valid (slot, k, expert_id) entries downstream needs.

A dense layout with sentinel rows would avoid the D2H, but multiplies all
downstream m-sized ops by ~K/n_local_experts — worse than the sync cost.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _filter_dispatch_kernel(
    recv_topk_idx_ptr,  # [n_recv*K] int64 (deep_ep.topk_idx_t is int64)
    expand_idx_ptr,  # [n_recv*K] int64, output — valid prefix
    k_idx_ptr,  # [n_recv*K] int64, output
    expert_idxs_ptr,  # [n_recv*K] int32, output
    valid_count_ptr,  # [1] int32, atomic counter, pre-zeroed
    total,  # n_recv * K
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    eid = tl.load(recv_topk_idx_ptr + offs, mask=mask, other=-1)
    valid = mask & (eid >= 0)
    zero_offs = tl.zeros([BLOCK], dtype=tl.int32)
    pos = tl.atomic_add(valid_count_ptr + zero_offs, 1, mask=valid)
    i = offs // K
    k = offs - i * K
    tl.store(expand_idx_ptr + pos, i.to(tl.int64), mask=valid)
    tl.store(k_idx_ptr + pos, k.to(tl.int64), mask=valid)
    tl.store(expert_idxs_ptr + pos, eid.to(tl.int32), mask=valid)


def prepare_dispatch_indices(
    recv_topk_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Returns (expand_idx[V] int64, k_idx[V] int64, expert_idxs[V] int32, n_recv)."""
    assert recv_topk_idx.dtype == torch.int64, (
        f"expected int64 recv_topk_idx, got {recv_topk_idx.dtype}"
    )
    n_recv, K = recv_topk_idx.shape[0], recv_topk_idx.shape[1]
    total = n_recv * K
    device = recv_topk_idx.device

    expand_idx_buf = torch.empty(total, dtype=torch.int64, device=device)
    k_idx_buf = torch.empty(total, dtype=torch.int64, device=device)
    expert_idxs_buf = torch.empty(total, dtype=torch.int32, device=device)
    valid_count = torch.zeros(1, dtype=torch.int32, device=device)

    BLOCK = 256
    grid = (triton.cdiv(total, BLOCK),)
    _filter_dispatch_kernel[grid](
        recv_topk_idx.reshape(-1),
        expand_idx_buf,
        k_idx_buf,
        expert_idxs_buf,
        valid_count,
        total,
        K=K,
        BLOCK=BLOCK,
    )
    # Default-stream sync: nothing concurrent to overlap with.
    n_valid = int(valid_count.item())
    expand_idx = expand_idx_buf[:n_valid]
    k_idx = k_idx_buf[:n_valid]
    expert_idxs = expert_idxs_buf[:n_valid]
    return expand_idx, k_idx, expert_idxs, n_recv
