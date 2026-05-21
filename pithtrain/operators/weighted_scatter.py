"""Fused weighted scatter_add for the deepep recv-side combine.

    output[i, h] = sum_{j: expand_idx[j]==i}
                     outs[j, h] * rtw_leaf[i, k_idx[j]].to(outs.dtype)

ATen's index_add runs ~5x off the HBM bandwidth roof on this shape, so we fuse
gather+multiply+atomic_add in one pass and ship custom backward kernels.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _weighted_scatter_fwd_kernel(
    outs_ptr,  # [V, H] bf16
    rtw_ptr,  # [n_recv, K] fp32 (or bf16 — handled via cast)
    expand_idx_ptr,  # [V] int64
    k_idx_ptr,  # [V] int64
    output_ptr,  # [n_recv, H] bf16, pre-zeroed
    H,
    K,
    BLOCK_H: tl.constexpr,
):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    h_offs = bh * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    i = tl.load(expand_idx_ptr + j)
    k = tl.load(k_idx_ptr + j)
    w_bf16 = tl.load(rtw_ptr + i * K + k).to(tl.bfloat16)

    o = tl.load(outs_ptr + j * H + h_offs, mask=h_mask, other=0.0)
    val = (o * w_bf16).to(tl.bfloat16)
    tl.atomic_add(output_ptr + i * H + h_offs, val, mask=h_mask, sem="relaxed")


@triton.jit
def _weighted_scatter_bwd_outs_kernel(
    d_output_ptr,  # [n_recv, H] bf16
    rtw_ptr,  # [n_recv, K] fp32
    expand_idx_ptr,  # [V] int64
    k_idx_ptr,  # [V] int64
    d_outs_ptr,  # [V, H] bf16, output
    H,
    K,
    BLOCK_H: tl.constexpr,
):
    j = tl.program_id(0)
    bh = tl.program_id(1)
    h_offs = bh * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    i = tl.load(expand_idx_ptr + j)
    k = tl.load(k_idx_ptr + j)
    w_bf16 = tl.load(rtw_ptr + i * K + k).to(tl.bfloat16)

    grad = tl.load(d_output_ptr + i * H + h_offs, mask=h_mask, other=0.0)
    val = (grad * w_bf16).to(tl.bfloat16)
    tl.store(d_outs_ptr + j * H + h_offs, val, mask=h_mask)


@triton.jit
def _weighted_scatter_bwd_rtw_kernel(
    d_output_ptr,  # [n_recv, H] bf16
    outs_ptr,  # [V, H] bf16
    expand_idx_ptr,  # [V] int64
    k_idx_ptr,  # [V] int64
    d_rtw_ptr,  # [n_recv, K] fp32, pre-zeroed
    H,
    K,
    BLOCK_H: tl.constexpr,
):
    j = tl.program_id(0)
    i = tl.load(expand_idx_ptr + j)
    k = tl.load(k_idx_ptr + j)

    acc = 0.0
    for bh_start in range(0, H, BLOCK_H):
        h_offs = bh_start + tl.arange(0, BLOCK_H)
        h_mask = h_offs < H
        grad = tl.load(d_output_ptr + i * H + h_offs, mask=h_mask, other=0.0)
        o = tl.load(outs_ptr + j * H + h_offs, mask=h_mask, other=0.0)
        acc += tl.sum(grad.to(tl.float32) * o.to(tl.float32))

    tl.atomic_add(d_rtw_ptr + i * K + k, acc, sem="relaxed")


class _WeightedScatterCombine(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        outs: torch.Tensor,
        rtw_leaf: torch.Tensor,
        expand_idx: torch.Tensor,
        k_idx: torch.Tensor,
        n_recv: int,
    ) -> torch.Tensor:
        assert outs.is_contiguous() and rtw_leaf.is_contiguous()
        assert expand_idx.is_contiguous() and k_idx.is_contiguous()
        assert expand_idx.dtype == torch.int64 and k_idx.dtype == torch.int64
        assert outs.dtype == torch.bfloat16, f"expected bf16 outs, got {outs.dtype}"
        assert rtw_leaf.dtype in (torch.float32, torch.bfloat16), (
            f"expected fp32/bf16 rtw_leaf, got {rtw_leaf.dtype}"
        )

        V, H = outs.shape
        _, K = rtw_leaf.shape
        output = torch.zeros(n_recv, H, dtype=outs.dtype, device=outs.device)

        if V > 0:
            BLOCK_H = 256
            grid = (V, triton.cdiv(H, BLOCK_H))
            _weighted_scatter_fwd_kernel[grid](
                outs,
                rtw_leaf,
                expand_idx,
                k_idx,
                output,
                H=H,
                K=K,
                BLOCK_H=BLOCK_H,
            )

        ctx.save_for_backward(outs, rtw_leaf, expand_idx, k_idx)
        ctx.n_recv = n_recv
        return output

    @staticmethod
    def backward(ctx, d_output: torch.Tensor):
        outs, rtw_leaf, expand_idx, k_idx = ctx.saved_tensors
        V, H = outs.shape
        _, K = rtw_leaf.shape
        d_output = d_output.contiguous()

        d_outs = torch.empty_like(outs)
        d_rtw = torch.zeros_like(rtw_leaf)

        if V > 0:
            BLOCK_H = 256
            grid_outs = (V, triton.cdiv(H, BLOCK_H))
            _weighted_scatter_bwd_outs_kernel[grid_outs](
                d_output,
                rtw_leaf,
                expand_idx,
                k_idx,
                d_outs,
                H=H,
                K=K,
                BLOCK_H=BLOCK_H,
            )
            grid_rtw = (V,)
            _weighted_scatter_bwd_rtw_kernel[grid_rtw](
                d_output,
                outs,
                expand_idx,
                k_idx,
                d_rtw,
                H=H,
                K=K,
                BLOCK_H=BLOCK_H,
            )

        return d_outs, d_rtw, None, None, None


def weighted_scatter_combine(
    outs: torch.Tensor,
    rtw_leaf: torch.Tensor,
    expand_idx: torch.Tensor,
    k_idx: torch.Tensor,
    n_recv: int,
) -> torch.Tensor:
    return _WeightedScatterCombine.apply(outs, rtw_leaf, expand_idx, k_idx, n_recv)
