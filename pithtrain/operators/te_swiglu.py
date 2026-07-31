"""Fused SwiGLU-style ``silu(gate) * up`` activation backed by TransformerEngine.

The gate and up projections are concatenated along the last dim and handed to
TE's fused SwiGLU kernel (``silu`` of the first half times the second half),
with a single fused backward. A simple ``(gate, up) -> out`` signature makes it
a drop-in for the SwiGLU MLP block.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformer_engine.pytorch.cpp_extensions import dswiglu as _te_dswiglu
from transformer_engine.pytorch.cpp_extensions import swiglu as _te_swiglu


class _TESwiGLU(torch.autograd.Function):
    """Fused ``silu(gate) * up`` on TE's SwiGLU kernel (gate first, up second)."""

    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([gate, up], dim=-1).contiguous()
        out = _te_swiglu(cat, None)
        ctx.save_for_backward(cat)
        ctx.gate_dim = gate.shape[-1]
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (cat,) = ctx.saved_tensors
        dcat = _te_dswiglu(grad_out.contiguous(), cat, None)
        h = ctx.gate_dim
        return dcat[..., :h], dcat[..., h:]


def te_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused ``silu(gate) * up`` for a SwiGLU MLP.

    Parameters
    ----------
    gate, up : torch.Tensor
        Same-shape, same-dtype tensors from the gate and up projections.

    Returns
    -------
    torch.Tensor
        Element-wise ``silu(gate) * up`` in the same dtype as the inputs.
    """
    if gate.numel() == 0:
        # TE's kernel rejects zero-row inputs; the native path keeps the graph.
        return F.silu(gate) * up
    return _TESwiGLU.apply(gate, up)
