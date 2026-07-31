from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

from pithtrain.dualpipe.utils import WeightGradStore


def _split_sizes(input: torch.Tensor, grouped_mm_offs: torch.Tensor, ks: Optional[list]) -> list:
    """Per-group row counts (m_splits) summing to ``input.shape[0]``.

    Prefers the precomputed ``ks`` list from token scatter; falls back to a
    host copy of ``grouped_mm_offs`` when unavailable.
    """
    if ks is not None:
        return ks
    sizes = torch.diff(grouped_mm_offs, prepend=grouped_mm_offs.new_zeros(1))
    return sizes.tolist()


class TEGroupLinearFunc(torch.autograd.Function):
    """
    BF16 grouped linear (MoE experts) backed by TransformerEngine's cuBLAS
    grouped GEMM. Same three-GEMM structure as the torch ``grouped_mm`` path:

    Forward: output      = grouped(input @ weight.T)     [layout TN]
    Dgrad:   grad_input  = grouped(grad_output @ weight)  [layout NN]
    Wgrad:   weight_grad = grouped(grad_output.T @ input) [layout NT]

    Wgrad is kept as a separate GEMM so DualPipeV's zero-bubble W-phase can
    defer it through WeightGradStore, freeing the critical path during stage3_b.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        m_splits: list,
    ) -> torch.Tensor:
        num_experts, out_features, _ = weight.shape
        output = torch.empty(input.shape[0], out_features, device=input.device, dtype=input.dtype)
        general_grouped_gemm(
            [weight[i] for i in range(num_experts)],
            list(torch.split(input, m_splits)),
            [output],
            [None] * num_experts,
            input.dtype,
            layout="TN",
            m_splits=m_splits,
            single_output=True,
        )
        ctx.save_for_backward(input, weight)
        ctx.m_splits = m_splits
        return output

    @staticmethod
    def backward(ctx, dy):
        input, weight = ctx.saved_tensors
        m_splits = ctx.m_splits
        num_experts, _, in_features = weight.shape

        dgrad = torch.empty(dy.shape[0], in_features, device=dy.device, dtype=dy.dtype)
        general_grouped_gemm(
            [weight[i] for i in range(num_experts)],
            list(torch.split(dy, m_splits)),
            [dgrad],
            [None] * num_experts,
            dy.dtype,
            layout="NN",
            m_splits=m_splits,
            single_output=True,
        )

        def wgrad_fn(dy, x, m_splits):
            wgrad = torch.empty_like(weight)
            general_grouped_gemm(
                list(torch.split(x, m_splits)),
                list(torch.split(dy, m_splits)),
                [wgrad[i] for i in range(num_experts)],
                [None] * num_experts,
                x.dtype,
                layout="NT",
                m_splits=m_splits,
            )
            weight.grad = wgrad if weight.grad is None else weight.grad.add_(wgrad)

        if WeightGradStore.enabled:
            WeightGradStore.put(partial(wgrad_fn, dy.detach(), input.detach(), m_splits))
        else:
            wgrad_fn(dy, input, m_splits)

        return dgrad, None, None


class TEGroupLinear(nn.Module):
    """
    Grouped linear layer (MoE experts) whose per-expert GEMMs run on
    TransformerEngine's cuBLAS grouped GEMM. Drop-in for ``GroupLinear``:
    identical ``weight[num_groups, out, in]`` parameter and call signature.
    """

    def __init__(self, num_groups: int, in_features: int, out_features: int):
        super().__init__()
        self.num_groups = num_groups
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((num_groups, out_features, in_features)))

    def forward(
        self,
        input: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: Optional[list] = None,
        ks_tensor: Optional[torch.Tensor] = None,
        group_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input.shape[0] == 0:
            # Preserve the autograd graph with a zero-row matmul (see GroupLinear).
            return input @ self.weight[0].T
        m_splits = _split_sizes(input, grouped_mm_offs, ks)
        return TEGroupLinearFunc.apply(input, self.weight, m_splits)
