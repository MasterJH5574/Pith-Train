"""BF16 grouped linear layer (MoE experts) backed by the ``pithtrain_ext`` CUDA kernels.

The expert grouped GEMMs are exposed as ``torch.library.custom_op``s (with
``register_fake``) so they trace under ``torch.compile(fullgraph=True)``.
"""

from functools import partial
from typing import Optional

import pithtrain_ext
import torch
import torch.nn as nn

from pithtrain.dualpipe.utils import WeightGradStore


@torch.library.custom_op("pithtrain::grouped_gemm_forward", mutates_args=())
def _grouped_gemm_forward(
    input: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    return pithtrain_ext.grouped_gemm_forward(input, weight, offs)


@_grouped_gemm_forward.register_fake
def _(input: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    return input.new_empty((input.shape[0], weight.shape[1]))


@torch.library.custom_op("pithtrain::grouped_gemm_dgrad", mutates_args=())
def _grouped_gemm_dgrad(
    grad_output: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    return pithtrain_ext.grouped_gemm_dgrad(grad_output, weight, offs)


@_grouped_gemm_dgrad.register_fake
def _(grad_output: torch.Tensor, weight: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    return grad_output.new_empty((grad_output.shape[0], weight.shape[2]))


@torch.library.custom_op("pithtrain::grouped_gemm_wgrad", mutates_args=())
def _grouped_gemm_wgrad(
    grad_output: torch.Tensor, input: torch.Tensor, offs: torch.Tensor
) -> torch.Tensor:
    return pithtrain_ext.grouped_gemm_wgrad(grad_output, input, offs)


@_grouped_gemm_wgrad.register_fake
def _(grad_output: torch.Tensor, input: torch.Tensor, offs: torch.Tensor) -> torch.Tensor:
    return grad_output.new_empty((offs.shape[0], grad_output.shape[1], input.shape[1]))


class NativeGroupLinearFunc(torch.autograd.Function):
    """
    Custom autograd Function for the BF16 grouped linear layer (MoE experts).

    Forward: output      = grouped_gemm_forward(input, weight)      [jagged on M]
    Dgrad:   grad_input  = grouped_gemm_dgrad(grad_output, weight)  [jagged on M]
    Wgrad:   weight_grad = grouped_gemm_wgrad(grad_output, input)   [jagged on K]

    The wgrad is split from dgrad so DualPipeV's zero-bubble W-phase can defer
    it via WeightGradStore, freeing the critical path during stage3_b.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
    ) -> torch.Tensor:
        output = _grouped_gemm_forward(input, weight, grouped_mm_offs)
        ctx.save_for_backward(input, weight, grouped_mm_offs)
        return output

    @staticmethod
    def backward(ctx, dy):
        input, weight, offs = ctx.saved_tensors
        dgrad = _grouped_gemm_dgrad(dy, weight, offs)

        def wgrad_fn(dy, x, offs):
            wgrad = _grouped_gemm_wgrad(dy, x, offs)
            weight.grad = wgrad if weight.grad is None else weight.grad.add_(wgrad)

        if WeightGradStore.enabled:
            WeightGradStore.put(partial(wgrad_fn, dy.detach(), input.detach(), offs.detach()))
        else:
            wgrad_fn(dy, input, offs)

        return dgrad, None, None


class NativeGroupLinear(nn.Module):
    """
    Grouped linear layer that partitions input data and applies a distinct
    linear transformation per group. This is useful for the MLP layers in
    the mixture-of-experts models.
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
            # Use a matmul instead of new_empty to preserve the autograd graph.
            # With 0 tokens the result is (0, out_features) and gradients are zero,
            # but the grad_fn must exist so that run_backward does not crash.
            return input @ self.weight[0].T
        return NativeGroupLinearFunc.apply(input, self.weight, grouped_mm_offs)
