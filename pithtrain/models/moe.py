"""Mixture-of-Experts blocks, expert banks, and routing gates."""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from pithtrain.dualpipe.utils import FP8WeightCacheControl
from pithtrain.layers.deepgemm_fp8_linear import FP8GroupLinearFunc
from pithtrain.layers.factory import ModelImplMode
from pithtrain.layers.group_linear import GroupLinearFunc
from pithtrain.models.spec import build_module
from pithtrain.modules.load_balance import MoELoadBalanceLossInjector, MoELoadBalanceLossTracker
from pithtrain.operators.clamped_swiglu import clamped_swiglu
from pithtrain.operators.deepgemm_fp8_quantize import fused_blockwise_transpose_cast_to_fp8_batched
from pithtrain.operators.indexed_bias_add import indexed_bias_add
from pithtrain.operators.silu_mul import silu_mul
from pithtrain.operators.token_scatter import (
    precompute_group_indices,
    scatter_for_grouped_gemm,
)

torch._dynamo.allow_in_graph(MoELoadBalanceLossInjector)

SWIGLU_ALPHA = 1.702  # sigmoid approximation of GELU.


class ScaledTopKGate(nn.Module):
    """Top-K routing gate with softmax scores and routed scaling."""

    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.num_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.load_balance_loss_fn = None
        self.weight = nn.Parameter(
            torch.empty((self.n_routed_experts, config.hidden_size)), requires_grad=True
        )

    @torch.compile(fullgraph=True)
    def compute(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Gate math + lb_loss injection (compiled).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
            topk_idx, topk_weight, lb_loss (None when not training or no loss fn).
        """
        _, _, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32), None)
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = topk_weight * self.routed_scaling_factor

        if self.training and self.load_balance_loss_fn is not None:
            lb_loss = self.load_balance_loss_fn(scores, topk_idx, self.n_routed_experts, self.top_k)
            topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss)
        else:
            lb_loss = None

        return topk_idx, topk_weight, lb_loss

    def forward(self, hidden_states):
        topk_idx, topk_weight, lb_loss = self.compute(hidden_states)

        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)

        return topk_idx, topk_weight


class TopKGate(nn.Module):
    """Top-K routing gate with softmax normalization."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)
        self.load_balance_loss_fn = None
        self.weight = nn.Parameter(
            torch.empty((self.num_experts, config.hidden_size)), requires_grad=True
        )

    @torch.compile(fullgraph=True)
    def compute(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Gate math + lb_loss injection (compiled).

        Includes linear + softmax + topk + normalize + load-balance loss
        computation + injection. Only MoELoadBalanceLossTracker.add() (a
        class-level side effect) stays outside in forward().

        Note: norm_topk_prob is applied before lb_loss injection. This is
        safe because MoELoadBalanceLossInjector is identity in forward and
        ones_like(lb_loss) in backward - gradient on topk_weight is unchanged.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]
            topk_idx, topk_weight, lb_loss (None when not training or no loss fn).
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        logits = F.linear(hidden_states, self.weight, None)
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.num_experts_per_tok, dim=-1, sorted=False)

        if self.norm_topk_prob:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

        if self.training and self.load_balance_loss_fn is not None:
            lb_loss = self.load_balance_loss_fn(
                scores, topk_idx, self.num_experts, self.num_experts_per_tok
            )
            topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss)
        else:
            lb_loss = None

        return topk_idx, topk_weight, lb_loss

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute routing weights and expert indices.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor of shape [batch, seq_len, hidden_size].

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            topk_idx: Expert indices of shape [batch*seq_len, num_experts_per_tok].
            topk_weight: Routing weights of shape [batch*seq_len, num_experts_per_tok].
        """
        topk_idx, topk_weight, lb_loss = self.compute(hidden_states)

        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)

        return topk_idx, topk_weight


class TopKRouter(nn.Module):
    """Top-K routing gate with post-softmax normalization and bias."""

    def __init__(self, hidden_size: int, num_experts: int, num_experts_per_tok: int):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.load_balance_loss_fn = None
        self.weight = nn.Parameter(torch.empty((num_experts, hidden_size)), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_experts))

    @torch.compile(fullgraph=True)
    def compute(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)

        logits = F.linear(hidden_states, self.weight, self.bias)

        topk_logits, topk_idx = torch.topk(logits, k=self.num_experts_per_tok, dim=-1, sorted=True)
        topk_weight = F.softmax(topk_logits, dim=-1, dtype=torch.float32)

        if self.training and self.load_balance_loss_fn is not None:
            scores = logits.softmax(dim=-1, dtype=torch.float32)
            lb_loss = self.load_balance_loss_fn(
                scores, topk_idx, self.num_experts, self.num_experts_per_tok
            )
            topk_weight = MoELoadBalanceLossInjector.apply(topk_weight, lb_loss)
        else:
            lb_loss = None

        return topk_idx, topk_weight, lb_loss

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        topk_idx, topk_weight, lb_loss = self.compute(hidden_states)
        if lb_loss is not None:
            MoELoadBalanceLossTracker.add(lb_loss)
        return topk_idx, topk_weight


class GroupedExperts(nn.Module):
    """Expert bank using grouped linear projections."""

    def __init__(
        self,
        config,
        num_experts: int,
        hidden_size: Optional[int] = None,
        moe_intermediate_size: Optional[int] = None,
        submodules=None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size or config.hidden_size
        self.moe_intermediate_size = moe_intermediate_size or config.moe_intermediate_size

        self.gate_proj = build_module(
            submodules["gate_proj"], num_experts, self.hidden_size, self.moe_intermediate_size
        )
        self.up_proj = build_module(
            submodules["up_proj"], num_experts, self.hidden_size, self.moe_intermediate_size
        )
        self.down_proj = build_module(
            submodules["down_proj"], num_experts, self.moe_intermediate_size, self.hidden_size
        )

    def forward(
        self,
        x: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list | None = None,
        ks_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gi = precompute_group_indices(grouped_mm_offs, x.shape[0])
        kwargs = dict(grouped_mm_offs=grouped_mm_offs, ks=ks, ks_tensor=ks_tensor, group_indices=gi)
        g = self.gate_proj(x, **kwargs)
        u = self.up_proj(x, **kwargs)
        return self.down_proj(silu_mul(g, u), **kwargs)


class FusedExperts(nn.Module):
    """Expert FFN with clamped SwiGLU and per-expert bias on fused weights."""

    def __init__(
        self, num_experts: int, hidden_size: int, intermediate_size: int, swiglu_limit: float
    ):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.swiglu_limit = swiglu_limit
        self.gate_up_proj = nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size)
        )
        self.gate_up_proj_bias = nn.Parameter(torch.zeros(num_experts, 2 * intermediate_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.down_proj_bias = nn.Parameter(torch.zeros(num_experts, hidden_size))

        # Expert projections are raw nn.Parameter (fused gate_up), so the
        # FP8GroupLinear module wrapper does not apply. FP8GroupLinearFunc is
        # dispatched directly on these parameters and the quantized-weight cache
        # is hosted here. Cache is dict-or-None so DualPipeV's
        # FP8WeightCacheControl.clear_caches (which sets _wq_cache=None) works.
        self._fp8 = ModelImplMode.fp8_training == "deep-gemm"
        self._wq_cache: dict[str, tuple] | None = None
        self._wq_version: int = -1

    def _quantized_weight(self, name: str, weight: torch.Tensor) -> tuple:
        if torch.compiler.is_compiling():
            return fused_blockwise_transpose_cast_to_fp8_batched(weight)
        ver = FP8WeightCacheControl._version
        cache = self._wq_cache
        if FP8WeightCacheControl.enabled and self._wq_version == ver and cache is not None:
            hit = cache.get(name)
            if hit is not None:
                return hit
        result = fused_blockwise_transpose_cast_to_fp8_batched(weight)
        if FP8WeightCacheControl.enabled:
            if self._wq_version != ver or cache is None:
                self._wq_cache = {name: result}
                self._wq_version = ver
            else:
                cache[name] = result
        return result

    def _group_linear(
        self,
        x: torch.Tensor,
        weight: nn.Parameter,
        name: str,
        offs: torch.Tensor,
        ks: list | None,
        ks_tensor: torch.Tensor | None,
        group_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if x.shape[0] == 0:
            return x @ weight[0].transpose(-2, -1)
        if self._fp8:
            return FP8GroupLinearFunc.apply(
                x, weight, offs, ks, ks_tensor, self._quantized_weight(name, weight), group_indices
            )
        return GroupLinearFunc.apply(x, weight, offs)

    def forward(
        self,
        x: torch.Tensor,
        grouped_mm_offs: torch.Tensor,
        ks: list | None = None,
        ks_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        group_ids = torch.searchsorted(
            grouped_mm_offs.to(torch.int64),
            torch.arange(x.shape[0], device=x.device, dtype=torch.int64),
            right=True,
        ).clamp_(max=self.num_experts - 1)

        # Hopper SM90 needs explicit per-row group indices for m_grouped FP8 GEMM;
        # Blackwell ignores it. Computed once and shared across both projections.
        gi = precompute_group_indices(grouped_mm_offs, x.shape[0]) if self._fp8 else None

        gate_up = self._group_linear(
            x, self.gate_up_proj, "gate_up_proj", grouped_mm_offs, ks, ks_tensor, gi
        )
        gate_up = indexed_bias_add(gate_up, self.gate_up_proj_bias, group_ids, grouped_mm_offs)
        activated = clamped_swiglu(gate_up, SWIGLU_ALPHA, self.swiglu_limit)

        out = self._group_linear(
            activated, self.down_proj, "down_proj", grouped_mm_offs, ks, ks_tensor, gi
        )
        out = indexed_bias_add(out, self.down_proj_bias, group_ids, grouped_mm_offs)
        return out


class SharedExpertMoE(nn.Module):
    """MoE block with routed experts plus an always-on shared expert."""

    def __init__(
        self,
        config,
        ep_group: Optional[dist.ProcessGroup] = None,
        layer_id: int = 0,
        submodules=None,
    ):
        super().__init__()
        self.config = config
        self.ep_group = ep_group
        self.num_experts_per_tok = config.num_experts_per_tok
        self.ep_size = getattr(config, "ep_size", 1)
        self.ep_rank = ep_group.rank() if ep_group is not None else 0
        self.experts_per_rank = config.n_routed_experts // self.ep_size
        self.n_routed_experts = config.n_routed_experts

        self.experts = build_module(
            submodules["experts"],
            config,
            self.experts_per_rank,
            moe_intermediate_size=config.moe_intermediate_size,
        )
        self.gate = build_module(submodules["gate"], config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = build_module(
                submodules["shared_experts"], config, intermediate_size=intermediate_size
            )

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    def moe_infer(self, x, topk_ids, topk_weight):
        assert self.ep_size == 1, "reference implementation only supports ep_size=1"
        expert_idxs = topk_ids.view(-1)
        sorted_tokens = (
            x.unsqueeze(1).expand(-1, self.num_experts_per_tok, -1).reshape(-1, x.shape[-1])
        )
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = (
            scatter_for_grouped_gemm(sorted_tokens, expert_idxs, self.experts_per_rank)
        )
        outs = self.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = outs[reverse_shuffle_idxs]

        final_out = (
            (outs.view(*topk_ids.shape, -1) * topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .to(outs.dtype)
        )
        return final_out


class GroupedMoE(nn.Module):
    """MoE block with expert parallelism support."""

    def __init__(
        self,
        config,
        ep_group: Optional[dist.ProcessGroup] = None,
        layer_id: int = 0,
        submodules=None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size

        self.ep_size = getattr(config, "ep_size", 1)
        self.ep_group = ep_group
        self.ep_rank = ep_group.rank() if ep_group is not None else 0
        self.experts_per_rank = self.num_experts // self.ep_size

        self.experts = build_module(
            submodules["experts"],
            config,
            self.experts_per_rank,
            moe_intermediate_size=config.moe_intermediate_size,
        )
        self.gate = build_module(submodules["gate"], config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        return y

    def moe_infer(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        """MoE inference with grouped GEMM."""
        assert self.ep_size == 1, "Reference implementation only supports ep_size=1"
        expert_idxs = topk_ids.view(-1)
        sorted_tokens = (
            x.unsqueeze(1).expand(-1, self.num_experts_per_tok, -1).reshape(-1, x.shape[-1])
        )
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = (
            scatter_for_grouped_gemm(sorted_tokens, expert_idxs, self.experts_per_rank)
        )
        outs = self.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = outs[reverse_shuffle_idxs]

        final_out = (
            (outs.view(*topk_ids.shape, -1) * topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .to(outs.dtype)
        )
        return final_out


class FusedMoE(nn.Module):
    """MoE block with fused-weight experts and expert parallelism support."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        intermediate_size: int,
        swiglu_limit: float,
        ep_size: int = 1,
        ep_group: Optional[dist.ProcessGroup] = None,
        submodules=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        self.ep_size = ep_size
        self.ep_group = ep_group
        self.ep_rank = ep_group.rank() if ep_group is not None else 0
        self.experts_per_rank = num_experts // ep_size

        self.experts = build_module(
            submodules["experts"],
            self.experts_per_rank,
            hidden_size,
            intermediate_size,
            swiglu_limit,
        )
        self.router = build_module(
            submodules["router"], hidden_size, num_experts, num_experts_per_tok
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.router(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        return y

    def moe_infer(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        assert self.ep_size == 1, "Reference implementation only supports ep_size=1"
        expert_idxs = topk_ids.view(-1)
        sorted_tokens = (
            x.unsqueeze(1).expand(-1, self.num_experts_per_tok, -1).reshape(-1, x.shape[-1])
        )
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = (
            scatter_for_grouped_gemm(sorted_tokens, expert_idxs, self.experts_per_rank)
        )
        outs = self.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = outs[reverse_shuffle_idxs]

        final_out = (
            (outs.view(*topk_ids.shape, -1) * topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .to(outs.dtype)
        )
        return final_out
