from typing import Dict, List, NamedTuple, Optional, Protocol

import torch
import torch.nn as nn

from pithtrain.operators.ep_backend import is_deepep_backend


def uses_deepep_dispatch(layer) -> bool:
    """True iff this layer's a2a goes through deepep (vs direct_all_to_all)."""
    return is_deepep_backend(getattr(layer.mlp, "ep_backend", None)) and layer.mlp.ep_size > 1


def needs_ep_prepare_dispatch(layer) -> bool:
    """True iff the torch all-to-all path is in use (deepep handles dedup
    inside buf.dispatch and skips moe_ep_prepare_dispatch)."""
    return not uses_deepep_dispatch(layer)


class ForwardAttnOutput(NamedTuple):
    """Output from forward_attn. The deepep backend uses hidden_states_post_attn
    + topk_ids; the torch backend uses sorted_tokens + splits."""

    sorted_tokens: torch.Tensor
    moe_local_idxs: torch.Tensor
    topk_weight: torch.Tensor
    output_splits: List[int]
    input_splits: List[int]
    expert_idxs: torch.Tensor
    residual: torch.Tensor
    expand_idx: Optional[torch.Tensor] = None
    dedup_input_splits: Optional[List[int]] = None
    dedup_output_splits: Optional[List[int]] = None
    hidden_states_post_attn: Optional[torch.Tensor] = None
    topk_ids: Optional[torch.Tensor] = None


class DecoderLayerMlpProtocol(Protocol):
    """
    Protocol for the MLP component of a decoder layer in DualPipeV.

    If the experts attribute is present, we treat the MLP as a MoE layer.
    """

    ep_size: int
    ep_group: torch.distributed.ProcessGroup


class DecoderLayerProtocol(Protocol):
    """Protocol for a decoder layer in DualPipeV."""

    idx: int
    mlp: DecoderLayerMlpProtocol

    def reference_forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Reference forward implementation for correctness validation."""

    def forward_attn(
        self,
        hidden_states: torch.Tensor,
    ) -> ForwardAttnOutput:
        """LN + Attn + LN + Expert selection."""

    def forward_mlp(
        self,
        gathered_tokens: torch.Tensor,
        expert_idxs: Optional[torch.Tensor] = None,
        expand_idx: Optional[torch.Tensor] = None,
        dispatch_state: Optional[dict] = None,
    ) -> torch.Tensor:
        """MLP forward."""

    def forward_aggregate(
        self,
        moe_outs: torch.Tensor,
        moe_local_idxs: Optional[torch.Tensor],
        topk_weight: Optional[torch.Tensor],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted expert output + residual connection."""


class ModelProtocol(Protocol):
    """Protocol for the DualPipeV model."""

    embed_tokens: Optional[nn.Module]
    norm: Optional[nn.Module]
    lm_head: Optional[nn.Module]
    layers: Dict[str, DecoderLayerProtocol]
