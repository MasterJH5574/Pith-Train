"""Generic decoder layer and model shared by every supported architecture.

Architecture-specific leaf modules (attention, MLP / MoE / experts / gate /
router, and the rotary embedding) live in the module-type files and are selected
from ``config`` through the module spec (``spec.get_layer_spec`` and
``spec.get_rotary_spec``). Only the layer / model skeleton is defined here.
"""

from dataclasses import fields
from typing import List, Optional

import torch
from torch import nn

from pithtrain.dualpipe.execution import EpilogArgs, IntermediateTensors, PrologArgs, PrologOuts
from pithtrain.dualpipe.layer_partition import layer_partition
from pithtrain.dualpipe.modeling import decoder_layer_backward, decoder_layer_forward
from pithtrain.dualpipe.utils import run_backward
from pithtrain.layers.factory import ModelImplMode
from pithtrain.models.interface import ForwardAttnOutput
from pithtrain.models.spec import build_module, get_layer_spec, get_rotary_spec
from pithtrain.operators.ep_dispatch import moe_ep_prepare_dispatch
from pithtrain.operators.token_scatter import padded_index_gather, scatter_for_grouped_gemm


class TransformerDecoderLayer(nn.Module):
    """Generic decoder layer implementing the DualPipeV 5-stage protocol.

    Attention / MLP / MoE leaf modules are supplied fully-specified through
    ``submodules``; this class owns only the layer skeleton and the stage
    forward / backward entry points.
    """

    def __init__(
        self,
        layer_id: int,
        hidden_size: int,
        rms_norm_eps: float,
        compile_attn: bool,
        submodules=None,
    ):
        super().__init__()
        self.idx = layer_id
        self.self_attn = build_module(submodules["self_attn"])
        self.mlp = build_module(submodules["mlp"])
        self.input_layernorm = build_module(
            submodules["input_layernorm"], hidden_size, eps=rms_norm_eps
        )
        self.post_attention_layernorm = build_module(
            submodules["post_attention_layernorm"], hidden_size, eps=rms_norm_eps
        )

        # deepseek / qwen3 compile the attn stage; ring attention (CP) unwraps it
        # to eager, as does gpt-oss (compile_attn=False).
        use_ring = getattr(self.self_attn, "use_ring_attn", False)
        if not compile_attn or use_ring:
            self._forward_attn_compute = self._forward_attn_compute.__wrapped__.__get__(
                self, type(self)
            )

    @torch.compile(fullgraph=True)
    def _forward_attn_compute(self, hidden_states: torch.Tensor):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        position_embeddings = getattr(self, "_position_embeddings", None)
        if position_embeddings is None:
            raise RuntimeError("Position embeddings must be set before calling forward_attn")

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # Shared experts (deepseek) fold into the residual; a no-op where absent.
        if hasattr(self.mlp, "shared_experts"):
            residual = residual + self.mlp.shared_experts(hidden_states)

        return hidden_states, residual

    def forward_attn(self, hidden_states: torch.Tensor) -> ForwardAttnOutput:
        """LN + Attn + LN + Expert selection."""
        hidden_states, residual = self._forward_attn_compute(hidden_states)

        # Dense (non-MoE) layer: no expert routing / dispatch.
        if not hasattr(self.mlp, "experts"):
            return ForwardAttnOutput(
                hidden_states,  # sorted_tokens
                None,  # moe_local_idxs
                None,  # topk_weight
                None,  # output_splits
                None,  # input_splits
                None,  # expert_idxs
                residual,
            )

        router = getattr(self.mlp, "gate", None) or getattr(self.mlp, "router", None)
        topk_ids, topk_weight = router(hidden_states)
        num_experts = getattr(self.mlp, "num_experts", None)
        if num_experts is None:
            num_experts = self.mlp.n_routed_experts
        (
            sorted_tokens,
            idxs,
            expert_idxs,
            expand_idx,
            dedup_input_splits,
            dedup_output_splits,
            input_splits,
            output_splits,
        ) = moe_ep_prepare_dispatch(
            hidden_states,
            topk_ids,
            num_experts,
            self.mlp.ep_size,
            self.mlp.experts_per_rank,
            self.mlp.ep_group,
        )
        return ForwardAttnOutput(
            sorted_tokens,
            idxs,
            topk_weight,
            output_splits,
            input_splits,
            expert_idxs,
            residual,
            expand_idx,
            dedup_input_splits,
            dedup_output_splits,
        )

    def forward_mlp(
        self,
        gathered_tokens: torch.Tensor,
        expert_idxs: Optional[torch.Tensor] = None,
        expand_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """MLP / Expert forward."""
        if not hasattr(self.mlp, "experts"):
            assert expert_idxs is None
            return self.mlp(gathered_tokens)

        assert expert_idxs is not None
        if expand_idx is not None:
            gathered_tokens = padded_index_gather(gathered_tokens, expand_idx)
        output_tokens, reverse_shuffle_idxs, grouped_mm_offs, ks, ks_tensor = (
            scatter_for_grouped_gemm(gathered_tokens, expert_idxs, self.mlp.experts_per_rank)
        )
        del gathered_tokens  # free expanded tokens; no longer needed after scatter
        outs = self.mlp.experts(output_tokens, grouped_mm_offs, ks=ks, ks_tensor=ks_tensor)
        outs = padded_index_gather(outs, reverse_shuffle_idxs)
        return outs

    @torch.compile(fullgraph=True)
    def forward_aggregate(
        self,
        moe_outs: torch.Tensor,
        moe_local_idxs: Optional[torch.Tensor],
        topk_weight: Optional[torch.Tensor],
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """Weighted expert output + residual connection.

        Shared expert output is already folded into ``residual`` by forward_attn.
        """
        if hasattr(self.mlp, "experts"):
            if self.mlp.ep_size > 1:
                assert moe_local_idxs is not None
                seq_len, topk = topk_weight.shape
                # Memory-efficient equivalent of
                # new_x[moe_local_idxs] = moe_outs followed by weighted sum.
                permuted_probs = topk_weight.view(-1)[moe_local_idxs]
                token_indices = moe_local_idxs // topk
                weighted = (moe_outs.float() * permuted_probs.unsqueeze(-1)).to(moe_outs.dtype)
                hidden_states = moe_outs.new_zeros(seq_len, moe_outs.shape[-1])
                hidden_states.scatter_add_(0, token_indices[:, None].expand_as(weighted), weighted)
                hidden_states = hidden_states.view(*residual.shape)
            else:
                assert moe_local_idxs is None
                new_x = moe_outs
                final_out = new_x.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(dim=-1)
                final_out = final_out.sum(dim=1).to(new_x.dtype)
                hidden_states = final_out.view(*residual.shape)
        else:
            assert moe_local_idxs is None
            assert topk_weight is None
            hidden_states = moe_outs

        hidden_states = residual + hidden_states
        return hidden_states

    def reference_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Standard (non-pipelined) forward for correctness validation."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        position_embeddings = getattr(self, "_position_embeddings", None)
        if position_embeddings is None:
            raise RuntimeError("Position embeddings must be set before calling reference_forward")

        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class TransformerModel(nn.Module):
    """Generic pipeline-stage model shared by every supported architecture.

    Holds ``embed_tokens`` (stage 0), the local slice of decoder layers, the
    final ``norm`` + ``lm_head`` (last stage), and a spec-selected ``rotary_emb``.
    Layer and rotary construction are derived from ``config``.
    """

    def __init__(
        self,
        config,
        num_stages: int,
        stage_id: int,
        ep_group=None,
        cp_group=None,
    ):
        super().__init__()
        self.config = config
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.cp_group = cp_group
        self.cp_rank = cp_group.rank() if cp_group is not None else 0
        self.cp_size = cp_group.size() if cp_group is not None else 1

        self.embed_tokens = (
            nn.Embedding(config.vocab_size, config.hidden_size) if stage_id == 0 else None
        )

        # Distribute decoder layers across pipeline stages; edge stages (holding
        # embed_tokens / norm+lm_head) get fewer layers to balance memory.
        num_local_layers = layer_partition(config.num_hidden_layers, num_stages)
        layer_id_begin = sum(num_local_layers[:stage_id])
        layer_id_end = layer_id_begin + num_local_layers[stage_id]
        self.layers = nn.ModuleDict(
            {
                str(i): build_module(get_layer_spec(config, i, ep_group, cp_group))
                for i in range(layer_id_begin, layer_id_end)
            }
        )

        if stage_id == num_stages - 1:
            self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.norm = None
            self.lm_head = None

        self.rotary_emb = build_module(get_rotary_spec(config))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pre-allocated intermediate_tensors from module attribute (set by DualPipeV).
        intermediate_tensors: Optional[IntermediateTensors] = getattr(
            self, "_intermediate_tensors", None
        )

        if self.embed_tokens is not None:
            hidden_states = self.embed_tokens(hidden_states)

        seq_len = hidden_states.shape[1]
        # Zigzag CP layout: local tokens come from two non-contiguous global chunks;
        # build the global position IDs from the front and mirror back blocks, then
        # gather cos/sin. At cp_size == 1 this reduces to arange(0, seq_len).
        block = seq_len // 2
        global_seq_len = seq_len * self.cp_size
        front_start = self.cp_rank * block
        back_start = (2 * self.cp_size - self.cp_rank - 1) * block
        position_ids = torch.cat(
            [
                torch.arange(front_start, front_start + block, device=hidden_states.device),
                torch.arange(back_start, back_start + block, device=hidden_states.device),
            ]
        )
        cos, sin = self.rotary_emb(hidden_states, seq_len=global_seq_len)
        position_embeddings = (
            cos[position_ids].unsqueeze(0).to(dtype=hidden_states.dtype),
            sin[position_ids].unsqueeze(0).to(dtype=hidden_states.dtype),
        )
        for layer in self.layers.values():
            layer._position_embeddings = position_embeddings

        if intermediate_tensors is None:
            for _, layer in self.layers.items():
                ret = decoder_layer_forward(layer, hidden_states)
                hidden_states = ret[0] if isinstance(ret, tuple) else ret
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)
                hidden_states = self.lm_head(hidden_states)
            return hidden_states

        layer_idx = 0
        if self.embed_tokens is not None:
            intermediate_tensors.prolog.args = PrologArgs()
            intermediate_tensors.prolog.outs = PrologOuts(hidden_states)
        for _, layer in self.layers.items():
            ret = decoder_layer_forward(layer, hidden_states)
            if len(ret) == 2:
                hidden_states, layer_record = ret
                # Copy into pre-allocated slot.
                dst = intermediate_tensors.layers[layer_idx]
                for field in fields(layer_record):
                    src_rec = getattr(layer_record, field.name)
                    dst_rec = getattr(dst, field.name)
                    for rf in fields(src_rec):
                        setattr(dst_rec, rf.name, getattr(src_rec, rf.name))
            else:
                hidden_states = ret[0]
                # Clear pre-allocated slot (layer didn't produce intermediate).
                dst = intermediate_tensors.layers[layer_idx]
                for field in fields(dst):
                    record = getattr(dst, field.name)
                    for rf in fields(record):
                        setattr(record, rf.name, None)
            layer_idx += 1

        if self.norm is not None:
            assert self.lm_head is not None
            if not ModelImplMode.use_reference_fwd:
                hidden_states = hidden_states.detach().requires_grad_()
            intermediate_tensors.epilog.args = EpilogArgs(hidden_states)
            hidden_states = self.norm(hidden_states)
            hidden_states = self.lm_head(hidden_states)

        return hidden_states

    @staticmethod
    def backward(
        module: "TransformerModel",
        dy: Optional[List[torch.Tensor]],
        loss: Optional[torch.Tensor],
        intermediate_tensors: IntermediateTensors,
    ):
        assert (dy is None) != (loss is None), "Either dy or loss should be provided"
        if loss is not None:
            assert module.norm is not None
            assert module.lm_head is not None
            loss.backward()
            loss.detach_()
            dy = (intermediate_tensors.epilog.args.hidden_states.grad,)
            # Clear tensor refs but keep pre-allocated record.
            intermediate_tensors.epilog.args = None
            loss = None
        else:
            assert module.norm is None
            assert module.lm_head is None

        dx = dy
        layers_list = [layer for _, layer in module.layers.items()]
        for layer, intermediate_tensors_layer in zip(
            reversed(layers_list), reversed(intermediate_tensors.layers)
        ):
            dx = (decoder_layer_backward(layer, dx, loss, intermediate_tensors_layer),)

        final_grads = dx
        if module.embed_tokens is not None:
            record = intermediate_tensors.prolog
            run_backward(record.outs, dx)
            # Clear tensor refs but keep pre-allocated record.
            for rf in fields(record):
                setattr(record, rf.name, None)
            final_grads = (None,)
        return final_grads
