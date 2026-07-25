"""Multi-head Latent Attention (MLA) and its interleaved rope apply."""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from pithtrain.models.attention import rotate_half
from pithtrain.models.spec import build_module
from pithtrain.operators.flash_attn_v4 import mla_flash_attn_func
from pithtrain.operators.ring_attention.standard import ring_attention_func


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MLASelfAttention(nn.Module):
    """Multi-head Latent Attention with low-rank KV compression."""

    def __init__(
        self,
        config,
        layer_id: int = 0,
        cp_group: Optional[dist.ProcessGroup] = None,
        submodules=None,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.cp_group = cp_group
        self.use_ring_attn = cp_group is not None and cp_group.size() > 1
        self._disable_ring_attn = False

        self.q_proj = build_module(
            submodules["q_proj"], self.hidden_size, self.num_heads * self.q_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = build_module(
            submodules["kv_a_proj_with_mqa"],
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = build_module(
            submodules["kv_a_layernorm"], config.kv_lora_rank, eps=config.rms_norm_eps
        )
        self.kv_b_proj = build_module(
            submodules["kv_b_proj"],
            config.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = build_module(
            submodules["o_proj"], self.num_heads * self.v_head_dim, self.hidden_size, bias=False
        )
        self.softmax_scale = self.q_head_dim ** (-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim)
        kv = self.kv_b_proj(self.kv_a_layernorm(compressed_kv)).view(
            bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )

        k_nope, value_states = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        cos, sin = position_embeddings
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, unsqueeze_dim=2)

        if self.use_ring_attn and not self._disable_ring_attn:
            query_states = torch.cat([q_nope, q_pe], dim=-1)
            key_states = torch.cat([k_nope, k_pe.expand(-1, -1, self.num_heads, -1)], dim=-1)
            attn_output = ring_attention_func(
                query_states,
                key_states,
                value_states.contiguous(),
                softmax_scale=self.softmax_scale,
                cp_group=self.cp_group,
            )
        else:
            attn_output = mla_flash_attn_func(
                q_nope,
                q_pe,
                k_nope,
                k_pe,
                value_states,
                softmax_scale=self.softmax_scale,
                qk_nope_head_dim=self.qk_nope_head_dim,
                causal=True,
            )

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output
