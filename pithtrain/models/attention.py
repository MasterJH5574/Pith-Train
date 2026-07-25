"""Dense-attention variants (grouped-query, attention-sink) and rope helpers."""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
from flash_attn.cute.interface import flash_attn_func as flash_attn_cute_func
from torch import nn

from pithtrain.models.spec import build_module
from pithtrain.operators.flash_attn_v4 import flash_attn_func
from pithtrain.operators.ring_attention.standard import ring_attention_func


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape [batch, seq_len, num_heads, head_dim].
    k : torch.Tensor
        Key tensor of shape [batch, seq_len, num_kv_heads, head_dim].
    cos : torch.Tensor
        Cosine embedding of shape [batch, seq_len, head_dim].
    sin : torch.Tensor
        Sine embedding of shape [batch, seq_len, head_dim].

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Rotated query and key tensors.
    """
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GQASelfAttention(nn.Module):
    """Grouped Query Attention using Flash Attention."""

    def __init__(
        self,
        config,
        cp_group: Optional[dist.ProcessGroup] = None,
        submodules=None,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.cp_group = cp_group
        self.use_ring_attn = cp_group is not None and cp_group.size() > 1
        self._disable_ring_attn = False

        attention_bias = getattr(config, "attention_bias", False)
        self.q_proj = build_module(
            submodules["q_proj"],
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=attention_bias,
        )
        self.k_proj = build_module(
            submodules["k_proj"],
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=attention_bias,
        )
        self.v_proj = build_module(
            submodules["v_proj"],
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=attention_bias,
        )
        self.o_proj = build_module(
            submodules["o_proj"],
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=attention_bias,
        )
        self.q_norm = build_module(submodules["q_norm"], self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = build_module(submodules["k_norm"], self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Forward pass for GQA attention.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor of shape [batch, seq_len, hidden_size].
        position_embeddings : Tuple[torch.Tensor, torch.Tensor]
            Tuple of (cos, sin) for rotary embeddings.

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch, seq_len, hidden_size].
        """
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_kv_heads, self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if not self.use_ring_attn or self._disable_ring_attn:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                softmax_scale=self.scaling,
                causal=True,
            )
        else:
            attn_output = ring_attention_func(
                query_states,
                key_states,
                value_states,
                softmax_scale=self.scaling,
                cp_group=self.cp_group,
            )

        attn_output = attn_output.reshape(bsz, seq_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output


class SinkSelfAttention(nn.Module):
    """
    Grouped Query Attention with attention sinks and optional sliding window.

    Backed by FlashAttention-4 (CUTE DSL).  learnable_sink is a per-head
    scalar fused into the softmax denominator inside the kernel — letting a
    head "dump" attention mass to the sink and produce near-zero attention to
    real tokens.  Causal + sliding window + GQA + per-head sink are all
    native kwargs, so attention runs in a single FA-4 kernel call.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_bias: bool = True,
        is_sliding: bool = False,
        sliding_window: int = 128,
        submodules=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        self.is_sliding = is_sliding
        self.sliding_window = sliding_window

        self.q_proj = build_module(
            submodules["q_proj"], hidden_size, num_attention_heads * head_dim, bias=attention_bias
        )
        self.k_proj = build_module(
            submodules["k_proj"], hidden_size, num_key_value_heads * head_dim, bias=attention_bias
        )
        self.v_proj = build_module(
            submodules["v_proj"], hidden_size, num_key_value_heads * head_dim, bias=attention_bias
        )
        self.o_proj = build_module(
            submodules["o_proj"], num_attention_heads * head_dim, hidden_size, bias=attention_bias
        )

        self.sinks = nn.Parameter(torch.zeros(num_attention_heads))

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(
            bsz, seq_len, self.num_kv_heads, self.head_dim
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # FA-4 expects (B, S, H, D); GQA is auto-detected from H_q vs H_kv.
        # Sliding window: (W-1, 0) means each query attends to W tokens (self
        # + W-1 prior).
        window_size: Tuple[Optional[int], Optional[int]] = (
            (self.sliding_window - 1, 0) if self.is_sliding else (None, None)
        )
        # FA-4 requires learnable_sink to match q/k/v dtype; the parameter
        # itself stays in fp32 for optimizer numerical stability.
        # flash_attn_func returns (out, lse); we only need out.
        attn_output, _ = flash_attn_cute_func(
            query_states,
            key_states,
            value_states,
            softmax_scale=self.scaling,
            causal=True,
            window_size=window_size,
            learnable_sink=self.sinks.to(query_states.dtype),
        )

        attn_output = attn_output.reshape(bsz, seq_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output
