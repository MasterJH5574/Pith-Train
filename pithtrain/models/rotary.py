"""Rotary position embeddings and YaRN scaling helpers."""

import math
from typing import Optional, Tuple

import torch
from torch import nn


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class YarnRotaryEmbedding(RotaryEmbedding):
    """YaRN-scaled rotary embedding with dual-mscale concentration."""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * _mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * _mscale).to(dtype), persistent=False)


class DynamicRotaryEmbedding(nn.Module):
    """Standard rotary embedding that grows its cos/sin cache on demand."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 40960,
        base: float = 1000000.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, dtype=torch.float32, device=device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device], dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the cos/sin cache up to ``seq_len`` positions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, used only for dtype.
        seq_len : int
            Required cache length (global, i.e. includes CP offset).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Cosine and sine tables of shape ``[seq_len, head_dim]``.
        """
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def _yarn_find_correction_dim(
    num_rotations: float, dim: int, base: float, max_position_embeddings: int
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


def _yarn_find_correction_range(
    low_rot: float,
    high_rot: float,
    dim: int,
    base: float,
    max_position_embeddings: int,
    truncate: bool,
) -> Tuple[float, float]:
    low = _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


class ScaledRotaryEmbedding(nn.Module):
    """YaRN-scaled rotary embedding with a single concentration factor."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 150000.0,
        scaling_factor: float = 32.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        truncate: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.truncate = truncate
        self._set_cos_sin_cache(max_position_embeddings, device, torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len: int, device: Optional[torch.device], dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
            self.truncate,
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Concentration factor (mscale).
        concentration = 0.1 * math.log(self.scaling_factor) + 1.0

        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", (emb.cos() * concentration).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * concentration).to(dtype), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len, x.device, x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
