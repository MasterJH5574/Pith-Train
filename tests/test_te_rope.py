"""
Correctness of the TransformerEngine fused rotary embedding used by
DeepSeek-V2-Lite's MLA against the DeepSeek Python rope it replaces.

DeepSeek MLA rope de-interleaves the pe head dim (GPT-J pairs) and applies a
NeoX rotation; TE's fused rope applies the identical rotation with
``interleaved=True``, keeping the interleaved output layout. The two are equal
up to that pair permutation, so:

* the TE output, de-interleaved, matches the reference element-wise, and
* the attention scores ``q_pe . k_pe`` (permutation-invariant) match directly.

BF16 fused math differs slightly from the reference, so a 1e-2 relative error
threshold is used.
"""

import pytest
import torch

pytest.importorskip("transformer_engine.pytorch")
from transformer_engine.pytorch.attention.rope import (  # noqa: E402
    apply_rotary_pos_emb as te_apply_rotary_pos_emb,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# DeepSeek-V2-Lite MLA rope geometry.
NUM_HEADS = 16
ROPE_DIM = 64
CASES = [(1, 64), (2, 128), (4, 96)]


def rel_err(actual: torch.Tensor, ref: torch.Tensor) -> float:
    actual, ref = actual.float(), ref.float()
    return ((actual - ref).pow(2).mean() / ref.pow(2).mean().clamp_min(1e-12)).sqrt().item()


def _deinterleave(x: torch.Tensor) -> torch.Tensor:
    """Interleaved (GPT-J) -> blocked (even components first, then odd)."""
    b, s, h, d = x.shape
    return x.view(b, s, h, d // 2, 2).transpose(-1, -2).reshape(b, s, h, d)


def _ref_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """DeepSeek Python MLA rope: de-interleave then NeoX rotation. x: [b,s,h,d]."""
    cos = cos.unsqueeze(2)  # [1,s,1,d]
    sin = sin.unsqueeze(2)
    x = _deinterleave(x)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    rot = torch.cat((-x2, x1), dim=-1)
    return x * cos + rot * sin


def _freqs_and_cossin(seq: int, dev, dt):
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, ROPE_DIM, 2, device=dev).float() / ROPE_DIM))
    t = torch.arange(seq, device=dev, dtype=torch.float32)
    angles = torch.outer(t, inv_freq)  # [seq, ROPE_DIM//2]
    # TE (interleaved): [a0,a0,a1,a1,...]; reference cos/sin (NeoX): cat([a,a]).
    freqs = torch.stack((angles, angles), dim=-1).reshape(seq, 1, 1, ROPE_DIM)
    emb = torch.cat((angles, angles), dim=-1)
    cos = emb.cos().to(dt).unsqueeze(0)  # [1,seq,ROPE_DIM]
    sin = emb.sin().to(dt).unsqueeze(0)
    return freqs, cos, sin


@requires_cuda
@pytest.mark.parametrize("B,S", CASES)
def test_te_rope_forward_vs_reference(B, S):
    dev, dt = torch.device("cuda"), torch.bfloat16
    q = torch.randn(B, S, NUM_HEADS, ROPE_DIM, device=dev, dtype=dt)
    k = torch.randn(B, S, 1, ROPE_DIM, device=dev, dtype=dt)  # MLA rope key: single head
    freqs, cos, sin = _freqs_and_cossin(S, dev, dt)

    q_te = te_apply_rotary_pos_emb(q, freqs, tensor_format="bshd", interleaved=True, fused=True)
    k_te = te_apply_rotary_pos_emb(k, freqs, tensor_format="bshd", interleaved=True, fused=True)
    q_ref = _ref_rope(q, cos, sin)
    k_ref = _ref_rope(k, cos, sin)

    assert torch.isfinite(q_te).all() and torch.isfinite(k_te).all()
    # element-wise match after removing the interleave permutation
    assert rel_err(_deinterleave(q_te), q_ref) < 1e-2, f"q fwd B={B} S={S}"
    assert rel_err(_deinterleave(k_te), k_ref) < 1e-2, f"k fwd B={B} S={S}"

    # attention scores (permutation-invariant) match directly
    score_te = torch.einsum(
        "bshd,bszd->bshz", q_te.float(), k_te.expand(-1, -1, NUM_HEADS, -1).float()
    )
    score_ref = torch.einsum(
        "bshd,bszd->bshz", q_ref.float(), k_ref.expand(-1, -1, NUM_HEADS, -1).float()
    )
    assert rel_err(score_te, score_ref) < 1e-2, f"score B={B} S={S}"


@requires_cuda
@pytest.mark.parametrize("B,S", [(2, 128), (1, 256)])
def test_te_rope_backward_vs_reference(B, S):
    """Gradients w.r.t. the pe inputs match through a permutation-invariant loss."""
    dev, dt = torch.device("cuda"), torch.bfloat16
    q0 = torch.randn(B, S, NUM_HEADS, ROPE_DIM, device=dev, dtype=dt)
    k0 = torch.randn(B, S, 1, ROPE_DIM, device=dev, dtype=dt)
    freqs, cos, sin = _freqs_and_cossin(S, dev, dt)

    def score_loss(q, k, fn):
        qr = fn(q)
        kr = fn(k)
        s = torch.einsum("bshd,bszd->bshz", qr.float(), kr.expand(-1, -1, NUM_HEADS, -1).float())
        return s.sum()

    q_te = q0.clone().requires_grad_(True)
    k_te = k0.clone().requires_grad_(True)
    score_loss(
        q_te,
        k_te,
        lambda x: te_apply_rotary_pos_emb(
            x, freqs, tensor_format="bshd", interleaved=True, fused=True
        ),
    ).backward()

    q_rf = q0.clone().requires_grad_(True)
    k_rf = k0.clone().requires_grad_(True)
    score_loss(q_rf, k_rf, lambda x: _ref_rope(x, cos, sin)).backward()

    for name, g in (("dq", q_te.grad), ("dk", k_te.grad)):
        assert torch.isfinite(g).all(), f"{name} non-finite B={B} S={S}"
    assert rel_err(q_te.grad, q_rf.grad) < 1e-2, f"dq B={B} S={S}"
    assert rel_err(k_te.grad, k_rf.grad) < 1e-2, f"dk B={B} S={S}"
