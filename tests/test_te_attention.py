"""
Correctness of the TransformerEngine-backed softmax-attention core used by
DeepSeek-V2-Lite's MLA against a float32 PyTorch reference and the FA4 path.

MLA has asymmetric head dims: query/key carry ``qk_nope_head_dim + qk_rope_head_dim``
while value carries ``v_head_dim``. TE expresses this via ``kv_channels=(k, v)``.
Shapes follow DeepSeek-V2-Lite (num_heads 16, qk_nope 128, qk_rope 64, v 128).
cuDNN-fused attention differs slightly from the reference softmax, so a relative
error threshold of 1e-2 is used.
"""

import pytest
import torch

pytest.importorskip("transformer_engine.pytorch")
import transformer_engine.pytorch as te  # noqa: E402

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# DeepSeek-V2-Lite MLA head geometry.
NUM_HEADS = 16
QK_NOPE = 128
QK_ROPE = 64
V_HEAD_DIM = 128
Q_HEAD_DIM = QK_NOPE + QK_ROPE  # key/query head dim (192)
SOFTMAX_SCALE = Q_HEAD_DIM ** (-0.5)

SHAPES = [(1, 64), (2, 128), (1, 256), (4, 96)]


def rel_err(actual: torch.Tensor, ref: torch.Tensor) -> float:
    actual, ref = actual.float(), ref.float()
    return ((actual - ref).pow(2).mean() / ref.pow(2).mean().clamp_min(1e-12)).sqrt().item()


def ref_causal_attention(q, k, v, scale):
    """Ground-truth causal SDPA in float32. q/k: [B,S,H,Dqk], v: [B,S,H,Dv]."""
    q = q.float().transpose(1, 2)  # [B,H,S,Dqk]
    k = k.float().transpose(1, 2)
    v = v.float().transpose(1, 2)  # [B,H,S,Dv]
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # [B,H,S,S]
    s = scores.shape[-1]
    mask = torch.triu(torch.ones(s, s, device=scores.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)  # [B,H,S,Dv]
    return out.transpose(1, 2)  # [B,S,H,Dv]


def make_core():
    return te.DotProductAttention(
        num_attention_heads=NUM_HEADS,
        kv_channels=(Q_HEAD_DIM, V_HEAD_DIM),
        qkv_format="bshd",
        attn_mask_type="causal",
        softmax_scale=SOFTMAX_SCALE,
        attention_dropout=0.0,
    ).cuda()


@requires_cuda
@pytest.mark.parametrize("B,S", SHAPES)
def test_te_attention_forward_vs_reference(B, S):
    dev, dt = torch.device("cuda"), torch.bfloat16
    q = torch.randn(B, S, NUM_HEADS, Q_HEAD_DIM, device=dev, dtype=dt)
    k = torch.randn(B, S, NUM_HEADS, Q_HEAD_DIM, device=dev, dtype=dt)
    v = torch.randn(B, S, NUM_HEADS, V_HEAD_DIM, device=dev, dtype=dt)

    core = make_core()
    out = core(q, k, v).view(B, S, NUM_HEADS, V_HEAD_DIM)
    ref = ref_causal_attention(q, k, v, SOFTMAX_SCALE)

    err = rel_err(out, ref)
    assert err < 1e-2, f"fwd B={B} S={S}: rel_err={err}"


@requires_cuda
@pytest.mark.parametrize("B,S", SHAPES)
def test_te_attention_backward_vs_reference(B, S):
    dev, dt = torch.device("cuda"), torch.bfloat16
    q = torch.randn(B, S, NUM_HEADS, Q_HEAD_DIM, device=dev, dtype=dt)
    k = torch.randn(B, S, NUM_HEADS, Q_HEAD_DIM, device=dev, dtype=dt)
    v = torch.randn(B, S, NUM_HEADS, V_HEAD_DIM, device=dev, dtype=dt)
    grad = torch.randn(B, S, NUM_HEADS * V_HEAD_DIM, device=dev, dtype=dt)

    core = make_core()

    q_te = q.detach().clone().requires_grad_(True)
    k_te = k.detach().clone().requires_grad_(True)
    v_te = v.detach().clone().requires_grad_(True)
    core(q_te, k_te, v_te).backward(grad)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref = ref_causal_attention(q_ref, k_ref, v_ref, SOFTMAX_SCALE)
    out_ref.reshape(B, S, NUM_HEADS * V_HEAD_DIM).backward(grad.float())

    for name, g in (("dq", q_te.grad), ("dk", k_te.grad), ("dv", v_te.grad)):
        assert torch.isfinite(g).all(), f"{name} B={B} S={S}: non-finite grad"
    assert rel_err(q_te.grad, q_ref.grad) < 1e-2, f"dq B={B} S={S}"
    assert rel_err(k_te.grad, k_ref.grad) < 1e-2, f"dk B={B} S={S}"
    assert rel_err(v_te.grad, v_ref.grad) < 1e-2, f"dv B={B} S={S}"


@requires_cuda
@pytest.mark.parametrize("B,S", [(2, 128), (1, 256)])
def test_te_attention_matches_fa4(B, S):
    """TE core matches the FA4 MLA path (the backend it replaces), fwd + bwd."""
    fa4 = pytest.importorskip("pithtrain.operators.flash_attn_v4")
    dev, dt = torch.device("cuda"), torch.bfloat16

    q_nope = torch.randn(B, S, NUM_HEADS, QK_NOPE, device=dev, dtype=dt)
    q_pe = torch.randn(B, S, NUM_HEADS, QK_ROPE, device=dev, dtype=dt)
    k_nope = torch.randn(B, S, NUM_HEADS, QK_NOPE, device=dev, dtype=dt)
    k_pe = torch.randn(B, S, 1, QK_ROPE, device=dev, dtype=dt)  # rope key: single head
    v = torch.randn(B, S, NUM_HEADS, V_HEAD_DIM, device=dev, dtype=dt)
    grad = torch.randn(B, S, NUM_HEADS, V_HEAD_DIM, device=dev, dtype=dt)

    core = make_core()

    def clones(*ts):
        return [t.detach().clone().requires_grad_(True) for t in ts]

    qn, qp, kn, kp, vv = clones(q_nope, q_pe, k_nope, k_pe, v)
    query = torch.cat([qn, qp], dim=-1)
    key = torch.cat([kn, kp.expand(-1, -1, NUM_HEADS, -1)], dim=-1)
    out_te = core(query, key, vv.contiguous()).view(B, S, NUM_HEADS, V_HEAD_DIM)
    out_te.backward(grad)

    qn2, qp2, kn2, kp2, vv2 = clones(q_nope, q_pe, k_nope, k_pe, v)
    out_fa = fa4.mla_flash_attn_func(
        qn2,
        qp2,
        kn2,
        kp2,
        vv2,
        softmax_scale=SOFTMAX_SCALE,
        qk_nope_head_dim=QK_NOPE,
        causal=True,
    )
    out_fa.backward(grad)

    assert rel_err(out_te, out_fa) < 1e-2, f"fwd vs FA4 B={B} S={S}"
    assert rel_err(qn.grad, qn2.grad) < 1e-2, f"dq_nope vs FA4 B={B} S={S}"
    assert rel_err(qp.grad, qp2.grad) < 1e-2, f"dq_pe vs FA4 B={B} S={S}"
    assert rel_err(kn.grad, kn2.grad) < 1e-2, f"dk_nope vs FA4 B={B} S={S}"
    assert rel_err(vv.grad, vv2.grad) < 1e-2, f"dv vs FA4 B={B} S={S}"
