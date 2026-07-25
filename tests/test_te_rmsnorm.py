"""
Correctness of the TransformerEngine RMSNorm used by DeepSeek-V2-Lite (decoder
input/post-attention norms, the MLA ``kv_a_layernorm``, and the final model norm)
against a float32 PyTorch reference (``F.rms_norm``).

TE's fused RMSNorm keeps the same single ``weight`` parameter (shape
``[normalized_shape]``) as ``nn.RMSNorm``, so state dicts are unaffected. BF16
fused reductions differ slightly from the reference, so a relative error
threshold of 1e-2 is used.
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformer_engine.pytorch")
import transformer_engine.pytorch as te  # noqa: E402

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

EPS = 1e-6
# (hidden, leading shape) — hidden_size (2048), kv_lora_rank (512).
CASES = [(2048, (2, 128)), (512, (4, 96)), (2048, (1, 256))]


def rel_err(actual: torch.Tensor, ref: torch.Tensor) -> float:
    actual, ref = actual.float(), ref.float()
    return ((actual - ref).pow(2).mean() / ref.pow(2).mean().clamp_min(1e-12)).sqrt().item()


@requires_cuda
def test_te_rmsnorm_preserves_weight_param():
    """Drop-in parameter contract: a single ``weight`` of shape [hidden]."""
    m = te.RMSNorm(2048, eps=EPS).cuda()
    ref = torch.nn.RMSNorm(2048, eps=EPS).cuda()
    assert [n for n, _ in m.named_parameters()] == ["weight"]
    assert m.weight.shape == ref.weight.shape
    # both initialise gamma to ones
    assert torch.allclose(m.weight.float(), torch.ones_like(m.weight.float()))


@requires_cuda
@pytest.mark.parametrize("hidden,lead", CASES)
def test_te_rmsnorm_forward_backward_vs_reference(hidden, lead):
    dev, dt = torch.device("cuda"), torch.bfloat16
    weight = torch.randn(hidden, device=dev, dtype=dt)

    m = te.RMSNorm(hidden, eps=EPS, params_dtype=dt).to(dev)
    with torch.no_grad():
        m.weight.copy_(weight)

    x = torch.randn(*lead, hidden, device=dev, dtype=dt, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)

    y = m(x)
    y_ref = F.rms_norm(x_ref, (hidden,), weight=w_ref, eps=EPS)
    assert torch.isfinite(y).all()
    assert rel_err(y, y_ref) < 1e-2, f"fwd hidden={hidden} lead={lead}"

    grad = torch.randn_like(y)
    y.backward(grad)
    y_ref.backward(grad)
    assert torch.isfinite(x.grad).all() and torch.isfinite(m.weight.grad).all()
    assert rel_err(x.grad, x_ref.grad) < 1e-2, f"dx hidden={hidden} lead={lead}"
    assert rel_err(m.weight.grad, w_ref.grad) < 1e-2, f"dw hidden={hidden} lead={lead}"
