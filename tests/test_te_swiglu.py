"""
Correctness of the TransformerEngine SwiGLU activation used by DeepSeek-V2-Lite's
dense/shared-expert MLP (and the routed-expert activation) against a float32
PyTorch reference (``silu(gate) * up``) and the Triton ``silu_mul`` it replaces.

``te_swiglu`` concatenates ``[gate, up]`` and runs TE's fused SwiGLU kernel
(``silu`` of the first half times the second half), with a fused backward.
BF16 fused math differs slightly from the reference, so a relative error
threshold of 1e-2 is used.
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformer_engine.pytorch")
from pithtrain.operators.silu_mul import silu_mul  # noqa: E402
from pithtrain.operators.te_swiglu import te_swiglu  # noqa: E402

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# (leading, feature) — dense MLP is 3-D [b, s, inter]; routed experts are 2-D [tokens, inter].
SHAPES = [(4, 32, 1408), (256, 1408), (2, 128, 10944)]


def rel_err(actual: torch.Tensor, ref: torch.Tensor) -> float:
    actual, ref = actual.float(), ref.float()
    return ((actual - ref).pow(2).mean() / ref.pow(2).mean().clamp_min(1e-12)).sqrt().item()


@requires_cuda
@pytest.mark.parametrize("shape", SHAPES)
def test_te_swiglu_forward_backward_vs_reference(shape):
    dev, dt = torch.device("cuda"), torch.bfloat16
    gate = torch.randn(*shape, device=dev, dtype=dt, requires_grad=True)
    up = torch.randn(*shape, device=dev, dtype=dt, requires_grad=True)
    g_ref = gate.detach().clone().requires_grad_(True)
    u_ref = up.detach().clone().requires_grad_(True)

    out = te_swiglu(gate, up)
    ref = F.silu(g_ref.float()) * u_ref.float()
    assert torch.isfinite(out).all()
    assert rel_err(out, ref) < 1e-2, f"fwd {shape}"

    grad = torch.randn_like(out)
    out.backward(grad)
    ref.to(dt).backward(grad)
    assert torch.isfinite(gate.grad).all() and torch.isfinite(up.grad).all()
    assert rel_err(gate.grad, g_ref.grad) < 1e-2, f"dgate {shape}"
    assert rel_err(up.grad, u_ref.grad) < 1e-2, f"dup {shape}"


@requires_cuda
def test_te_swiglu_matches_silu_mul():
    """TE SwiGLU matches the Triton ``silu_mul`` it replaces, fwd + bwd."""
    dev, dt = torch.device("cuda"), torch.bfloat16
    gate = torch.randn(256, 1408, device=dev, dtype=dt)
    up = torch.randn(256, 1408, device=dev, dtype=dt)
    grad = torch.randn(256, 1408, device=dev, dtype=dt)

    def run(fn):
        g = gate.detach().clone().requires_grad_(True)
        u = up.detach().clone().requires_grad_(True)
        out = fn(g, u)
        out.backward(grad)
        return out, g.grad, u.grad

    o_te, dg_te, du_te = run(te_swiglu)
    o_tr, dg_tr, du_tr = run(silu_mul)
    assert rel_err(o_te, o_tr) < 1e-2
    assert rel_err(dg_te, dg_tr) < 1e-2
    assert rel_err(du_te, du_tr) < 1e-2


@requires_cuda
def test_te_swiglu_empty_rows():
    """Zero-row inputs (possible in the routed-expert path) keep the graph."""
    dev, dt = torch.device("cuda"), torch.bfloat16
    gate = torch.randn(0, 1408, device=dev, dtype=dt, requires_grad=True)
    up = torch.randn(0, 1408, device=dev, dtype=dt, requires_grad=True)
    out = te_swiglu(gate, up)
    assert out.shape == (0, 1408)
    out.sum().backward()
    assert gate.grad is not None and up.grad is not None
