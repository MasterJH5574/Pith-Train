"""
Correctness of the TransformerEngine-backed grouped linear (MoE experts)
against the reference ``F.grouped_mm`` path, for forward, dgrad, and wgrad.

Shapes follow DeepSeek-V2-Lite experts (hidden 2048, moe_intermediate 1408) in
both projection directions, with jagged per-expert token counts including empty
and partial (non-aligned) groups. BF16 through TE differs slightly from the torch
grouped GEMM, so a relative-error threshold of 1e-2 is used.
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformer_engine.pytorch")

from pithtrain.layers.te_group_linear import TEGroupLinear, TEGroupLinearFunc  # noqa: E402

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

# (name, num_experts, group_sizes) — jagged, with empty and partial groups.
GROUP_CONFIGS = [
    ("uniform", 6, [128, 128, 128, 128, 128, 128]),
    ("jagged", 8, [128, 0, 256, 64, 384, 128, 256, 64]),
    ("partial+empty", 8, [100, 0, 200, 37, 0, 128, 13, 256]),
    ("single-expert", 1, [512]),
]

# (name, in_features(K), out_features(N)) — both DeepSeek projection directions.
PROJ_DIRS = [
    ("gate_up", 2048, 1408),
    ("down", 1408, 2048),
]


def rel_err(actual: torch.Tensor, ref: torch.Tensor) -> float:
    actual, ref = actual.float(), ref.float()
    return ((actual - ref).pow(2).mean() / ref.pow(2).mean().clamp_min(1e-12)).sqrt().item()


def _offs(group_sizes, device):
    return torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)


@requires_cuda
def test_te_grouped_gemm_forward():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    for gname, num_experts, group_sizes in GROUP_CONFIGS:
        for pname, K, N in PROJ_DIRS:
            M = sum(group_sizes)
            x = torch.randn(M, K, device=device, dtype=dtype)
            w = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.02
            offs = _offs(group_sizes, device)

            ref = F.grouped_mm(x, w.transpose(1, 2), offs=offs)
            out = TEGroupLinearFunc.apply(x, w, list(group_sizes))

            err = rel_err(out, ref)
            assert err < 1e-2, f"forward {gname}/{pname}: rel_err={err}"


@requires_cuda
def test_te_grouped_gemm_dgrad_wgrad():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    for gname, num_experts, group_sizes in GROUP_CONFIGS:
        for pname, K, N in PROJ_DIRS:
            M = sum(group_sizes)
            x = torch.randn(M, K, device=device, dtype=dtype)
            w = torch.randn(num_experts, N, K, device=device, dtype=dtype) * 0.02
            offs = _offs(group_sizes, device)
            dy = torch.randn(M, N, device=device, dtype=dtype)

            ref_dgrad = F.grouped_mm(dy, w, offs=offs)
            ref_wgrad = F.grouped_mm(dy.transpose(0, 1), x, offs=offs)

            x_te = x.detach().clone().requires_grad_(True)
            out = TEGroupLinearFunc.apply(x_te, w, list(group_sizes))
            out.backward(dy)

            derr = rel_err(x_te.grad, ref_dgrad)
            werr = rel_err(w.grad, ref_wgrad)
            assert derr < 1e-2, f"dgrad {gname}/{pname}: rel_err={derr}"
            assert werr < 1e-2, f"wgrad {gname}/{pname}: rel_err={werr}"


@requires_cuda
def test_te_group_linear_module_matches_grouped_mm():
    """TEGroupLinear module fwd+bwd matches the reference F.grouped_mm path."""
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_experts, K, N = 8, 2048, 1408
    group_sizes = [128, 0, 256, 64, 384, 128, 256, 64]
    ks = list(group_sizes)
    M = sum(group_sizes)
    offs = _offs(group_sizes, device)

    weight = torch.empty(num_experts, N, K, device=device, dtype=dtype)
    torch.nn.init.normal_(weight, std=0.02)
    te = TEGroupLinear(num_experts, K, N).to(device, dtype)
    te.weight.data.copy_(weight)

    x_raw = torch.randn(M, K, device=device, dtype=dtype)
    grad = torch.randn(M, N, device=device, dtype=dtype)

    # Reference: F.grouped_mm forward + autograd dgrad/wgrad.
    x_ref = x_raw.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    out_ref = F.grouped_mm(x_ref, w_ref.transpose(1, 2), offs=offs)
    out_ref.backward(grad)

    x_te = x_raw.detach().clone().requires_grad_(True)
    out_te = te(x_te, offs, ks=ks)
    out_te.backward(grad)

    assert rel_err(out_te, out_ref) < 1e-2
    assert rel_err(x_te.grad, x_ref.grad) < 1e-2
    assert rel_err(te.weight.grad, w_ref.grad) < 1e-2


@requires_cuda
def test_te_group_linear_weight_grad_store():
    """TEGroupLinear defers weight grad via WeightGradStore and matches the eager path."""
    from pithtrain.dualpipe.utils import WeightGradStore

    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_experts, K, N = 4, 2048, 1408
    group_sizes = [128, 64, 96, 32]
    ks = list(group_sizes)
    M = sum(group_sizes)
    offs = _offs(group_sizes, device)

    gl = TEGroupLinear(num_experts, K, N).to(device, dtype)
    torch.nn.init.normal_(gl.weight, std=0.02)
    x_raw = torch.randn(M, K, device=device, dtype=dtype)
    grad = torch.randn(M, N, device=device, dtype=dtype)

    # Eager reference.
    x_ref = x_raw.detach().clone().requires_grad_(True)
    gl(x_ref, offs, ks=ks).backward(grad)
    ref_wgrad = gl.weight.grad.clone()
    ref_dgrad = x_ref.grad.clone()
    gl.weight.grad = None

    # Deferred path.
    x_def = x_raw.detach().clone().requires_grad_(True)
    WeightGradStore.enabled = True
    try:
        gl(x_def, offs, ks=ks).backward(grad)
        assert gl.weight.grad is None, "weight grad must be deferred"
        assert x_def.grad is not None, "input grad stays on the critical path"
        assert rel_err(x_def.grad, ref_dgrad) < 1e-2

        WeightGradStore.flush()
        WeightGradStore.pop()

        assert gl.weight.grad is not None, "weight grad must exist after pop"
        assert gl.weight.grad.shape == gl.weight.shape
        assert rel_err(gl.weight.grad, ref_wgrad) < 1e-2
    finally:
        WeightGradStore.enabled = False
        WeightGradStore.clear()
