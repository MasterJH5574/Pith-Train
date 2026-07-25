"""
Correctness test for the CUDA grouped-GEMM expert kernels.

Compares ``NativeGroupLinear`` and the underlying grouped-GEMM ops against
``F.grouped_mm`` for the forward, input-gradient, and weight-gradient paths on
representative DeepSeek-V2-Lite expert shapes (hidden 2048, moe_intermediate
1408) with jagged per-expert token counts, including a partial-tile group.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

try:
    from pithtrain.layers.native_ext import (
        NativeGroupLinear,
        _grouped_gemm_dgrad,
        _grouped_gemm_forward,
        _grouped_gemm_wgrad,
    )

    HAS_EXT = True
except ImportError:
    HAS_EXT = False

requires_ext = pytest.mark.skipif(not HAS_EXT, reason="pithtrain_ext not built")

# Normalized squared-error threshold for bf16 accumulation.
ERR_THRESHOLD = 1e-3

# (hidden, moe_intermediate) for DeepSeek-V2-Lite.
HIDDEN = 2048
INTERMEDIATE = 1408

# Jagged per-expert token counts, mixing empty, partial-tile (not a multiple of
# the 16-row kernel tile), and tile-aligned groups.
GROUP_SIZES = [40, 0, 128, 23, 16, 7]


def calc_diff(x, y):
    x, y = x.detach().double(), y.detach().double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


def _offs(group_sizes, device):
    return torch.tensor(group_sizes, device=device).cumsum(0).to(torch.int32)


@requires_ext
@pytest.mark.parametrize(
    "in_features,out_features",
    [(HIDDEN, INTERMEDIATE), (INTERMEDIATE, HIDDEN)],
)
def test_native_grouped_gemm_forward(in_features, out_features):
    device = torch.device("cuda")
    num_groups = len(GROUP_SIZES)
    m_total = sum(GROUP_SIZES)
    offs = _offs(GROUP_SIZES, device)

    x = torch.randn(m_total, in_features, device=device, dtype=torch.bfloat16)
    weight = torch.randn(num_groups, out_features, in_features, device=device, dtype=torch.bfloat16)
    weight *= 0.02

    out = _grouped_gemm_forward(x, weight, offs)
    ref = F.grouped_mm(x, weight.transpose(1, 2), offs=offs)

    diff = calc_diff(out[:m_total], ref[:m_total])
    assert diff < ERR_THRESHOLD, f"forward diff={diff}"


@requires_ext
@pytest.mark.parametrize(
    "in_features,out_features",
    [(HIDDEN, INTERMEDIATE), (INTERMEDIATE, HIDDEN)],
)
def test_native_grouped_gemm_dgrad(in_features, out_features):
    device = torch.device("cuda")
    num_groups = len(GROUP_SIZES)
    m_total = sum(GROUP_SIZES)
    offs = _offs(GROUP_SIZES, device)

    dy = torch.randn(m_total, out_features, device=device, dtype=torch.bfloat16)
    weight = torch.randn(num_groups, out_features, in_features, device=device, dtype=torch.bfloat16)
    weight *= 0.02

    di = _grouped_gemm_dgrad(dy, weight, offs)
    ref = F.grouped_mm(dy, weight, offs=offs)

    diff = calc_diff(di[:m_total], ref[:m_total])
    assert diff < ERR_THRESHOLD, f"dgrad diff={diff}"


@requires_ext
@pytest.mark.parametrize(
    "in_features,out_features",
    [(HIDDEN, INTERMEDIATE), (INTERMEDIATE, HIDDEN)],
)
def test_native_grouped_gemm_wgrad(in_features, out_features):
    device = torch.device("cuda")
    num_groups = len(GROUP_SIZES)
    m_total = sum(GROUP_SIZES)
    offs = _offs(GROUP_SIZES, device)

    dy = torch.randn(m_total, out_features, device=device, dtype=torch.bfloat16)
    x = torch.randn(m_total, in_features, device=device, dtype=torch.bfloat16)

    dw = _grouped_gemm_wgrad(dy, x, offs)
    ref = F.grouped_mm(dy.transpose(0, 1), x, offs=offs)

    assert dw.shape == (num_groups, out_features, in_features)
    diff = calc_diff(dw, ref)
    assert diff < ERR_THRESHOLD, f"wgrad diff={diff}"


@requires_ext
def test_native_grouped_gemm_forward_overallocated():
    """Rows beyond offs[-1] are ignored; the packed prefix stays correct."""
    device = torch.device("cuda")
    num_groups = len(GROUP_SIZES)
    m_total = sum(GROUP_SIZES)
    extra = 96
    offs = _offs(GROUP_SIZES, device)

    x = torch.zeros(m_total + extra, HIDDEN, device=device, dtype=torch.bfloat16)
    x[:m_total].normal_()
    weight = torch.randn(num_groups, INTERMEDIATE, HIDDEN, device=device, dtype=torch.bfloat16)
    weight *= 0.02

    out = _grouped_gemm_forward(x, weight, offs)
    ref = F.grouped_mm(x[:m_total], weight.transpose(1, 2), offs=offs)

    diff = calc_diff(out[:m_total], ref)
    assert diff < ERR_THRESHOLD, f"over-allocated forward diff={diff}"


@requires_ext
def test_native_group_linear_module():
    """End-to-end forward + backward against an F.grouped_mm reference."""
    from pithtrain.layers.group_linear import GroupLinear

    device = torch.device("cuda")
    num_groups = len(GROUP_SIZES)
    m_total = sum(GROUP_SIZES)
    offs = _offs(GROUP_SIZES, device)

    ref_mod = GroupLinear(num_groups, HIDDEN, INTERMEDIATE).to(device, torch.bfloat16)
    torch.nn.init.normal_(ref_mod.weight, std=0.02)

    native_mod = NativeGroupLinear(num_groups, HIDDEN, INTERMEDIATE).to(device, torch.bfloat16)
    native_mod.weight.data.copy_(ref_mod.weight.data)

    x_raw = torch.randn(m_total, HIDDEN, device=device, dtype=torch.bfloat16)
    grad = torch.randn(m_total, INTERMEDIATE, device=device, dtype=torch.bfloat16)

    x_ref = x_raw.detach().clone().requires_grad_(True)
    out_ref = ref_mod(x_ref, offs)
    out_ref.backward(grad)

    x_native = x_raw.detach().clone().requires_grad_(True)
    out_native = native_mod(x_native, offs)
    out_native.backward(grad)

    assert calc_diff(out_native, out_ref) < ERR_THRESHOLD, "module forward mismatch"
    assert calc_diff(x_native.grad, x_ref.grad) < ERR_THRESHOLD, "module input-grad mismatch"
    assert calc_diff(native_mod.weight.grad, ref_mod.weight.grad) < ERR_THRESHOLD, (
        "module weight-grad mismatch"
    )


@requires_ext
def test_native_group_linear_weight_grad_store():
    """Weight gradients are deferred through WeightGradStore and match the direct path."""
    from pithtrain.dualpipe.utils import WeightGradStore

    device = torch.device("cuda")
    num_groups = len(GROUP_SIZES)
    m_total = sum(GROUP_SIZES)
    offs = _offs(GROUP_SIZES, device)

    mod = NativeGroupLinear(num_groups, HIDDEN, INTERMEDIATE).to(device, torch.bfloat16)
    torch.nn.init.normal_(mod.weight, std=0.02)

    x_raw = torch.randn(m_total, HIDDEN, device=device, dtype=torch.bfloat16)
    grad = torch.randn(m_total, INTERMEDIATE, device=device, dtype=torch.bfloat16)

    x_direct = x_raw.detach().clone().requires_grad_(True)
    mod(x_direct, offs).backward(grad)
    direct_weight_grad = mod.weight.grad.clone()
    direct_input_grad = x_direct.grad.clone()

    mod.weight.grad = None
    x_def = x_raw.detach().clone().requires_grad_(True)
    WeightGradStore.enabled = True
    try:
        mod(x_def, offs).backward(grad)
        assert mod.weight.grad is None, "weight grad should be deferred"
        assert x_def.grad is not None, "input grad should be on the critical path"
        assert calc_diff(x_def.grad, direct_input_grad) < ERR_THRESHOLD

        WeightGradStore.flush()
        WeightGradStore.pop()

        assert mod.weight.grad is not None, "weight grad should exist after pop"
        assert calc_diff(mod.weight.grad, direct_weight_grad) < ERR_THRESHOLD
    finally:
        WeightGradStore.enabled = False
        WeightGradStore.clear()
