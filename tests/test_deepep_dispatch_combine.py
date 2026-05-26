"""Standalone unit tests for the DeepEP v2 dispatch/combine fwd+bwd path.

Pithtrain only ever sends BF16 over EP — FP8 quantization stays inside Linear /
GroupLinear — so these tests use `use_fp8_dispatch=False` everywhere.

Run with:
    bash tests/test_deepep_dispatch_combine.sh
i.e. `torchrun --standalone --nproc-per-node=4 -m pytest <this file> -v`.

Skips gracefully if deep_ep is not importable or fewer than 4 GPUs are visible,
so `pytest --collect-only` works on no-GPU dev boxes.
"""

import os

import pytest
import torch
import torch.distributed as dist

deep_ep = pytest.importorskip("deep_ep")


def _need_gpu_world(min_world: int = 4):
    """Skip the test (don't fail) when the env can't run a 4-rank torchrun job."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("not running under torchrun")
    if int(os.environ["WORLD_SIZE"]) < min_world:
        pytest.skip(f"need world_size >= {min_world}, have {os.environ['WORLD_SIZE']}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        pytest.skip("no CUDA device visible to this rank")


@pytest.fixture(scope="module", autouse=True)
def _dist_setup():
    """Init NCCL once per torchrun process group. Tears down at module exit."""
    _need_gpu_world()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{os.environ['LOCAL_RANK']}"),
        )
    yield
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def _make_buffer(num_topk=4, num_experts=16, hidden=512, num_max_tokens=64):
    """Construct an ElasticBuffer over WORLD; pithtrain always uses BF16 over EP."""
    group = dist.group.WORLD
    buf = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=False,
    )
    return buf, group


def _make_inputs(N, H, K, E, *, weights_grad: bool = True):
    """Random x / topk_idx / topk_weights matching DeepEP's expected dtypes."""
    x = torch.randn(N, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    topk_idx = torch.stack([torch.randperm(E, device="cuda")[:K] for _ in range(N)]).to(
        deep_ep.topk_idx_t
    )
    w = torch.rand(N, K, dtype=torch.float32, device="cuda")
    w = w / w.sum(dim=-1, keepdim=True)
    w = w.detach().requires_grad_(weights_grad)
    return x, topk_idx, w


def test_buffer_construction():
    """Buffer init must allocate non-zero bytes and report sane logical sizes."""
    buf, _ = _make_buffer()
    assert buf.num_bytes > 0
    assert buf.num_ranks == int(os.environ["WORLD_SIZE"])


def test_dispatch_combine_shapes_bf16():
    """One forward dispatch + identity combine. Checks dtypes/shapes/handle layout."""
    buf, _ = _make_buffer()
    N, H, K, E = 64, 512, 4, 16
    x, topk_idx, topk_weights = _make_inputs(N, H, K, E, weights_grad=False)
    num_sms = buf.get_theoretical_num_sms(E, K)

    recv_x, recv_topk_idx, recv_topk_weights, handle, ev = buf.dispatch(
        x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=E,
        num_max_tokens_per_rank=N,
        expert_alignment=1,
        num_sms=num_sms,
        async_with_compute_stream=True,
    )
    ev.current_stream_wait()

    assert recv_x.dim() == 2 and recv_x.shape[1] == H
    assert recv_x.dtype == torch.bfloat16
    assert recv_topk_idx.shape == (recv_x.shape[0], K)
    assert recv_topk_idx.dtype == deep_ep.topk_idx_t
    assert recv_topk_weights.shape == (recv_x.shape[0], K)
    assert recv_topk_weights.dtype == torch.float32
    assert handle.num_recv_tokens_per_expert_list is not None

    y = recv_x.detach().clone()  # identity expert
    combined_x, _, ev2 = buf.combine(
        y,
        handle=handle,
        num_sms=num_sms,
        async_with_compute_stream=True,
    )
    ev2.current_stream_wait()
    assert combined_x.shape == (N, H)
    assert combined_x.dtype == torch.bfloat16


class _DispatchFn(torch.autograd.Function):
    """Same wrapper pattern as the Stage 1 demo at $DEEPEP_DIR/example_deepep_v2.py."""

    @staticmethod
    def forward(ctx, x, topk_idx, topk_weights, buf, num_experts, num_max_tokens, num_sms):
        recv_x, recv_topk_idx, recv_topk_w, handle, ev = buf.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens,
            expert_alignment=1,
            num_sms=num_sms,
            async_with_compute_stream=True,
        )
        ev.current_stream_wait()
        ctx.buf, ctx.handle, ctx.num_sms = buf, handle, num_sms
        ctx.mark_non_differentiable(recv_topk_idx)
        return recv_x, recv_topk_idx, recv_topk_w, handle

    @staticmethod
    def backward(ctx, grad_recv_x, _g_idx, grad_recv_w, _g_handle):
        gw = (
            grad_recv_w.contiguous()
            if grad_recv_w is not None and grad_recv_w.abs().sum() > 0
            else None
        )
        grad_x, grad_w, ev = ctx.buf.combine(
            grad_recv_x.contiguous(),
            handle=ctx.handle,
            topk_weights=gw,
            num_sms=ctx.num_sms,
            async_with_compute_stream=True,
        )
        ev.current_stream_wait()
        return grad_x, None, grad_w, None, None, None, None


class _CombineFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, handle, buf, num_sms):
        combined_x, _, ev = buf.combine(
            y,
            handle=handle,
            num_sms=num_sms,
            async_with_compute_stream=True,
        )
        ev.current_stream_wait()
        ctx.buf, ctx.handle, ctx.num_sms = buf, handle, num_sms
        return combined_x

    @staticmethod
    def backward(ctx, grad_combined_x):
        grad_y, _, _, _, ev = ctx.buf.dispatch(
            grad_combined_x.contiguous(),
            handle=ctx.handle,
            num_sms=ctx.num_sms,
            async_with_compute_stream=True,
        )
        ev.current_stream_wait()
        return grad_y, None, None, None


def _finite_nonzero(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item() and t.abs().sum().item() > 0


def test_backward_chains_fire():
    """dispatch.bwd is combine; combine.bwd is dispatch. Verifies the autograd
    path is wired correctly by checking that x and a downstream learnable
    weight both receive finite, non-zero gradients. Strict numerical
    correctness against a reference is deferred to Stage 3 integration tests
    (which compare against pithtrain's existing dispatch/combine semantics)."""
    buf, _ = _make_buffer()
    N, H, K, E = 64, 512, 4, 16
    x, topk_idx, topk_weights = _make_inputs(N, H, K, E, weights_grad=True)
    expert_w = torch.randn(H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    num_sms = buf.get_theoretical_num_sms(E, K)

    recv_x, _, _, handle = _DispatchFn.apply(x, topk_idx, topk_weights, buf, E, N, num_sms)
    y = recv_x * expert_w  # toy "expert"
    combined_x = _CombineFn.apply(y, handle, buf, num_sms)
    loss = combined_x.sum()
    loss.backward()

    assert torch.isfinite(combined_x).all().item(), "combined_x has non-finite entries"
    assert x.grad is not None and _finite_nonzero(x.grad), "x.grad missing/non-finite/zero"
    assert expert_w.grad is not None and _finite_nonzero(expert_w.grad), (
        "expert_w.grad missing/non-finite/zero"
    )
    # topk_weights.grad is allowed to be None: nothing in the graph consumes
    # recv_topk_weights, so PyTorch elides its gradient computation.
