"""Replay must survive live micro-batches, parameter updates and FSDP storage replacement."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_capture_preserves_nested_compiled_grad_mode():
    from pithtrain.pipeline.cuda_graphs import StaticGraph

    @torch.compile(fullgraph=True)
    def nested(x):
        return x.sin()

    graph = StaticGraph(lambda args: [nested(args[0])], (0,), (), backward=False)
    x = torch.randn(16, device="cuda")
    with torch.no_grad():
        graph([x])
    # DualPipeV captures at the next step boundary, outside AOT's no-grad invocation.
    assert torch.is_grad_enabled()
    with torch.compiler.set_stance("default", skip_guard_eval_unsafe=True):
        graph.capture()
    assert torch.is_grad_enabled()
    x = torch.randn_like(x)
    (out,) = graph([x])
    torch.testing.assert_close(out, x.sin())
    assert graph.replays == 1


def test_release_graphs_before_nccl_teardown(tmp_path):
    from pithtrain.pipeline.cuda_graphs import StaticGraph, release_graphs

    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1
    )
    try:

        def collective(args):
            (x,) = args
            torch.distributed.all_reduce(x)
            return [x]

        graph = StaticGraph(collective, (0,), (0,), backward=False)
        x = torch.ones(8, device="cuda")
        graph([x])
        graph.capture()
        x.fill_(3)
        graph([x])
        torch.testing.assert_close(x, torch.full_like(x, 3))
        release_graphs()
        assert graph.graph is None and graph.key is None
        assert not hasattr(graph, "inputs") and not hasattr(graph, "outputs")
    finally:
        release_graphs()
        torch.distributed.destroy_process_group()


def test_scratch_replay_aliases_mutations_and_fallback():
    from pithtrain.pipeline.cuda_graphs import StaticGraph

    def fn(args):
        x, weight, grad, unused = args
        grad.add_(x @ weight)
        return [grad.square(), x[:, 1:], unused, grad.sum().expand(5)]

    graph = StaticGraph(fn, (0, 1, 2), (2,), backward=True)
    unused = torch.empty(10, device="cuda")
    unused.untyped_storage().resize_(0)
    x = torch.randn(8, 8, device="cuda")
    weight = torch.randn(8, 8, device="cuda")
    grad = torch.zeros_like(x)
    graph([x, weight, grad, unused])
    graph.capture()
    saved = []
    for step in range(5):
        # Offset into another allocation, unlike the scratch input's zero offset.
        x = torch.randn(10, 8, device="cuda")[1:9]
        weight = torch.randn_like(weight)
        grad = torch.zeros_like(grad)
        expected = x @ weight
        out, alias, passthrough, expanded = graph([x, weight, grad, unused])
        torch.testing.assert_close(grad, expected)
        torch.testing.assert_close(out, expected.square())
        torch.testing.assert_close(alias, x[:, 1:])
        assert alias.untyped_storage()._cdata == x.untyped_storage()._cdata
        assert passthrough is unused
        assert expanded.stride() == (0,)
        torch.testing.assert_close(expanded, expected.sum().expand(5))
        saved.append((out, expected.square()))
    for out, expected in saved:
        torch.testing.assert_close(out, expected)
    before = graph.replays
    graph([x[:4], weight, torch.zeros_like(x[:4]), unused])
    assert graph.replays == before
    # Same metadata, but a newly introduced writable alias must also take ordinary execution.
    expected = x + x @ weight
    graph([x, weight, x, unused])
    torch.testing.assert_close(x, expected)
    assert graph.replays == before
    assert graph.fallbacks == 3  # warmup, mismatched shape, writable alias


def test_compiled_forward_backward_with_live_microbatches():
    from pithtrain.contexts import training
    from pithtrain.operators.linear import BF16LinearFunc
    from pithtrain.pipeline.cuda_graphs import capture_pending, compile_stage, graph_stats

    @compile_stage(options={"cpp_wrapper": True})
    def region(x, weight, bias, main_grad, counter):
        counter.add_(1)
        # Like global-batch load balancing, updated state affects this micro-batch's gradient.
        return (BF16LinearFunc.apply(x, weight, main_grad) + bias).sin() / counter.to(x.dtype)

    torch.manual_seed(1234)
    weight0 = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16) * 0.05
    bias0 = torch.randn(32, device="cuda", dtype=torch.bfloat16) * 0.05
    inputs = [torch.randn(16, 32, device="cuda", dtype=torch.bfloat16) for _ in range(12)]
    results = []
    old_flag = training.cuda_graphs
    try:
        for enabled in (False, True):
            training.cuda_graphs = enabled
            weight = weight0.clone().requires_grad_()
            bias = bias0.clone().requires_grad_()
            main_grad = torch.empty_like(weight, dtype=torch.float32)
            counter = torch.zeros((), device="cuda", dtype=torch.int64)
            observed = []
            instance_counts = []
            for step in range(4):
                capture_pending()
                main_grad.untyped_storage().resize_(0)
                xs = [x.clone().requires_grad_() for x in inputs[step * 3 : step * 3 + 3]]
                ys = [region(x, weight, bias, main_grad, counter) for x in xs]
                main_grad.untyped_storage().resize_(main_grad.numel() * main_grad.element_size())
                main_grad.zero_()
                for y in reversed(ys):
                    y.float().square().sum().backward()
                observed.extend([y.detach().clone() for y in ys])
                observed.extend([x.grad.clone() for x in xs])
                observed.extend([main_grad.clone(), bias.grad.clone(), counter.clone()])
                with torch.no_grad():
                    weight.add_(main_grad.to(weight.dtype), alpha=-0.01)
                    bias.add_(bias.grad, alpha=-0.01)
                    bias.grad = None
                    counter.zero_()
                # Replace weights as FSDP does, rather than relying on fixed device addresses.
                weight = weight.detach().clone().requires_grad_()
                bias = bias.detach().clone().requires_grad_()
                instance_counts.append(graph_stats()["instances"])
            if enabled:
                assert instance_counts[-1] == instance_counts[-2] > 0
            results.append(observed)
        for reference, captured in zip(*results):
            torch.testing.assert_close(captured, reference, rtol=0, atol=0)
        stats = graph_stats()
        assert stats["forward_replays"] >= 9
        assert stats["backward_replays"] >= 9
    finally:
        training.cuda_graphs = old_flag
