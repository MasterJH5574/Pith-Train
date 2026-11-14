"""Replay must survive live micro-batches, parameter updates and FSDP storage replacement."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("static_side", ["source", "destination"])
@torch.no_grad()
def test_mixed_copy_words_preserve_bits_versions_and_updates(static_side):
    from pithtrain.pipeline import cuda_graphs as cg

    # Include arbitrary IEEE bits (NaNs included), dense transposes, nonzero aligned offsets,
    # and scalar counters. The plan changes transfers, never their numeric interpretation.
    torch.manual_seed(83)
    bits = torch.randint(-(2**31), 2**31 - 1, (66,), device="cuda", dtype=torch.int32)
    bits[2:8] = torch.tensor(
        [0, -2147483648, 2139095040, -8388608, 2143289345, 2139095041],
        device="cuda",
        dtype=torch.int32,
    )
    sources = [
        bits[2:].view(torch.float32).view(8, 8).T,
        torch.arange(34, device="cuda", dtype=torch.bfloat16)[2:].view(4, 8),
        torch.arange(16, device="cuda", dtype=torch.int64),
        torch.tensor(13, device="cuda", dtype=torch.int32),
        torch.randint(0, 256, (32,), device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn),
    ]
    destinations = [
        torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device) for x in sources
    ]
    static = sources if static_side == "source" else destinations
    groups = cg.copy_groups(static)
    words = cg.static_copy_words(static, groups)
    assert words is not None
    options = {"source_words" if static_side == "source" else "destination_words": words}
    for update in range(3):
        if update:
            bits.add_(117)
            sources[1].add_(1)
        versions = [x._version for x in destinations]
        assert cg.copy_grouped(destinations, sources, groups, **options)
        for source, destination, version in zip(sources, destinations, versions):
            torch.testing.assert_close(
                cg.copy_word_view(destination), cg.copy_word_view(source), rtol=0, atol=0
            )
            assert destination._version == version + 1
        # Only the graph's static side may be cached; dynamic addresses change between calls.
        if static_side == "source":
            destinations = [
                torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
                for x in sources
            ]
        else:
            sources = [x.clone(memory_format=torch.preserve_format) for x in sources]


@pytest.mark.parametrize(
    "case", ["offset", "dlpack", "negative", "conjugate", "strided", "odd", "homogeneous"]
)
@torch.no_grad()
def test_mixed_copy_words_keep_logical_view_and_layout_fallback(case):
    from pithtrain.pipeline import cuda_graphs as cg

    first = torch.arange(16, device="cuda", dtype=torch.float32)
    second = torch.arange(8, device="cuda", dtype=torch.bfloat16)
    if case == "conjugate":
        first = torch.complex(first, first + 1).conj()
    elif case == "negative":
        first = torch._neg_view(first)
    elif case == "offset":
        second = torch.arange(9, device="cuda", dtype=torch.bfloat16)[1:]
    elif case == "dlpack":
        second = torch.from_dlpack(torch.arange(9, device="cuda", dtype=torch.bfloat16)[1:])
        assert second.storage_offset() == 0 and second.data_ptr() % 4 == 2
    elif case == "strided":
        second = torch.arange(16, device="cuda", dtype=torch.bfloat16)[::2]
    elif case == "odd":
        second = second[:7]
    elif case == "homogeneous":
        second = second.float()
    sources = [first, second]
    destinations = [
        torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device) for x in sources
    ]
    groups = cg.copy_groups(destinations)
    words = cg.static_copy_words(destinations, groups)
    assert not cg.copy_grouped(destinations, sources, groups, destination_words=words)
    for actual, expected in zip(destinations, sources):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mixed_copy_words_preserve_grad_enabled_copy_semantics():
    from pithtrain.pipeline import cuda_graphs as cg

    source = torch.arange(8, device="cuda", dtype=torch.float32, requires_grad=True)
    destinations = [torch.empty_like(source), torch.zeros((), device="cuda", dtype=torch.int32)]
    sources = [source, torch.ones_like(destinations[1])]
    groups = cg.copy_groups(destinations)
    words = cg.static_copy_words(destinations, groups)
    assert words is not None
    assert not cg.copy_grouped(destinations, sources, groups, destination_words=words)
    # The pinned foreach copy builds an autograd node but deliberately has no derivative.
    # Match its behavior rather than treating a bit view as a differentiable copy.
    reference = [torch.empty_like(source), torch.zeros_like(destinations[1])]
    torch._foreach_copy_(reference, sources)
    assert destinations[0].requires_grad == reference[0].requires_grad
    assert destinations[0].grad_fn is not None and reference[0].grad_fn is not None
    for result in (reference[0], destinations[0]):
        with pytest.raises(RuntimeError, match="derivative for _foreach_copy_ is not implemented"):
            result.sum().backward()
    assert source.grad is None
    leaf = torch.empty_like(source, requires_grad=True)
    destinations[0] = leaf
    words = cg.static_copy_words(destinations, groups)
    with pytest.raises(RuntimeError, match="leaf Variable"):
        cg.copy_grouped(destinations, sources, groups, destination_words=words)


def test_mixed_copy_words_mutation_is_visible_to_autograd():
    from pithtrain.pipeline import cuda_graphs as cg

    value = torch.arange(8, device="cuda", dtype=torch.float32, requires_grad=True)
    loss = value.square().sum()
    destinations = [value, torch.zeros((), device="cuda", dtype=torch.int32)]
    sources = [torch.full_like(value, 9), torch.ones_like(destinations[1])]
    groups = cg.copy_groups(destinations)
    with torch.no_grad():
        words = cg.static_copy_words(destinations, groups)
        assert cg.copy_grouped(destinations, sources, groups, destination_words=words)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@torch.no_grad()
def test_mixed_copy_replay_readonly_aliases_and_release():
    import weakref

    from pithtrain.pipeline import cuda_graphs as cg

    def function(args):
        first, second = args
        return [first + 1, second + 2, first]

    cg.release_graphs()
    for cycle in range(2):
        graph = cg.StaticGraph(function, (0, 1), (), False)
        raw = torch.full((32,), 2**32 + cycle, dtype=torch.int64, device="cuda")
        args = [raw.view(torch.float32), raw]
        graph(args)
        graph.capture()
        words = [weakref.ref(x) for x in graph.input_word_views + graph.output_word_views]
        for update in range(3):
            raw = torch.full((32,), 2**32 + update, dtype=torch.int64, device="cuda")
            args = [raw.view(torch.float32), raw]
            first, second, alias = graph(list(args))
            torch.testing.assert_close(first, args[0] + 1, rtol=0, atol=0)
            torch.testing.assert_close(second, args[1] + 2, rtol=0, atol=0)
            assert alias is args[0]
        assert graph.word_input_copies >= 3 and graph.word_output_copies == 3
        cg.release_graphs()
        assert not hasattr(graph, "input_word_views")
        assert not hasattr(graph, "output_word_views")
        assert all(reference() is None for reference in words)


@pytest.mark.parametrize("save_internal", [False, True], ids=["no-private-save", "private-save"])
def test_executor_stage1_stage5_retains_fresh_residual_after_reuse(monkeypatch, save_internal):
    from pithtrain.contexts import distributed, training
    from pithtrain.pipeline import cuda_graphs as cg
    from pithtrain.pipeline.execution import ExecutionCtx, stage1_b, stage1_f, stage5_b, stage5_f

    cg.release_graphs()
    monkeypatch.setattr(training, "cuda_graphs", True)
    monkeypatch.setattr(distributed, "ep_size", 2, raising=False)

    class Layer(torch.nn.Module):
        idx = 0

        def __init__(self):
            super().__init__()
            self.norm = torch.nn.Identity()
            self.weight = torch.nn.Parameter(torch.randn(16, 16, device="cuda") * 0.1)
            self.save_internal = save_internal
            cg.declare_pipeline_outputs(self, Layer, ("norm",), ("forward_stage1", "compute"))

        @cg.compile_stage(options={}, pipeline_outputs=(0,))
        def compute(self, x):
            residual = x.sin()
            intermediate = residual @ self.weight if self.save_internal else residual
            return self.norm(intermediate).cos(), residual

        def forward_stage1(self, x, rotary, cu_seqlens=None):
            hidden, residual = self.compute(x)
            # The EP>1 dispatch consumer always materializes a separate token allocation.
            return hidden.clone(), residual, None

        def forward_stage5(self, moe, indices, weights, residual):
            return moe + residual

    layer = Layer()
    ctx = ExecutionCtx(comm_stream=torch.cuda.Stream())
    retained = []
    original = layer.forward_stage5

    def retaining_consumer(moe, indices, weights, residual):
        retained.append((residual.detach(), residual.detach().clone()))
        return original(moe, indices, weights, residual)

    try:
        for step in range(3):
            cg.capture_pending()
            with cg.graph_step([layer], max_leases=2, executor_outputs=True):
                for pair in range(2):
                    pending = []
                    for mb in range(2):
                        x = torch.full((8, 16), step + pair + mb + 0.25, device="cuda")
                        first, dispatch, residual, _ = stage1_f(ctx, layer, x, None)
                        assert dispatch.untyped_storage()._cdata not in cg.owners[layer].saved
                        # Change the downstream consumer after the producing forward.
                        layer.forward_stage5 = retaining_consumer
                        fifth, result = stage5_f(ctx, layer, dispatch, None, residual)
                        intermediate = x.sin() @ layer.weight if save_internal else x.sin()
                        torch.testing.assert_close(result, intermediate.cos() + x.sin())
                        pending.append((x, first, fifth))
                    for x, first, fifth in reversed(pending):
                        gradients = stage5_b(ctx, layer, fifth, (torch.ones_like(x),))
                        actual = stage1_b(ctx, layer, first, (gradients[0], gradients[2], None))
                        term = (
                            (x.sin() @ layer.weight).sin() @ layer.weight.T
                            if save_internal
                            else x.sin().sin()
                        )
                        expected = x.cos() * (1 - term)
                        torch.testing.assert_close(actual, expected)
                    for actual, expected in retained:
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        stats = cg.graph_stats()
        assert stats["forward_replays"] > 0 and stats["backward_replays"] > 0
        assert bool(stats["pipeline_output_replays"]) == save_internal
        assert stats["pipeline_output_slots"] == (2 if save_internal else 0)
        assert stats["live_saved_storages"] == 0
    finally:
        cg.release_graphs()
        torch._dynamo.reset()


def test_executor_default_device_context_does_not_authorize_observing_modes(monkeypatch):
    from torch.overrides import TorchFunctionMode
    from torch.utils._device import DeviceContext
    from torch.utils._python_dispatch import TorchDispatchMode

    from pithtrain.contexts import distributed
    from pithtrain.pipeline import cuda_graphs as cg

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.Identity()
            cg.declare_pipeline_outputs(self, Layer, ("norm",), ("forward",))

        def forward(self, x):
            return self.norm(x)

    class FunctionObserver(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    class DispatchObserver(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    class DeviceObserver(DeviceContext):
        pass

    monkeypatch.setattr(distributed, "ep_size", 2, raising=False)
    layer = Layer()
    with cg.graph_step([layer], executor_outputs=True), torch.device("cuda"):
        assert cg.pipeline_outputs_allowed(layer)
        with FunctionObserver():
            assert not cg.pipeline_outputs_allowed(layer)
        with DispatchObserver():
            assert not cg.pipeline_outputs_allowed(layer)
        with DeviceObserver("cuda"):
            assert not cg.pipeline_outputs_allowed(layer)
        mode = DeviceContext("cuda")
        with mode:
            mode.__torch_function__ = FunctionObserver().__torch_function__
            assert not cg.pipeline_outputs_allowed(layer)


@torch._dynamo.config.patch(recompile_limit=32)
def test_executor_activation_lease_keeps_observers_and_direct_calls_fresh(monkeypatch):
    from pithtrain.contexts import distributed, training
    from pithtrain.pipeline import cuda_graphs as cg
    from pithtrain.pipeline.execution import run_backward

    cg.release_graphs()
    monkeypatch.setattr(training, "cuda_graphs", True)
    monkeypatch.setattr(distributed, "ep_size", 2, raising=False)
    # Retaining hooks deliberately change their list-length guard; test the replay backend's
    # bounded variant fallback rather than stopping at Dynamo's smaller default cache limit.

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.Identity()
            self.weight = torch.nn.Parameter(torch.randn(16, 16, device="cuda"))
            self.tracked = []
            self.before_compute = None
            cg.declare_pipeline_outputs(self, Layer, ("norm",), ("forward", "compute"))

        @cg.compile_stage(options={}, pipeline_outputs=(0,))
        def compute(self, x):
            residual = (x @ self.weight).sin()
            hidden = self.norm(residual).cos()
            return hidden, residual, hidden.sum()

        def forward(self, x):
            if self.before_compute is not None:
                self.before_compute()
            hidden, residual, loss = self.compute(x)
            self.tracked.append(loss.detach())
            return hidden, residual, loss

    layer = Layer()
    retained, expected_tracked = [], []

    def pair(mode, value):
        outputs = []
        for index in range(2):
            x = torch.full((8, 16), value + index, device="cuda", requires_grad=True)
            out = (
                cg.pipeline_call(layer, layer.forward, x)
                if mode == "executor"
                else layer.forward(x)
            )
            expected = (x @ layer.weight).sin()
            torch.testing.assert_close(out[0], expected.cos())
            torch.testing.assert_close(out[1], expected)
            expected_tracked.append(out[2].detach().clone())
            # A downstream consumer can retain a residual after its producer's backward.
            retained.append((out[1], out[1].detach().clone()))
            outputs.append(out)
            if mode != "executor":
                retained.extend((t, t.detach().clone()) for t in out)
        for out in reversed(outputs):
            run_backward(list(out), [torch.ones_like(t) for t in out], graph_owner=layer)

    try:
        for step in range(4):
            cg.capture_pending()
            with cg.graph_step([layer], max_leases=2, executor_outputs=True):
                pair("executor", step + 1.0)
                before = cg.graph_stats()["pipeline_output_replays"]
                # An objective can install a retaining hook in the middle of the step.
                hooks = []

                def install_hook():
                    hooks.append(
                        layer.norm.register_forward_hook(
                            lambda module, args, out: retained.append((out, out.detach().clone()))
                        )
                    )
                    layer.before_compute = None

                # Like a router-replay callback: installs the observer after executor entry.
                layer.before_compute = install_hook
                pair("executor", step + 3.0)
                hooks[0].remove()
                assert cg.graph_stats()["pipeline_output_replays"] == before
                pair("direct", step + 5.0)
                assert cg.graph_stats()["pipeline_output_replays"] == before
                assert "forward" not in layer.norm.__dict__
                layer.norm.forward = lambda x: x + 0
                pair("executor", step + 7.0)
                del layer.norm.forward
                assert cg.graph_stats()["pipeline_output_replays"] == before
                pair("executor", step + 9.0)
            for actual, expected in retained:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for actual, expected in zip(layer.tracked, expected_tracked):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        stats = cg.graph_stats()
        assert stats["pipeline_output_replays"] > 0
        assert stats["pipeline_output_slots"] == 2
        assert stats["live_saved_storages"] == 0
    finally:
        cg.release_graphs()
        torch._dynamo.reset()


@pytest.mark.parametrize("restored_candidates", [False, True])
def test_cpp_wrapper_autotuned_reduction_resolves_future_once(monkeypatch, restored_candidates):
    from torch._inductor import config
    from torch._inductor.async_compile import AsyncCompile
    from torch._inductor.codecache import CodeCacheFuture, LambdaFuture, StaticAutotunerFuture
    from torch._inductor.runtime import triton_lazy_compile

    from pithtrain.pipeline import cuda_graphs as cg

    cg.install_lazy_kernel_metadata_fix()
    helper = triton_lazy_compile.run_triton_kernel_with_autotune
    resolutions = {}
    candidates = []

    def observe_helper(pending, name, stream, args):
        future = pending[name]
        if isinstance(future, CodeCacheFuture):
            resolutions[future] = 0
        return helper(pending, name, stream, args)

    def observe_result(original):
        def result(future, *args, **kwargs):
            kernel = original(future, *args, **kwargs)
            if future in resolutions:
                resolutions[future] += 1
                candidates.append(len(kernel.launchers))
                if restored_candidates and len(kernel.launchers) > 1:
                    # Match the observed restore state: a valid cached numerical launcher
                    # survives alongside a rebuilt candidate list. Its fast path must not
                    # bypass selection before the helper derives grid and binary metadata.
                    kernel._cached_launcher = kernel.launchers[0]
            return kernel

        return result

    monkeypatch.setattr(triton_lazy_compile, "run_triton_kernel_with_autotune", observe_helper)
    for cls in (LambdaFuture, StaticAutotunerFuture):
        monkeypatch.setattr(cls, "result", observe_result(cls.result))

    # Force a real asynchronous, multi-candidate reduction rather than a mocked
    # future: resolving it again can restore the candidates after autotuning.
    with config.patch(compile_threads=2, force_disable_caches=True):
        AsyncCompile.process_pool().submit(AsyncCompile._get_ready).result()

        @torch.compile(fullgraph=True, options={"cpp_wrapper": True})
        def reduction(x, weight):
            x.add_(1)
            return (x * weight).sum(dim=0)

        x = torch.randn(4096, 2880, device="cuda")
        weight = torch.randn(2880, device="cuda")
        for _ in range(2):
            expected_x = x + 1
            expected = (expected_x * weight).sum(dim=0)
            torch.testing.assert_close(reduction(x, weight), expected)
            torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
    assert resolutions, "fixture did not exercise asynchronous kernel resolution"
    assert all(count == 1 for count in resolutions.values())
    assert max(candidates) > 1, "fixture did not require launcher selection"
    torch._dynamo.reset()


def test_cpp_wrapper_reused_kernel_keeps_its_own_binary(monkeypatch):
    from dataclasses import replace

    from pithtrain.kernels import stage_compile_options
    from pithtrain.pipeline import cuda_graphs as cg

    cg.install_lazy_kernel_metadata_fix()
    repaired = []

    def observe_metadata(result, **fields):
        repaired.append(result.cubin_path != fields["cubin_path"])
        return replace(result, **fields)

    monkeypatch.setattr(cg, "replace", observe_metadata)
    options = {"cpp_wrapper": True, **stage_compile_options()}

    @torch.compile(fullgraph=True, options=options)
    def first(x, y, weight):
        x.add_(1)
        return (x + y) @ weight

    @torch.compile(fullgraph=True, options=options)
    def different(x, y, z, weight):
        x.add_(1)
        return (x + y + z) @ weight

    @torch.compile(fullgraph=True, options=options)
    def reuse(x, y, weight):
        x.add_(1)
        return ((x + y) @ weight).sin()

    x, y, z, weight = [torch.randn(32, 32, device="cuda") for _ in range(4)]
    # The add kernel has the same descriptive name but a different pointer count in
    # `different`. `reuse` needs the original cached kernel inside a new C++ module.
    for fn, args in (
        (first, (x, y, weight)),
        (different, (x, y, z, weight)),
        (reuse, (x, y, weight)),
    ):
        for _ in range(2):
            expected_x = x + 1
            expected = (
                (expected_x + y + z) @ weight if fn is different else (expected_x + y) @ weight
            )
            if fn is reuse:
                expected = expected.sin()
            torch.testing.assert_close(fn(*args), expected)
            # Metadata refresh may never execute the numerical kernel a second time.
            torch.testing.assert_close(x, expected_x, rtol=0, atol=0)
    assert any(repaired), "fixture did not exercise the pinned compiler's stale metadata"
    torch._dynamo.reset()


@pytest.mark.parametrize("mutated", (False, True))
def test_repeated_outputs_and_output_only_inputs(mutated):
    from pithtrain.pipeline.cuda_graphs import StaticGraph

    def fn(args):
        x, passthrough = args
        if mutated:
            passthrough.zero_()
        y = x.sin()
        sliced = x.cos()[:, 1:]
        y_view = y.view_as(y)
        sliced_view = sliced.view_as(sliced)
        return [y, y_view, y_view, sliced, sliced_view, sliced_view, passthrough]

    def inputs():
        return [torch.randn(16, 32, device="cuda"), torch.randn(1024, 1024, device="cuda")]

    graph = StaticGraph(fn, (0,), (1,) if mutated else (), backward=False)
    graph(inputs())
    graph.capture()
    saved = []
    for _ in range(3):
        args = inputs()
        expected = args[0].sin()
        expected_slice = args[0].cos()[:, 1:]
        expected_passthrough = torch.zeros_like(args[1]) if mutated else args[1].clone()
        y, y_view, repeated, sliced, sliced_view, repeated_slice, passthrough = graph(args.copy())
        assert y_view is repeated and sliced_view is repeated_slice
        assert y_view is not y and sliced_view is not sliced
        assert y_view.untyped_storage()._cdata == y.untyped_storage()._cdata
        assert sliced_view.untyped_storage()._cdata == sliced.untyped_storage()._cdata
        assert passthrough is args[1]
        torch.testing.assert_close(passthrough, expected_passthrough, rtol=0, atol=0)
        assert sliced.stride() == expected[:, 1:].stride()
        y.add_(1)
        torch.testing.assert_close(repeated, expected + 1, rtol=0, atol=0)
        y_view.transpose_(0, 1)
        sliced_view.transpose_(0, 1)
        assert y.shape == expected.shape and sliced.shape == expected_slice.shape
        assert repeated.shape == expected.T.shape
        assert repeated_slice.shape == expected_slice.T.shape
        torch.testing.assert_close(repeated, (expected + 1).T, rtol=0, atol=0)
        torch.testing.assert_close(repeated_slice, expected_slice.T, rtol=0, atol=0)
        saved.append((y, expected + 1, sliced, expected_slice, y_view, sliced_view))
    for y, expected, sliced, expected_slice, y_view, sliced_view in saved:
        torch.testing.assert_close(y, expected, rtol=0, atol=0)
        torch.testing.assert_close(sliced, expected_slice, rtol=0, atol=0)
        torch.testing.assert_close(y_view, expected.T, rtol=0, atol=0)
        torch.testing.assert_close(sliced_view, expected_slice.T, rtol=0, atol=0)
    assert graph.replays == 3
    assert (graph.inputs[1].untyped_storage().nbytes() > 0) == mutated


@pytest.mark.parametrize("conjugate", (False, True))
def test_shared_output_storage_with_different_logical_values(conjugate):
    from pithtrain.pipeline.cuda_graphs import StaticGraph

    def fn(args):
        y = args[0].square()
        return [y, y.conj() if conjugate else torch._neg_view(y)]

    dtype = torch.complex64 if conjugate else torch.float32
    x = torch.randn(32, device="cuda", dtype=dtype)
    graph = StaticGraph(fn, (0,), (), backward=False)
    graph([x])
    graph.capture()
    x = torch.randn_like(x)
    expected = fn([x])
    actual = graph([x])
    for out, reference in zip(actual, expected):
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
    assert graph.replays == 1


def test_replay_copies_mixed_dtypes_and_layouts():
    from pithtrain.pipeline.cuda_graphs import StaticGraph

    def inputs():
        dense = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
        return [
            dense,
            dense.float().t(),
            torch.randn(8, 32, device="cuda", dtype=torch.bfloat16)[:, ::2],
            torch.randint(0, 8, (8, 16), device="cuda"),
            dense.to(torch.float8_e4m3fn),
            torch.randn(8, 1, device="cuda").expand(8, 16),
        ]

    def fn(args):
        dense, transposed, sliced, integer, fp8, expanded = args
        transposed.add_(dense.t().float())
        return [
            dense + sliced,
            transposed.square().t(),
            (sliced * 2)[:, ::2],
            integer + 1,
            fp8.clone(),
            expanded.sum().expand(3),
        ]

    graph = StaticGraph(fn, tuple(range(6)), (1,), backward=False)
    graph(inputs())
    graph.capture()
    saved = []
    for _ in range(3):
        actual = inputs()
        expected = [x.clone() for x in actual]
        reference = fn(expected)
        replay = graph(actual.copy())
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        saved.extend(zip(replay, reference))
    for actual, expected in saved:
        # Float8 comparison through float preserves the represented values.
        torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
    assert graph.replays == 3


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

    @torch.compile(fullgraph=True, options={"cpp_wrapper": True})
    def aggregate(x, probabilities):
        return (x * probabilities.unsqueeze(-1)).sum(dim=1)

    @compile_stage(options={"cpp_wrapper": True})
    def region(x, weight, bias, main_grad, counter):
        counter.add_(1)
        # Like global-batch load balancing, updated state affects this micro-batch's gradient.
        return (BF16LinearFunc.apply(x, weight, main_grad) + bias).sin() / counter.to(x.dtype)

    torch.manual_seed(1234)
    weight0 = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16) * 0.05
    bias0 = torch.randn(32, device="cuda", dtype=torch.bfloat16) * 0.05
    inputs = [torch.randn(16, 4, 32, device="cuda", dtype=torch.bfloat16) for _ in range(12)]
    probabilities = [torch.randn(16, 4, device="cuda", dtype=torch.bfloat16) for _ in inputs]
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
                ps = [p.clone().requires_grad_() for p in probabilities[step * 3 : step * 3 + 3]]
                # Aggregation's backward stays ordinary and must receive fresh escaped gradients.
                ys = [
                    region(aggregate(x, p), weight, bias, main_grad, counter)
                    for x, p in zip(xs, ps)
                ]
                main_grad.untyped_storage().resize_(main_grad.numel() * main_grad.element_size())
                main_grad.zero_()
                for y in reversed(ys):
                    y.float().square().sum().backward()
                observed.extend([y.detach().clone() for y in ys])
                observed.extend([x.grad.clone() for x in xs])
                observed.extend([p.grad.clone() for p in ps])
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


def test_owned_layer_replay_lifetimes_and_refresh(monkeypatch):
    from pithtrain.contexts import training
    from pithtrain.pipeline import cuda_graphs as cg

    cg.release_graphs()
    monkeypatch.setattr(training, "cuda_graphs", True)
    layers = [
        torch.nn.Linear(32, 16, bias=False, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    for layer in layers:
        layer.main_grad = torch.zeros_like(layer.weight, dtype=torch.float32)

    def forward(args):
        x, weight, main_grad = args
        return [x @ weight.T, x, weight, main_grad]

    def backward(args):
        dy, x, weight, main_grad = args
        torch.addmm(main_grad, dy.T, x, out_dtype=torch.float32, out=main_grad)
        return [dy @ weight]

    fwd = cg.StaticGraph(forward, (0, 1), (), backward=False)
    bwd = cg.StaticGraph(backward, (0, 1, 2, 3), (3,), backward=True)
    optimizer = torch.optim.SGD([layer.weight for layer in layers], lr=0.001)
    saved_outputs = []
    for step in range(4):
        cg.capture_pending()
        # Simulate the authoritative first-backward zero and final reduce-scatter read. The
        # separate hook test exercises the actual retained-buffer lifecycle implementation.
        for layer in layers:
            layer.main_grad.zero_()
        with cg.graph_step(layers):
            pending = []
            expected_grads = [torch.zeros_like(layer.main_grad) for layer in layers]
            for microbatch in range(3):
                for i, layer in enumerate(layers):
                    x = torch.randn(8, 32, device="cuda", dtype=torch.bfloat16)
                    dy = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
                    expected = x @ layer.weight.T
                    with cg.graph_stage(layer):
                        out, saved_x, weight, main_grad = fwd([x, layer.weight, layer.main_grad])
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    assert weight is layer.weight and main_grad is layer.main_grad
                    pending.append((i, dy, saved_x, weight, main_grad))
                    saved_outputs.append((out, expected))
            for i, dy, x, weight, main_grad in reversed(pending):
                layer = layers[i]
                torch.addmm(
                    expected_grads[i], dy.T, x, out_dtype=torch.float32, out=expected_grads[i]
                )
                with cg.graph_stage(layer):
                    (dx,) = bwd([dy, x, weight, main_grad])
                torch.testing.assert_close(dx, dy @ weight, rtol=0, atol=0)
            for i, layer in enumerate(layers):
                torch.testing.assert_close(layer.main_grad, expected_grads[i], rtol=0, atol=0)
                layer.weight.grad = layer.main_grad.to(layer.weight.dtype)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        for out, expected in saved_outputs:
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert cg.graph_stats()["instances"] == 4
    for owner in (cg.owners[layer] for layer in layers):
        assert len(owner.graphs) == 2
        assert all(g.replays == 9 for g in owner.graphs.values())
    assert layers[0].main_grad.cuda_graph_retain
    # Distinct layers may reuse the same FSDP address: layer identity still selects distinct
    # mirrors. Replacing weights is observed in the next epoch without capturing more graphs.
    layers[1].weight = layers[0].weight
    with cg.graph_step(layers):
        x = torch.randn(8, 32, device="cuda", dtype=torch.bfloat16)
        for layer in layers:
            with cg.graph_stage(layer):
                out = fwd([x, layer.weight, layer.main_grad])[0]
            torch.testing.assert_close(out, x @ layer.weight.T, rtol=0, atol=0)
    mirrors = [cg.owners[layer].mirrors[("weight", "weight")] for layer in layers]
    assert mirrors[0].data_ptr() != mirrors[1].data_ptr()
    assert all(not mirror.requires_grad and mirror.grad_fn is None for mirror in mirrors)
    # Replacement accumulator and unsupported layout take the ordinary callable. No stale
    # captured pointer is used, and another owner's current gradient remains untouched.
    layer = layers[0]
    prior = layer.main_grad
    layer.main_grad = torch.zeros_like(prior)
    dy = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(8, 32, device="cuda", dtype=torch.bfloat16)
    with cg.graph_step(layers), cg.graph_stage(layer):
        bwd([dy, x, layer.weight, layer.main_grad])
        assert cg.owners[layer].graphs[bwd].fallbacks > 3
    expected = torch.zeros_like(prior)
    torch.addmm(expected, dy.T, x, out_dtype=torch.float32, out=expected)
    torch.testing.assert_close(layer.main_grad, expected, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="test cleanup"):
        with cg.graph_step(layers), cg.graph_stage(layer):
            raise RuntimeError("test cleanup")
    assert cg.active_owner is None and cg.step_owners is None
    cg.release_graphs()
    assert not prior.cuda_graph_retain


def test_retained_main_grad_zero_and_reduce_once():
    from types import SimpleNamespace

    from pithtrain.modules.wgrad_accumulation import _attach_main_grads

    main_grad = torch.empty(16, 32, device="cuda")
    main_grad.untyped_storage().resize_(0)
    main_grad.cuda_graph_retain = True
    param = SimpleNamespace(unsharded_accumulated_grad=None)
    reduced = []
    group = SimpleNamespace(reduce_grads=False, fsdp_params=[param], pre_backward=lambda: None)

    def reduce():
        if group.reduce_grads and param.unsharded_accumulated_grad is not None:
            reduced.append(param.unsharded_accumulated_grad.clone())
            param.unsharded_accumulated_grad = None

    group.post_backward = reduce
    _attach_main_grads(group, [(param, main_grad)])
    pointer = None
    for step in range(3):
        # DualPipeV also drains FSDP after forward-only execution between training steps.
        group.reduce_grads = True
        group.post_backward()
        assert len(reduced) == step
        group.reduce_grads = False
        for microbatch in range(4):
            group.pre_backward()
            torch.testing.assert_close(
                main_grad, torch.full_like(main_grad, microbatch), rtol=0, atol=0
            )
            main_grad.add_(1)
            group.post_backward()
        if pointer is None:
            pointer = main_grad.data_ptr()
        assert main_grad.data_ptr() == pointer
        group.reduce_grads = True
        group.post_backward()
        assert main_grad.data_ptr() == pointer
        assert len(reduced) == step + 1
        torch.testing.assert_close(reduced[-1], torch.full_like(main_grad, 4), rtol=0, atol=0)
    main_grad.cuda_graph_retain = False
    group.pre_backward()
    group.post_backward()
    assert main_grad.untyped_storage().nbytes() == 0


def test_owned_shared_scratch_rejects_stream_mask_and_shape_changes(monkeypatch):
    from pithtrain.contexts import training
    from pithtrain.pipeline import cuda_graphs as cg

    cg.release_graphs()
    monkeypatch.setattr(training, "cuda_graphs", True)
    layers = [torch.nn.Linear(32, 16, bias=False, device="cuda") for _ in range(3)]
    for layer in layers:
        layer.main_grad = torch.zeros_like(layer.weight)

    def fn(args):
        x, weight, main_grad = args
        return [x @ weight.T, main_grad]

    graph = cg.StaticGraph(fn, (0, 1), (), backward=False)
    x = torch.randn(8, 32, device="cuda")
    with cg.graph_step(layers), cg.graph_stage(layers[0]):
        graph([x, layers[0].weight, layers[0].main_grad])
    cg.capture_pending()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), cg.graph_step(layers), cg.graph_stage(layers[1]):
        actual = graph([x, layers[1].weight, layers[1].main_grad])[0]
        expected = x @ layers[1].weight.T
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert cg.owners[layers[1]].graphs[graph].key is None
    assert not getattr(layers[1].main_grad, "cuda_graph_retain", False)
    # The same signature with an unowned matrix must never borrow layer 0's weight mirror.
    value = torch.randn_like(layers[2].weight)
    with cg.graph_step(layers), cg.graph_stage(layers[2]):
        actual = graph([x, value, layers[2].main_grad])[0]
    torch.testing.assert_close(actual, x @ value.T, rtol=0, atol=0)
    assert cg.owners[layers[2]].graphs[graph].key is None
    assert not getattr(layers[2].main_grad, "cuda_graph_retain", False)
    # A new compiled variant with changed parameter geometry may not capture the old mirror.
    layer = layers[0]
    layer.weight = torch.nn.Parameter(torch.randn(24, 32, device="cuda"))
    layer.main_grad = torch.zeros_like(layer.weight)
    variant = cg.StaticGraph(fn, (0, 1), (), backward=False)
    with cg.graph_step(layers), cg.graph_stage(layer):
        actual = variant([x, layer.weight, layer.main_grad])[0]
    torch.testing.assert_close(actual, x @ layer.weight.T, rtol=0, atol=0)
    assert cg.owners[layer].graphs[variant].key is None
    assert not getattr(layer.main_grad, "cuda_graph_retain", False)
    assert cg.graph_stats()["instances"] == 1
    cg.release_graphs()


@pytest.mark.parametrize("recompute", (False, True))
def test_private_saves_compiled_optimizer_live_microbatches(monkeypatch, recompute):
    from pithtrain.contexts import training
    from pithtrain.pipeline import cuda_graphs as cg
    from pithtrain.pipeline.execution import run_backward

    cg.release_graphs()
    monkeypatch.setattr(training, "cuda_graphs", True)
    monkeypatch.setattr(training, "fp8", False, raising=False)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(32, 32, device="cuda") * 0.05)
            self.bias = torch.nn.Parameter(torch.randn(32, device="cuda") * 0.05)

        def compute(self, x):
            if recompute:
                x = torch.nn.functional.rms_norm(x, (32,))
            y = x @ self.weight + self.bias
            return y.sin().cos() + y.square()

        @cg.compile_stage(options={"cpp_wrapper": True})
        def forward(self, x):
            if recompute:
                from pithtrain.modules.recompute import norm_recompute

                return norm_recompute(self.compute, 32, x)
            return self.compute(x)

    layer = Layer()
    reference = Layer()
    reference.load_state_dict(layer.state_dict())
    optim = torch.optim.SGD(layer.parameters(), lr=0.0001)
    ref_optim = torch.optim.SGD(reference.parameters(), lr=0.0001)
    escaped = []
    for step in range(4):
        cg.capture_pending()
        xs = [torch.randn(8, 32, device="cuda").requires_grad_() for _ in range(3)]
        refs = [x.detach().clone().requires_grad_() for x in xs]
        expected = []
        with monkeypatch.context() as flags:
            flags.setattr(training, "cuda_graphs", False)
            expected = [reference(x) for x in refs]
            for y in reversed(expected):
                y.sum().backward()
        with cg.graph_step([layer], max_leases=3):
            with cg.graph_stage(layer):
                actual = [layer(x) for x in xs]
                with pytest.raises(RuntimeError, match="private saves"):
                    with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
                        pass
            assert cg.graph_stats()["live_saved_storages"] > 0
            for y in reversed(actual):
                run_backward([y], [torch.ones_like(y)], graph_owner=layer)
            assert cg.graph_stats()["live_saved_storages"] == 0
        for x, ref in zip(xs, refs):
            torch.testing.assert_close(x.grad, ref.grad)
        for y, ref in zip(actual, expected):
            torch.testing.assert_close(y, ref)
            escaped.append((y.detach(), ref.detach().clone()))
        for p, ref in zip(layer.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, ref.grad)
        optim.step()
        ref_optim.step()
        optim.zero_grad(set_to_none=True)
        ref_optim.zero_grad(set_to_none=True)
        for y, ref in escaped:
            torch.testing.assert_close(y, ref)
    owner = cg.owners[layer]
    assert len(owner.pools) == 1
    assert len(next(iter(owner.pools.values()))) == 3
    assert all(g.replays > 0 for g in owner.graphs.values())
    assert len(owner.graphs) == 6  # three private-save slots and their bound backwards
    assert cg.graph_stats()["direct_saved_instances"] == 3
    assert cg.graph_stats()["direct_saved_replays"] == 9
    cg.release_graphs()


def test_private_save_pool_bounds_aliases_hooks_and_streams(monkeypatch):
    from pithtrain.contexts import training
    from pithtrain.pipeline import cuda_graphs as cg

    cg.release_graphs()
    monkeypatch.setattr(training, "cuda_graphs", True)
    layer = torch.nn.Module()

    def forward(args):
        x = args[0]
        public = x.sin()
        private = x.cos()
        return [public, public[:, 1:], x[:, 1:], private[:, 1:], private]

    def backward(args):
        saved = args[0]
        return [saved[:, 1:], saved, saved, torch._neg_view(saved)]

    forward_graph = cg.StaticGraph(forward, (0,), (), False, saved_outputs=(1, 2, 3, 4))
    backward_graph = cg.StaticGraph(backward, (0,), (), True)
    saved_results = []
    for step in range(3):
        cg.capture_pending()
        with cg.graph_step([layer], max_leases=2):
            pending = []
            for mb in range(3):
                x = torch.randn(8, 32, device="cuda")
                with cg.graph_stage(layer):
                    result = forward_graph([x])
                pending.append((result, x.cos()))
                saved_results.append((result[0], x.sin()))
            owner = cg.owners[layer]
            assert len(owner.pools[forward_graph]) == 2
            assert sum(g.busy for g in owner.pools[forward_graph]) == 2
            for result, expected in reversed(pending):
                # Backward may use another stream; its returned aliases still must escape.
                stream = torch.cuda.Stream() if step == 2 else torch.cuda.current_stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream), cg.graph_stage(layer, backward=True):
                    outputs = backward_graph([result[4]])
                torch.cuda.current_stream().wait_stream(stream)
                assert outputs[1] is outputs[2]
                assert outputs[0].untyped_storage()._cdata == outputs[1].untyped_storage()._cdata
                torch.testing.assert_close(outputs[0], expected[:, 1:])
                torch.testing.assert_close(outputs[3], -expected)
                saved_results.append((outputs[1], expected))
            assert not owner.saved
            for g in owner.pools[forward_graph]:
                if g.graph is not None:
                    assert g.borrowed_outputs == {3, 4}
        for value, expected in saved_results:
            torch.testing.assert_close(value, expected)
    # An installed pack hook may retain arbitrary internal saves; borrowing must be disabled.
    with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
        with cg.graph_step([layer], max_leases=2), cg.graph_stage(layer):
            before = len(cg.owners[layer].saved)
            outputs = forward_graph([torch.randn(8, 32, device="cuda")])
            assert len(cg.owners[layer].saved) == before == 0
    assert forward_graph.fallbacks >= 3  # full pool on each step
    # Standalone callers lack a retirement contract and always receive escaping copies.
    cg.capture_pending()
    plain = cg.StaticGraph(forward, (0,), (), False, saved_outputs=(1, 2, 3, 4))
    x = torch.randn(8, 32, device="cuda")
    plain([x])
    plain.capture()
    assert not plain.borrowed_outputs
    first = plain([x])[4]
    plain([torch.randn_like(x)])
    torch.testing.assert_close(first, x.cos())
    cg.release_graphs()


def test_private_save_failed_backward_quarantines_slot(monkeypatch):
    from pithtrain.contexts import training
    from pithtrain.pipeline import cuda_graphs as cg

    cg.release_graphs()
    monkeypatch.setattr(training, "cuda_graphs", True)
    layer = torch.nn.Module()
    graph = cg.StaticGraph(
        lambda args: [args[0].sin(), args[0].cos()], (0,), (), False, saved_outputs=(1,)
    )
    with pytest.raises(RuntimeError, match="backward failed"):
        with cg.graph_step([layer], max_leases=1):
            with cg.graph_stage(layer):
                graph([torch.randn(8, device="cuda")])
            with cg.graph_stage(layer, backward=True):
                raise RuntimeError("backward failed")
    assert cg.active_owner is None and cg.step_owners is None
    with pytest.raises(RuntimeError, match="outstanding saved tensors"):
        with cg.graph_step([layer], max_leases=1):
            pass
    cg.release_graphs()


def test_private_save_cross_stream_reuse_and_variant_bound(monkeypatch):
    from pithtrain.contexts import training
    from pithtrain.pipeline import cuda_graphs as cg

    cg.release_graphs()
    monkeypatch.setattr(training, "cuda_graphs", True)
    layer = torch.nn.Module()
    forward = cg.StaticGraph(
        lambda args: [args[0].sin(), args[0].cos()], (0,), (), False, saved_outputs=(1,)
    )
    backward = cg.StaticGraph(lambda args: [args[0].square()], (0,), (), True)
    side = torch.cuda.Stream()
    foreign = torch.cuda.Stream()
    for step in range(3):
        cg.capture_pending()
        with cg.graph_step([layer], max_leases=1):
            x = torch.randn(4096, device="cuda")
            with cg.graph_stage(layer):
                first = forward([x])
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), cg.graph_stage(layer, backward=True):
                torch.cuda._sleep(20_000_000)
                (grad,) = backward([first[1]])
            # A foreign-stream forward falls back. Its wait must not consume the event
            # needed by the next actual replay on this slot's captured stream.
            if step:
                foreign.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(foreign), cg.graph_stage(layer):
                    fallback = forward([x + 2])
            # No caller wait or host assertion before reuse: the lease must wait for side.
            with cg.graph_stage(layer):
                second = forward([x + 1])
            with cg.graph_stage(layer, backward=True):
                backward([second[1]])
            torch.cuda.current_stream().wait_stream(side)
            torch.testing.assert_close(grad, x.cos().square())
            if step:
                torch.cuda.current_stream().wait_stream(foreign)
                torch.testing.assert_close(fallback[0], (x + 2).sin())
    owner = cg.owners[layer]
    # A changing compiler variant may fall back, but may not leave unbounded empty pools.
    with cg.graph_step([layer], max_leases=1):
        for i in range(cg.MAX_GRAPHS + 4):
            variant = cg.StaticGraph(
                lambda args: [args[0], args[0].cos()], (0,), (), False, saved_outputs=(1,)
            )
            with cg.graph_stage(layer):
                outputs = variant([x])
            with cg.graph_stage(layer, backward=True):
                backward([outputs[1]])
        assert len(owner.variants) == cg.MAX_GRAPHS
        assert len(owner.pools) <= cg.MAX_GRAPHS
    cg.release_graphs()


def test_private_saves_never_borrow_public_intermediate_bases(monkeypatch):
    from pithtrain.contexts import training
    from pithtrain.pipeline import cuda_graphs as cg
    from pithtrain.pipeline.execution import run_backward

    cg.release_graphs()
    monkeypatch.setattr(training, "cuda_graphs", True)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(32, 32, device="cuda"))

        @cg.compile_stage(options={"cpp_wrapper": True})
        def forward(self, x):
            z = (x @ self.weight).sin()
            # AOT may reconstruct these views from an extra intermediate-base output.
            return z[:, 1:], z[:, 2:]

    layer = Layer()
    previous = []
    for step in range(3):
        cg.capture_pending()
        with cg.graph_step([layer], max_leases=1):
            for mb in range(3):
                x = torch.randn(8, 32, device="cuda").requires_grad_()
                expected = (x @ layer.weight).sin().detach()
                outputs = layer(x)
                run_backward(
                    list(outputs), [torch.ones_like(t) for t in outputs], graph_owner=layer
                )
                previous.append((outputs, expected))
        for outputs, expected in previous:
            torch.testing.assert_close(outputs[0], expected[:, 1:])
            torch.testing.assert_close(outputs[1], expected[:, 2:])
            assert outputs[0].untyped_storage()._cdata == outputs[1].untyped_storage()._cdata
    assert cg.graph_stats()["forward_slots"] > 0
    cg.release_graphs()


def test_private_saves_survive_interleaved_slots_with_large_temporaries(monkeypatch):
    from pithtrain.contexts import training
    from pithtrain.pipeline import cuda_graphs as cg

    cg.release_graphs()
    monkeypatch.setattr(training, "cuda_graphs", True)
    layers = [torch.nn.Module() for _ in range(8)]

    def fn(args):
        x = args[0]
        public, private = x.sin(), x.cos()
        temporary = x.repeat(16).square()
        private.add_(temporary[: x.numel()])
        return [public, private]

    forward = cg.StaticGraph(fn, (0,), (), False, saved_outputs=(1,))
    backward = cg.StaticGraph(lambda args: [args[0]], (0,), (), True)
    for step in range(3):
        cg.capture_pending()
        with cg.graph_step(layers, max_leases=2):
            pending = []
            for layer in layers:
                for mb in range(2):
                    x = torch.randn(512 * 1024, device="cuda")
                    with cg.graph_stage(layer):
                        outputs = forward([x])
                    pending.append((layer, outputs[1], x.cos() + x.square()))
            # F0,F1,B0,F0,B1: replay an earlier captured slot while later captures' saves live.
            layer, saved, expected = pending.pop(0)
            with cg.graph_stage(layer, backward=True):
                backward([saved])
            x = torch.randn_like(saved)
            with cg.graph_stage(layer):
                outputs = forward([x])
            pending.append((layer, outputs[1], x.cos() + x.square()))
            for layer, saved, expected in reversed(pending):
                torch.testing.assert_close(saved, expected)
                with cg.graph_stage(layer, backward=True):
                    backward([saved])
        assert cg.graph_stats()["live_saved_storages"] == 0
    cg.release_graphs()


@pytest.mark.parametrize("saved_case", ("view", "negative", "mutated"))
def test_direct_saved_views_and_copying_fallbacks(monkeypatch, saved_case):
    from pithtrain.contexts import training
    from pithtrain.pipeline import cuda_graphs as cg

    cg.release_graphs()
    monkeypatch.setattr(training, "cuda_graphs", True)
    layer = torch.nn.Module()
    forward = cg.StaticGraph(
        lambda args: [args[0].sin(), args[0].cos().T], (0,), (), False, saved_outputs=(1,)
    )

    def backward_fn(args):
        saved, dy = args
        if saved_case == "mutated":
            saved.add_(dy)
            value = saved.square()
        else:
            value = (saved * dy).square()
        return [value, saved[:, 1:]]

    backward = cg.StaticGraph(backward_fn, (0, 1), (0,) if saved_case == "mutated" else (), True)
    escaped = []

    def forward_one():
        x = torch.randn(32, 64, device="cuda")
        with cg.graph_stage(layer):
            outputs = forward([x])
        saved, expected = outputs[1][:, 1:], x.cos().T[:, 1:]
        if saved_case == "negative":
            saved, expected = torch._neg_view(saved), -expected
        return saved, expected.clone()

    def backward_one(saved, expected, fallback=False):
        # Changed tangent signature broadcasts correctly in ordinary backward.
        dy = torch.ones((), device="cuda") if fallback else torch.randn_like(saved)
        if saved_case == "mutated":
            expected = expected + dy
            value = expected.square()
        else:
            value = (expected * dy).square()
        with cg.graph_stage(layer, backward=True):
            result, alias = backward([saved, dy])
        torch.testing.assert_close(result, value)
        torch.testing.assert_close(alias, expected[:, 1:])
        escaped.append((alias, expected[:, 1:].clone()))

    for step in range(3):
        cg.capture_pending()
        with cg.graph_step([layer], max_leases=2):
            first, second = forward_one(), forward_one()
            backward_one(*first)
            third = forward_one()
            backward_one(*second, fallback=step == 2)
            backward_one(*third)
        for alias, expected in escaped:
            torch.testing.assert_close(alias, expected)
    children = [g for g in cg.owners[layer].graphs.values() if g.backward]
    assert len(children) == 2 and all(g.graph is not None for g in children)
    if saved_case == "view":
        assert all(set(g.saved_inputs) == {0} for g in children)
        assert all(0 not in {i for i, dims in g.copy_sources} for g in children)
    else:
        assert all(not g.saved_inputs for g in children)
        assert all(0 in {i for i, dims in g.copy_sources} for g in children)
    assert cg.graph_stats()["live_saved_storages"] == 0
    cg.release_graphs()


def test_owned_refresh_foreign_first_stream(monkeypatch):
    from pithtrain.contexts import training
    from pithtrain.pipeline import cuda_graphs as cg

    monkeypatch.setattr(training, "cuda_graphs", True)
    cg.release_graphs()
    layer = torch.nn.Linear(32, 32, bias=False, device="cuda")
    with torch.no_grad():
        layer.weight.fill_(1)
    x = torch.ones(32, 32, device="cuda")
    graph = cg.StaticGraph(lambda args: [args[0] @ args[1].T], (0, 1), (), False)
    side = torch.cuda.Stream()
    try:
        with cg.graph_step([layer]), cg.graph_stage(layer):
            graph([x, layer.weight])
        cg.capture_pending()
        with cg.graph_step([layer]), cg.graph_stage(layer):
            (warm,) = graph([x, layer.weight])
        torch.testing.assert_close(warm, torch.full_like(warm, 32), rtol=0, atol=0)
        with torch.no_grad():
            layer.weight.fill_(2)
        # Public operands and the previous epoch are fully drained before the probe.
        torch.cuda.synchronize()
        before = cg.graph_stats()
        with cg.graph_step([layer]):
            with torch.cuda.stream(side):
                # Delay only the implementation's internal mirror refresh. The caller's
                # original weight and x are already ready for both streams.
                torch.cuda._sleep(200_000_000)
                with cg.graph_stage(layer):
                    (foreign,) = graph([x, layer.weight])
            # No user-visible operand depends on side. A stream-guard fallback must not
            # leave an unordered implementation-private write to the captured mirror.
            with cg.graph_stage(layer):
                (actual,) = graph([x, layer.weight])
        torch.cuda.synchronize()
        after = cg.graph_stats()
        torch.testing.assert_close(foreign, torch.full_like(foreign, 64), rtol=0, atol=0)
        torch.testing.assert_close(actual, torch.full_like(actual, 64), rtol=0, atol=0)
        assert after["forward_replays"] - before["forward_replays"] == 1
    finally:
        torch.cuda.synchronize()
        cg.release_graphs()


@pytest.mark.parametrize("enabled", (False, True), ids=("ordinary", "graphs"))
def test_router_hooks_keep_full_guards_after_trusted_step(monkeypatch, enabled):
    from pithtrain.contexts import distributed, training
    from pithtrain.pipeline import cuda_graphs as cg
    from pithtrain.pipeline.guard_skip import GuardSkip

    cg.release_graphs()
    torch._dynamo.reset()
    monkeypatch.setattr(training, "cuda_graphs", enabled)
    monkeypatch.setattr(training, "fp8", False, raising=False)
    for name in ("ep_size", "cp_size", "pp_size"):
        monkeypatch.setattr(distributed, name, 1, raising=False)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = torch.nn.Linear(32, 32, bias=False, device="cuda")
            self.router.router_replay = None
            self.router.load_balance_loss_fn = None

        @cg.compile_stage(options={"cpp_wrapper": True})
        def compute(self, x):
            return self.router(x).sin()

    layer, guard = Layer(), GuardSkip()
    if not guard.enabled:
        pytest.skip("guard skipping is disabled on this Torch version")
    x = torch.ones(32, 32, device="cuda")
    state = {"offset": 1}

    def hook(module, args, output):
        return output + state["offset"]

    def step():
        cg.capture_pending()
        with cg.graph_step([layer]):
            return layer.compute(x)

    handle = None
    try:
        expected = torch.nn.functional.linear(x, layer.router.weight)
        torch.testing.assert_close(guard.run(layer, [], step), expected.sin())
        assert guard.trusted
        handle = layer.router.register_forward_hook(hook)
        assert guard.signature(layer, []) is None
        for offset in (1, 3):
            state["offset"] = offset
            torch.testing.assert_close(guard.run(layer, [], step), (expected + offset).sin())
        handle.remove()
        handle = None
        torch.testing.assert_close(guard.run(layer, [], step), expected.sin())
    finally:
        if handle is not None:
            handle.remove()
        cg.release_graphs()
        torch._dynamo.reset()


@torch._dynamo.config.patch(recompile_limit=32)
@pytest.mark.parametrize(
    "observer", ["none", "prehook", "latehook", "producer", "temporary_compute"]
)
def test_private_producer_input_retained_until_backward(monkeypatch, observer):
    import pithtrain.pipeline.cuda_graphs as cg
    from pithtrain.contexts import distributed, training
    from pithtrain.pipeline.execution import ExecutionCtx, stage1_stage5_b, stage5_stage1_f

    cg.release_graphs()
    torch._dynamo.reset()
    monkeypatch.setattr(training, "cuda_graphs", True)
    monkeypatch.setattr(distributed, "ep_size", 2, raising=False)

    class Layer(torch.nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.idx = idx
            self.callback = None
            self.norm = torch.nn.Identity()
            self.weight = torch.nn.Parameter(torch.randn(16, 16, device="cuda") / 10)
            cg.declare_pipeline_outputs(
                self,
                Layer,
                ("norm",),
                ("forward_stage1", "compute", "forward_stage5"),
                producer_methods=("forward_stage5",),
            )

        @cg.compile_stage(options={}, pipeline_outputs=(0,))
        def compute(self, x):
            x = self.norm(x)
            intermediate = x.sin() @ self.weight
            return intermediate.cos(), x.cos()

        def forward_stage1(self, x, rotary, cu_seqlens=None):
            if self.callback is not None:
                self.callback()
            cg.private_input_after_callback(self)
            hidden, residual = self.compute(x)
            return hidden.clone(), residual, None

        def forward_stage5(self, moe, indices, weights, residual):
            return moe + residual

    first, second = Layer(0), Layer(1)
    ctx = ExecutionCtx(comm_stream=torch.cuda.Stream())
    observed = []

    def install_hook():
        second.norm.register_forward_pre_hook(
            lambda module, args: observed.append((args[0], args[0].detach().clone()))
        )
        second.callback = None

    if observer == "prehook":
        install_hook()
    elif observer == "latehook":
        second.callback = install_hook
    elif observer == "temporary_compute":
        original_compute = second.compute

        def install_wrapper():
            def temporary_compute(x):
                observed.append((x, x.detach().clone()))
                second.compute = original_compute
                return original_compute(x)

            second.compute = temporary_compute

        second.callback = install_wrapper
    elif observer == "producer":
        original = first.forward_stage5

        def producer(*args):
            value = original(*args)
            observed.append((value, value.detach().clone()))
            return value

        first.forward_stage5 = producer
    try:
        for step in range(3):
            cg.capture_pending()
            with cg.graph_step([first, second], max_leases=2, executor_outputs=True):
                for pair in range(2):
                    pending = []
                    for mb in range(2):
                        moe, residual = [torch.randn(8, 16, device="cuda") for _ in range(2)]
                        x = moe + residual
                        args, outs, dispatch, public, _ = stage5_stage1_f(
                            ctx, first, second, moe, None, residual, None
                        )
                        torch.testing.assert_close(dispatch, (x.sin() @ second.weight).cos())
                        torch.testing.assert_close(public, x.cos())
                        # The fresh producer output is independent of its caller-owned inputs.
                        moe.add_(2)
                        residual.sub_(3)
                        reference = None
                        if observer in ("producer", "temporary_compute"):
                            ref_x = x.detach().clone().requires_grad_()
                            with monkeypatch.context() as patch:
                                patch.setattr(training, "cuda_graphs", False)
                                ref_outputs = second.compute(ref_x)
                            ref_x.data.add_(0.25)
                            reference = ref_x, ref_outputs
                            # A custom producer may expose a .data alias without a version bump.
                            # Declining private retention must preserve this current input read.
                            value, expected_value = observed[-1]
                            value.data.add_(0.25)
                            expected_value.add_(0.25)
                        pending.append((x, args, outs, reference))
                        # Same compiled parent, but this direct public input must use ordinary
                        # scratch without overwriting either outstanding retained input.
                        if observer == "none":
                            ordinary = torch.randn_like(x, requires_grad=True)
                            a, b, _ = cg.pipeline_call(
                                second, second.forward_stage1, ordinary, None
                            )
                            with cg.graph_stage(second, backward=True):
                                (grad,) = torch.autograd.grad((a + b).sum(), ordinary)
                            expected = -(ordinary.sin() @ second.weight).sin() @ second.weight.T
                            torch.testing.assert_close(
                                grad, expected * ordinary.cos() - ordinary.sin()
                            )
                    for x, args, outs, reference in reversed(pending):
                        g = torch.ones_like(x)
                        actual, _, residual_grad = stage1_stage5_b(
                            ctx, second, first, outs, args, (g, g, None)
                        )
                        expected = -(x.sin() @ second.weight).sin() @ second.weight.T
                        expected = expected * x.cos() - x.sin()
                        if reference is not None:
                            ref_x, ref_outputs = reference
                            (expected,) = torch.autograd.grad(
                                sum(t.sum() for t in ref_outputs), ref_x
                            )
                        torch.testing.assert_close(actual, expected)
                        torch.testing.assert_close(residual_grad, expected)
            with torch.no_grad():
                second.weight.add_(0.01)
            for value, expected in observed:
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
        stats = cg.graph_stats()
        assert stats["live_saved_storages"] == 0
        assert bool(stats["retained_input_replays"]) == (observer == "none")
        if observer == "none":
            assert stats["retained_input_slots"] == 2
            retained = [g for g in cg.live_graphs() if g.retained_saved_outputs]
            assert len({g.inputs[g.retained_inputs[0]].data_ptr() for g in retained}) == 2
            for graph in cg.live_graphs():
                if (
                    graph.backward
                    and graph.saved_from is not None
                    and graph.saved_from.retained_inputs
                ):
                    indices = {
                        i
                        for i, item in graph.saved_inputs.items()
                        if item[0] in graph.saved_from.retained_saved_outputs
                    }
                    assert indices
                    assert indices.isdisjoint(i for i, dims in graph.copy_sources)
    finally:
        cg.release_graphs()
        torch._dynamo.reset()


@torch.no_grad()
def test_captured_transfers_addresses_versions_and_bounded_uploads():
    from pithtrain.pipeline import cuda_graphs as cg

    cg.release_graphs()
    owner = cg.StageOwner()

    def compute(args):
        x, counter = args
        counter.add_(x.sum())
        y = x * 3
        return [y, y, x, counter]

    graph = cg.StaticGraph(compute, (0, 1), (1,), False, owner=owner)
    x = torch.arange(32, device="cuda", dtype=torch.float32)
    counter = torch.zeros((), device="cuda")
    graph([x, counter])
    graph.capture()
    assert graph.transfers is not None
    # Capture must never execute a stateful counter update, including the planning capture.
    torch.testing.assert_close(counter, x.sum(), rtol=0, atol=0)
    kept = []
    for index in range(2 * cg.TRANSFER_UPLOAD_SLOTS + 3):
        value = torch.full_like(x, index + 1)
        accumulator = torch.tensor(float(index), device="cuda")
        version = accumulator._version
        out, repeated, alias, updated = graph([value, accumulator])
        assert repeated is out and alias is value and updated is accumulator
        assert accumulator._version == version + 1
        kept.append((out, accumulator, index))
    for out, accumulator, index in kept:
        torch.testing.assert_close(out, torch.full_like(out, 3 * (index + 1)), rtol=0, atol=0)
        torch.testing.assert_close(
            accumulator,
            torch.tensor(float(index + 32 * (index + 1)), device="cuda"),
            rtol=0,
            atol=0,
        )
    assert graph.transfer_replays == len(kept)
    assert cg.graph_stats()["transfer_upload_slots"] == cg.TRANSFER_UPLOAD_SLOTS
    cg.release_graphs()
    assert not cg.transfer_uploads and graph.transfers is None


@torch.no_grad()
def test_captured_transfers_foreign_storage_and_logical_fallback():
    from pithtrain.pipeline import cuda_graphs as cg

    cg.release_graphs()
    owner = cg.StageOwner()
    graph = cg.StaticGraph(lambda args: [args[0] * 2, args[1] + 1], (0, 1), (), False, owner=owner)
    graph([torch.ones(4096, device="cuda"), torch.ones(4, device="cuda", dtype=torch.bfloat16)])
    graph.capture()
    assert graph.transfers is not None
    current, producer = torch.cuda.current_stream(), torch.cuda.Stream()
    kept = []
    for index in range(50):
        with torch.cuda.stream(producer):
            x = torch.full((4096,), index + 1, device="cuda", dtype=torch.float32)
            y = torch.full((4,), index + 2, device="cuda", dtype=torch.bfloat16)
        current.wait_stream(producer)
        kept.append(graph([x, y]))
        del x, y
        with torch.cuda.stream(producer):
            torch.empty(4096, device="cuda").fill_(-100)
    for index, (x, y) in enumerate(kept):
        torch.testing.assert_close(x, torch.full_like(x, 2 * (index + 1)), rtol=0, atol=0)
        torch.testing.assert_close(y, torch.full_like(y, index + 3), rtol=0, atol=0)
    assert graph.transfer_replays == len(kept)
    before = graph.transfer_replays
    x = torch._neg_view(torch.ones(4096, device="cuda"))
    y = torch.ones(5, device="cuda", dtype=torch.bfloat16)[1:]
    a, b = graph([x, y])
    torch.testing.assert_close(a, x * 2, rtol=0, atol=0)
    torch.testing.assert_close(b, y + 1, rtol=0, atol=0)
    assert graph.transfer_replays == before
    cg.release_graphs()


@torch.no_grad()
def test_captured_transfer_pending_upload_backpressure(monkeypatch):
    from pithtrain.pipeline import cuda_graphs as cg

    cg.release_graphs()
    monkeypatch.setattr(cg, "TRANSFER_UPLOAD_SLOTS", 2)
    owner = cg.StageOwner()
    graph = cg.StaticGraph(lambda args: [args[0] * 2], (0,), (), False, owner=owner)
    graph([torch.ones(64, device="cuda")])
    graph.capture()
    assert graph.transfers is not None
    pool = graph.transfers.uploads
    values = [torch.full((64,), float(i), device="cuda") for i in range(6)]
    warm = [graph([value])[0] for value in values]
    torch.cuda.synchronize()
    del warm
    before = pool.waits
    # The third table submission must encounter a pending HtoD, not just a reused ready slot.
    torch.cuda._sleep(100_000_000)
    kept = [graph([value])[0] for value in values]
    assert pool.waits > before
    assert len(pool.slots) == 2
    for i, out in enumerate(kept):
        torch.testing.assert_close(out, torch.full_like(out, 2 * i), rtol=0, atol=0)
    cg.release_graphs()


@torch.no_grad()
def test_captured_transfer_overlapping_outputs_and_fallback_recovery():
    from pithtrain.pipeline import cuda_graphs as cg

    cg.release_graphs()
    owner = cg.StageOwner()

    def compute(args):
        output = args[0] * 3
        return [output, output[4:12]]

    graph = cg.StaticGraph(compute, (0,), (), False, owner=owner)
    graph([torch.ones(16, device="cuda")])
    graph.capture()
    assert graph.transfers is not None
    before = graph.transfer_replays
    kept = []
    for value in (
        torch.ones(16, device="cuda"),
        torch._neg_view(torch.ones(16, device="cuda")),
        torch.full((16,), 4.0, device="cuda"),
    ):
        outputs = graph([value])
        kept.append((outputs, value * 3))
    assert graph.transfer_replays == before + 2
    for (full, part), expected in kept:
        torch.testing.assert_close(full, expected, rtol=0, atol=0)
        torch.testing.assert_close(part, expected[4:12], rtol=0, atol=0)
    cg.release_graphs()


@torch.no_grad()
def test_captured_transfer_prepare_capture_replay_streams(monkeypatch):
    from pithtrain.pipeline import cuda_graphs as cg

    cg.release_graphs()
    monkeypatch.setattr(cg, "TRANSFER_UPLOAD_SLOTS", 2)
    owner = cg.StageOwner()
    first, capture_stream, producer = torch.cuda.Stream(), torch.cuda.Stream(), torch.cuda.Stream()
    graph = cg.StaticGraph(lambda args: [args[0] * 7], (0,), (), False, owner=owner)
    with torch.cuda.stream(first):
        graph([torch.ones(4096, device="cuda")])
    with torch.cuda.stream(capture_stream):
        graph.capture()
    assert graph.transfers is not None
    assert graph.transfers.stream.cuda_stream == first.cuda_stream
    with torch.cuda.stream(producer):
        values = [torch.full((4096,), float(i), device="cuda") for i in range(6)]
    with torch.cuda.stream(first):
        first.wait_stream(producer)
        warm = [graph([value])[0] for value in values]
        first.synchronize()
        del warm
        before = graph.transfers.uploads.waits
        torch.cuda._sleep(100_000_000)
        outputs = [graph([value])[0] for value in values]
        assert graph.transfers.uploads.waits > before
    del values
    with torch.cuda.stream(producer):
        torch.empty((6, 4096), device="cuda").fill_(-100)
    torch.cuda.current_stream().wait_stream(first)
    for index, output in enumerate(outputs):
        torch.testing.assert_close(output, torch.full_like(output, index * 7), rtol=0, atol=0)
    assert graph.transfer_replays == 12
    cg.release_graphs()


@pytest.mark.parametrize("change", ("stride", "offset", "negative"))
def test_direct_saved_guard_checks_each_current_view(monkeypatch, change):
    from pithtrain.contexts import training
    from pithtrain.pipeline import cuda_graphs as cg

    cg.release_graphs()
    monkeypatch.setattr(training, "cuda_graphs", True)
    layer = torch.nn.Module()
    forward = cg.StaticGraph(
        lambda args: [args[0].sin(), args[0].cos()], (0,), (), False, saved_outputs=(1,)
    )
    backward = cg.StaticGraph(lambda args: [args[0].square()], (0,), (), True)
    try:
        for step in range(4):
            cg.capture_pending()
            with cg.graph_step([layer], max_leases=2):
                pending = []
                for mb in range(2):
                    x = torch.randn(8, 9, device="cuda") + step + mb
                    with cg.graph_stage(layer):
                        outputs = forward([x])
                    saved, expected = outputs[1][:, :8], x.cos()[:, :8]
                    if step == 2:
                        # The slot identity stays live, but current view metadata changes.
                        if change == "stride":
                            saved, expected = saved.T, expected.T
                        elif change == "offset":
                            saved, expected = outputs[1][:, 1:], x.cos()[:, 1:]
                        else:
                            saved, expected = torch._neg_view(saved), -expected
                    pending.append((saved, expected))
                for saved, expected in reversed(pending):
                    before = cg.graph_stats()["backward_replays"]
                    with cg.graph_stage(layer, backward=True):
                        (actual,) = backward([saved])
                    torch.testing.assert_close(actual, expected.square())
                    assert cg.graph_stats()["backward_replays"] - before == int(step in (1, 3))
                assert cg.graph_stats()["live_saved_storages"] == 0
        # The global weak registry can still hold released wrappers from other tests.
        # Check this layer's children, whose lifetime and replay counts this fixture owns.
        children = tuple(cg.owners[layer].graphs.values())
        assert all(child.parent in (forward, backward) for child in children)
        assert sum(child.replays for child in children if not child.backward) == 6
        assert sum(child.replays for child in children if child.backward) == 4
    finally:
        cg.release_graphs()


@pytest.mark.parametrize(
    "candidate", ("exact", "stride", "offset", "dtype", "negative", "duplicate", "mutated")
)
def test_retained_input_filter_requires_one_complete_view_match(monkeypatch, candidate):
    from pithtrain.contexts import training
    from pithtrain.pipeline import cuda_graphs as cg

    cg.release_graphs()
    monkeypatch.setattr(training, "cuda_graphs", True)
    layer = torch.nn.Module()

    def forward_fn(args):
        if candidate == "mutated":
            args[0].add_(0.25)
        return [args[0].sin(), args[0].cos()]

    forward = cg.StaticGraph(
        forward_fn,
        (0, 1),
        (0,) if candidate == "mutated" else (),
        False,
        saved_outputs=(1,),
        pipeline_outputs=(0,),
    )
    backward = cg.StaticGraph(lambda args: [args[0].square()], (0,), (), True)
    try:
        for step in range(3):
            cg.capture_pending()
            with cg.graph_step([layer], max_leases=1, executor_outputs=True):
                value = torch.randn(8, 8, device="cuda") + step
                x = value
                if candidate == "stride":
                    x = value.T
                elif candidate == "offset":
                    x = value[1:]
                elif candidate == "dtype":
                    x = value.view(torch.int32)
                elif candidate == "negative":
                    x = torch._neg_view(value)
                other = value if candidate == "duplicate" else torch.randn_like(value)
                with monkeypatch.context() as state:
                    # Model/executor authorization is tested elsewhere; isolate the view filter.
                    state.setattr(cg, "active_pipeline_layer", layer)
                    state.setattr(cg, "active_private_input", (layer, value))
                    with cg.graph_stage(layer):
                        outputs = forward([x, other])
                torch.testing.assert_close(outputs[0], x.sin())
                with cg.graph_stage(layer, backward=True):
                    (actual,) = backward([outputs[1]])
                torch.testing.assert_close(actual, x.cos().square())
                assert cg.graph_stats()["live_saved_storages"] == 0
        forwards = [g for g in cg.owners[layer].graphs.values() if not g.backward]
        assert len(forwards) == 1 and forwards[0].replays == 2
        assert forwards[0].retained_inputs == ((0,) if candidate == "exact" else ())
    finally:
        cg.release_graphs()


@pytest.mark.parametrize(
    "change", ("stride", "empty_stride", "singleton_stride", "dtype", "scalar", "rank")
)
def test_native_signature_rechecks_mutable_metadata(change):
    from pithtrain.pipeline import cuda_graphs as cg

    x = (
        torch.empty_strided((0, 8), (8, 1), device="cuda")
        if change == "empty_stride"
        else torch.randn(8, 8, device="cuda")
    )
    if change == "singleton_stride":
        x = torch.empty_strided((1, 8), (8, 1), device="cuda")
    args = [x, 3, 0.5, None, True]
    graph = cg.StaticGraph(lambda args: args, (), (), False)
    graph.key, graph.signature_guard = cg.signature(args), cg.make_signature_guard(args)
    assert graph.matches_signature(args) and graph.native_signature_matches == 1
    if change == "stride":
        x.transpose_(0, 1)
    elif change in ("empty_stride", "singleton_stride"):
        x.as_strided_(x.shape, (32, 1))
    elif change == "dtype":
        args[0] = x.to(torch.bfloat16)
    elif change == "scalar":
        args[1] = 3.0
    else:
        args[0] = torch.ones((), device="cuda")
    assert cg.signature(args) != graph.key
    assert not graph.matches_signature(args)
    assert graph.native_signature_matches == 1 and graph.python_signature_checks == 1


def test_native_signature_preserves_python_fallback_and_releases_storage():
    import gc
    import weakref

    from pithtrain.pipeline import cuda_graphs as cg

    graph = cg.StaticGraph(lambda args: args, (), (), False)
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    x = torch.empty(1024 * 1024, device="cuda")
    graph.key, graph.signature_guard = cg.signature([x]), cg.make_signature_guard([x])
    assert graph.matches_signature([x])
    x.requires_grad_()
    assert not graph.signature_guard.check([x])
    assert graph.matches_signature([x])  # requires_grad is not part of the original signature
    parameter = torch.nn.Parameter(x)
    assert not graph.signature_guard.check([parameter])
    assert graph.matches_signature([parameter])
    ref = weakref.ref(x)
    del parameter, x
    gc.collect()
    assert ref() is None and torch.cuda.memory_allocated() == before
    assert graph.native_signature_matches == 1 and graph.python_signature_checks == 2


@pytest.mark.parametrize("mode_kind", ("function", "default_device", "patched_device"))
def test_native_signature_observer_modes_keep_original_queries(monkeypatch, mode_kind):
    from torch.overrides import TorchFunctionMode
    from torch.utils._device import DeviceContext

    from pithtrain.pipeline import cuda_graphs as cg

    x = torch.randn(8, 8, device="cuda")
    graph = cg.StaticGraph(lambda args: args, (), (), False)
    graph.key, graph.signature_guard = cg.signature([x]), cg.make_signature_guard([x])
    seen = []

    class MetadataObserver(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            if func is torch.Tensor.stride:
                seen.append(func)
                return (1, 8)
            return func(*args, **(kwargs or {}))

    if mode_kind == "function":
        mode = MetadataObserver()
    else:
        mode = DeviceContext("cuda")
        if mode_kind == "patched_device":
            monkeypatch.setattr(
                DeviceContext, "__torch_function__", MetadataObserver.__torch_function__
            )
    with mode:
        actual = graph.matches_signature([x])
        expected = cg.signature([x]) == graph.key
        assert actual == expected
    if mode_kind == "default_device":
        assert actual and graph.native_signature_matches == 1
    else:
        assert not actual and len(seen) == 2
        assert graph.native_signature_matches == 0 and graph.python_signature_checks == 1


@pytest.mark.parametrize("layout", ("coo", "csr", "mkldnn", "nested"))
def test_native_signature_layout_miss_restores_function_state(layout):
    from torch.overrides import TorchFunctionMode

    from pithtrain.pipeline import cuda_graphs as cg

    x = torch.randn(8, 8, device="cuda")
    graph = cg.StaticGraph(lambda args: args, (), (), False)
    graph.key, graph.signature_guard = cg.signature([x]), cg.make_signature_guard([x])
    if layout == "nested":
        changed = torch.nested.nested_tensor([x[:2], x[:3]])
        with pytest.raises(RuntimeError):
            graph.matches_signature([changed])
    else:
        changed = (
            x.to_sparse()
            if layout == "coo"
            else x.to_sparse_csr()
            if layout == "csr"
            else x.cpu().to_mkldnn()
        )
        assert not graph.matches_signature([changed])
    seen = []

    class Observer(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            seen.append(func)
            return func(*args, **(kwargs or {}))

    with Observer():
        assert cg.make_signature_guard([x]) is None
        assert graph.matches_signature([x])
    assert seen and graph.native_signature_matches == 0
    assert graph.python_signature_checks == 2


@pytest.mark.parametrize("field", ("dtype", "device"))
@pytest.mark.parametrize("mode_kind", ("function", "patched_device"))
def test_native_signature_does_not_cache_observer_metadata(monkeypatch, field, mode_kind):
    from torch.overrides import TorchFunctionMode
    from torch.utils._device import DeviceContext

    from pithtrain.pipeline import cuda_graphs as cg

    x = torch.randn(8, 8, device="cuda")
    getter = getattr(torch.Tensor, field).__get__
    reported = torch.bfloat16 if field == "dtype" else torch.device("cpu")

    class MetadataObserver(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            if func == getter:
                return reported
            return func(*args, **(kwargs or {}))

    if mode_kind == "function":
        mode = MetadataObserver()
    else:
        mode = DeviceContext("cuda")
        monkeypatch.setattr(
            DeviceContext, "__torch_function__", MetadataObserver.__torch_function__
        )
    graph = cg.StaticGraph(lambda args: args, (), (), False)
    with mode:
        graph.key = cg.signature([x])
        assert graph.key[0][2 if field == "dtype" else 3] == reported
        graph.signature_guard = cg.make_signature_guard([x], graph.key)
        assert graph.signature_guard is None
    assert not graph.matches_signature([x])
    assert graph.native_signature_matches == 0


@torch.no_grad()
def test_native_signature_strict_miss_keeps_real_replay():
    from pithtrain.pipeline import cuda_graphs as cg

    cg.release_graphs()
    graph = cg.StaticGraph(lambda args: [args[0].sin()], (0,), (), False)
    x = torch.randn(8, 8, device="cuda")
    try:
        graph([x])
        graph.capture()
        (first,) = graph([x])
        torch.testing.assert_close(first, x.sin())
        assert graph.native_signature_matches == 1 and graph.replays == 1
        x.requires_grad_()
        x.add_(0.25)
        (second,) = graph([x])
        torch.testing.assert_close(second, x.sin())
        parameter = torch.nn.Parameter(x.clone() + 0.5)
        (third,) = graph([parameter])
        torch.testing.assert_close(third, parameter.sin())
        torch.testing.assert_close(first, (x - 0.25).sin())
        assert graph.replays == 3
        assert graph.native_signature_matches == 1 and graph.python_signature_checks == 2
    finally:
        cg.release_graphs()


@pytest.mark.parametrize("retarget", ["none", "data", "set"])
def test_retained_input_foreign_backward_preserves_allocation_before_slot_reuse(
    monkeypatch, retarget
):
    from pithtrain.contexts import training
    from pithtrain.pipeline import cuda_graphs as cg

    cg.release_graphs()
    torch.cuda.synchronize()
    monkeypatch.setattr(training, "cuda_graphs", True)
    layer = torch.nn.Module()

    def compute(args):
        y = args[0] * 2
        return [y, args[0], y]

    replacement = None

    def backward_compute(args):
        result = args[0] * 3
        if replacement is not None:
            if retarget == "data":
                args[0].data = replacement
            elif retarget == "set":
                args[0].set_(replacement)
        args.clear()
        return [result]

    forward = cg.StaticGraph(compute, (0,), (), False, saved_outputs=(1, 2), pipeline_outputs=(0,))
    backward = cg.StaticGraph(backward_compute, (0,), (), True)
    side = torch.cuda.Stream()
    count = 1024 * 1024
    try:
        for _ in range(2):
            cg.capture_pending()
            with cg.graph_step([layer], max_leases=1, executor_outputs=True):
                x = torch.full((count,), 2.0, device="cuda")
                with monkeypatch.context() as fields:
                    fields.setattr(cg, "active_pipeline_layer", layer)
                    fields.setattr(cg, "active_private_input", (layer, x))
                    with cg.graph_stage(layer):
                        out = forward([x])
                with cg.graph_stage(layer, backward=True):
                    expected = backward([out[1]])
            torch.testing.assert_close(expected[0], torch.full_like(expected[0], 6), rtol=0, atol=0)
        del x, out, expected
        torch.cuda.synchronize()
        with cg.graph_step([layer], max_leases=1, executor_outputs=True):
            x = torch.full((count,), 2.0, device="cuda")
            pointer = x.data_ptr()
            replacement = torch.full_like(x, 7)
            with monkeypatch.context() as fields:
                fields.setattr(cg, "active_pipeline_layer", layer)
                fields.setattr(cg, "active_private_input", (layer, x))
                with cg.graph_stage(layer):
                    out = forward([x])
            lease = next(g for g in cg.owners[layer].graphs.values() if not g.backward and g.busy)
            assert lease.graph is not None and lease.retained_saved_outputs == {1}
            assert lease.live_outputs[1] is x
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), cg.graph_stage(layer, backward=True):
                torch.cuda._sleep(150_000_000)
                result = backward([out[1]])[0]
            assert lease.ready_event is not None and not lease.ready_event.query()
            assert not lease.live_outputs and not lease.busy
            del out, x
            # Reallocate on the original stream without reusing the lease or waiting for B.
            # The slot-reuse event alone cannot protect this original input allocation.
            churn = [torch.full((count,), 99.0, device="cuda") for _ in range(16)]
            reused = any(t.data_ptr() == pointer for t in churn)
            pending = not lease.ready_event.query()
            torch.cuda.synchronize()
            assert pending, "Delay did not cover the allocator churn"
            torch.testing.assert_close(result, torch.full_like(result, 6), rtol=0, atol=0)
            assert not reused
    finally:
        cg.release_graphs()
