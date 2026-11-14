"""Scheduler-owned combine storage and direct graph-input lifetime checks."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_all_to_all_explicit_destination_and_invalid_aliases():
    from pithtrain.operators.all_to_all import all_to_all_async

    class Group:
        def all_to_all_single(self, output, source, outputs, inputs, options):
            assert outputs == inputs == [2, 3]
            output.copy_(source)
            return self

    source = torch.randn(5, 8, device="cuda")
    destination = torch.empty_like(source)
    group = Group()
    result, work = all_to_all_async(source, [2, 3], [2, 3], group, out=destination)
    assert result is destination and work is group
    torch.testing.assert_close(result, source)
    for invalid in [source, source[:4], torch.empty(8, 5, device="cuda").T]:
        with pytest.raises(ValueError, match="destination"):
            all_to_all_async(source, [2, 3], [2, 3], group, out=invalid)


def test_combine_lease_survives_detach_and_stale_retirement():
    from pithtrain.pipeline.graph_buffers import (
        CombineBufferPool,
        detach_combine_buffer,
        release_combine_buffer,
        scheduled_graph_buffers,
    )

    source = torch.ones(4, 8, device="cuda")
    pool = CombineBufferPool(2)
    with scheduled_graph_buffers(pool):
        a = detach_combine_buffer(pool.acquire(source, 4, torch.cuda.current_stream()))
        b = detach_combine_buffer(pool.acquire(source, 4, torch.cuda.current_stream()))
        assert a.data_ptr() != b.data_ptr()
        assert pool.acquire(source, 4, torch.cuda.current_stream()) is None
        release_combine_buffer(a)
        c = pool.acquire(source, 4, torch.cuda.current_stream())
        assert c.data_ptr() == a.data_ptr()
        release_combine_buffer(a)  # This old record must not retire c's newer lease.
        assert pool.live == 2 and pool.binding(c) is not None
        release_combine_buffer(b)
        release_combine_buffer(c)
    assert pool.peak_live == 2 and pool.reuses == 1


def test_combine_slot_reuse_orders_different_streams():
    from pithtrain.pipeline.graph_buffers import CombineBufferPool, scheduled_graph_buffers

    pool = CombineBufferPool(1)
    source = torch.zeros(128, device="cuda")
    consumer, producer = torch.cuda.Stream(), torch.cuda.Stream()
    with scheduled_graph_buffers(pool):
        value = pool.acquire(source, 128, consumer)
        with torch.cuda.stream(consumer):
            value.fill_(7)
            torch.cuda._sleep(1_000_000)
            saved = value.clone()
            pool.release(value)
        reused = pool.acquire(source, 128, producer)
        with torch.cuda.stream(producer):
            reused.fill_(13)
            pool.release(reused)
        consumer.synchronize()
        producer.synchronize()
    torch.testing.assert_close(saved, torch.full_like(saved, 7))
    torch.testing.assert_close(reused, torch.full_like(reused, 13))


def test_scheduled_graph_inputs_replay_with_two_live_microbatches():
    from pithtrain.pipeline import cuda_graphs as cg
    from pithtrain.pipeline.graph_buffers import (
        CombineBufferPool,
        detach_combine_buffer,
        scheduled_graph_buffers,
    )

    cg.release_graphs()
    pool = CombineBufferPool(2)

    @cg.compile_stage(options={})
    def compute(x, weight):
        return (x @ weight).sin()

    from pithtrain.contexts import training

    previous = training.cuda_graphs
    training.cuda_graphs = True
    weight = torch.randn(8, 8, device="cuda", requires_grad=True)
    escaped = []
    try:
        for step in range(4):
            with scheduled_graph_buffers(pool):
                cg.capture_pending()
                inputs, outputs, references = [], [], []
                for mb in range(2):
                    source = torch.randn(4, 8, device="cuda")
                    value = pool.acquire(source, 4, torch.cuda.current_stream())
                    value.copy_(source)
                    x = detach_combine_buffer(value)
                    ref_x = source.detach().clone().requires_grad_()
                    ref_w = weight.detach().clone().requires_grad_()
                    inputs.append(x)
                    outputs.append(compute(x, weight))
                    references.append((ref_x, ref_w, (ref_x @ ref_w).sin()))
                weight.grad = None
                expected_grad = torch.zeros_like(weight)
                for index in [1, 0]:
                    ref_x, ref_w, ref_y = references[index]
                    torch.testing.assert_close(outputs[index], ref_y)
                    torch.autograd.backward(outputs[index], torch.ones_like(outputs[index]))
                    ref_y.sum().backward()
                    torch.testing.assert_close(inputs[index].grad, ref_x.grad)
                    expected_grad.add_(ref_w.grad)
                    pool.release(inputs[index])
                torch.testing.assert_close(weight.grad, expected_grad)
                escaped.extend((y, y.detach().clone()) for y in outputs)
                with torch.no_grad():
                    weight.add_(0.001)
            for value, snapshot in escaped:
                torch.testing.assert_close(value, snapshot, rtol=0, atol=0)
        direct = [g for g in pool.graphs.values() if g.graph is not None]
        assert any(not g.backward and g.replays for g in direct)
        assert any(g.backward and g.replays for g in direct)
        for graph in direct:
            assert set(graph.scheduled_inputs).isdisjoint(i for i, _ in graph.copy_sources)
        assert all(g.nonowning_output_bytes > 0 for g in direct)
        forward = [g for g in direct if not g.backward]
        pointers = [{x.data_ptr() for x in g.output_copy_sources() if x.numel()} for g in forward]
        assert len(pointers) == 2 and pointers[0] & pointers[1]

    finally:
        cg.release_graphs()
        training.cuda_graphs = previous


def test_combine_spare_precedes_recent_retirement_and_reuse_is_fifo():
    from pithtrain.pipeline.graph_buffers import CombineBufferPool, scheduled_graph_buffers

    source = torch.ones(4, 8, device="cuda")
    stream = torch.cuda.current_stream()
    pool = CombineBufferPool(3)
    with scheduled_graph_buffers(pool):
        first = pool.acquire(source, 4, stream)
        second = pool.acquire(source, 4, stream)
        pool.release(second)
        spare = pool.acquire(source, 4, stream)
        assert spare.data_ptr() not in (first.data_ptr(), second.data_ptr())
        pool.release(first)
        reused = pool.acquire(source, 4, stream)
        assert reused.data_ptr() == second.data_ptr()  # retired before the lower-index slot
        pool.release(second)  # stale lease cannot enqueue this slot a second time
        assert len(pool.available) == 1
        pool.release(reused)
        pool.release(spare)
    assert pool.live == 0 and len(pool.slots) == 3


def test_scheduled_step_rejects_late_saved_tensor_hooks():
    from pithtrain.pipeline.graph_buffers import CombineBufferPool, scheduled_graph_buffers

    with scheduled_graph_buffers(CombineBufferPool(1)):
        with pytest.raises(RuntimeError, match="scheduler-owned combine buffers"):
            with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
                raise AssertionError("observer should be rejected before installation")


def test_scheduled_stage_rejects_late_method_replacement(monkeypatch):
    from pithtrain.contexts import distributed
    from pithtrain.pipeline import cuda_graphs as cg
    from pithtrain.pipeline.graph_buffers import CombineBufferPool, scheduled_graph_buffers

    monkeypatch.setattr(distributed, "ep_size", 2, raising=False)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.Identity()
            cg.declare_pipeline_outputs(self, Layer, ("norm",), ("compute",))

        def compute(self, x):
            return self.norm(x)

    layer = Layer()
    called = []
    original = layer.compute
    with scheduled_graph_buffers(CombineBufferPool(1)):
        cg.check_scheduled_layer(layer)
        layer.compute = lambda x: called.append(x)
        with pytest.raises(RuntimeError, match="stage mutation"):
            cg.pipeline_call(layer, layer.compute, torch.ones(4, device="cuda"))
        layer.compute = original
        layer.norm.router_replay = lambda tokens, topk: called.append(tokens)
        with pytest.raises(RuntimeError, match="routing callbacks"):
            cg.pipeline_call(layer, layer.compute, torch.ones(4, device="cuda"))
    assert not called


def test_only_registered_force_balance_callbacks_are_observer_free():
    from pithtrain.contexts.router_replay import is_observer_free_replay
    from pithtrain.modules.load_balance import force_balance

    callback = force_balance(8)
    assert is_observer_free_replay(callback)
    assert not is_observer_free_replay(lambda tokens, topk: callback(tokens, topk))
    assert not is_observer_free_replay(None)


def test_nonowning_outputs_preserve_aliases_with_delayed_auxiliary_stream():
    from pithtrain.pipeline.cuda_graphs import nonowning_graph_outputs

    pool = torch.cuda.graph_pool_handle()
    stream, auxiliary = torch.cuda.Stream(), torch.cuda.Stream()
    first = torch.full((8, 16), 2.0, device="cuda")
    second = torch.full_like(first, 5.0)
    torch.cuda.synchronize()

    def capture(value):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            intermediate = value * 2
            auxiliary.wait_stream(stream)
            with torch.cuda.stream(auxiliary):
                torch.cuda._sleep(1_000_000)
                result = intermediate + 3
            stream.wait_stream(auxiliary)
        outputs, released = nonowning_graph_outputs(
            [result, result, result[:, 1::2], value], [value], graph, pool
        )
        assert released == result.untyped_storage().nbytes()
        assert outputs[0] is outputs[1]
        assert outputs[3] is value  # Input storage must retain its original owner.
        assert outputs[2].stride() == result[:, 1::2].stride()
        assert outputs[2].storage_offset() == 1
        return graph, outputs

    g0, y0 = capture(first)
    g1, y1 = capture(second)
    assert y0[0].data_ptr() == y1[0].data_ptr()
    with torch.cuda.stream(stream):
        g0.replay()
        saved0 = [x.clone() for x in y0]
        g1.replay()
        saved1 = [x.clone() for x in y1]
        g0.replay()
        saved_again = y0[0].clone()
    stream.synchronize()  # No host synchronization between replay and the next variant.
    g0.reset()
    g1.reset()
    pressure = torch.empty(1024 * 1024, device="cuda").fill_(99)
    for outputs, expected, source in [(saved0, 7, first), (saved1, 13, second)]:
        for tensor in outputs[:3]:
            torch.testing.assert_close(tensor, torch.full_like(tensor, expected))
        torch.testing.assert_close(outputs[3], source)
    torch.testing.assert_close(saved_again, torch.full_like(saved_again, 7))
    del pressure
