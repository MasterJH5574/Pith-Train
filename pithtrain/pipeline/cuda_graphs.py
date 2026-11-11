"""
CUDA graphs for static compiled compute regions, below AOTAutograd and above Inductor.

Each compiled forward/backward owns one scratch invocation. Inputs are refreshed on every call,
mutations are copied back, and outputs escape to ordinary allocator storage. Saved activations
therefore follow the pipeline's actual lifetimes, not a CUDA graph per outstanding micro-batch.
In particular, neither FSDP's unsharded weights nor main_grad have stable addresses across steps.
"""

import atexit
import time
import weakref
from functools import cache, partial, wraps

import torch

from pithtrain.contexts import training

__all__ = ["capture_pending", "compile_stage", "graph_stats", "release_graphs"]

# Shape changes use ordinary execution after this bounded set has been captured. No eviction:
# a graph may still have an AOT backward referring to it when another shape arrives.
MAX_GRAPHS = 12
graphs = []


def live_graphs():
    graphs[:] = [ref for ref in graphs if ref() is not None]
    return [ref() for ref in graphs]


def release_graphs():
    """Release captured NCCL references before destroying their process groups."""
    live = live_graphs()
    if any(g.graph is not None for g in live):
        torch.cuda.synchronize()
    for graph in live:
        if graph.graph is not None:
            graph.graph.reset()
            graph.graph = None
        # Allocation or capture failure may leave only some of these buffers initialized.
        for name in ("inputs", "copy_destinations", "outputs", "output_specs"):
            if hasattr(graph, name):
                delattr(graph, name)
        graph.key = None


@cache
def register_cleanup():
    # Register on first use, after setup_distributed registers process-group destruction.
    # atexit runs in reverse order: NCCL shutdown waits for graphs that still reference it.
    atexit.register(release_graphs)


def graph_stats() -> dict:
    live = live_graphs()
    return {
        "instances": sum(g.graph is not None for g in live),
        "forward_replays": sum(g.replays for g in live if not g.backward),
        "backward_replays": sum(g.replays for g in live if g.backward),
        "fallbacks": sum(g.fallbacks for g in live),
        "capture_seconds": sum(g.capture_seconds for g in live),
    }


def capture_pending():
    """Capture after a complete warmup step, before the next step issues any communication."""
    if training.cuda_graphs and torch.is_grad_enabled():
        for graph in live_graphs():
            if graph.key is not None and graph.graph is None:
                graph.capture()


def signature(args):
    return tuple(
        (x.shape, x.stride() if x.layout == torch.strided else None, x.dtype, x.device)
        if isinstance(x, torch.Tensor)
        else (type(x), x)
        if isinstance(x, (int, float, bool, type(None)))
        else type(x)
        for x in args
    )


def broadcast_dims(x):
    return tuple(i for i, (n, s) in enumerate(zip(x.shape, x.stride())) if n > 1 and s == 0)


def copy_view(x, dims):
    # Copy just one element of every broadcast dimension into an equally expanded scratch view.
    for dim in dims:
        x = x.narrow(dim, 0, 1)
    return x


def copy_groups(tensors):
    # Foreach's fused copy requires a homogeneous dtype and dense, non-overlapping layouts.
    # Keep strided slices separate so one RoPE view cannot force every copy onto the slow path.
    # Source/destination metadata match exactly and are guarded before each replay.
    groups = {}
    for i, tensor in enumerate(tensors):
        key = (tensor.dtype, torch.ops.aten.is_non_overlapping_and_dense(tensor))
        groups.setdefault(key, []).append(i)
    return tuple(tuple(indices) for indices in groups.values())


def copy_grouped(destinations, sources, groups):
    for indices in groups:
        torch._foreach_copy_([destinations[i] for i in indices], [sources[i] for i in indices])


class StaticGraph:
    def __init__(self, fn, read_inputs, mutated_inputs, backward):
        self.fn = fn
        self.read_inputs = read_inputs
        self.mutated_inputs = tuple(mutated_inputs)
        self.backward = backward
        self.graph = None
        self.key = None
        self.replays = self.fallbacks = 0
        self.capture_seconds = 0.0
        graphs.append(weakref.ref(self))

    def eligible(self, args):
        tensors = [x for x in args if isinstance(x, torch.Tensor)]
        if not tensors or any(
            x.device.type != "cuda" or x.layout != torch.strided for x in tensors
        ):
            return False
        if len({x.device for x in tensors}) != 1:
            return False
        return all(isinstance(x, (torch.Tensor, int, float, bool, type(None))) for x in args)

    def valid_storage(self, args):
        storages = {i: args[i].untyped_storage() for i in self.tensor_inputs}
        if any(storages[i].nbytes() == 0 for i in self.read_nonempty):
            return False
        # Writable aliases need a shared scratch storage; leave that uncommon case ordinary.
        identities = (
            [storage._cdata for storage in storages.values()] if self.mutated_inputs else []
        )
        for i in self.mutated_inputs:
            if identities.count(storages[i]._cdata) > 1:
                return False
        return True

    def copy_inputs(self, args):
        if self.copy_destinations:
            copy_grouped(
                self.copy_destinations,
                [copy_view(args[i], dims) for i, dims in self.copy_sources],
                self.input_copy_groups,
            )

    def prepare(self, args):
        register_cleanup()
        self.tensor_inputs = tuple(i for i, x in enumerate(args) if isinstance(x, torch.Tensor))
        self.read_nonempty = tuple(
            i for i in self.read_inputs if isinstance(args[i], torch.Tensor) and args[i].numel()
        )
        if not self.valid_storage(args):
            return
        self.key = signature(args)
        self.stream = torch.cuda.current_stream().cuda_stream
        self.grad_enabled = torch.is_grad_enabled()
        self.inputs = [
            torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
            if isinstance(x, torch.Tensor)
            else x
            for x in args
        ]
        self.copy_sources = [
            (i, broadcast_dims(args[i]))
            for i in self.read_inputs
            if isinstance(args[i], torch.Tensor)
        ]
        self.copy_destinations = [copy_view(self.inputs[i], dims) for i, dims in self.copy_sources]
        self.input_copy_groups = copy_groups(self.copy_destinations)
        self.copy_inputs(args)

    def capture(self):
        start = time.perf_counter()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        # The preceding complete step warmed both external kernels and Inductor. Capture only
        # records operations; there is no extra warmup that could mutate state or run collectives.
        # AOT invokes its inner callable without autograd. Nested compiled helpers (e.g. ring
        # attention's softmax merge) guard that mode, even though the enclosing step enables it.
        with torch.set_grad_enabled(self.grad_enabled), torch.cuda.graph(graph):
            self.outputs = self.fn(list(self.inputs))
        self.output_specs = []
        for out in self.outputs:
            alias = None
            meta = None
            dims = ()
            if isinstance(out, torch.Tensor):
                meta = (out.shape, out.stride(), out.dtype, out.device)
                dims = broadcast_dims(out)
                for i, arg in enumerate(self.inputs):
                    if (
                        isinstance(arg, torch.Tensor)
                        and out.untyped_storage()._cdata == arg.untyped_storage()._cdata
                    ):
                        if out.dtype != arg.dtype:
                            raise ValueError(
                                "CUDA graph outputs may not reinterpret an input dtype"
                            )
                        alias = (i, out.storage_offset() - arg.storage_offset())
                        break
                if alias is not None:
                    i, offset = alias
                    if (
                        offset == 0
                        and out.shape == self.inputs[i].shape
                        and out.stride() == self.inputs[i].stride()
                    ):
                        meta = None  # return the current input object itself
            source = copy_view(out, dims) if meta is not None and alias is None else out
            self.output_specs.append((alias, meta, dims, source))
        self.output_copy_groups = copy_groups(
            [self.inputs[i] for i in self.mutated_inputs]
            + [
                out
                for alias, meta, dims, out in self.output_specs
                if alias is None and meta is not None
            ]
        )
        self.graph = graph
        self.capture_seconds += time.perf_counter() - start

    def __call__(self, args):
        if self.key is not None:
            if (
                signature(args) != self.key
                or torch.cuda.current_stream().cuda_stream != self.stream
                or not self.valid_storage(args)
            ):
                self.fallbacks += 1
                return self.fn(args)
        if self.graph is None:
            if (
                self.key is None
                and self.eligible(args)
                and sum(g.key is not None for g in live_graphs()) < MAX_GRAPHS
            ):
                self.prepare(args)
            self.fallbacks += 1
            return self.fn(args)
        self.copy_inputs(args)
        self.graph.replay()
        self.replays += 1
        destinations = [args[i] for i in self.mutated_inputs]
        sources = [self.inputs[i] for i in self.mutated_inputs]
        result = []
        for alias, meta, dims, out in self.output_specs:
            if alias is not None:
                i, offset = alias
                arg = args[i]
                if meta is None:
                    result.append(arg)
                else:
                    result.append(arg.as_strided(meta[0], meta[1], arg.storage_offset() + offset))
            elif meta is None:
                result.append(out)
            else:
                # Keep exact strides: compiled backward guards them, including non-dense views.
                escaped = torch.empty_strided(meta[0], meta[1], dtype=meta[2], device=meta[3])
                destinations.append(copy_view(escaped, dims))
                sources.append(out)
                result.append(escaped)
        if destinations:
            copy_grouped(destinations, sources, self.output_copy_groups)
        args.clear()
        return result


def compile_graph(gm, example_inputs, **kwargs):
    from torch._inductor.compile_fx import compile_fx_inner
    from torch._inductor.output_code import CompiledFxGraph

    read_inputs = tuple(
        i
        for i, node in enumerate(n for n in gm.graph.nodes if n.op == "placeholder")
        if any(user.op != "output" for user in node.users)
    )
    compiled = compile_fx_inner(gm, example_inputs, **kwargs)
    if isinstance(compiled, CompiledFxGraph):
        compiled.current_callable = StaticGraph(
            compiled.current_callable,
            read_inputs,
            compiled.mutated_input_idxs,
            kwargs.get("is_backward", False),
        )
    return compiled


def graph_backend(gm, example_inputs, *, options):
    from torch._functorch import config as aot_config
    from torch._inductor.compile_fx import compile_fx

    # AOT cache deserialization would restore Inductor without our runtime wrapper.
    with aot_config.patch(enable_autograd_cache=False):
        return compile_fx(
            gm,
            example_inputs,
            inner_compile=compile_graph,
            config_patches={**options, "triton.cudagraphs": False},
        )


def compile_stage(*, options):
    """Compile with the model's options, selecting scratch replay only when requested."""

    def decorate(fn):
        ordinary = torch.compile(fn, fullgraph=True, options=options)
        captured = torch.compile(
            fn, fullgraph=True, backend=partial(graph_backend, options=options)
        )

        @wraps(fn)
        def call(*args, **kwargs):
            selected = captured if training.cuda_graphs and torch.is_grad_enabled() else ordinary
            return selected(*args, **kwargs)

        return call

    return decorate
