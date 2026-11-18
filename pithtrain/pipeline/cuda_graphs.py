"""
CUDA graphs for compiled compute regions, below AOTAutograd and above Inductor.

compile_stage compiles a model method with a backend that wraps each Inductor forward and
backward callable in a StaticGraph. A StaticGraph records its first eligible call's input layout,
is captured at the start of the next step (capture_pending), and from then on replays:

- every read input is copied into scratch tensors the graph reads,
- mutated inputs (main_grad) are copied back after the replay,
- every output is copied out to ordinary allocator storage.

Nothing outside the graph points into graph memory, so outputs and saved activations follow the
pipeline's real lifetimes and no address has to stay stable across replays, steps or FSDP
reshards. Calls whose layout or stream differs from the capture run the compiled code instead.
"""

import atexit
import time
import weakref
from functools import cache, partial, wraps
from typing import NamedTuple, Optional

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
        graph.reset()


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
        # Bytes each replay moves across the graph boundary, summed over replays.
        **{
            f"{kind}_bytes": sum(g.replays * g.boundary_bytes.get(kind, 0) for g in live)
            for kind in ("parameter_in", "activation_in", "mutation_in", "mutation_out", "output")
        },
    }


def capture_pending():
    """Capture every prepared graph. Called at the start of a step, while the pipeline is drained."""
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


class OutputSpec(NamedTuple):
    """How one output of the compiled callable is returned after a replay."""

    tensor: Optional[torch.Tensor]  # the captured output, or the non-tensor value itself
    meta: Optional[tuple]  # (shape, stride, dtype, device); None returns the aliased input itself
    alias: Optional[tuple]  # (input index, storage offset) when the output views an input
    dims: tuple  # broadcast dimensions, copied as a single element

    @property
    def copied(self) -> bool:
        return self.alias is None and self.meta is not None


class StaticGraph:
    def __init__(self, fn, read_inputs, mutated_inputs, backward):
        self.fn = fn
        self.read_inputs = read_inputs
        self.mutated_inputs = tuple(mutated_inputs)
        self.backward = backward
        self.replays = self.fallbacks = 0
        self.capture_seconds = 0.0
        self.reset()
        graphs.append(weakref.ref(self))

    def reset(self):
        self.graph = None
        self.key = None  # the prepared input signature; None until prepare succeeds
        self.inputs = self.outputs = self.output_specs = None
        self.copy_sources = self.copy_destinations = ()
        self.boundary_bytes = {}

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
        """Record the call's layout and allocate scratch inputs; capture follows next step."""
        register_cleanup()
        self.tensor_inputs = tuple(i for i, x in enumerate(args) if isinstance(x, torch.Tensor))
        self.read_nonempty = tuple(
            i for i in self.read_inputs if isinstance(args[i], torch.Tensor) and args[i].numel()
        )
        if not self.valid_storage(args):
            return
        self.key = signature(args)
        self.parameter_inputs = {i for i, x in enumerate(args) if isinstance(x, torch.nn.Parameter)}
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

    def describe_output(self, out):
        if not isinstance(out, torch.Tensor):
            return OutputSpec(out, None, None, ())
        meta = (out.shape, out.stride(), out.dtype, out.device)
        dims = broadcast_dims(out)
        alias = None
        for i, arg in enumerate(self.inputs):
            if (
                isinstance(arg, torch.Tensor)
                and out.untyped_storage()._cdata == arg.untyped_storage()._cdata
            ):
                if out.dtype != arg.dtype:
                    raise ValueError("CUDA graph outputs may not reinterpret an input dtype")
                alias = (i, out.storage_offset() - arg.storage_offset())
                if alias[1] == 0 and out.shape == arg.shape and out.stride() == arg.stride():
                    meta = None  # return the current input object itself
                break
        return OutputSpec(copy_view(out, dims) if alias is None else out, meta, alias, dims)

    def capture(self):
        start = time.perf_counter()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        # The preceding step warmed both external kernels and Inductor. Capture only records
        # operations; there is no extra warmup that could mutate state or run collectives.
        # AOT invokes its inner callable without autograd. Nested compiled helpers (e.g. ring
        # attention's softmax merge) guard that mode, even though the enclosing step enables it.
        with torch.set_grad_enabled(self.grad_enabled), torch.cuda.graph(graph):
            self.outputs = self.fn(list(self.inputs))
        self.output_specs = [self.describe_output(out) for out in self.outputs]
        copied = [spec.tensor for spec in self.output_specs if spec.copied]
        mutated = [self.inputs[i] for i in self.mutated_inputs]
        self.output_copy_groups = copy_groups(mutated + copied)
        self.account_bytes(mutated, copied)
        self.graph = graph
        self.capture_seconds += time.perf_counter() - start

    def account_bytes(self, mutated, copied):
        """Bytes one replay copies across the graph boundary, by kind (see graph_stats)."""

        def nbytes(tensors):
            return sum(t.numel() * t.element_size() for t in tensors)

        inputs = {"parameter_in": [], "activation_in": [], "mutation_in": []}
        for (i, _), view in zip(self.copy_sources, self.copy_destinations):
            if i in self.mutated_inputs:
                inputs["mutation_in"].append(view)
            elif i in self.parameter_inputs:
                inputs["parameter_in"].append(view)
            else:
                inputs["activation_in"].append(view)
        self.boundary_bytes = {kind: nbytes(views) for kind, views in inputs.items()}
        self.boundary_bytes["mutation_out"] = nbytes(mutated)
        self.boundary_bytes["output"] = nbytes(copied)

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
        for spec in self.output_specs:
            if spec.alias is not None:
                i, offset = spec.alias
                arg = args[i]
                if spec.meta is None:
                    result.append(arg)
                else:
                    shape, stride = spec.meta[0], spec.meta[1]
                    result.append(arg.as_strided(shape, stride, arg.storage_offset() + offset))
            elif spec.meta is None:
                result.append(spec.tensor)
            else:
                # Keep exact strides: compiled backward guards them, including non-dense views.
                shape, stride, dtype, device = spec.meta
                escaped = torch.empty_strided(shape, stride, dtype=dtype, device=device)
                destinations.append(copy_view(escaped, spec.dims))
                sources.append(spec.tensor)
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
    """
    torch.compile(fullgraph=True) with the model's options. When training.cuda_graphs is set and
    autograd is enabled, calls go through the CUDA-graph backend instead of the ordinary one.
    """

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
