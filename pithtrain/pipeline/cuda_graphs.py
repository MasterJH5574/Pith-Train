"""
CUDA graphs for static compiled compute regions, below AOTAutograd and above Inductor.

Standalone calls use refreshed scratch inputs and escaping outputs. An explicit drained pipeline
step may bind layer-owned weights/main_grad and lend private saves from a bounded set of forward
slots. Audited executor entries may also lend the normalized dispatch activation until backward;
all other public results escape. FSDP weight refresh uses an explicit epoch, never pointer identity.
"""

import atexit
import time
import weakref
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from functools import cache, partial, wraps

import torch
from torch.distributed.tensor import DTensor
from torch.utils._device import DeviceContext

from pithtrain.contexts import distributed, training
from pithtrain.operators.graph_copy import GraphCopy
from pithtrain.pipeline import graph_buffers

__all__ = ["capture_pending", "compile_stage", "graph_stats", "release_graphs"]

# Shape changes use ordinary execution after this bounded set has been captured. No eviction:
# a graph may still have an AOT backward referring to it when another shape arrives.
MAX_GRAPHS = 12
graphs = []

# Only an explicitly drained DualPipeV step authorizes reuse of a layer's weights. Standalone
# compiled calls retain the per-call scratch contract. The stage context is restored on exit;
# pipeline backward disables autograd multithreading.
owners = weakref.WeakKeyDictionary()
step_owners = None
active_owner = None
active_borrow = False
active_pipeline_layer = None
active_private_input = None
pipeline_output_leases = False
step_epoch = 0
DEVICE_CONTEXT_HANDLER = DeviceContext.__torch_function__


def declare_pipeline_outputs(module, cls, consumers, methods, *, producer_methods=()):
    """Declare built-in consumers of an executor-owned stage1 activation result.

    Called by the model constructor, before FSDP wraps its stage entries. Subclasses and
    replacement methods are not authorized. FSDP updates only its own known wrappers.
    """
    if type(module) is not cls:
        return
    observed = tuple(
        (path + ("." + name if name else ""), child)
        for path in consumers
        for name, child in module.get_submodule(path).named_modules()
    )
    module.cuda_graph_output_contract = {
        "methods": {name: getattr(module, name) for name in methods},
        "producer_methods": tuple((cls, name, getattr(cls, name)) for name in producer_methods),
        "consumers": tuple((path, child, type(child), child.forward) for path, child in observed),
        "modules": tuple(child for _, child in observed),
    }


def observer_modes_absent():
    from torch.overrides import _get_current_function_mode_stack
    from torch.utils._python_dispatch import _get_current_dispatch_mode_stack

    return (
        all(
            type(mode) is DeviceContext
            and getattr(mode.__torch_function__, "__func__", None) is DEVICE_CONTEXT_HANDLER
            for mode in _get_current_function_mode_stack()
        )
        and not _get_current_dispatch_mode_stack()
        and not torch._C._are_functorch_transforms_active()
    )


def private_producer_allowed(producer, consumer):
    """Prove the built-in parameter-free producer and its immediate consumer entry.

    Aggregation invokes tensor operations, not attention/router modules. Their observers are
    checked by compile_stage after the replay callback, immediately before retained-input use.
    """
    contract = getattr(producer, "cuda_graph_output_contract", None)
    downstream = getattr(consumer, "cuda_graph_output_contract", None)
    if not (pipeline_output_leases and contract and downstream and observer_modes_absent()):
        return False
    declared = contract["producer_methods"]
    return (
        bool(declared)
        and all(
            getattr(cls, name) is method
            and getattr(producer, name) == contract["methods"].get(name)
            for cls, name, method in declared
        )
        and all(getattr(consumer, name) == method for name, method in downstream["methods"].items())
    )


def pipeline_outputs_allowed(module, *, scheduled=False):
    """Check observers and the constructor's consumer contract immediately before compute."""
    contract = getattr(module, "cuda_graph_output_contract", None)
    allowed = (
        (pipeline_output_leases or scheduled)
        and distributed.ep_size > 1
        and contract is not None
        and observer_modes_absent()
    )
    if allowed:
        hooks = torch.nn.modules.module
        allowed = not any(
            getattr(hooks, name)
            for name in (
                "_global_forward_hooks",
                "_global_forward_pre_hooks",
                "_global_backward_hooks",
                "_global_backward_pre_hooks",
            )
        ) and all(
            getattr(module, name) == expected for name, expected in contract["methods"].items()
        )
    if allowed:
        allowed = all(
            module.get_submodule(path) is child and type(child) is cls and child.forward == forward
            for path, child, cls, forward in contract["consumers"]
        ) and all(
            not (
                child._forward_hooks
                or child._forward_pre_hooks
                or child._backward_hooks
                or child._backward_pre_hooks
            )
            for child in contract["modules"]
            # FSDP's hooks on the layer's __call__ are not invoked by its stage methods.
            if child is not module
        )
    return allowed


def pipeline_call(module, method, *args):
    """Authorize the executor's immediate dispatch consumer, never a direct model call.

    EP dispatch materializes its own tokens before returning. Residuals, routing results and
    tracked loss values still escape, since their consumers may outlive or change after this call.
    The compiled entry rechecks observers after any router-replay callback has run.
    """
    global active_pipeline_layer
    if not pipeline_output_leases:
        return method(*args)
    previous = active_pipeline_layer
    contract = getattr(module, "cuda_graph_output_contract", None)
    active_pipeline_layer = (
        module if contract is not None and method in contract["methods"].values() else None
    )
    try:
        return method(*args)
    finally:
        active_pipeline_layer = previous


def private_input_after_callback(module):
    """A temporary stage wrapper must not erase evidence that it could observe this input."""
    global active_private_input
    if active_private_input is not None and active_private_input[0] is module:
        contract = getattr(module, "cuda_graph_output_contract", None)
        if not (
            contract
            and observer_modes_absent()
            and all(getattr(module, name) == method for name, method in contract["methods"].items())
        ):
            active_private_input = None


@contextmanager
def private_stage_input(module, tensor, producer, producer_inputs, *, allowed):
    """Authorize one unexposed, fresh built-in aggregation result for its next stage only."""
    global active_private_input
    previous = active_private_input
    fresh = (
        allowed
        and private_producer_allowed(producer, module)
        and all(
            type(value) is torch.Tensor
            for value in producer_inputs
            if isinstance(value, torch.Tensor)
        )
        and type(tensor) is torch.Tensor
        and not tensor.is_conj()
        and not tensor.is_neg()
        and tensor.is_contiguous()
        and tensor.storage_offset() == 0
        and all(
            tensor.untyped_storage()._cdata != value.untyped_storage()._cdata
            for value in producer_inputs
            if isinstance(value, torch.Tensor)
        )
    )
    active_private_input = (module, tensor) if fresh else None
    try:
        yield
    finally:
        active_private_input = previous


def tensor_view_key(tensor):
    return (
        tensor.untyped_storage()._cdata,
        tensor.storage_offset(),
        tensor.shape,
        tensor.stride(),
        tensor.dtype,
        tensor.device,
        tensor.is_conj(),
        tensor.is_neg(),
    )


class StageOwner:
    def __init__(self):
        self.epoch = -1
        self.stream = None
        self.refreshes = 0
        self.graphs = {}
        self.mirrors = {}
        self.current = {}
        self.lookup = {}
        self.variants = set()
        self.pools = {}
        self.saved = {}
        self.completed = []
        self.max_leases = 0

    def child(self, parent, key, lease=False, saved_from=None, public=False, retained_inputs=()):
        child = self.graphs.get(key)
        if child is None:
            if parent not in self.variants and len(self.variants) >= MAX_GRAPHS:
                return None
            self.variants.add(parent)
            child = StaticGraph(
                parent.fn,
                parent.read_inputs,
                parent.mutated_inputs,
                parent.backward,
                parent=parent,
                owner=self,
                saved_outputs=parent.saved_outputs if lease else (),
                saved_from=saved_from,
                pipeline_outputs=parent.pipeline_outputs if public else (),
                retained_inputs=retained_inputs,
            )
            child.lease = lease
            if lease:
                # A later capture's live save can occupy an earlier graph's freed temporary
                # in a shared pool. Interleaved slot replay therefore needs isolated arenas;
                # sharing only the explicitly retained input scratch remains safe.
                child.pool = torch.cuda.graph_pool_handle()
            self.graphs[key] = child
        return child

    def call(self, parent, args):
        on_stream = torch.cuda.current_stream().cuda_stream == self.stream
        if not parent.backward and not on_stream:
            # A foreign call must not stage weights or touch this stream's private saves.
            parent.fallbacks += 1
            return parent.fn(args)
        if not parent.backward and parent.saved_outputs and active_borrow and self.max_leases:
            if parent not in self.variants and len(self.variants) >= MAX_GRAPHS:
                parent.fallbacks += 1
                return parent.fn(args)
            public = (
                active_pipeline_layer is not None
                and owners.get(active_pipeline_layer) is self
                and bool(parent.pipeline_outputs)
            )
            retained = ()
            if public and active_private_input is not None:
                module, value = active_private_input
                if owners.get(module) is self:
                    value_key = tensor_view_key(value)
                    retained = tuple(
                        i
                        for i in parent.read_inputs
                        if isinstance(args[i], torch.Tensor)
                        and args[i].untyped_storage()._cdata == value_key[0]
                        and tensor_view_key(args[i]) == value_key
                        and i not in parent.mutated_inputs
                    )
                    if len(retained) != 1:
                        retained = ()
            pool_key = (
                (parent, public, retained) if retained else (parent, True) if public else parent
            )
            pool = self.pools.setdefault(pool_key, [])
            child = next((g for g in pool if not g.busy), None)
            if child is None and len(pool) < self.max_leases:
                child = self.child(
                    parent,
                    (parent, public, retained, len(pool)),
                    lease=True,
                    public=public,
                    retained_inputs=retained,
                )
                if child is not None:
                    pool.append(child)
            if child is None:
                parent.fallbacks += 1
                return parent.fn(args)
            if child.ready_event is not None:
                current = torch.cuda.current_stream()
                current.wait_event(child.ready_event)
                # A foreign-stream call can fall back without replaying this slot. Preserve
                # its dependency until the captured stream itself has consumed the event.
                if current.cuda_stream == child.stream:
                    child.ready_event = None
            original = list(args)
            before = child.replays
            result = child(args)
            if child.key is not None and (child.graph is None or child.replays > before):
                indices = (
                    child.borrowed_outputs
                    if child.graph is not None
                    else child.lease_outputs(result, original)
                )
                retained_saves = child.retained_saves(result, original) if indices else set()
                indices = set(indices) | retained_saves
                child.live_keys = {result[i].untyped_storage()._cdata for i in indices}
                if child.live_keys:
                    assert not child.live_keys.intersection(self.saved)
                    child.busy, child.lease_epoch = True, step_epoch
                    child.live_outputs = {i: result[i] for i in indices}
                    self.saved.update({key: child for key in child.live_keys})
            return result
        leases = {
            self.saved[x.untyped_storage()._cdata]
            for x in args
            if parent.backward
            and isinstance(x, torch.Tensor)
            and x.untyped_storage()._cdata in self.saved
        }
        if len(leases) > 1:
            raise RuntimeError("ambiguous CUDA graph saved-tensor leases")
        lease = next(iter(leases), None)
        if lease is not None and (not lease.busy or lease.lease_epoch != step_epoch):
            raise RuntimeError("CUDA graph saved tensors crossed a step boundary")
        child = (
            self.child(parent, (parent, lease) if lease is not None else parent, saved_from=lease)
            if on_stream
            else None
        )
        if child is None:
            if lease is not None and not on_stream:
                # Slot-reuse events cannot protect an original retained allocation released
                # by this fallback. Record the actual inputs before a boxed callable clears
                # args or retargets a Tensor's storage, and before retirement drops its saves.
                stream = torch.cuda.current_stream()
                for value in args:
                    if isinstance(value, torch.Tensor) and value.is_cuda:
                        value.record_stream(stream)
            parent.fallbacks += 1
            result = parent.fn(args)
        else:
            result = child(args)
        if lease is not None:
            # Ordinary/input-alias outputs may otherwise expose a private save as a gradient.
            private_keys = lease.live_keys | {
                lease.inputs[i].untyped_storage()._cdata for i in lease.retained_inputs
            }
            result = escape_borrowed(result, private_keys)
            self.completed.append((lease, torch.cuda.current_stream()))
        return result

    def retire(self):
        # Called after the entire autograd engine invocation succeeds, not just its inner AOT
        # callable. Cross-stream fallbacks establish an explicit dependency before slot reuse.
        for lease, stream in self.completed:
            if stream.cuda_stream != lease.stream:
                lease.ready_event = torch.cuda.Event()
                lease.ready_event.record(stream)
            for key in lease.live_keys:
                del self.saved[key]
            lease.live_keys.clear()
            lease.live_outputs.clear()
            lease.busy = False
        self.completed.clear()

    @torch.no_grad()
    def refresh(self, module):
        if self.epoch == step_epoch:
            return
        stream = torch.cuda.current_stream().cuda_stream
        if self.stream is None:
            self.stream = stream
        if stream != self.stream:
            # Late replay guards cannot undo an unordered mirror copy. Refresh only on the
            # owner's original stream; a foreign-first call leaves this epoch unrefreshed.
            return
        self.current = {
            ("weight", name): value
            for name, value in module.named_parameters()
            if not isinstance(value, DTensor)
        }
        self.current.update(
            {
                ("gradient", name): child.main_grad
                for name, child in module.named_modules()
                if isinstance(getattr(child, "main_grad", None), torch.Tensor)
            }
        )
        self.lookup = {tensor_view_key(value): key for key, value in self.current.items()}
        # Refresh after this layer's FSDP pre-forward/all-gather, once in every explicit epoch.
        # A repeated address or Tensor version is never a reason to skip this refresh.
        for key, mirror in self.mirrors.items():
            current = self.current.get(key)
            if current is not None and signature([current]) == signature([mirror]):
                mirror.copy_(current)
        self.epoch = step_epoch
        self.refreshes += 1

    @torch.no_grad()
    def bind(self, value):
        key = self.lookup.get(tensor_view_key(value))
        if key is None:
            return None
        if key[0] == "gradient":
            # The FSDP hook retains this actual accumulator, preserving the original in-place
            # addmm order and ordinary fallback writes; there is no contribution merge.
            return key, value
        if key in self.mirrors and signature([value]) != signature([self.mirrors[key]]):
            return key, None  # existing captures still own the old mirror
        if key not in self.mirrors:
            self.mirrors[key] = torch.empty_strided(
                value.shape, value.stride(), dtype=value.dtype, device=value.device
            )
            self.mirrors[key].copy_(value)
        return key, self.mirrors[key]


@contextmanager
def graph_step(layers, *, max_leases=0, executor_outputs=False):
    """Authorize immutable weights for one drained ZeRO-1 step over a fixed set of layers."""
    global step_owners, step_epoch, pipeline_output_leases
    if step_owners is not None:
        raise RuntimeError("nested CUDA graph steps are unsupported")
    step_epoch += 1
    pipeline_output_leases = executor_outputs
    step_owners = {}
    for layer in layers:
        owner = owners.setdefault(layer, StageOwner())
        if owner.saved or owner.completed:
            step_owners = None
            pipeline_output_leases = False
            raise RuntimeError("previous CUDA graph step has outstanding saved tensors")
        if owner.pools and owner.max_leases != max_leases:
            step_owners = None
            pipeline_output_leases = False
            raise RuntimeError("release CUDA graphs before changing the lease bound")
        owner.max_leases = max_leases
        step_owners[layer] = owner
    try:
        yield
        if any(owner.saved or owner.completed for owner in step_owners.values()):
            raise RuntimeError("CUDA graph step ended with unconsumed saved tensors")
    finally:
        for owner in step_owners.values():
            owner.current.clear()
            owner.lookup.clear()
        step_owners = None
        pipeline_output_leases = False


@contextmanager
def graph_stage(module, *, backward=False):
    """Identify the layer explicitly; borrowed saves retire only after its backward succeeds."""
    global active_owner, active_borrow
    previous, previous_borrow = active_owner, active_borrow
    active_owner = step_owners.get(module) if step_owners is not None else None
    active_borrow = False
    try:
        with ExitStack() as scope:
            if active_owner is not None:
                active_owner.refresh(module)
                if not backward and active_owner.max_leases:
                    try:
                        scope.enter_context(
                            torch.autograd.graph.disable_saved_tensors_hooks(
                                "CUDA graph private saves cannot escape through saved-tensor hooks"
                            )
                        )
                    except RuntimeError:
                        pass  # Existing hooks retain the ordinary escaping/copying path.
                    else:
                        active_borrow = True
            yield
            if backward and active_owner is not None:
                active_owner.retire()
    finally:
        active_owner, active_borrow = previous, previous_borrow


def private_outputs(outputs, inputs, candidates):
    # Exclude entire storage groups touching a caller input or a public/mutation output.
    protected = {x.untyped_storage()._cdata for x in inputs if isinstance(x, torch.Tensor)}
    protected.update(
        x.untyped_storage()._cdata
        for i, x in enumerate(outputs)
        if i not in candidates and isinstance(x, torch.Tensor)
    )
    return {
        i
        for i in candidates
        if isinstance(outputs[i], torch.Tensor)
        and outputs[i].untyped_storage()._cdata not in protected
        and not outputs[i].is_conj()
        and not outputs[i].is_neg()
    }


def escape_borrowed(outputs, storages):
    """An input-gradient alias of a private save must not expose reusable slot storage."""
    copies, objects, result = {}, {}, []
    for out in outputs:
        if not isinstance(out, torch.Tensor) or out.untyped_storage()._cdata not in storages:
            result.append(out)
            continue
        if id(out) not in objects:
            storage = out.untyped_storage()
            if storage._cdata not in copies:
                raw = torch.empty(0, dtype=torch.uint8, device=out.device).set_(
                    storage, 0, (storage.nbytes(),), (1,)
                )
                copies[storage._cdata] = raw.clone().untyped_storage()
            escaped = torch.empty(0, dtype=out.dtype, device=out.device).set_(
                copies[storage._cdata], out.storage_offset(), out.shape, out.stride()
            )
            if out.is_conj():
                escaped = escaped.conj()
            if out.is_neg():
                escaped = torch._neg_view(escaped)
            objects[id(out)] = escaped
        result.append(objects[id(out)])
    return result


def live_graphs():
    graphs[:] = [ref for ref in graphs if ref() is not None]
    return [ref() for ref in graphs]


def release_graphs():
    """Release captured NCCL references before destroying their process groups."""
    live = live_graphs()
    if any(g.key is not None or g.graph is not None for g in live):
        torch.cuda.synchronize()
    # Backward captures may read private forward storage directly. Reset every executable
    # before freeing any operands, with backward references released first.
    for graph in sorted(live, key=lambda g: not g.backward):
        if graph.graph is not None:
            graph.graph.reset()
            graph.graph = None
    transfer_uploads.clear()
    for graph in live:
        graph.transfers = None
        # Allocation or capture failure may leave only some of these buffers initialized.
        for name in (
            "inputs",
            "copy_destinations",
            "outputs",
            "output_specs",
            "output_repeats",
            "scratch_inputs",
            "saved_scratch_inputs",
            "retained_scratch_inputs",
            "saved_retained_scratch_inputs",
            "scheduled_scratch_inputs",
            "input_word_views",
            "output_word_views",
            "signature_guard",
        ):
            if hasattr(graph, name):
                delattr(graph, name)
        graph.key = None
        graph.saved_inputs.clear()
        graph.retained_saved_outputs.clear()
        graph.saved_from = None
        graph.live_outputs.clear()
        graph.live_keys.clear()
        graph.busy = False
        graph.ready_event = None
    for owner in owners.values():
        for graph in owner.graphs.values():
            for i, (key, tensor) in graph.bound.items():
                if key[0] == "gradient":
                    tensor.cuda_graph_retain = False
        owner.graphs.clear()
        owner.variants.clear()
        owner.pools.clear()
        owner.saved.clear()
        owner.completed.clear()
        owner.max_leases = 0
        owner.mirrors.clear()
        owner.epoch = -1
        owner.stream = None
    for graph in live:
        graph.bound.clear()
    for pool in graph_buffers.pools:
        pool.graphs.clear()


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
        "scheduled_instances": sum(g.graph is not None and bool(g.scheduled_inputs) for g in live),
        "scheduled_direct_replays": sum(g.replays for g in live if g.scheduled_inputs),
        "combine_buffer_slots": sum(len(pool.slots) for pool in graph_buffers.pools),
        "combine_buffer_bytes": sum(
            slot.tensor.numel() * slot.tensor.element_size()
            for pool in graph_buffers.pools
            for slot in pool.slots
        ),
        "combine_buffer_max_slots": sum(pool.max_slots for pool in graph_buffers.pools),
        "combine_buffer_peak_live": max(
            (pool.peak_live for pool in graph_buffers.pools), default=0
        ),
        "combine_buffer_live": sum(pool.live for pool in graph_buffers.pools),
        "owned_instances": sum(g.graph is not None and g.owner is not None for g in live),
        "owners": len(owners),
        "max_leases_per_owner": max((o.max_leases for o in owners.values()), default=0),
        "forward_slots": sum(len(pool) for o in owners.values() for pool in o.pools.values()),
        "live_saved_storages": sum(len(o.saved) for o in owners.values()),
        "direct_saved_instances": sum(g.graph is not None and bool(g.saved_inputs) for g in live),
        "direct_saved_replays": sum(g.replays for g in live if g.saved_inputs),
        "pipeline_output_slots": sum(
            bool(set(g.pipeline_outputs) & g.borrowed_outputs) for g in live
        ),
        "pipeline_output_replays": sum(
            g.replays for g in live if set(g.pipeline_outputs) & g.borrowed_outputs
        ),
        "retained_input_slots": sum(bool(g.retained_saved_outputs) for g in live),
        "retained_input_replays": sum(
            g.replays
            for g in live
            if g.backward
            and g.saved_from is not None
            and any(v[0] in g.saved_from.retained_saved_outputs for v in g.saved_inputs.values())
        ),
        "transfer_instances": sum(g.transfers is not None for g in live),
        "transfer_replays": sum(g.transfer_replays for g in live),
        "transfer_upload_slots": sum(len(pool.slots) for pool in transfer_uploads.values()),
        "transfer_upload_waits": sum(pool.waits for pool in transfer_uploads.values()),
        "transfer_pinned_bytes": sum(len(pool.slots) for pool in transfer_uploads.values())
        * TRANSFER_TABLE_SIZE
        * 8,
        "word_input_copies": sum(g.word_input_copies for g in live),
        "word_output_copies": sum(g.word_output_copies for g in live),
        "native_signature_matches": sum(g.native_signature_matches for g in live),
        "python_signature_checks": sum(g.python_signature_checks for g in live),
        "weight_refreshes": sum(owner.refreshes for owner in owners.values()),
        "weight_mirror_bytes": sum(
            t.numel() * t.element_size()
            for owner in owners.values()
            for t in owner.mirrors.values()
        ),
    }


def capture_pending():
    """Capture after a complete warmup step, before the next step issues any communication."""
    if training.cuda_graphs and torch.is_grad_enabled():
        for graph in sorted(live_graphs(), key=lambda g: g.backward):
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


def make_signature_guard(args, expected=None):
    # Use the pinned native metadata checker only for concrete, ordinary arguments. It owns
    # metadata, not tensor storage. Its extra dispatch/grad checks may reject more than our
    # signature does, so a miss must still consult the original comparison.
    from torch._C._dynamo.guards import RootGuardManager
    from torch._dynamo.guards import GuardManagerType

    if not observer_modes_absent() or any(
        type(x) not in (torch.Tensor, torch.nn.Parameter, int, float, bool, type(None))
        for x in args
    ):
        return None
    if expected is not None and signature(args) != expected:
        return None
    guard = RootGuardManager()
    guard.add_type_match_guard(id(list), [], None)
    guard.add_length_check_guard(len(args), [], None)
    for i, x in enumerate(args):
        child = guard.list_getitem_manager(i, f"args[{i}]", x, GuardManagerType.GUARD_MANAGER)
        if isinstance(x, torch.Tensor):
            child.add_tensor_match_guard(
                x,
                list(x.shape),
                list(x.stride()),
                f"args[{i}]",
                [],
                None,
                type(x),
                torch._C._dispatch_keys(x),
            )
        else:
            child.add_type_match_guard(id(type(x)), [], None)
            child.add_equals_match_guard(x, [], None)
    return guard


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


def copy_word_view(tensor):
    # Identical dense layouts copy in physical order, including dense transposes. Reinterpret
    # bits, never convert values; four-byte words keep foreach's vectorized copy width.
    return tensor.as_strided((tensor.numel(),), (1,)).view(torch.int32)


def static_copy_words(tensors, groups):
    # A homogeneous foreach already launches once. Only merge mixed dtype/layout groups when
    # every operand is dense and word-aligned; other cases retain their existing copy semantics.
    if len(groups) < 2 or any(
        tensor.is_conj()
        or tensor.is_neg()
        or tensor.numel() * tensor.element_size() % 4
        or tensor.storage_offset() * tensor.element_size() % 4
        or tensor.data_ptr() % 4
        or not torch.ops.aten.is_non_overlapping_and_dense(tensor)
        for tensor in tensors
    ):
        return None
    return [copy_word_view(tensor) for tensor in tensors]


def copy_grouped(destinations, sources, groups, *, destination_words=None, source_words=None):
    static_words = destination_words if destination_words is not None else source_words
    # AOT inner callables run without autograd. Dtype views are non-differentiable, so
    # direct grad-enabled callers must keep foreach's ordinary CopyBackward/leaf checks.
    if static_words is not None and not torch.is_grad_enabled():
        dynamic = sources if destination_words is not None else destinations
        # Shape/stride/dtype were guarded by the caller, but storage offset and logical view
        # bits can change independently. A conjugate/negative copy must copy logical values.
        if all(
            not tensor.is_conj()
            and not tensor.is_neg()
            and tensor.storage_offset() * tensor.element_size() % 4 == 0
            and tensor.data_ptr() % 4 == 0
            for tensor in dynamic
        ):
            words = [copy_word_view(tensor) for tensor in dynamic]
            if destination_words is not None:
                torch._foreach_copy_(static_words, words)
            else:
                torch._foreach_copy_(words, static_words)
            return True
    for indices in groups:
        torch._foreach_copy_([destinations[i] for i in indices], [sources[i] for i in indices])
    return False


# Each stream has a bounded ring of immutable in-flight host pointer tables. Reuse waits only
# when the corresponding HtoD has not completed. CUDA operands record their actual use stream.
TRANSFER_TABLE_SIZE = 256
TRANSFER_UPLOAD_SLOTS = 128
transfer_uploads = {}


class TransferUploads:
    def __init__(self):
        self.slots = []
        self.cursor = 0
        self.waits = 0

    def upload(self, table, tensors, stream):
        if len(self.slots) < TRANSFER_UPLOAD_SLOTS:
            host = torch.empty(
                TRANSFER_TABLE_SIZE, dtype=torch.int64, device="cpu", pin_memory=True
            )
            slot = (host, host.numpy(), torch.cuda.Event())
            self.slots.append(slot)
        else:
            slot = self.slots[self.cursor]
            self.cursor = (self.cursor + 1) % TRANSFER_UPLOAD_SLOTS
        host, array, event = slot
        if not event.query():
            self.waits += 1
            event.synchronize()
        array[: len(tensors)] = [tensor.data_ptr() for tensor in tensors]
        for tensor in tensors:
            tensor.record_stream(stream)
        table.copy_(host, non_blocking=True)
        event.record(stream)


def transfer_static_eligible(tensors):
    return all(
        type(tensor) is torch.Tensor
        and not tensor.is_conj()
        and not tensor.is_neg()
        and tensor.numel() * tensor.element_size() % 4 == 0
        and tensor.storage_offset() * tensor.element_size() % 4 == 0
        and tensor.data_ptr() % 4 == 0
        and torch.ops.aten.is_non_overlapping_and_dense(tensor)
        for tensor in tensors
    )


class CapturedTransfers:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.table = torch.empty(
            TRANSFER_TABLE_SIZE,
            dtype=torch.int64,
            device=inputs[0].device if inputs else outputs[0].device,
        )
        self.stream = torch.cuda.current_stream()
        key = (self.table.device, self.stream.cuda_stream)
        self.uploads = transfer_uploads.setdefault(key, TransferUploads())
        self.input_copy = GraphCopy(self.table, inputs)
        self.output_copy = GraphCopy(self.table, outputs, base=len(inputs), into_static=False)
        self.publish(inputs, outputs)

    def publish(self, inputs, outputs):
        self.uploads.upload(self.table, inputs + outputs, self.stream)

    def copy_in(self):
        self.input_copy()

    def copy_out(self):
        self.output_copy()

    def eligible(self, dynamic):
        return not torch.is_grad_enabled() and all(
            type(tensor) is torch.Tensor
            and not tensor.is_conj()
            and not tensor.is_neg()
            and tensor.storage_offset() * tensor.element_size() % 4 == 0
            and tensor.data_ptr() % 4 == 0
            for tensor in dynamic
        )


class StaticGraph:
    def __init__(
        self,
        fn,
        read_inputs,
        mutated_inputs,
        backward,
        *,
        parent=None,
        owner=None,
        saved_outputs=(),
        saved_from=None,
        pipeline_outputs=(),
        retained_inputs=(),
        scheduled_inputs=None,
        scheduled_pool=None,
    ):
        self.fn = fn
        self.parent = parent
        self.owner = weakref.ref(owner) if owner is not None else None
        self.bound = {}
        self.saved_outputs = tuple(saved_outputs)
        self.pipeline_outputs = tuple(pipeline_outputs)
        self.retained_inputs = tuple(retained_inputs)
        self.scheduled_inputs = scheduled_inputs or {}
        self.scheduled_pool = weakref.ref(scheduled_pool) if scheduled_pool is not None else None
        self.scheduled_input_keys = {}
        self.retained_saved_outputs = set()
        self.saved_from = saved_from
        self.saved_inputs = {}
        self.lease = self.busy = False
        self.ready_event = None
        self.live_keys = set()
        self.live_outputs = {}
        self.borrowed_outputs = set()
        if parent is not None and parent.pool is None:
            parent.pool = torch.cuda.graph_pool_handle()
        self.pool = parent.pool if parent is not None else None
        self.read_inputs = read_inputs
        self.mutated_inputs = tuple(mutated_inputs)
        self.backward = backward
        self.graph = None
        self.key = None
        self.signature_guard = None
        self.native_signature_matches = self.python_signature_checks = 0
        self.replays = self.fallbacks = 0
        self.capture_seconds = 0.0
        self.word_input_copies = self.word_output_copies = 0
        self.transfers = None
        self.transfer_replays = 0
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

    def lease_outputs(self, outputs, inputs):
        selected = private_outputs(outputs, inputs, self.saved_outputs + self.pipeline_outputs)
        saved_keys = {
            outputs[i].untyped_storage()._cdata
            for i in self.saved_outputs
            if isinstance(outputs[i], torch.Tensor)
        }
        # Backward finds the slot through an actual AOT save. A result that is never saved
        # cannot by itself keep a loan alive, even if the executor would like to borrow it.
        if not any(outputs[i].untyped_storage()._cdata in saved_keys for i in selected):
            return set()
        return selected

    def retained_saves(self, outputs, inputs):
        # The input remains the original AOT save, preserving its version checks. Only the
        # backward's internal read may use the immutable slot copy. Public aliases veto this.
        selected = set()
        for i in self.retained_inputs:
            key = inputs[i].untyped_storage()._cdata
            if any(
                isinstance(out, torch.Tensor) and out.untyped_storage()._cdata == key
                for j, out in enumerate(outputs)
                if j not in self.saved_outputs
            ):
                continue
            selected.update(
                j
                for j in self.saved_outputs
                if isinstance(outputs[j], torch.Tensor)
                and outputs[j].untyped_storage()._cdata == key
                and outputs[j].dtype == inputs[i].dtype
                and not outputs[j].is_conj()
                and not outputs[j].is_neg()
            )
        return selected

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
            self.word_input_copies += copy_grouped(
                self.copy_destinations,
                [copy_view(args[i], dims) for i, dims in self.copy_sources],
                self.input_copy_groups,
                destination_words=self.input_word_views,
            )

    def map_saved_inputs(self, args):
        self.saved_inputs = {}
        if self.saved_from is None:
            return
        for i in self.read_inputs:
            x = args[i]
            if (
                i in self.mutated_inputs
                or not isinstance(x, torch.Tensor)
                or x.is_conj()
                or x.is_neg()
            ):
                continue
            for index, output in self.saved_from.live_outputs.items():
                if (
                    x.dtype == output.dtype
                    and x.untyped_storage()._cdata == output.untyped_storage()._cdata
                ):
                    self.saved_inputs[i] = (
                        index,
                        x.storage_offset() - output.storage_offset(),
                        x.shape,
                        x.stride(),
                        signature([output]),
                    )
                    break

    def bind_saved_inputs(self):
        if self.saved_from is None:
            return True
        source = self.saved_from
        if source.graph is None:
            return False
        for i, (index, offset, shape, stride, output_key) in self.saved_inputs.items():
            if index not in source.borrowed_outputs | source.retained_saved_outputs:
                return False
            output = source.outputs[index]
            if signature([output]) != output_key:
                return False
            try:
                self.inputs[i] = output.as_strided(shape, stride, output.storage_offset() + offset)
            except RuntimeError:
                return False
        return True

    def prepare(self, args):
        register_cleanup()
        self.tensor_inputs = tuple(i for i, x in enumerate(args) if isinstance(x, torch.Tensor))
        self.read_nonempty = tuple(
            i for i in self.read_inputs if isinstance(args[i], torch.Tensor) and args[i].numel()
        )
        if not self.valid_storage(args):
            return
        self.key = signature(args)
        self.signature_guard = make_signature_guard(args, self.key)
        self.replay_stream = torch.cuda.current_stream()
        self.stream = self.replay_stream.cuda_stream
        self.grad_enabled = torch.is_grad_enabled()
        owner = self.owner() if self.owner is not None else None
        parent = self.parent
        self.map_saved_inputs(args)
        prefix = (
            "scheduled_"
            if self.scheduled_inputs
            else "saved_retained_"
            if self.saved_from is not None and self.saved_from.retained_inputs
            else "saved_"
            if self.saved_from is not None
            else "retained_"
            if self.retained_inputs
            else ""
        )
        shared = getattr(parent, prefix + "scratch_inputs", None)
        if shared is not None and (
            self.key != getattr(parent, prefix + "scratch_key")
            or self.stream != getattr(parent, prefix + "scratch_stream")
            or self.retained_inputs != getattr(parent, prefix + "scratch_retained")
        ):
            self.key = None
            return
        if owner is not None:
            self.bound = {
                i: binding
                for i, value in enumerate(args)
                if isinstance(value, torch.Tensor) and (binding := owner.bind(value)) is not None
            }
        if any(
            i in self.mutated_inputs and key[0] == "weight" for i, (key, _) in self.bound.items()
        ):
            # Parameters are immutable only in the authorized stage. An unexpected writer
            # executes ordinarily and invalidates the epoch snapshot for the next invocation.
            owner.epoch = -1
            self.bound.clear()
            self.key = None
            return
        if any(tensor is None for _, tensor in self.bound.values()) or (
            shared is not None
            and set(self.bound) | set(self.saved_inputs) | set(self.scheduled_inputs)
            != getattr(parent, prefix + "scratch_bound")
        ):
            self.bound.clear()
            self.key = None
            return
        for key, tensor in self.bound.values():
            if key[0] == "gradient":
                tensor.cuda_graph_retain = True
        self.copy_mutations = tuple(i for i in self.mutated_inputs if i not in self.bound)
        self.inputs = [
            x.detach()
            if i in self.scheduled_inputs
            else None
            if i in self.saved_inputs
            else self.bound[i][1]
            if i in self.bound
            else shared[i]
            if shared is not None and i not in self.retained_inputs
            else torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device=x.device)
            if isinstance(x, torch.Tensor)
            else x
            for i, x in enumerate(args)
        ]
        if parent is not None and shared is None:
            # Bound operands differ by layer/slot. Direct saves use metadata-only placeholders
            # until their forward capture exists; only the remaining scratch is shared.
            setattr(
                parent,
                prefix + "scratch_inputs",
                [
                    None if i in self.retained_inputs or i in self.scheduled_inputs else value
                    for i, value in enumerate(self.inputs)
                ],
            )
            setattr(parent, prefix + "scratch_stream", self.stream)
            setattr(parent, prefix + "scratch_retained", self.retained_inputs)
            setattr(parent, prefix + "scratch_key", self.key)
            setattr(
                parent,
                prefix + "scratch_bound",
                set(self.bound) | set(self.saved_inputs) | set(self.scheduled_inputs),
            )

        # Output-only placeholders (notably forward main_grad) need metadata and a distinct
        # storage identity, but no backing allocation. Never discard a writable input's storage.
        for i, x in enumerate(self.inputs):
            if (
                isinstance(x, torch.Tensor)
                and i not in self.read_inputs
                and i not in self.mutated_inputs
                and i not in self.bound
            ):
                x.untyped_storage().resize_(0)
        self.copy_sources = [
            (i, broadcast_dims(args[i]))
            for i in self.read_inputs
            if isinstance(args[i], torch.Tensor)
            and i not in self.bound
            and i not in self.saved_inputs
            and i not in self.scheduled_inputs
        ]
        self.copy_destinations = [copy_view(self.inputs[i], dims) for i, dims in self.copy_sources]
        self.input_copy_groups = copy_groups(self.copy_destinations)
        self.input_word_views = static_copy_words(self.copy_destinations, self.input_copy_groups)
        self.copy_inputs(args)
        self.scheduled_input_keys = {i: tensor_view_key(args[i]) for i in self.scheduled_inputs}

    def capture(self):
        # Capture can be requested from a different current stream at the drained step
        # boundary. Initialize pointer tables and upload events on the actual replay stream.
        with torch.cuda.stream(self.replay_stream):
            self.capture_on_stream()

    def capture_on_stream(self):
        if not self.bind_saved_inputs():
            self.key = None
            return  # A changed/unsupported saved-output layout retains ordinary execution.
        start = time.perf_counter()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        # The preceding complete step warmed both external kernels and Inductor. Capture only
        # records operations; there is no extra warmup that could mutate state or run collectives.
        # AOT invokes its inner callable without autograd. Nested compiled helpers (e.g. ring
        # attention's softmax merge) guard that mode, even though the enclosing step enables it.
        with torch.set_grad_enabled(self.grad_enabled), torch.cuda.graph(graph, pool=self.pool):
            self.outputs = self.fn(list(self.inputs))
        self.describe_outputs()
        output_sources = self.output_copy_sources()
        stream_key = (self.inputs[self.tensor_inputs[0]].device, self.stream)
        if (
            self.owner is not None
            and not self.grad_enabled
            and 0 < len(self.copy_destinations) + len(output_sources) <= TRANSFER_TABLE_SIZE
            and transfer_static_eligible(self.copy_destinations + output_sources)
            and (stream_key in transfer_uploads or len(transfer_uploads) < MAX_GRAPHS)
        ):
            transfers = CapturedTransfers(self.copy_destinations, output_sources)
            # Compile both transfer signatures with self-copies outside capture. The compute
            # probe is never replayed: doing so would update stateful load-balance counters.
            transfers.copy_in()
            transfers.copy_out()
            torch.cuda.synchronize()
            graph.reset()
            self.outputs = self.output_specs = self.output_word_views = None
            transfers.outputs = None
            del output_sources
            graph = torch.cuda.CUDAGraph()
            with torch.set_grad_enabled(self.grad_enabled), torch.cuda.graph(graph, pool=self.pool):
                transfers.copy_in()
                self.outputs = self.fn(list(self.inputs))
                transfers.copy_out()
            self.describe_outputs()
            transfers.outputs = self.output_copy_sources()
            transfers.output_copy.bind(transfers.outputs)
            self.transfers = transfers
        self.graph = graph
        self.capture_seconds += time.perf_counter() - start

    def output_copy_sources(self):
        # The probe's indices describe the compiled output ABI; final captured Tensor objects
        # replace probe objects before any backward binds private saved inputs.
        return [self.inputs[i] for i in self.copy_mutations] + [
            copy_view(self.outputs[i], dims) for i, dims in self.escaping_copies
        ]

    def describe_outputs(self):
        self.bound_pointers = {i: tensor.data_ptr() for i, (_, tensor) in self.bound.items()}
        # These wrappers belong to the captured graph, never to a public invocation. Compare
        # each current operand with the captured view, not a newly rebuilt expected tuple.
        # Retained producer inputs still use their current original AOT save below.
        self.saved_input_keys = {i: tensor_view_key(self.inputs[i]) for i in self.saved_inputs}
        self.borrowed_outputs = (
            self.lease_outputs(self.outputs, self.inputs) if self.lease else set()
        )
        self.retained_saved_outputs = self.retained_saves(self.outputs, self.inputs)
        self.output_specs = []
        self.output_repeats = {}
        seen_outputs = {}
        seen_objects = {}
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
                        alias = (
                            i,
                            out.storage_offset() - arg.storage_offset(),
                            out.is_conj() != arg.is_conj(),
                            out.is_neg() != arg.is_neg(),
                        )
                        break
                if alias is not None:
                    i, offset, conjugate, negative = alias
                    if (
                        offset == 0
                        and out.shape == self.inputs[i].shape
                        and out.stride() == self.inputs[i].stride()
                    ):
                        meta = None  # return the current input object itself
            source = copy_view(out, dims) if meta is not None and alias is None else out
            if meta is not None and alias is None:
                # AOT can return a result again among its saved tensors. Escape that exact view
                # once, retaining its ordinary alias and lifetime within this invocation only.
                key = (
                    out.untyped_storage()._cdata,
                    out.storage_offset(),
                    meta,
                    out.is_conj(),
                    out.is_neg(),
                )
                index = len(self.output_specs)
                if id(out) in seen_objects:
                    self.output_repeats[index] = (seen_objects[id(out)], True)
                elif key in seen_outputs:
                    self.output_repeats[index] = (seen_outputs[key], False)
                else:
                    seen_outputs[key] = index
                seen_objects.setdefault(id(out), index)
            self.output_specs.append((alias, meta, dims, source))
        self.escaping_copies = [
            (i, dims)
            for i, (alias, meta, dims, out) in enumerate(self.output_specs)
            if alias is None
            and meta is not None
            and i not in self.output_repeats
            and i not in self.borrowed_outputs
        ]
        output_sources = self.output_copy_sources()
        self.output_copy_groups = copy_groups(output_sources)
        self.output_word_views = static_copy_words(output_sources, self.output_copy_groups)

    def matches_signature(self, args):
        # Native guards suppress __torch_function__ internally. Observing modes must retain
        # the original Python queries; an unchanged default-device context is harmless.
        if self.signature_guard is not None and observer_modes_absent():
            try:
                # RootGuardManager itself does not restore TorchFunction TLS if a tensor
                # metadata query throws (e.g. an unsupported layout). Own that restoration.
                with torch._C.DisableTorchFunction():
                    matches = self.signature_guard.check(args)
            except RuntimeError:
                matches = False
            if matches:
                self.native_signature_matches += 1
                return True
        self.python_signature_checks += 1
        return signature(args) == self.key

    def __call__(self, args):
        pool = graph_buffers.active_pool
        if self.scheduled_pool is not None:
            # Binding is valid only during the same scheduler's live combine lease. View
            # metadata is checked too; storage identity alone does not authorize a new view.
            if pool is not self.scheduled_pool() or any(
                pool.binding(args[i]) is not slot
                or (
                    self.key is not None
                    and tensor_view_key(args[i]) != self.scheduled_input_keys[i]
                )
                for i, slot in self.scheduled_inputs.items()
            ):
                self.fallbacks += 1
                return self.fn(args)
        elif pool is not None and self.owner is None:
            bindings = {
                i: slot
                for i in self.read_inputs
                if i not in self.mutated_inputs and (slot := pool.binding(args[i])) is not None
            }
            if bindings:
                key = (self, tuple((i, tensor_view_key(args[i])) for i in bindings))
                child = pool.graphs.get(key)
                if child is None:
                    if len(pool.graphs) >= pool.max_slots * MAX_GRAPHS:
                        self.fallbacks += 1
                        return self.fn(args)
                    child = StaticGraph(
                        self.fn,
                        self.read_inputs,
                        self.mutated_inputs,
                        self.backward,
                        parent=self,
                        scheduled_inputs=bindings,
                        scheduled_pool=pool,
                    )
                    # No graph-owned result escapes: outputs/saves are copied before the
                    # next replay. These variants can reuse their parent's temporary arena.
                    pool.graphs[key] = child
                return child(args)
        if self.owner is None and active_owner is not None:
            return active_owner.call(self, args)
        if self.saved_from is not None and self.graph is not None:
            source = self.saved_from
            if (
                self.owner() is not active_owner
                or not source.busy
                or source.lease_epoch != step_epoch
                or any(
                    tensor_view_key(args[i])
                    != (
                        (
                            source.live_outputs[index].untyped_storage()._cdata,
                            source.live_outputs[index].storage_offset() + offset,
                            shape,
                            stride,
                            source.live_outputs[index].dtype,
                            source.live_outputs[index].device,
                            False,
                            False,
                        )
                        if index in source.retained_saved_outputs
                        else self.saved_input_keys[i]
                    )
                    for i, (index, offset, shape, stride, _) in self.saved_inputs.items()
                )
            ):
                self.fallbacks += 1
                return self.fn(args)
        if self.bound and self.graph is not None:
            owner = self.owner()
            if owner is not active_owner or any(
                key not in owner.current
                or (
                    args[i] is not owner.current[key]
                    and tensor_view_key(args[i]) != tensor_view_key(owner.current[key])
                )
                or (
                    key[0] == "gradient"
                    and args[i] is not tensor
                    and tensor_view_key(args[i]) != tensor_view_key(tensor)
                )
                or tensor.data_ptr() != self.bound_pointers[i]
                for i, (key, tensor) in self.bound.items()
            ):
                self.fallbacks += 1
                return self.fn(args)
        if self.key is not None:
            if (
                not self.matches_signature(args)
                or torch.cuda.current_stream().cuda_stream != self.stream
                or not self.valid_storage(args)
            ):
                self.fallbacks += 1
                return self.fn(args)
        if self.graph is None:
            if (
                self.key is None
                and self.eligible(args)
                and (
                    self.owner is not None
                    or self.scheduled_pool is not None
                    or sum(g.key is not None and g.owner is None for g in live_graphs())
                    < MAX_GRAPHS
                )
            ):
                self.prepare(args)
            self.fallbacks += 1
            return self.fn(args)
        destinations = [args[i] for i in self.copy_mutations]
        sources = [self.inputs[i] for i in self.copy_mutations]
        result = []
        borrowed_objects = {}
        for i, (alias, meta, dims, out) in enumerate(self.output_specs):
            if i in self.borrowed_outputs:
                value = self.outputs[i]
                if self.pipeline_outputs:
                    # AOT attaches public autograd metadata. Keep graph-owned Tensor objects
                    # untouched and preserve repeated-object aliases within this invocation.
                    if id(value) not in borrowed_objects:
                        borrowed_objects[id(value)] = value.detach()
                    value = borrowed_objects[id(value)]
                result.append(value)
            elif i in self.output_repeats:
                previous, same_object = self.output_repeats[i]
                escaped = result[previous]
                # Distinct views share data, not mutable shape/stride metadata. Rebase their
                # offset onto the escaping allocation rather than the captured graph storage.
                result.append(
                    escaped
                    if same_object
                    else escaped.as_strided(meta[0], meta[1], escaped.storage_offset())
                )
            elif alias is not None:
                i, offset, conjugate, negative = alias
                arg = args[i]
                if meta is None:
                    escaped = arg
                else:
                    escaped = arg.as_strided(meta[0], meta[1], arg.storage_offset() + offset)
                if conjugate:
                    escaped = escaped.conj()
                if negative:
                    escaped = torch._neg_view(escaped)
                result.append(escaped)
            elif meta is None:
                result.append(out)
            else:
                # Keep exact strides: compiled backward guards them, including non-dense views.
                escaped = torch.empty_strided(meta[0], meta[1], dtype=meta[2], device=meta[3])
                destinations.append(copy_view(escaped, dims))
                sources.append(out)
                result.append(escaped)
        input_sources = [copy_view(args[i], dims) for i, dims in self.copy_sources]
        transfers = self.transfers
        captured_copies = transfers is not None and transfers.eligible(input_sources + destinations)
        if captured_copies:
            transfers.publish(input_sources, destinations)
        else:
            self.copy_inputs(args)
            if transfers is not None:
                # Logical views keep ordinary copies. Point captured transfers back at static
                # storage so they become harmless self-copies on this invocation only.
                transfers.publish(transfers.inputs, transfers.outputs)
        self.graph.replay()
        self.replays += 1
        if captured_copies:
            torch.autograd.graph.increment_version(self.copy_destinations + destinations)
            self.transfer_replays += 1
        elif destinations:
            self.word_output_copies += copy_grouped(
                destinations, sources, self.output_copy_groups, source_words=self.output_word_views
            )
        args.clear()
        return result


def compile_graph(gm, example_inputs, *, pipeline_outputs=(), **kwargs):
    from torch._inductor.compile_fx import compile_fx_inner
    from torch._inductor.output_code import CompiledFxGraph

    read_inputs = tuple(
        i
        for i, node in enumerate(n for n in gm.graph.nodes if n.op == "placeholder")
        if any(user.op != "output" for user in node.users)
    )
    output = next(n for n in gm.graph.nodes if n.op == "output")
    visible = output.meta.get("user_visible_output_idxs")
    saved_outputs = ()
    leased_outputs = ()
    context = torch._guards.TracingContext.try_get()
    metadata = context.fw_metadata if context is not None else None
    boundary = getattr(metadata, "num_forward", None)
    if (
        not kwargs.get("is_backward", False)
        and not kwargs.get("is_inference", False)
        and visible
        and isinstance(boundary, int)
        and max(visible) < boundary <= len(output.args[0])
        and not getattr(metadata, "tokens", None)
    ):
        # AOT's prefix includes mutations, public outputs, intermediate alias bases and RNG
        # offsets. Intermediate bases may reconstruct caller-visible views at runtime, so
        # Inductor's user-visible indices alone do not define the private saved-tensor tail.
        saved_outputs = tuple(range(boundary, len(output.args[0])))
        offset = metadata.num_mutated_inp_runtime_indices
        proposed = tuple(offset + i for i in pipeline_outputs)
        if proposed and all(i in visible for i in proposed):
            leased_outputs = proposed
    compiled = compile_fx_inner(gm, example_inputs, **kwargs)
    if isinstance(compiled, CompiledFxGraph):
        compiled.current_callable = StaticGraph(
            compiled.current_callable,
            read_inputs,
            compiled.mutated_input_idxs,
            kwargs.get("is_backward", False),
            saved_outputs=saved_outputs,
            pipeline_outputs=leased_outputs,
        )
    return compiled


def graph_backend(gm, example_inputs, *, options, pipeline_outputs=()):
    from torch._functorch import config as aot_config
    from torch._inductor.compile_fx import compile_fx

    # Selective checkpoint tracing uses hooks on fake tensors. Allow them only while the
    # compiler builds the callable, restoring the guard before any runtime saved tensors escape.
    disabled = (
        torch._C._autograd._saved_tensors_hooks_get_disabled_error_message()
        if active_borrow
        else None
    )
    if disabled is not None:
        torch._C._autograd._saved_tensors_hooks_enable()
    try:
        # AOT cache deserialization would restore Inductor without our runtime wrapper.
        with aot_config.patch(enable_autograd_cache=False):
            return compile_fx(
                gm,
                example_inputs,
                inner_compile=partial(compile_graph, pipeline_outputs=pipeline_outputs),
                config_patches={**options, "triton.cudagraphs": False},
            )
    finally:
        if disabled is not None:
            torch._C._autograd._saved_tensors_hooks_disable(disabled)


@cache
def install_lazy_kernel_metadata_fix():
    """Keep a C++ wrapper's first launch tied to its own compiled Triton kernel."""
    from torch._inductor.codecache import CodeCacheFuture, CudaKernelParamCache
    from torch._inductor.runtime import triton_lazy_compile

    original = triton_lazy_compile.run_triton_kernel_with_autotune

    @wraps(original)
    def resolve(pending, name, stream, args):
        # The pinned Inductor caches autotuners by source, but binary metadata by a
        # descriptive name shared across modules. A reused autotuner's fast launcher
        # skips publishing metadata, so the helper can return another kernel's ABI.
        # Run exactly once, then publish from the actual launcher before C++ loads it.
        kernel = pending[name]
        if isinstance(kernel, CodeCacheFuture):
            kernel = kernel.result()
            # LambdaFuture.result() restores/precompiles again on each invocation,
            # rebuilding the candidate list. Resolve once so the original helper
            # and this metadata repair observe the same selected launcher.
            pending[name] = kernel
        if len(kernel.launchers) != 1 and kernel._cached_launcher is not None:
            # Restoring a previously used autotuner may rebuild candidates but retain its
            # fast launcher. That fast path skips selection; the helper would then derive
            # grid/ABI metadata from an unrelated first candidate. Re-select on this cold
            # wrapper initialization, never on steady C++ or CUDA-graph replays.
            kernel._cached_launcher = None
        result = original(pending, name, stream, args)
        if len(kernel.launchers) != 1:
            raise RuntimeError("CUDA kernel metadata requires one selected launcher")
        launcher = kernel.launchers[0]
        kernel.save_gpu_kernel(stream, launcher)
        params = CudaKernelParamCache.get(name)
        if (
            params is None
            or params["triton_meta"] is not kernel.triton_meta
            or params["inductor_meta"] is not kernel.inductor_meta
            or params["def_args"] != launcher.def_args
        ):
            raise RuntimeError("CUDA kernel metadata changed owner during initialization")
        # Retain the local metadata reference: another registry replacement cannot change it.
        return replace(
            result,
            **{
                key: params[key]
                for key in (
                    "cubin_path",
                    "mangled_name",
                    "num_warps",
                    "shared_mem",
                    "global_scratch",
                    "profile_scratch",
                )
            },
        )

    triton_lazy_compile.run_triton_kernel_with_autotune = resolve


def compile_stage(*, options, pipeline_outputs=()):
    """Compile with the model's options, selecting scratch replay only when requested."""

    def decorate(fn):
        # Install before any generated C++ module caches the Python helper pointer. Both
        # ordinary and captured stages use the same wrapper and compiler configuration.
        install_lazy_kernel_metadata_fix()
        # Nested router hooks must remain observable when added/removed after compilation.
        # The pinned Dynamo default omits these guards even under normal guarded execution.
        torch._dynamo.config.skip_nnmodule_hook_guards = False
        ordinary = torch.compile(fn, fullgraph=True, options=options)
        captured = torch.compile(
            fn,
            fullgraph=True,
            backend=partial(graph_backend, options=options, pipeline_outputs=pipeline_outputs),
        )

        @wraps(fn)
        def call(*args, **kwargs):
            global active_pipeline_layer
            selected = captured if training.cuda_graphs and torch.is_grad_enabled() else ordinary
            if selected is captured and step_owners is not None:
                previous = active_pipeline_layer
                if previous is args[0] and not pipeline_outputs_allowed(args[0]):
                    active_pipeline_layer = None
                try:
                    with graph_stage(args[0]):
                        return selected(*args, **kwargs)
                finally:
                    active_pipeline_layer = previous
            return selected(*args, **kwargs)

        return call

    return decorate
