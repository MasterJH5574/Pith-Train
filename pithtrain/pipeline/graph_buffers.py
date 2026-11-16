"""Combine destinations retained by DualPipeV through aggregation backward.

These slots replace the ordinary combine allocations. They are not graph scratch copies.
Every acquisition creates a distinct lease; retiring an old record cannot release a later
microbatch that reuses its storage. CUDA events order reuse on the communication stream.
"""

import weakref
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass

import torch

active_pool = None
pools = weakref.WeakSet()


@dataclass(eq=False)
class CombineSlot:
    tensor: torch.Tensor
    ready: torch.cuda.Event | None = None
    lease: "CombineLease | None" = None


@dataclass(eq=False)
class CombineLease:
    slot: CombineSlot
    released: bool = False


class CombineBufferPool:
    def __init__(self, max_slots, *, lend_outputs=False):
        self.max_slots = max_slots
        self.lend_outputs = lend_outputs
        self.output_slots = {}
        self.shape_key = None
        self.slots = []
        self.available = deque()
        self.by_storage = {}
        self.live = self.peak_live = self.acquisitions = self.reuses = 0
        self.graphs = {}
        pools.add(self)

    def acquire(self, source, rows, stream):
        key = ((rows, *source.shape[1:]), source.dtype, source.device)
        if self.shape_key not in (None, key):
            return None  # A changed geometry keeps the ordinary allocation/capture path.
        # Fill the bounded pool before recycling. The scheduler includes one spare so the
        # first combine after a backward can use independent storage while that backward
        # is still reading its input on the compute stream.
        if len(self.slots) < self.max_slots:
            slot = CombineSlot(source.new_empty(key[0]))
            self.slots.append(slot)
            self.by_storage[slot.tensor.untyped_storage()._cdata] = slot
            self.shape_key = key
        elif self.available:
            # Retire order, not allocation index: prefer storage whose last read was queued
            # longest ago. The event remains mandatory; FIFO is not a synchronization proof.
            slot = self.available.popleft()
            self.reuses += 1
            if slot.ready is not None:
                stream.wait_event(slot.ready)
        else:
            return None
        lease = CombineLease(slot)
        slot.lease = lease
        value = slot.tensor.detach()
        value.cuda_graph_combine_lease = lease
        self.live += 1
        self.peak_live = max(self.peak_live, self.live)
        self.acquisitions += 1
        return value

    def release(self, value):
        lease = getattr(value, "cuda_graph_combine_lease", None)
        if lease is None or lease.released:
            return
        slot = lease.slot
        if (
            self.by_storage.get(value.untyped_storage()._cdata) is not slot
            or slot.lease is not lease
        ):
            raise RuntimeError("combine buffer retired by a different lease or pool")
        if slot.ready is None:
            slot.ready = torch.cuda.Event()
        slot.ready.record(torch.cuda.current_stream(value.device))
        lease.released = True
        slot.lease = None
        self.available.append(slot)
        self.live -= 1

    def register_outputs(self, outputs, slots):
        """Keep graph-produced saves tied to their original combine lifetime."""
        unique = {id(slot): slot for slot in slots}
        if len(unique) != 1:
            raise RuntimeError("graph outputs require one combine-buffer lifetime")
        slot = next(iter(unique.values()))
        for value in outputs:
            key = value.untyped_storage()._cdata
            prior = self.output_slots.setdefault(key, slot)
            if prior is not slot:
                raise RuntimeError("graph output storage overlaps another live slot")

    def binding(self, value, *, include_outputs=True):
        if not isinstance(value, torch.Tensor) or not value.is_cuda:
            return None
        key = value.untyped_storage()._cdata
        slot = self.by_storage.get(key)
        if slot is None and include_outputs:
            slot = self.output_slots.get(key)
        if slot is None or slot.lease is None or value.is_conj() or value.is_neg():
            return None
        return slot


@contextmanager
def scheduled_graph_buffers(pool):
    global active_pool
    if active_pool is not None or pool.live:
        raise RuntimeError("CUDA graph buffer step must start with a drained pipeline")
    active_pool = pool
    try:
        with torch.autograd.graph.disable_saved_tensors_hooks(
            "scheduler-owned combine buffers cannot escape through saved-tensor hooks"
        ):
            yield
            if pool.live:
                raise RuntimeError("CUDA graph buffer step ended with unconsumed combine leases")
    finally:
        active_pool = None


def acquire_combine_buffer(source, rows, stream):
    if active_pool is None:
        return None
    return active_pool.acquire(source, rows, stream)


def detach_combine_buffer(value):
    lease = getattr(value, "cuda_graph_combine_lease", None)
    detached = value.detach().requires_grad_()
    if lease is not None:
        detached.cuda_graph_combine_lease = lease
    return detached


def release_combine_buffer(value):
    if active_pool is not None:
        active_pool.release(value)


def is_combine_buffer(value):
    return active_pool is not None and active_pool.binding(value) is not None
