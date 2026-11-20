"""
FSDP2 all-gathers without the copy-out after the collective, for marked groups.

FSDP2 packs a group's shards into one flat buffer, all-gathers it rank-major and copies every
parameter out of it (split_with_sizes_copy). For a parameter sharded on dim 0, its own all-gather
output is the unsharded parameter (FSDP pads only past the last row). So one all-gather per
parameter, coalesced into a single NCCL group, writes straight into the storage the unsharded
parameter views and needs no copy-out. What stays is the cast of the fp32 shard to bf16.

The reduce-scatter keeps FSDP's flat copy-in: an in-place one (one coalesced reduce-scatter per
gradient) ran fine alone, but with the coalesced all-gathers on the other communicator it stalled
whole steps at random (7 s steps taking 10 to 23 s at 16k / cp4).

install_in_place replaces foreach_all_gather and foreach_all_gather_copy_out in FSDP's
param-group module with functions that run this for the parameters mark_in_place marked and call
FSDP's own otherwise, so FSDP's state machine (unsharded parameters, events) is unchanged and
unmarked groups (every MoE model) run FSDP's code as before. Only DenseZero3DualPipeV marks groups.

Lifetimes: the unsharded storages are allocated on the stream that issues the gather, before the
collective, as FSDP's copy-out allocates them on the stream that waits.
"""

from typing import List

import torch
import torch.distributed as dist
import torch.distributed.fsdp._fully_shard._fsdp_param_group as fsdp_param_group
from torch.distributed.distributed_c10d import _coalescing_manager
from torch.distributed.fsdp._fully_shard._fsdp_collectives import AllGatherResult
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam, ShardedState

__all__ = ["install_in_place", "mark_in_place"]

MARK = "_pith_in_place"


def supports_in_place(fsdp_param: FSDPParam) -> bool:
    """
    Whether FSDP's all-gather input of the parameter is its plain shard cast to param_dtype (its
    foreach-copy path) and its sharding is on dim 0.
    """
    return (
        fsdp_param.param_dtype is not None
        and not fsdp_param.offload_to_cpu
        and not fsdp_param.is_dtensor
        and not hasattr(fsdp_param._sharded_local_tensor, "fsdp_pre_all_gather")
        and fsdp_param.fsdp_placement.dim == 0
    )


def mark_in_place(group: fsdp_param_group.FSDPParamGroup) -> bool:
    """
    Mark the group for the in-place collectives if it shards over more than one rank. Whether its
    parameters support them (supports_in_place, one param_dtype) is settled at its first
    collective: FSDP sets their dtypes in its lazy init, after the engine marks the groups.
    Returns whether it marked the group.
    """
    params = group.fsdp_params
    if not params or group.mesh_info.shard_mesh_size <= 1:
        return False
    for p in params:
        setattr(p, MARK, None)
    return True


def is_marked(fsdp_params: List[FSDPParam]) -> bool:
    if not fsdp_params:
        return False
    marked = getattr(fsdp_params[0], MARK, False)
    if marked is None:
        marked = all(supports_in_place(p) for p in fsdp_params) and len({p.param_dtype for p in fsdp_params}) == 1  # fmt: skip
        for p in fsdp_params:
            setattr(p, MARK, marked)
    return marked


def in_place_all_gather(fsdp_params, group, async_op, copy_in_stream, all_gather_stream, device, all_gather_comm):  # fmt: skip
    if not is_marked(fsdp_params) or any(p.sharded_state != ShardedState.SHARDED for p in fsdp_params):  # fmt: skip
        return ORIGINAL["foreach_all_gather"](fsdp_params, group, async_op, copy_in_stream, all_gather_stream, device, all_gather_comm)  # fmt: skip
    world_size = group.size()
    current = torch.cuda.current_stream()
    shards = [p._sharded_param_data for p in fsdp_params]
    numels = [s.numel() for s in shards]
    dtype = fsdp_params[0].param_dtype
    # The unsharded storages, from the issuing stream's pool: what was freed there before is no
    # longer read by it, and the collective below is ordered after this point.
    outputs = []
    for p, numel in zip(fsdp_params, numels):
        p.init_all_gather_outputs([numel], [dtype], world_size, device)
        p.alloc_all_gather_outputs()
        outputs.append(p.all_gather_outputs[0])
    if copy_in_stream != current:
        copy_in_stream.wait_stream(current)
    with torch.cuda.stream(copy_in_stream):
        flat = torch.empty(sum(numels), dtype=dtype, device=device)
        inputs = torch.split(flat, numels)
        torch._foreach_copy_(inputs, shards)
    all_gather_stream.wait_stream(copy_in_stream)
    with torch.cuda.stream(all_gather_stream):
        with _coalescing_manager(group=group, async_ops=async_op) as manager:
            for output, input in zip(outputs, inputs):
                dist.all_gather_into_tensor(output, input, group=group, async_op=async_op)
        work = manager.works[0] if async_op and manager.works else None
        event = all_gather_stream.record_event()
    # The flat input rides in all_gather_output, so it lives as long as FSDP keeps the result.
    return AllGatherResult(flat, event, work, [], [], [])


def in_place_all_gather_copy_out(all_gather_result, fsdp_params, group) -> None:
    if not is_marked(fsdp_params) or all_gather_result.param_all_gather_input_numels:
        return ORIGINAL["foreach_all_gather_copy_out"](all_gather_result, fsdp_params, group)
    if all_gather_result.all_gather_event is not None:
        torch.cuda.current_stream().wait_event(all_gather_result.all_gather_event)
    if isinstance(all_gather_result.all_gather_work, dist.distributed_c10d.Work):
        all_gather_result.all_gather_work.wait()


ORIGINAL = {}


def install_in_place() -> None:
    """
    Route FSDP's param-group all-gathers through the in-place versions (idempotent).
    """
    replacements = {
        "foreach_all_gather": in_place_all_gather,
        "foreach_all_gather_copy_out": in_place_all_gather_copy_out,
    }
    for name, replacement in replacements.items():
        current = getattr(fsdp_param_group, name)
        if current is replacement or name in ORIGINAL:
            continue
        ORIGINAL[name] = current
        setattr(fsdp_param_group, name, replacement)
