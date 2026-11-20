"""
DualPipeV for dense models with per-layer sharding inside the step (ZeRO-3).

DualPipeV runs every step ZeRO-1 style: each rank keeps its stage's weights unsharded and full-size
fp32 gradients from the first use to the end-of-step reduce. With expert parallelism sharding the
experts that is cheap; for a dense model it is 6 bytes per parameter of the stage (about 150 GB for
a 27B model at pp_size 1). DenseZero3DualPipeV keeps DualPipeV's schedule and changes two things:

- Each forward/backward pair runs as the backward and then the forward, not interleaved stage by
  stage. Interleaving hides expert all-to-alls; a dense model has none (stages 2 and 4 pass
  through), and the sequential order keeps about one micro-batch of activations live.
- A ShardedUnit per FSDP module (decoder layer, embed_tokens, norm, lm_head) gathers it one unit
  ahead of use, waits explicitly before use, releases it after its forward, and after its backward
  of each micro-batch reduce-scatters its gradients into the sharded fp32 gradient and releases it.
  So at most the current and the next unit are unsharded, and one unit's fp32 main_grad exists.
  The prefetch also crosses compute chunks: the schedule fixes their order, so the previous step
  of the same shape tells which units the next chunk starts with. They are gathered during this
  chunk's last unit, and a forward that ends with them (a stage's last layer, or its epilog before
  the loss backward) keeps them rather than releasing and gathering them again.

FSDP's own hooks are not relied on. DualPipeV keeps every state in PRE_BACKWARD after its first
backward of the step (the root callback runs at the end), so later micro-batches register no
pre- or post-backward hooks at all; the units drive unshard, main_grad allocation, reduce and
reshard themselves. For the hooks that the first micro-batch does register, each group's
pre_backward is replaced on the instance by one that does not prefetch (FSDP's backward prefetch
follows its recorded forward order, which spans micro-batches and would gather the wrong layers).
Everything is installed on instances of this model; MoE models and DualPipeV are untouched.
"""

from collections import Counter
from types import MethodType
from typing import Dict, List, Optional, Tuple

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule, fully_shard

from pithtrain.contexts import distributed
from pithtrain.contexts.weight_grad_store import WeightGradStore
from pithtrain.modules.fsdp_in_place import install_in_place, mark_in_place
from pithtrain.operators.linear import BF16AssignLinear, BF16Linear
from pithtrain.pipeline.dualpipev import DualPipeV, Microbatch
from pithtrain.pipeline.execution import (
    clear_layer_records,
    epilog_f,
    layer_backward,
    layer_forward,
    prolog_f,
    run_backward,
)

__all__ = ["DenseZero3DualPipeV", "ShardedUnit", "is_dense"]


def is_dense(modules: nn.Sequential) -> bool:
    """
    No decoder layer has experts: the condition setup_model also uses to shard experts apart.
    """
    return not any(hasattr(layer.mlp, "experts") for m in modules for layer in m.layers.values())


class ShardedUnit:
    """
    Explicit gather, release and per-micro-batch reduce of one FSDP module's own parameter groups.
    """

    def __init__(self, module: FSDPModule, name: str, rs_stream: torch.cuda.Stream) -> None:
        self.module, self.name, self.rs_stream = module, name, rs_stream
        self.groups = list(fully_shard.state(module)._fsdp_param_groups)
        # The fp32 main_grad buffers install_wgrad_accumulation gave this module's linear weights.
        owners = [m for m in module.modules() if isinstance(getattr(m, "main_grad", None), torch.Tensor)]  # fmt: skip
        # Each micro-batch's backward gets a fresh main_grad and reduces it right after, and every
        # linear runs once per forward: its weight gradient can overwrite main_grad, which then
        # needs no zero-fill.
        for m in owners:
            if type(m) is BF16Linear:
                m.__class__ = BF16AssignLinear
        self.main_grads = [m.main_grad for m in owners]
        self.assigns = all(isinstance(m, BF16AssignLinear) for m in owners)
        # Groups whose all-gathers skip FSDP's copy-out (fsdp_in_place).
        for group in self.groups:
            mark_in_place(group)
        self.handle = None
        self.companions: List[ShardedUnit] = [self]  # gathered together with this unit
        self.stats: Counter = Counter()
        for group in self.groups:
            original = group.pre_backward  # main_grad's wrapper, or FSDP's own

            def pre_backward(default_prefetch, *unused, _original=original, _group=group):
                if not _group.is_unsharded:
                    self.stats["hook_unshard"] += 1  # a gather the schedule missed
                _original(False, *unused)

            group.pre_backward = pre_backward

    def is_unsharded(self) -> bool:
        return all(group.is_unsharded for group in self.groups)

    def gather(self) -> None:
        """
        Start the all-gather if the module is sharded and none is in flight.
        """
        if self.handle is not None or self.is_unsharded():
            return
        self.handle = self.module.unshard(async_op=True)
        self.stats["gather"] += 1

    def gather_with_companions(self) -> None:
        for unit in self.companions:
            unit.gather()

    def wait(self) -> None:
        """
        Make the parameters usable now, on the current stream.
        """
        if self.handle is None and not self.is_unsharded():
            self.stats["late_gather"] += 1  # not prefetched: gathered here, not overlapped
            self.gather()
        if self.handle is not None:
            self.handle.wait()
            self.handle = None
        assert self.is_unsharded(), f"{self.name} is not unsharded after its gather"

    def alloc_grads(self) -> None:
        """
        What main_grad's pre_backward does, which FSDP only calls on the first micro-batch.
        """
        for main_grad in self.main_grads:
            if main_grad.untyped_storage().size() == 0:
                main_grad.untyped_storage().resize_(main_grad.numel() * main_grad.element_size())
                if not self.assigns:
                    main_grad.zero_()

    def reduce(self) -> None:
        """
        Reduce-scatter this micro-batch's gradients into the sharded fp32 gradient and reshard.
        """
        module = self.module
        module.set_requires_gradient_sync(True, recurse=False)
        module.set_reshard_after_backward(True, recurse=False)
        try:
            for group in reversed(self.groups):
                # DualPipeV's end-of-step reduce moves the reduce-scatter onto the compute stream;
                # per-layer reduces must overlap compute.
                group.comm_ctx.reduce_scatter_stream = self.rs_stream
                group.post_backward()
        finally:
            module.set_requires_gradient_sync(False, recurse=False)
            module.set_reshard_after_backward(False, recurse=False)
        self.stats["reduce"] += 1

    def release(self) -> None:
        if self.handle is not None:
            self.wait()
        self.module.reshard()


class StageUnits:
    """
    The units of one stage module, in forward order.
    """

    def __init__(self, module: nn.Module, rs_stream: torch.cuda.Stream) -> None:
        def unit(child: Optional[nn.Module], name: str) -> Optional[ShardedUnit]:
            return ShardedUnit(child, name, rs_stream) if isinstance(child, FSDPModule) else None

        prefix = f"stage{module.stage_index}"
        self.embed = unit(getattr(module, "embed_tokens", None), f"{prefix}.embed_tokens")
        self.layers = [unit(layer, f"{prefix}.layers.{name}") for name, layer in module.layers.items()]  # fmt: skip
        assert all(u is not None for u in self.layers), "every dense layer must be fully_shard'ed"
        epilog = [unit(getattr(module, name, None), f"{prefix}.{name}") for name in ("norm", "lm_head")]  # fmt: skip
        self.epilog = [u for u in epilog if u is not None]
        # The final norm does almost no work, so lm_head gathered behind it would not overlap:
        # the epilog gathers as one.
        for u in self.epilog:
            u.companions = self.epilog

    def all(self) -> List[ShardedUnit]:
        return [u for u in (self.embed, *self.layers, *self.epilog) if u is not None]


def dense_forward(module: nn.Module, hidden_states: torch.Tensor, cu_seqlens: Optional[torch.Tensor] = None) -> torch.Tensor:  # fmt: skip
    """
    model_forward with each unit gathered one ahead, waited on before use and released after.
    Installed as the stage module's forward (instance attribute), inside its FSDP root forward.
    """
    units: StageUnits = module.zero3_units
    chunk_record = module.chunk_record
    forward_only = not torch.is_grad_enabled()
    first, last = module.stage_index == 0, module.stage_index == module.stage_count - 1
    order = ([units.embed] if first else []) + units.layers + (units.epilog if last else [])
    # The units the next compute chunk of the step starts with: gathered during this chunk's last
    # unit, and kept rather than released if this chunk ends with them.
    following = module.zero3_following()
    tail = units.epilog if last else [order[-1]]
    keep = [unit for unit in following if unit in tail]
    order[0].gather_with_companions()

    def advance(i: int) -> None:  # wait on unit i, prefetch unit i + 1 (or the next chunk's)
        order[i].wait()
        for unit in [order[i + 1]] if i + 1 < len(order) else following:
            unit.gather_with_companions()

    def release(unit: ShardedUnit) -> None:
        if unit not in keep:
            unit.release()

    position = 0
    if first:
        advance(position)
        hidden_states = prolog_f(module, hidden_states, chunk_record.prolog)
        release(units.embed)
        position += 1
        if forward_only:
            chunk_record.prolog.outs = None

    rotary_posemb = module.forward_posemb(hidden_states.shape[1], cu_seqlens)
    for i, (layer, layer_record, unit) in enumerate(zip(module.layers.values(), chunk_record.layers, units.layers)):  # fmt: skip
        advance(position)
        hidden_states = layer_forward(layer, hidden_states, rotary_posemb, layer_record, cu_seqlens)
        release(unit)
        if i == 0 and not forward_only:
            # Behind a layer of queued compute, so its wait on the last reduce-scatter is hidden.
            module.release_reduce_inputs()
        if not forward_only:
            # The MLP output is a record root only: stage 5's add saves neither input, and the
            # backward needs its grad_fn, not its values (as DualPipeV frees it for experts).
            layer_record.stage3.outs.moe_outs.untyped_storage().resize_(0)
        position += 1
        if forward_only:
            clear_layer_records(layer_record)

    if last:
        for _ in units.epilog:
            advance(position)
            position += 1
        hidden_states = epilog_f(module, hidden_states, chunk_record.epilog)
        for unit in units.epilog:
            release(unit)
        if forward_only:
            chunk_record.epilog.args = None
    return hidden_states


class DenseZero3DualPipeV(DualPipeV):
    """
    DualPipeV's schedule for dense models, sharding each layer within the step. See the module
    docstring.
    """

    def __init__(self, modules: nn.Sequential) -> None:
        super().__init__(modules)
        assert is_dense(modules), "DenseZero3DualPipeV is for models without experts"
        self.rs_stream = torch.cuda.Stream(device=torch.cuda.current_device())
        self.deferred_forward = False
        self.stage_units: List[StageUnits] = []
        install_in_place()
        for module in self.module:
            units = StageUnits(module, self.rs_stream)
            module.zero3_units = units
            module.forward = MethodType(dense_forward, module)
            module.release_reduce_inputs = self.release_reduce_inputs
            module.zero3_following = self.following_units
            self.stage_units.append(units)
        for unit in self.units():
            # The units release themselves (or stay gathered for the next chunk); FSDP's reshard
            # after forward (set for embed_tokens, norm and lm_head) would undo the latter.
            unit.module.set_reshard_after_forward(False, recurse=False)
        self.fsdp_group = self.dedicate_communicators()
        # The order of compute chunks (("F" or "B", phase)) of the last step per (micro-batch
        # count, training), which DualPipeV's schedule fixes; this step's so far, and whether it
        # has followed that order.
        self.call_plans: Dict[Tuple[int, bool], List[Tuple[str, int]]] = {}
        self.plan: Optional[List[Tuple[str, int]]] = None
        self.calls: List[Tuple[str, int]] = []
        self.on_plan = False

    def dedicate_communicators(self) -> dist.ProcessGroup:
        """
        Give the units' all-gathers and reduce-scatters their own communicators over the same ranks.
        At dp_size 1 the FSDP group (dp x cp) is the context-parallel group itself, and NCCL runs a
        process group's collectives in order on one stream: a prefetched layer all-gather would
        hold back the context-parallel exchanges that compute waits on. A separate reduce-scatter
        communicator likewise keeps a layer's gradient reduce from delaying the next prefetch.
        Returns the all-gather group.
        """
        existing = {id(g.mesh_info): g.mesh_info for u in self.units() for g in u.groups}
        ranks = {tuple(dist.get_process_group_ranks(m.shard_process_group)) for m in existing.values()}  # fmt: skip
        assert len(ranks) == 1, f"dense units span several FSDP groups: {ranks}"
        (mine,) = ranks
        # new_group is collective over the world: every rank creates every stage's pair, in order.
        stages = distributed.attn_mesh.mesh.reshape(distributed.pp_size, -1).tolist()
        assert list(mine) in stages, (mine, stages)
        dedicated = {}
        for stage in stages:
            pair = (dist.new_group(stage), dist.new_group(stage))
            if stage == list(mine):
                dedicated = pair
        all_gather_group, reduce_scatter_group = dedicated
        for mesh_info in existing.values():
            assert (
                all_gather_group.rank() == reduce_scatter_group.rank() == mesh_info.shard_mesh_rank
            )
            mesh_info.shard_process_group = all_gather_group
            mesh_info.reduce_scatter_process_group = reduce_scatter_group
        return all_gather_group

    def release_reduce_inputs(self) -> None:
        """
        FSDP keeps each root's last reduce-scatter input (the fp32 gradients of its last reduced
        unit) until that root's next reduce, to order the allocator's reuse after the collective. In
        a step of several micro-batches that is a whole forward and the loss backward later: at
        pp_size 1, embed_tokens' 4.7 GiB (Qwen3.8-27B) and a layer's 1.6 GiB sit through the
        activation peak. Order the compute stream after those reduce-scatters and drop them now.
        """
        for module in self.module:
            # Set by the root's lazy init, at its first forward.
            states = getattr(fully_shard.state(module)._comm_ctx, "reduce_scatter_states", [])
            for state in states:
                if state.event is not None:
                    torch.cuda.current_stream().wait_event(state.event)
            states.clear()

    def units(self) -> List[ShardedUnit]:
        return [u for units in self.stage_units for u in units.all()]

    def record_call(self, kind: str, phase: int) -> None:
        call = (kind, phase)
        i = len(self.calls)
        self.on_plan = self.on_plan and i < len(self.plan) and self.plan[i] == call
        self.calls.append(call)

    def following_units(self) -> List[ShardedUnit]:
        """
        The units the next compute chunk starts with, by the order the previous step of this shape
        ran; none on the first step or once the step departs from that order.
        """
        i = len(self.calls)
        if not self.on_plan or i >= len(self.plan):
            return []
        kind, phase = self.plan[i]
        module, units = self.module[phase], self.stage_units[phase]
        if kind == "F":
            return [units.embed if module.stage_index == 0 else units.layers[0]]
        return list(units.epilog) if module.stage_index == module.stage_count - 1 else [units.layers[-1]]  # fmt: skip

    def run_step(self, microbatches: List[Microbatch], objective) -> list:
        # Every gather and reduce is a collective over the FSDP group (dp x cp of this stage), so
        # its ranks must run the same number of micro-batches, or the collectives cross.
        if self.fsdp_group.size() > 1:
            count = torch.tensor([len(microbatches), -len(microbatches)], device="cuda")
            dist.all_reduce(count, op=dist.ReduceOp.MAX, group=self.fsdp_group)
            if count[0].item() != -count[1].item():
                raise RuntimeError(
                    f"ranks of the FSDP group run {-count[1].item()}..{count[0].item()} "
                    "micro-batches; per-layer sharding needs the same count on every rank"
                )
        key = (len(microbatches), torch.is_grad_enabled())
        self.plan = self.call_plans.get(key)
        self.on_plan = self.plan is not None
        self.calls = []
        outputs = super().run_step(microbatches, objective)
        self.call_plans[key] = self.calls
        return outputs

    def forward_compute_chunk(self, phase: int) -> None:
        self.record_call("F", phase)
        super().forward_compute_chunk(phase)

    def forward_chunk(self, phase: int, recv: bool = True, send: bool = True) -> None:
        """
        On the last pipeline rank, step 4's first iteration runs the forward of phase 0 before the
        backward of phase 1 (DualPipeV.run_step), holding 1.5 micro-batches of activations; at
        pp_size 1 every rank is the last. That forward is the one call with recv=False, send=False
        at F0 chunk pp_size (step 2's are chunks 0 .. pp_size - 1): defer its compute to right after
        the backward that follows. It neither receives nor sends, so the point-to-point order is
        unchanged.
        """
        if (
            phase == 0
            and not recv
            and not send
            and self.is_last_pp_rank
            and not self.forward_only
            and self.current_f_chunk_id[0] == distributed.pp_size
        ):
            self.commit_and_wait_comm()
            self.deferred_forward = True
            return
        super().forward_chunk(phase, recv, send)

    def backward_chunk(
        self, phase: int, enable_zb: bool = False, recv: bool = True, send: bool = True
    ) -> None:
        super().backward_chunk(phase, enable_zb, recv, send)
        if self.deferred_forward:
            self.deferred_forward = False
            self.forward_compute_chunk(0)

    def forward_backward_compute_chunk(self, phase0: int, phase1: int) -> None:
        # Backward first: it frees a micro-batch's activations before the forward allocates more.
        if not self.forward_only:
            self.backward_compute_chunk(phase1)
        self.forward_compute_chunk(phase0)

    def backward_compute_chunk(self, phase: int, enable_zb: bool = False) -> None:
        """
        DualPipeV.backward_compute_chunk with dense_model_backward. WeightGradStore keeps its
        bookkeeping (empty groups for dense layers) so the schedule's W phases pop what they push.
        """
        if self.forward_only:
            return

        self.record_call("B", phase)
        chunk_id = self.current_b_chunk_id[phase]
        self.current_b_chunk_id[phase] += 1
        is_last_stage = self.is_first_pp_rank and phase == 1

        nvtx.range_push(f"backward chunk {chunk_id} (phase{phase})")
        WeightGradStore.enabled = enable_zb
        if is_last_stage:
            loss = self.loss_chunks[chunk_id]
            assert loss is not None, (
                "the objective must return a loss when gradients are enabled; "
                "a None loss is only valid under torch.no_grad()"
            )
            input_grads = self.dense_model_backward(phase, None, loss, self.chunk_records[phase][chunk_id])  # fmt: skip
            loss.detach_()
        else:
            outputs = self.output_chunks[phase][chunk_id]
            self.output_chunks[phase][chunk_id] = None
            output_grads = self.output_grad_chunks[phase][chunk_id]
            self.output_grad_chunks[phase][chunk_id] = None
            non_empty = [(out, grad) for out, grad in zip(outputs, output_grads) if grad is not None]  # fmt: skip
            outputs, output_grads = list(zip(*non_empty))
            input_grads = self.dense_model_backward(phase, output_grads, None, self.chunk_records[phase][chunk_id])  # fmt: skip
        WeightGradStore.enabled = False
        if enable_zb:
            WeightGradStore.flush()
        nvtx.range_pop()

        self.input_chunks[phase][chunk_id] = None
        if self.is_last_pp_rank and phase == 1:
            self.output_grad_chunks[0].append(input_grads)
        else:
            self.input_grad_chunks[phase].append(input_grads)

    def dense_model_backward(self, phase: int, output_grads, loss, chunk_record):
        """
        execution.model_backward with each unit gathered one ahead (descending), its main_grad
        allocated, and after its backward its gradients reduced and the unit released.
        """
        module, units = self.module[phase], self.stage_units[phase]
        first = module.stage_index == 0
        order = list(units.epilog if loss is not None else []) + units.layers[::-1]
        if first:
            order.append(units.embed)
        following = self.following_units()
        order[0].gather_with_companions()
        position = 0

        def begin(i: int) -> ShardedUnit:
            unit = order[i]
            unit.wait()
            unit.alloc_grads()
            # Prefetch unit i + 1, or the next chunk's units. One that is this last unit itself
            # is resharded by its reduce, so it is gathered again after that.
            for u in [order[i + 1]] if i + 1 < len(order) else [u for u in following if u is not unit]:  # fmt: skip
                u.gather_with_companions()
            return unit

        if loss is not None:
            epilog = [begin(position + k) for k in range(len(units.epilog))]
            position += len(units.epilog)
            loss.backward()
            loss.detach_()
            output_grads = (chunk_record.epilog.args.hidden_states.grad,)
            chunk_record.epilog.args = None
            for unit in epilog:
                unit.reduce()
                unit.release()

        input_grads = output_grads
        for layer, layer_record in zip(reversed(list(module.layers.values())), reversed(chunk_record.layers)):  # fmt: skip
            unit = begin(position)
            position += 1
            input_grads = (layer_backward(layer, input_grads, layer_record),)
            unit.reduce()
            unit.release()

        final_grads = input_grads
        if first:
            unit = begin(position)
            run_backward(chunk_record.prolog.outs, input_grads)
            chunk_record.prolog.outs = None
            unit.reduce()
            unit.release()
            final_grads = (None,)
        for unit in following:
            unit.gather()
        return final_grads
