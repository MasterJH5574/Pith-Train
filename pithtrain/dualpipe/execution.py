"""
Execution for each stage in the schedule.

Stage Mapping:
    - Stage 1: Attention (LN + Attn + LN + Expert selection)
    - Stage 2: Dispatch (All-to-all dispatch for expert parallelism)
    - Stage 3: MLP (Expert/MLP computation)
    - Stage 4: Combine (All-to-all combine for expert parallelism)
    - Stage 5: Aggregate (Weighted expert output + residual connection)
"""

from dataclasses import dataclass
from typing import Any, List, NamedTuple, Optional, Union

import torch
import torch.cuda.nvtx as nvtx

try:
    import deep_ep
except ImportError:
    deep_ep = None

from pithtrain.dualpipe.utils import WeightGradStore, run_backward
from pithtrain.models.interface import DecoderLayerProtocol, ModelProtocol, uses_deepep_dispatch
from pithtrain.operators.all_to_all import direct_all_to_all
from pithtrain.operators.ep_backend import (
    DeepEPEventWork,
    deepep_combine,
    deepep_dispatch,
    is_deepep_backend,
    is_deepep_dispatch_state,
)


@dataclass(init=False, slots=True)
class ExecutionCtx:
    """Shared context for the overlapped forward-backward execution loop."""

    comp_stream: torch.cuda.Stream
    """Main compute stream for forward/backward kernels."""
    comm_stream: torch.cuda.Stream
    """Separate stream for asynchronous all-to-all communication."""
    fwd_event: torch.cuda.Event
    """Event recorded after forward compute; comm_stream waits on it before dispatch."""
    bwd_event: torch.cuda.Event
    """Event recorded after backward compute; comm_stream waits on it before combine."""
    fwd_comm_work: Optional[torch.distributed.Work]
    """Async work handle for the in-flight forward all-to-all (dispatch or combine)."""
    bwd_comm_work: Optional[torch.distributed.Work]
    """Async work handle for the in-flight backward all-to-all."""
    fwd_comm_deferred_free: List[torch.Tensor]
    """Tensors whose storage should be freed after the next fwd_comm_work.wait().

    Callers append tensors here after launching async forward comms (e.g.
    all-to-all in Stage 2 / Stage 4).  The subsequent stage that waits on
    fwd_comm_work drains and frees this list automatically.
    """
    ep_backend: Optional[Any]
    """EP backend dict (from make_ep_backend); None for the torch path."""
    fwd_comm_prev_event: Optional[Any]
    """Comp_stream snapshot after forward compute, consumed by forward-direction comm."""


# ------------------------------------------------------------
# STAGE1(F/B)
# ------------------------------------------------------------


class Stage1Args(NamedTuple):
    prev_hidden_states: torch.Tensor
    next_hidden_states: torch.Tensor


class Stage1OutsMoe(NamedTuple):
    sorted_tokens: torch.Tensor
    topk_weight: torch.Tensor
    residual: torch.Tensor


class Stage1OutsMlp(NamedTuple):
    sorted_tokens: torch.Tensor
    residual: torch.Tensor


@dataclass(init=False, slots=True)
class Stage1Record:
    args: Stage1Args
    outs: Union[Stage1OutsMoe, Stage1OutsMlp]


def stage1_f(
    ctx: ExecutionCtx,
    layer: DecoderLayerProtocol,
    hidden_states: torch.Tensor,
):
    """Stage1 forward."""
    nvtx.range_push("layer%02d.stage1_f" % layer.idx)
    record = Stage1Record()

    prev_hidden_states = hidden_states
    next_hidden_states = hidden_states.detach().requires_grad_()
    record.args = Stage1Args(prev_hidden_states, next_hidden_states)

    output = layer.forward_attn(next_hidden_states)
    ctx.comp_stream.record_event(ctx.fwd_event)
    if is_deepep_backend(ctx.ep_backend):
        ctx.fwd_comm_prev_event = deep_ep.EventHandle()

    if hasattr(layer.mlp, "experts"):
        # deepep: backprop through hidden_states_post_attn (input to buf.dispatch).
        # torch: backprop through sorted_tokens (input to direct_all_to_all).
        record.outs = Stage1OutsMoe(
            output.hidden_states_post_attn if uses_deepep_dispatch(layer) else output.sorted_tokens,
            output.topk_weight,
            output.residual,
        )
    else:
        record.outs = Stage1OutsMlp(output.sorted_tokens, output.residual)

    nvtx.range_pop()
    return record, output


def stage1_b(
    ctx: ExecutionCtx,
    layer: DecoderLayerProtocol,
    record: Stage1Record,
    grad_tensors: Union[Stage1OutsMoe, Stage1OutsMlp],
):
    """Stage1 backward."""
    nvtx.range_push("layer%02d.stage1_b" % layer.idx)

    if ctx.bwd_comm_work is not None:
        ctx.bwd_comm_work.wait()

    run_backward(record.outs, grad_tensors)

    hidden_states_grad = record.args.next_hidden_states.grad
    record.args.prev_hidden_states.grad = hidden_states_grad

    nvtx.range_pop()
    return hidden_states_grad


# ------------------------------------------------------------
# STAGE2(F/B)
# ------------------------------------------------------------


@dataclass(slots=True)
class Stage2Record:
    a2a_ctx: Optional[tuple] = None
    dispatch_state: Optional[dict] = None


def stage2_f(
    ctx: ExecutionCtx,
    layer: DecoderLayerProtocol,
    sorted_tokens: torch.Tensor,
    output_splits: Optional[List[int]],
    input_splits: Optional[List[int]],
    ep_group: Optional[torch.distributed.ProcessGroup] = None,
    expert_idxs: Optional[torch.Tensor] = None,
    expand_idx: Optional[torch.Tensor] = None,
    hidden_states_post_attn: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weight: Optional[torch.Tensor] = None,
):
    """Stage2 forward: pure communication — all-to-all dispatch.
    Returns (record, gathered_tokens, expert_idxs, expand_idx). On the deepep
    path expert_idxs / expand_idx are None — forward_mlp computes them from
    recv_topk_idx."""
    nvtx.range_push("layer%02d.stage2_f" % layer.idx)
    record = Stage2Record()

    if uses_deepep_dispatch(layer):
        # Stage 2/3 autograd boundary for activations is stage3_f's
        # gathered_tokens.detach().requires_grad_(); for the routing weights
        # the boundary is created inside forward_mlp.
        recv_x, record.dispatch_state, ctx.fwd_comm_work = deepep_dispatch(
            ctx.ep_backend,
            hidden_states_post_attn,
            topk_ids,
            topk_weight,
            comp_stream=ctx.comp_stream,
            previous_event=ctx.fwd_comm_prev_event,
        )
        nvtx.range_pop()
        return record, recv_x, None, None

    ctx.comm_stream.wait_event(ctx.fwd_event)
    sorted_tokens = sorted_tokens.detach()
    if output_splits is not None:
        with torch.cuda.stream(ctx.comm_stream):
            gathered_tokens = direct_all_to_all(
                sorted_tokens, output_splits, input_splits, ep_group
            )
        record.a2a_ctx = (output_splits, input_splits, ep_group)
    else:
        gathered_tokens = sorted_tokens

    ctx.fwd_comm_work = getattr(gathered_tokens, "comm_work", None)
    setattr(gathered_tokens, "comm_work", None)

    nvtx.range_pop()
    return record, gathered_tokens, expert_idxs, expand_idx


def stage2_b(
    ctx: ExecutionCtx,
    layer: DecoderLayerProtocol,
    record: Stage2Record,
    grad_tensors: tuple,
):
    """Stage2 backward: pure communication — reverse all-to-all.
    Returns (hidden_states_grad, topk_weight_grad). topk_weight_grad is None
    on the torch path (grad flows through stage5_b instead)."""
    nvtx.range_push("layer%02d.stage2_b" % layer.idx)

    if is_deepep_dispatch_state(record.dispatch_state):
        ep = ctx.ep_backend
        buf = ep["buffer"]
        dispatch_state = record.dispatch_state
        grad_recv_x = grad_tensors[0]
        rtw_leaf = dispatch_state["recv_topk_weights_leaf"]
        if rtw_leaf is not None and rtw_leaf.requires_grad and rtw_leaf.grad is not None:
            grad_recv_topk_weights = rtw_leaf.grad.contiguous()
        else:
            grad_recv_topk_weights = None

        grad_h_flat, grad_tw_flat, ev = buf.combine(
            grad_recv_x.contiguous(),
            handle=dispatch_state["dispatch_handle"],
            topk_weights=grad_recv_topk_weights,
            num_sms=ep["num_sms"],
            async_with_compute_stream=True,
            allocate_on_comm_stream=True,
        )
        # Defer the wait: stage1_b in the next iteration calls bwd_comm_work.wait()
        # before consuming. Letting comp_stream advance lets stage3_w overlap
        # with this buf.combine.
        grad_h_flat.record_stream(ctx.comp_stream)
        if grad_tw_flat is not None:
            grad_tw_flat.record_stream(ctx.comp_stream)
        ctx.bwd_comm_work = DeepEPEventWork(ev)

        h_grad = grad_h_flat.view(*dispatch_state["input_h_shape"])
        if grad_tw_flat is not None and dispatch_state["input_tw_shape"] is not None:
            tw_grad = grad_tw_flat.view(*dispatch_state["input_tw_shape"])
        else:
            tw_grad = None
        nvtx.range_pop()
        return h_grad, tw_grad

    ctx.comm_stream.wait_event(ctx.bwd_event)

    if record.a2a_ctx is not None:
        output_splits, input_splits, group = record.a2a_ctx
        with torch.cuda.stream(ctx.comm_stream):
            sorted_tokens_grad = direct_all_to_all(
                grad_tensors[0], input_splits, output_splits, group
            )
        ctx.bwd_comm_work = sorted_tokens_grad.comm_work
        sorted_tokens_grad.comm_work = None
    else:
        sorted_tokens_grad = grad_tensors[0]
        ctx.bwd_comm_work = None

    nvtx.range_pop()
    return sorted_tokens_grad, None


# ------------------------------------------------------------
# STAGE3(F/B/W)
# ------------------------------------------------------------


class Stage3Args(NamedTuple):
    gathered_tokens: torch.Tensor


class Stage3Outs(NamedTuple):
    moe_outs: torch.Tensor


@dataclass(init=False, slots=True)
class Stage3Record:
    args: Stage3Args
    outs: Stage3Outs


def _drain_deferred_free(ctx: ExecutionCtx) -> None:
    """Free tensor storage that was deferred until after the comm wait."""
    for t in ctx.fwd_comm_deferred_free:
        t.untyped_storage().resize_(0)
    ctx.fwd_comm_deferred_free.clear()


def stage3_f(
    ctx: ExecutionCtx,
    layer: DecoderLayerProtocol,
    gathered_tokens: torch.Tensor,
    expert_idxs: Optional[torch.Tensor],
    expand_idx: Optional[torch.Tensor] = None,
    dispatch_state: Optional[dict] = None,
):
    """Stage3 forward — pure compute. Dispatches to layer.forward_mlp."""
    nvtx.range_push("layer%02d.stage3_f" % layer.idx)
    record = Stage3Record()

    gathered_tokens = gathered_tokens.detach().requires_grad_()
    record.args = Stage3Args(gathered_tokens)

    if ctx.fwd_comm_work is not None:
        ctx.fwd_comm_work.wait()
    _drain_deferred_free(ctx)

    moe_outs = layer.forward_mlp(
        gathered_tokens, expert_idxs, expand_idx, dispatch_state=dispatch_state
    )
    record.outs = Stage3Outs(moe_outs)
    # Free the args storage - only safe for MoE layers with EP where
    # padded_index_gather is the first consumer and doesn't save the input.
    # When ep_size==1, gathered_tokens shares storage with sorted_tokens.
    if hasattr(layer.mlp, "experts") and ctx.fwd_comm_work is not None:
        gathered_tokens.untyped_storage().resize_(0)

    ctx.comp_stream.record_event(ctx.fwd_event)
    if is_deepep_backend(ctx.ep_backend):
        ctx.fwd_comm_prev_event = deep_ep.EventHandle()

    nvtx.range_pop()
    return record, moe_outs


def stage3_b(
    ctx: ExecutionCtx,
    layer: DecoderLayerProtocol,
    record: Stage3Record,
    grad_tensors: Stage3Outs,
):
    """Stage3 backward for input."""
    nvtx.range_push("layer%02d.stage3_b" % layer.idx)

    if ctx.bwd_comm_work is not None:
        ctx.bwd_comm_work.wait()

    WeightGradStore.enabled = True
    run_backward(record.outs, grad_tensors)
    WeightGradStore.enabled = False

    ctx.comp_stream.record_event(ctx.bwd_event)

    gathered_tokens_grad = record.args.gathered_tokens.grad

    nvtx.range_pop()
    return gathered_tokens_grad


def stage3_w(ctx: ExecutionCtx, layer: DecoderLayerProtocol):
    """Stage3 backward for weight."""
    nvtx.range_push("layer%02d.stage3_w" % layer.idx)

    WeightGradStore.flush()
    WeightGradStore.pop()

    nvtx.range_pop()


# ------------------------------------------------------------
# STAGE4(F/B)
# ------------------------------------------------------------


@dataclass(slots=True)
class Stage4Record:
    a2a_ctx: Optional[tuple] = None
    dispatch_state: Optional[dict] = None


def stage4_f(
    ctx: ExecutionCtx,
    layer: DecoderLayerProtocol,
    moe_outs: torch.Tensor,
    input_splits: Optional[List[int]],
    output_splits: Optional[List[int]],
    ep_group: Optional[torch.distributed.ProcessGroup] = None,
    dispatch_state: Optional[dict] = None,
):
    """Stage4 forward: pure communication — all-to-all combine."""
    nvtx.range_push("layer%02d.stage4_f" % layer.idx)
    record = Stage4Record()

    if is_deepep_dispatch_state(dispatch_state):
        combined, ctx.fwd_comm_work = deepep_combine(
            ctx.ep_backend,
            moe_outs,
            dispatch_state,
            comp_stream=ctx.comp_stream,
            previous_event=ctx.fwd_comm_prev_event,
        )
        record.dispatch_state = dispatch_state
        nvtx.range_pop()
        return record, combined

    moe_outs = moe_outs.detach()
    ctx.comm_stream.wait_event(ctx.fwd_event)

    if output_splits is not None:
        with torch.cuda.stream(ctx.comm_stream):
            moe_outs = direct_all_to_all(moe_outs, input_splits, output_splits, ep_group)
        record.a2a_ctx = (input_splits, output_splits, ep_group)

    ctx.fwd_comm_work = getattr(moe_outs, "comm_work", None)
    setattr(moe_outs, "comm_work", None)

    nvtx.range_pop()
    return record, moe_outs


def stage4_b(
    ctx: ExecutionCtx,
    layer: DecoderLayerProtocol,
    record: Stage4Record,
    grad_tensors: tuple,
):
    """Stage4 backward: pure communication — reverse all-to-all. On the deepep
    path this calls buf.dispatch (reverse of buf.combine)."""
    nvtx.range_push("layer%02d.stage4_b" % layer.idx)

    if is_deepep_dispatch_state(record.dispatch_state):
        ep = ctx.ep_backend
        buf = ep["buffer"]
        dispatch_state = record.dispatch_state
        grad_combined = grad_tensors[0]
        # Snapshot comp_stream so comm_stream waits only on grad_combined-ready.
        prev_event = deep_ep.EventHandle()
        grad_combined.record_stream(ep["comm_stream"])

        with torch.cuda.stream(ep["comm_stream"]):
            moe_outs_grad, _, _, _, ev = buf.dispatch(
                grad_combined.contiguous(),
                handle=dispatch_state["dispatch_handle"],
                num_sms=ep["num_sms"],
                previous_event=prev_event,
                async_with_compute_stream=True,
                allocate_on_comm_stream=True,
            )
        moe_outs_grad.record_stream(ctx.comp_stream)
        ctx.bwd_comm_work = DeepEPEventWork(ev)  # stage3_b waits before using
        nvtx.range_pop()
        return moe_outs_grad

    ctx.comm_stream.wait_event(ctx.bwd_event)

    if record.a2a_ctx is not None:
        output_splits, input_splits, group = record.a2a_ctx
        with torch.cuda.stream(ctx.comm_stream):
            moe_outs_grad = direct_all_to_all(grad_tensors[0], input_splits, output_splits, group)
        ctx.bwd_comm_work = moe_outs_grad.comm_work
        moe_outs_grad.comm_work = None
    else:
        moe_outs_grad = grad_tensors[0]
        ctx.bwd_comm_work = None

    nvtx.range_pop()
    return moe_outs_grad


# ------------------------------------------------------------
# STAGE5(F/B)
# ------------------------------------------------------------


class Stage5Args(NamedTuple):
    moe_outs: torch.Tensor
    topk_weight: torch.Tensor
    residual: torch.Tensor


class Stage5Outs(NamedTuple):
    hidden_states: torch.Tensor


@dataclass(init=False, slots=True)
class Stage5Record:
    args: Stage5Args
    outs: Stage5Outs


def stage5_f(
    ctx: ExecutionCtx,
    layer: DecoderLayerProtocol,
    moe_outs: torch.Tensor,
    moe_local_idxs,
    topk_weight: torch.Tensor,
    residual: torch.Tensor,
):
    """Stage5 forward."""
    nvtx.range_push("layer%02d.stage5_f" % layer.idx)
    record = Stage5Record()

    moe_outs = moe_outs.detach().requires_grad_()
    topk_weight = topk_weight.detach().requires_grad_() if topk_weight is not None else None
    residual = residual.detach().requires_grad_()
    record.args = Stage5Args(moe_outs, topk_weight, residual)

    if ctx.fwd_comm_work is not None:
        ctx.fwd_comm_work.wait()
    _drain_deferred_free(ctx)

    hidden_states = layer.forward_aggregate(moe_outs, moe_local_idxs, topk_weight, residual)
    record.outs = Stage5Outs(hidden_states)

    nvtx.range_pop()
    return record, hidden_states


def stage5_b(
    ctx: ExecutionCtx,
    layer: DecoderLayerProtocol,
    record: Stage5Record,
    grad_tensors: Stage5Outs,
):
    """Stage5 backward."""
    nvtx.range_push("layer%02d.stage5_b" % layer.idx)

    run_backward(record.outs, grad_tensors)

    ctx.comp_stream.record_event(ctx.bwd_event)

    moe_outs_grad, topk_weight_grad, residual_grad = [
        t.grad if t is not None else None for t in record.args
    ]

    nvtx.range_pop()
    return moe_outs_grad, topk_weight_grad, residual_grad


# ------------------------------------------------------------
# STAGE5_AND_STAGE1(F/B) - Merged stage 5 + stage 1
# ------------------------------------------------------------


def stage5_and_stage1_f(
    ctx: ExecutionCtx,
    prev_layer: DecoderLayerProtocol,
    next_layer: DecoderLayerProtocol,
    moe_outs: torch.Tensor,
    moe_local_idxs,
    topk_weight: torch.Tensor,
    residual: torch.Tensor,
):
    """
    Merged Stage5 and Stage1 forward.
    Returns (stage5_args, stage1_outs, output) for storage in separate layer records.
    """
    nvtx.range_push("layer%02d_stage5_f_layer%02d_stage1_f" % (prev_layer.idx, next_layer.idx))

    moe_outs = moe_outs.detach().requires_grad_()
    topk_weight = topk_weight.detach().requires_grad_() if topk_weight is not None else None
    residual = residual.detach().requires_grad_()
    stage5_args = Stage5Args(moe_outs, topk_weight, residual)

    if ctx.fwd_comm_work is not None:
        ctx.fwd_comm_work.wait()
    _drain_deferred_free(ctx)

    hidden_states = prev_layer.forward_aggregate(moe_outs, moe_local_idxs, topk_weight, residual)

    output = next_layer.forward_attn(hidden_states)
    ctx.comp_stream.record_event(ctx.fwd_event)
    if is_deepep_backend(ctx.ep_backend):
        ctx.fwd_comm_prev_event = deep_ep.EventHandle()

    if hasattr(next_layer.mlp, "experts"):
        stage1_outs = Stage1OutsMoe(
            output.hidden_states_post_attn
            if uses_deepep_dispatch(next_layer)
            else output.sorted_tokens,
            output.topk_weight,
            output.residual,
        )
    else:
        stage1_outs = Stage1OutsMlp(output.sorted_tokens, output.residual)

    nvtx.range_pop()
    return stage5_args, stage1_outs, output


def stage5_and_stage1_b(
    ctx: ExecutionCtx,
    next_layer: DecoderLayerProtocol,
    prev_layer: DecoderLayerProtocol,
    stage1_outs: Union[Stage1OutsMoe, Stage1OutsMlp],
    stage5_args: Stage5Args,
    grad_tensors: Union[Stage1OutsMoe, Stage1OutsMlp],
):
    """
    Merged Stage5 and Stage1 backward.
    Takes stage1_outs (from next layer) and stage5_args (from prev layer) separately.
    """
    nvtx.range_push("layer%02d_stage5_b_layer%02d_stage1_b" % (prev_layer.idx, next_layer.idx))

    if ctx.bwd_comm_work is not None:
        ctx.bwd_comm_work.wait()

    run_backward(stage1_outs, grad_tensors)

    ctx.comp_stream.record_event(ctx.bwd_event)

    moe_outs_grad, topk_weight_grad, residual_grad = [
        t.grad if t is not None else None for t in stage5_args
    ]

    nvtx.range_pop()
    return moe_outs_grad, topk_weight_grad, residual_grad


# ------------------------------------------------------------
# PROLOG(F/B)
# ------------------------------------------------------------


class PrologArgs(NamedTuple):
    pass


class PrologOuts(NamedTuple):
    hidden_states: torch.Tensor


@dataclass(init=False, slots=True)
class PrologRecord:
    args: PrologArgs
    outs: PrologOuts


def prolog_f(module: ModelProtocol, hidden_states: torch.Tensor):
    """Prolog forward."""
    nvtx.range_push("prolog_f")
    record = PrologRecord()

    record.args = PrologArgs()
    hidden_states = module.embed_tokens(hidden_states)
    record.outs = PrologOuts(hidden_states)

    nvtx.range_pop()
    return record, hidden_states


def prolog_b(module: ModelProtocol, record: PrologRecord, grad_tensors: PrologOuts):
    """Prolog backward."""
    nvtx.range_push("prolog_b")

    run_backward(record.outs, grad_tensors)

    nvtx.range_pop()
    return


# ------------------------------------------------------------
# EPILOG(F/B)
# ------------------------------------------------------------


class EpilogArgs(NamedTuple):
    hidden_states: torch.Tensor


@dataclass(init=False, slots=True)
class EpilogRecord:
    args: EpilogArgs


def epilog_f(module: ModelProtocol, hidden_states: torch.Tensor):
    """
    Epilog forward: norm + lm_head.

    The backward is handled by ``loss.backward()`` which traverses the autograd
    graph through norm -> lm_head -> criterion.  The only thing the caller needs
    from the record is ``args.hidden_states.grad`` (populated by autograd).
    """
    nvtx.range_push("epilog_f")
    record = EpilogRecord()

    hidden_states = hidden_states.detach().requires_grad_()
    record.args = EpilogArgs(hidden_states)
    hidden_states = module.norm(hidden_states)
    logits = module.lm_head(hidden_states)

    nvtx.range_pop()
    return record, logits


# ------------------------------------------------------------
# INTERMEDIATE TENSORS
# ------------------------------------------------------------


@dataclass(init=False, slots=True)
class IntermediateTensorsLayer:
    stage1: Stage1Record
    stage2: Stage2Record
    stage3: Stage3Record
    stage4: Stage4Record
    stage5: Stage5Record


@dataclass(init=False, slots=True)
class IntermediateTensors:
    prolog: Optional[PrologRecord]
    epilog: Optional[EpilogRecord]
    layers: List[IntermediateTensorsLayer]


def create_intermediate_tensors_layer() -> IntermediateTensorsLayer:
    """Create a pre-allocated IntermediateTensorsLayer with all records."""
    layer = IntermediateTensorsLayer()
    layer.stage1 = Stage1Record()
    layer.stage2 = Stage2Record()
    layer.stage3 = Stage3Record()
    layer.stage4 = Stage4Record()
    layer.stage5 = Stage5Record()
    return layer


def create_intermediate_tensors(
    num_layers: int, has_prolog: bool, has_epilog: bool
) -> IntermediateTensors:
    """Create a pre-allocated IntermediateTensors structure for reuse across iterations."""
    tensors = IntermediateTensors()
    tensors.prolog = PrologRecord() if has_prolog else None
    tensors.epilog = EpilogRecord() if has_epilog else None
    tensors.layers = [create_intermediate_tensors_layer() for _ in range(num_layers)]
    return tensors
