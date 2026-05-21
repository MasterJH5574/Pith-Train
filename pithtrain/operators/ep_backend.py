"""EP backend selection + DeepEP dispatch/combine wrappers.

``None`` selects the torch ``direct_all_to_all`` path; a dict from
``make_ep_backend`` selects the DeepEP path. The dispatch/combine wrappers
here are reused by both the overlapped DualPipeV scheduler and the
non-overlapped ``decoder_layer_forward`` path.
"""

import atexit
import os
from typing import Literal, Optional, Tuple

import torch
import torch.distributed as dist

try:
    import deep_ep
except ImportError:
    deep_ep = None

EPBackendKind = Literal["auto", "deepep", "torch"]


def is_deepep_backend(ep_backend) -> bool:
    """True iff ``ep_backend`` was produced by the deepep path of
    ``make_ep_backend`` (i.e., its ``"kind"`` is ``"deepep"``)."""
    return ep_backend is not None and ep_backend.get("kind") == "deepep"


def is_deepep_dispatch_state(dispatch_state) -> bool:
    """True iff ``dispatch_state`` was produced by ``deepep_dispatch`` (i.e.,
    its ``"backend"`` tag is ``"deepep"``). ``None`` returns False."""
    return dispatch_state is not None and dispatch_state.get("backend") == "deepep"


class DeepEPEventWork:
    """Duck-typed torch.distributed.Work shim around DeepEP's EventOverlap.
    ``.wait()`` is a stream-side dependency (non-blocking on CPU)."""

    __slots__ = ("ev",)

    def __init__(self, ev):
        self.ev = ev

    def wait(self) -> None:
        self.ev.current_stream_wait()


def _make_deepep_state(
    group: dist.ProcessGroup,
    num_max_tokens_per_rank: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
) -> dict:
    buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        use_fp8_dispatch=False,
        explicitly_destroy=True,
    )
    # Tear down before NCCL pg destruction (atexit is LIFO; pg destroy was
    # registered by setup_default_process_group earlier).
    destroyed = [False]

    def _safe_destroy() -> None:
        if destroyed[0]:
            return
        destroyed[0] = True
        try:
            buffer.destroy()
        except Exception:
            pass

    atexit.register(_safe_destroy)

    # num_sms is populated on the first dispatch via buf.get_theoretical_num_sms
    # so subsequent dispatch/combine calls (incl. backward direction) reuse it.
    return {
        "kind": "deepep",
        "buffer": buffer,
        "comm_stream": buffer.get_comm_stream(),
        "num_max_tokens_per_rank": num_max_tokens_per_rank,
        "num_topk": num_topk,
        "num_experts": num_experts,
        "num_sms": 0,
    }


def make_ep_backend(
    kind: EPBackendKind,
    group: dist.ProcessGroup,
    num_max_tokens_per_rank: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
) -> Optional[dict]:
    """auto: try deepep, fall back to torch (None); deepep: force, raise if
    unavailable; torch: return None."""
    rank0 = group.rank() == 0

    def _log(msg: str) -> None:
        if rank0:
            print(f"[pithtrain] EP backend: {msg}", flush=True)

    if kind == "torch":
        _log(f"torch ({group.size()} ranks)")
        return None

    if kind in ("auto", "deepep"):
        if deep_ep is None:
            if kind == "deepep":
                raise RuntimeError("ep_backend=deepep requested but deep_ep not installed")
            _log(f"torch ({group.size()} ranks; deep_ep not installed)")
            return None
        try:
            # Probe DeepEP runtime deps (NCCL, IBGDA) with a minimum-size buffer.
            probe = deep_ep.ElasticBuffer(
                group,
                num_max_tokens_per_rank=1,
                hidden=256,
                num_topk=1,
                use_fp8_dispatch=False,
                explicitly_destroy=True,
            )
            probe.destroy()
        except (RuntimeError, AssertionError) as e:
            if kind == "deepep":
                raise RuntimeError(f"ep_backend=deepep requested but unavailable: {e}") from e
            _log(f"torch ({group.size()} ranks; deepep probe failed: {e})")
            return None

        backend = _make_deepep_state(group, num_max_tokens_per_rank, hidden, num_topk, num_experts)
        _log(f"deepep ({group.size()} ranks)")
        return backend
    raise ValueError(f"unknown ep_backend kind {kind!r}")


def resolve_ep_backend_kind(requested: EPBackendKind) -> EPBackendKind:
    """Apply PITHTRAIN_EP_BACKEND override if set."""
    env = os.environ.get("PITHTRAIN_EP_BACKEND")
    if env is None:
        return requested
    if env not in ("auto", "deepep", "torch"):
        raise ValueError(
            f"PITHTRAIN_EP_BACKEND must be one of 'auto', 'deepep', 'torch'; got {env!r}"
        )
    return env  # type: ignore[return-value]


def deepep_dispatch(
    ep_backend: dict,
    hidden_states_post_attn: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weight: Optional[torch.Tensor],
    *,
    comp_stream: torch.cuda.Stream,
    previous_event=None,
) -> Tuple[torch.Tensor, dict, DeepEPEventWork]:
    """Direct ``buf.dispatch``. Returns ``(recv_x, dispatch_state, work)``.

    The bundle is consumed by stage 3 (forward_mlp), stage 4_f (buf.combine),
    stage 2_b (buf.combine reverse), and stage 4_b (buf.dispatch reverse).
    """
    buf = ep_backend["buffer"]
    H = hidden_states_post_attn.shape[-1]
    K = ep_backend["num_topk"]
    num_experts = ep_backend["num_experts"]
    assert hidden_states_post_attn.is_contiguous(), "hidden_states_post_attn must be contiguous"
    assert topk_ids.dtype == deep_ep.topk_idx_t and topk_ids.is_contiguous(), (
        f"topk_ids must be {deep_ep.topk_idx_t} and contiguous; "
        f"got {topk_ids.dtype}, contig={topk_ids.is_contiguous()}"
    )
    x_flat = hidden_states_post_attn.reshape(-1, H)
    topk_ids_flat = topk_ids.reshape(-1, K)
    if topk_weight is not None:
        assert topk_weight.is_contiguous()
        topk_weight_flat = topk_weight.reshape(-1, K)
    else:
        topk_weight_flat = None

    num_sms = buf.get_theoretical_num_sms(num_experts, K)
    ep_backend["num_sms"] = num_sms

    if previous_event is not None:
        x_flat.record_stream(ep_backend["comm_stream"])
        topk_ids_flat.record_stream(ep_backend["comm_stream"])
        if topk_weight_flat is not None:
            topk_weight_flat.record_stream(ep_backend["comm_stream"])

    recv_x, recv_topk_idx, recv_topk_weights, dispatch_handle, ev = buf.dispatch(
        x_flat,
        topk_idx=topk_ids_flat,
        topk_weights=topk_weight_flat,
        num_experts=num_experts,
        num_max_tokens_per_rank=ep_backend["num_max_tokens_per_rank"],
        expert_alignment=1,
        num_sms=num_sms,
        previous_event=previous_event,
        async_with_compute_stream=True,
        allocate_on_comm_stream=True,
    )

    recv_x.record_stream(comp_stream)
    recv_topk_idx.record_stream(comp_stream)
    if recv_topk_weights is not None:
        recv_topk_weights.record_stream(comp_stream)

    dispatch_state = {
        "backend": "deepep",
        "dispatch_handle": dispatch_handle,
        "recv_topk_idx": recv_topk_idx,
        "recv_topk_weights": recv_topk_weights,
        "recv_topk_weights_leaf": None,  # populated by forward_mlp, read by stage2_b
        "input_h_shape": hidden_states_post_attn.shape,
        "input_tw_shape": topk_weight.shape if topk_weight is not None else None,
    }
    return recv_x, dispatch_state, DeepEPEventWork(ev)


def deepep_combine(
    ep_backend: dict,
    moe_outs: torch.Tensor,
    dispatch_state: dict,
    *,
    comp_stream: torch.cuda.Stream,
    previous_event=None,
) -> Tuple[torch.Tensor, DeepEPEventWork]:
    """Direct ``buf.combine``. Returns ``(combined, work)``. Detaches ``moe_outs``."""
    buf = ep_backend["buffer"]
    # buf.combine runs inside ``with torch.cuda.stream(...)``; DeepEP's default
    # stream_wait asserts s_0 != s_1, so always pass a non-None previous_event.
    if previous_event is None:
        previous_event = deep_ep.EventHandle()
    moe_outs = moe_outs.detach()
    moe_outs.record_stream(ep_backend["comm_stream"])
    with torch.cuda.stream(ep_backend["comm_stream"]):
        combined, _, ev = buf.combine(
            moe_outs,
            handle=dispatch_state["dispatch_handle"],
            num_sms=ep_backend["num_sms"],
            previous_event=previous_event,
            async_with_compute_stream=True,
            allocate_on_comm_stream=True,
        )
    combined.record_stream(comp_stream)
    return combined, DeepEPEventWork(ev)
