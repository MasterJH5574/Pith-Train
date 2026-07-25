"""Module specs and builder for model construction."""

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

from torch import nn

from pithtrain.layers.factory import get_group_linear_cls, get_linear_cls


@dataclass
class ModuleSpec:
    module: Union[Callable, None]
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    submodules: Dict[str, "ModuleSpec"] = field(default_factory=dict)


def build_module(spec: Optional[ModuleSpec], *args, **kwargs) -> Optional[nn.Module]:
    """Instantiate ``spec``, threading nested submodules to targets that accept them."""
    if spec is None or spec.module is None:
        return None
    cls = spec.module
    merged = {**spec.kwargs, **kwargs}
    if (
        spec.submodules
        and isinstance(cls, type)
        and "submodules" in inspect.signature(cls).parameters
    ):
        merged.setdefault("submodules", spec.submodules)
    return cls(*args, *spec.args, **merged)


class Backend:
    """Resolves leaf projection / norm classes for the active linear implementation."""

    def linear(self) -> ModuleSpec:
        return ModuleSpec(get_linear_cls())

    def grouped_linear(self) -> ModuleSpec:
        return ModuleSpec(get_group_linear_cls())

    def norm(self) -> ModuleSpec:
        return ModuleSpec(nn.RMSNorm)

    def dense_mlp_submodules(self) -> Dict[str, ModuleSpec]:
        return {"gate_proj": self.linear(), "up_proj": self.linear(), "down_proj": self.linear()}

    def grouped_expert_submodules(self) -> Dict[str, ModuleSpec]:
        return {
            "gate_proj": self.grouped_linear(),
            "up_proj": self.grouped_linear(),
            "down_proj": self.grouped_linear(),
        }


def _attention_spec(config, backend: Backend, layer_id: int, cp_group) -> ModuleSpec:
    if getattr(config, "kv_lora_rank", None) is not None:
        from pithtrain.models.mla_attention import MLASelfAttention

        return ModuleSpec(
            MLASelfAttention,
            args=(config, layer_id),
            kwargs={"cp_group": cp_group},
            submodules={
                "q_proj": backend.linear(),
                "kv_a_proj_with_mqa": backend.linear(),
                "kv_a_layernorm": backend.norm(),
                "kv_b_proj": backend.linear(),
                "o_proj": backend.linear(),
            },
        )

    if config.model_type == "gpt_oss":
        from pithtrain.models.attention import SinkSelfAttention

        hidden_size = config.hidden_size
        num_attention_heads = config.num_attention_heads
        head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            layer_types = [
                "sliding_attention" if i % 2 == 0 else "full_attention"
                for i in range(config.num_hidden_layers)
            ]
        return ModuleSpec(
            SinkSelfAttention,
            kwargs={
                "hidden_size": hidden_size,
                "num_attention_heads": num_attention_heads,
                "num_key_value_heads": config.num_key_value_heads,
                "head_dim": head_dim,
                "attention_bias": getattr(config, "attention_bias", True),
                "is_sliding": layer_types[layer_id] == "sliding_attention",
                "sliding_window": getattr(config, "sliding_window", 128),
            },
            submodules={
                "q_proj": backend.linear(),
                "k_proj": backend.linear(),
                "v_proj": backend.linear(),
                "o_proj": backend.linear(),
            },
        )

    from pithtrain.models.attention import GQASelfAttention

    return ModuleSpec(
        GQASelfAttention,
        args=(config,),
        kwargs={"cp_group": cp_group},
        submodules={
            "q_proj": backend.linear(),
            "k_proj": backend.linear(),
            "v_proj": backend.linear(),
            "o_proj": backend.linear(),
            "q_norm": backend.norm(),
            "k_norm": backend.norm(),
        },
    )


def _dense_mlp_spec(config, backend: Backend) -> ModuleSpec:
    from pithtrain.models.mlp import GatedMLP

    return ModuleSpec(GatedMLP, args=(config,), submodules=backend.dense_mlp_submodules())


def _fused_moe_spec(config, backend: Backend, ep_group) -> ModuleSpec:
    from pithtrain.models.moe import FusedExperts, FusedMoE, TopKRouter

    return ModuleSpec(
        FusedMoE,
        kwargs={
            "hidden_size": config.hidden_size,
            "num_experts": getattr(config, "num_local_experts", 128),
            "num_experts_per_tok": getattr(config, "num_experts_per_tok", 4),
            "intermediate_size": config.intermediate_size,
            "swiglu_limit": float(getattr(config, "swiglu_limit", 7.0)),
            "ep_size": getattr(config, "ep_size", 1),
            "ep_group": ep_group,
        },
        submodules={
            "experts": ModuleSpec(FusedExperts),
            "router": ModuleSpec(TopKRouter),
        },
    )


def _shared_moe_spec(config, backend: Backend, layer_id: int, ep_group) -> ModuleSpec:
    from pithtrain.models.mlp import GatedMLP
    from pithtrain.models.moe import GroupedExperts, ScaledTopKGate, SharedExpertMoE

    return ModuleSpec(
        SharedExpertMoE,
        args=(config,),
        kwargs={"ep_group": ep_group, "layer_id": layer_id},
        submodules={
            "gate": ModuleSpec(ScaledTopKGate),
            "experts": ModuleSpec(GroupedExperts, submodules=backend.grouped_expert_submodules()),
            "shared_experts": ModuleSpec(GatedMLP, submodules=backend.dense_mlp_submodules()),
        },
    )


def _grouped_moe_spec(config, backend: Backend, layer_id: int, ep_group) -> ModuleSpec:
    from pithtrain.models.moe import GroupedExperts, GroupedMoE, TopKGate

    return ModuleSpec(
        GroupedMoE,
        args=(config,),
        kwargs={"ep_group": ep_group, "layer_id": layer_id},
        submodules={
            "experts": ModuleSpec(GroupedExperts, submodules=backend.grouped_expert_submodules()),
            "gate": ModuleSpec(TopKGate),
        },
    )


def _is_dense_layer(config, layer_id: int) -> bool:
    """Whether the routing schedule keeps ``layer_id`` as a plain MLP."""
    if getattr(config, "n_routed_experts", None) is not None:
        return layer_id < config.first_k_dense_replace or layer_id % config.moe_layer_freq != 0
    if getattr(config, "num_experts", 0) > 0:
        step = getattr(config, "decoder_sparse_step", 1)
        mlp_only_layers = getattr(config, "mlp_only_layers", []) or []
        return (layer_id + 1) % step != 0 or layer_id in mlp_only_layers
    return False


def _mlp_spec(config, backend: Backend, layer_id: int, ep_group) -> ModuleSpec:
    if _is_dense_layer(config, layer_id):
        return _dense_mlp_spec(config, backend)
    if getattr(config, "swiglu_limit", None) is not None:
        return _fused_moe_spec(config, backend, ep_group)
    if getattr(config, "n_shared_experts", None) is not None:
        return _shared_moe_spec(config, backend, layer_id, ep_group)
    return _grouped_moe_spec(config, backend, layer_id, ep_group)


def get_layer_spec(config, layer_id: int, ep_group=None, cp_group=None) -> ModuleSpec:
    """Build the spec for one generic decoder layer."""
    from pithtrain.models.attention import SinkSelfAttention
    from pithtrain.models.transformer import TransformerDecoderLayer

    backend = Backend()
    attn = _attention_spec(config, backend, layer_id, cp_group)
    mlp = _mlp_spec(config, backend, layer_id, ep_group)

    # FA-4 sink attention runs eager; every other attention stage compiles.
    compile_attn = attn.module is not SinkSelfAttention

    return ModuleSpec(
        TransformerDecoderLayer,
        kwargs={
            "layer_id": layer_id,
            "hidden_size": config.hidden_size,
            "rms_norm_eps": config.rms_norm_eps,
            "compile_attn": compile_attn,
        },
        submodules={
            "self_attn": attn,
            "mlp": mlp,
            "input_layernorm": backend.norm(),
            "post_attention_layernorm": backend.norm(),
        },
    )


def get_rotary_spec(config) -> ModuleSpec:
    """Build the spec for a model's rotary-embedding submodule."""
    if getattr(config, "qk_rope_head_dim", None) is not None:
        from pithtrain.models.rotary import YarnRotaryEmbedding

        scaling_factor = config.rope_scaling["factor"]
        rope_kwargs = {
            key: config.rope_scaling[key]
            for key in [
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            ]
            if key in config.rope_scaling
        }
        rope_theta = getattr(config, "rope_theta", None) or (config.rope_scaling or {}).get(
            "rope_theta"
        )
        return ModuleSpec(
            YarnRotaryEmbedding,
            kwargs={
                "dim": config.qk_rope_head_dim,
                "max_position_embeddings": config.max_position_embeddings,
                "scaling_factor": scaling_factor,
                "base": rope_theta,
                **rope_kwargs,
            },
        )

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    rope_scaling = getattr(config, "rope_scaling", None) or {}
    if rope_scaling.get("rope_type") == "yarn":
        from pithtrain.models.rotary import ScaledRotaryEmbedding

        return ModuleSpec(
            ScaledRotaryEmbedding,
            kwargs={
                "dim": head_dim,
                "max_position_embeddings": config.max_position_embeddings,
                "base": getattr(config, "rope_theta", 150000.0),
                "scaling_factor": float(rope_scaling.get("factor", 32.0)),
                "original_max_position_embeddings": int(
                    rope_scaling.get("original_max_position_embeddings", 4096)
                ),
                "beta_fast": float(rope_scaling.get("beta_fast", 32.0)),
                "beta_slow": float(rope_scaling.get("beta_slow", 1.0)),
                "truncate": bool(rope_scaling.get("truncate", False)),
            },
        )

    from pithtrain.models.rotary import DynamicRotaryEmbedding

    return ModuleSpec(
        DynamicRotaryEmbedding,
        kwargs={
            "dim": head_dim,
            "max_position_embeddings": config.max_position_embeddings,
            "base": getattr(config, "rope_theta", 1000000.0),
        },
    )
