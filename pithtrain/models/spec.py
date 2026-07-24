"""Module specs and builder for model construction."""

import importlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

from torch import nn

from pithtrain.layers.factory import get_group_linear_cls, get_linear_cls


@dataclass
class ModuleSpec:
    module: Union[str, Callable, None]
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    submodules: Dict[str, "ModuleSpec"] = field(default_factory=dict)


# key -> "module.path:attr"; populated only for the string-keyed variant.
MODULE_REGISTRY: Dict[str, str] = {}


def _resolve(path: str) -> Callable:
    module_path, attr = path.split(":")
    return getattr(importlib.import_module(module_path), attr)


def build_module(spec: Optional[ModuleSpec], *args, **kwargs) -> Optional[nn.Module]:
    """Instantiate ``spec``, threading nested submodules to targets that accept them."""
    if spec is None or spec.module is None:
        return None
    cls = _resolve(MODULE_REGISTRY[spec.module]) if isinstance(spec.module, str) else spec.module
    merged = {**spec.kwargs, **kwargs}
    if (
        spec.submodules
        and isinstance(cls, type)
        and "submodules" in inspect.signature(cls).parameters
    ):
        merged.setdefault("submodules", spec.submodules)
    return cls(*args, *spec.args, **merged)


# Shared building blocks (reused across architectures).
def linear_spec() -> ModuleSpec:
    return ModuleSpec(get_linear_cls())


def group_linear_spec() -> ModuleSpec:
    return ModuleSpec(get_group_linear_cls())


def norm_spec() -> ModuleSpec:
    return ModuleSpec(nn.RMSNorm)


def dense_mlp_submodules() -> Dict[str, ModuleSpec]:
    return {"gate_proj": linear_spec(), "up_proj": linear_spec(), "down_proj": linear_spec()}


def grouped_experts_submodules() -> Dict[str, ModuleSpec]:
    return {
        "gate_proj": group_linear_spec(),
        "up_proj": group_linear_spec(),
        "down_proj": group_linear_spec(),
    }


def get_layer_spec(model_type, config, layer_id, ep_group=None, cp_group=None) -> ModuleSpec:
    if model_type == "deepseek_v2":
        from pithtrain.models.deepseek_v2_lite import (
            DeepseekV2LiteAttention,
            DeepseekV2LiteDecoderLayer,
            DeepseekV2LiteExperts,
            DeepseekV2LiteMLP,
            DeepseekV2LiteMoEGate,
            DeepseekV2LiteMoEWithGroupGeMM,
        )

        attn = ModuleSpec(
            DeepseekV2LiteAttention,
            submodules={
                "q_proj": linear_spec(),
                "kv_a_proj_with_mqa": linear_spec(),
                "kv_a_layernorm": norm_spec(),
                "kv_b_proj": linear_spec(),
                "o_proj": linear_spec(),
            },
        )
        use_moe = (
            config.n_routed_experts is not None
            and layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        )
        if use_moe:
            moe_sub = {
                "gate": ModuleSpec(DeepseekV2LiteMoEGate),
                "experts": ModuleSpec(
                    DeepseekV2LiteExperts, submodules=grouped_experts_submodules()
                ),
            }
            if config.n_shared_experts is not None:
                moe_sub["shared_experts"] = ModuleSpec(
                    DeepseekV2LiteMLP, submodules=dense_mlp_submodules()
                )
            mlp = ModuleSpec(
                DeepseekV2LiteMoEWithGroupGeMM,
                kwargs={"ep_group": ep_group, "layer_id": layer_id},
                submodules=moe_sub,
            )
        else:
            mlp = ModuleSpec(DeepseekV2LiteMLP, submodules=dense_mlp_submodules())
        layer_cls = DeepseekV2LiteDecoderLayer
    elif model_type == "qwen3_moe":
        from pithtrain.models.qwen3_moe import (
            Qwen3MoeAttention,
            Qwen3MoeDecoderLayer,
            Qwen3MoeExperts,
            Qwen3MoeGate,
            Qwen3MoeMLP,
            Qwen3MoeMoE,
        )

        attn = ModuleSpec(
            Qwen3MoeAttention,
            submodules={
                "q_proj": linear_spec(),
                "k_proj": linear_spec(),
                "v_proj": linear_spec(),
                "o_proj": linear_spec(),
                "q_norm": norm_spec(),
                "k_norm": norm_spec(),
            },
        )
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
        mlp_only_layers = getattr(config, "mlp_only_layers", []) or []
        use_moe = (
            config.num_experts > 0
            and (layer_id + 1) % decoder_sparse_step == 0
            and layer_id not in mlp_only_layers
        )
        if use_moe:
            mlp = ModuleSpec(
                Qwen3MoeMoE,
                kwargs={"ep_group": ep_group, "layer_id": layer_id},
                submodules={
                    "experts": ModuleSpec(Qwen3MoeExperts, submodules=grouped_experts_submodules()),
                    "gate": ModuleSpec(Qwen3MoeGate),
                },
            )
        else:
            mlp = ModuleSpec(Qwen3MoeMLP, submodules=dense_mlp_submodules())
        layer_cls = Qwen3MoeDecoderLayer
    elif model_type == "gpt_oss":
        from pithtrain.models.gpt_oss import (
            GptOssAttention,
            GptOssDecoderLayer,
            GptOssExperts,
            GptOssMLP,
            GptOssTopKRouter,
        )

        attn = ModuleSpec(
            GptOssAttention,
            submodules={
                "q_proj": linear_spec(),
                "k_proj": linear_spec(),
                "v_proj": linear_spec(),
                "o_proj": linear_spec(),
            },
        )
        mlp = ModuleSpec(
            GptOssMLP,
            kwargs={"ep_size": getattr(config, "ep_size", 1), "ep_group": ep_group},
            submodules={
                "experts": ModuleSpec(GptOssExperts),
                "router": ModuleSpec(GptOssTopKRouter),
            },
        )
        layer_cls = GptOssDecoderLayer
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return ModuleSpec(
        layer_cls,
        submodules={
            "self_attn": attn,
            "mlp": mlp,
            "input_layernorm": norm_spec(),
            "post_attention_layernorm": norm_spec(),
        },
    )
