"""
The compiled stages of the dense Qwen3.5 decoder layer over a fixed set of sequence lengths, as
an RL backend that pads micro-batches to length buckets feeds them, in both grad modes: each
stage holds at most variants x (lengths + 2) graphs, and lengths seen before compile nothing.

Forward-only graphs keep the token count symbolic: one static graph for the first length, one
dynamic graph after. Training graphs do not: AOTAutograd's backward compile restrides saved
activations to the forward's concrete strides, which specializes the token count, so training
compiles one graph per length per variant. Layer variants: forward_stage1 has three under CP
(full attention, linear attention that reshards to contiguous, linear attention already
contiguous) and two without (linear, full); forward_stage5 has two under CP (reshards back to
zigzag or not) and one without.

usage: pytest tests/test_qwen35_recompiles.py -v (one GPU without CP, two with)
"""

import json

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from pithtrain.contexts import distributed, training
from pithtrain.modules.distributed import DistributedCfg
from pithtrain.operators.linear import BF16Linear
from tests.utilities import launch

MODEL = "examples/pretrain_lm/qwen3.5-27b/config.json"

LENGTHS = (1024, 1218, 1448, 1722)
"""Rank-local token counts, four buckets of an octave. Even, for zigzag."""


def small_config(num_layers: int) -> Qwen3_5TextConfig:
    """
    The 27B config with its attention and Gated DeltaNet head geometry, a narrow residual stream
    and the first num_layers layers (linear, linear, linear, full, ...).
    """
    with open(MODEL) as f:
        config = json.load(f)
    config = config.get("text_config", config)
    config.update(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=num_layers,
        layer_types=config["layer_types"][:num_layers],
    )
    return Qwen3_5TextConfig(**config)


def cache_entries(stage) -> int:
    """
    The graphs Dynamo holds for a compiled stage: what recompile_limit counts.
    """
    code = getattr(stage, "_torchdynamo_orig_callable", stage).__code__
    return len(torch._dynamo.eval_frame._debug_get_cache_entry_list(code))


def verify_bounded_recompiles() -> None:
    # Imported here: the model module needs a GPU at import, and collection must not.
    from pithtrain.models.qwen35 import Qwen35DecoderLayer

    training.Linear, training.fp8 = BF16Linear, False  # what setup_model binds for BF16
    torch._dynamo.reset()
    device = distributed.device
    config = small_config(num_layers=8)
    torch.manual_seed(0)
    layers = [Qwen35DecoderLayer(config, i).to(device, torch.bfloat16) for i in range(8)]
    rotary_dim = int(config.head_dim * config.rope_parameters.get("partial_rotary_factor", 1.0))

    def step(S: int, grad: bool) -> None:
        hidden_states = torch.randn(1, S, config.hidden_size, device=device, dtype=torch.bfloat16)
        rotary_posemb = tuple(
            torch.randn(1, S, rotary_dim, device=device, dtype=torch.bfloat16) for _ in range(2)
        )
        with torch.set_grad_enabled(grad):
            out = hidden_states
            for layer in layers:
                # As DualPipeV calls them: each stage on leaves that require grad.
                out, residual, _ = layer.forward_stage1(out.detach().requires_grad_(), rotary_posemb)  # fmt: skip
                mlp_out = layer.forward_stage3(out.detach().requires_grad_())
                out = layer.forward_stage5(mlp_out.detach().requires_grad_(), None, None, residual.detach().requires_grad_())  # fmt: skip
            if grad:
                out.float().square().mean().backward()

    stages = {"forward_stage1": Qwen35DecoderLayer.forward_stage1, "forward_stage5": Qwen35DecoderLayer.forward_stage5}  # fmt: skip
    variants = {"forward_stage1": 3, "forward_stage5": 2} if distributed.cp_size > 1 else {"forward_stage1": 2, "forward_stage5": 1}  # fmt: skip

    for grad in (False, True):
        for S in LENGTHS:
            step(S, grad)
    seen = {name: cache_entries(stage) for name, stage in stages.items()}
    for name, count in seen.items():
        bound = variants[name] * (len(LENGTHS) + 2)
        if count > bound:
            raise AssertionError(f"{name}: {count} graphs over {len(LENGTHS)} lengths, expected <= {bound}")  # fmt: skip

    for grad in (True, False):
        for S in reversed(LENGTHS):
            step(S, grad)
    again = {name: cache_entries(stage) for name, stage in stages.items()}
    if again != seen:
        raise AssertionError(f"lengths seen before recompiled: {seen} -> {again}")


@pytest.mark.parametrize("cp_size", [1, 2])
def test_bounded_recompiles(cp_size: int) -> None:
    cfg = DistributedCfg()
    cfg.context_parallel_size = cp_size
    launch(cfg, verify_bounded_recompiles)
