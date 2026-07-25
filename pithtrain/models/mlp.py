"""Dense gated MLP."""

from typing import Optional

from torch import nn

from pithtrain.models.spec import build_module
from pithtrain.operators.silu_mul import silu_mul


class GatedMLP(nn.Module):
    """SwiGLU dense MLP (gate/up/down projections)."""

    def __init__(
        self,
        config,
        hidden_size: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        submodules=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size or config.hidden_size
        self.intermediate_size = intermediate_size or config.intermediate_size

        self.gate_proj = build_module(
            submodules["gate_proj"], self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = build_module(
            submodules["up_proj"], self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = build_module(
            submodules["down_proj"], self.intermediate_size, self.hidden_size, bias=False
        )

    def forward(self, x):
        return self.down_proj(silu_mul(self.gate_proj(x), self.up_proj(x)))
