"""Capture-time binding of fixed pointer tables for the raw replay transfer kernel."""

import torch

from pithtrain.kernels import graph_copy


class GraphCopy:
    def __init__(self, table, tensors, *, base=0, into_static=True):
        self.table, self.base, self.into_static = table, base, into_static
        self.word_counts = tuple(t.numel() * t.element_size() // 4 for t in tensors)
        self.fixed = torch.empty(len(tensors), dtype=torch.int64, device=table.device)
        self.bind(tensors)

    def bind(self, tensors):
        # Called outside capture after final graph outputs replace disposable probe outputs.
        self.fixed.copy_(
            torch.tensor(
                [t.data_ptr() for t in tensors], dtype=torch.int64, device=self.table.device
            )
        )

    def __call__(self):
        graph_copy(
            self.table, self.fixed, self.word_counts, base=self.base, into_static=self.into_static
        )
