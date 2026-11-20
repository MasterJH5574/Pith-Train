"""
DenseZero3DualPipeV, the per-layer sharded DualPipeV that setup_model uses for dense models, on the
dense Qwen3.5 config (cut to 8 layers by tests/test_dualpipev.py). It runs test_dualpipev's checks
against the single-device reference (loss, gradients, canonical export, forward-only step), and
counts FSDP's collectives per unit and micro-batch: two all-gathers and one reduce-scatter in
training, one all-gather forward-only, in the same order on every rank, with at most three units
unsharded at once.

usage: torchrun --nproc-per-node=8 tests/test_dense_zero3.py --pp-size 2 --cp-size 2
"""

import argparse
from types import SimpleNamespace

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from pithtrain.contexts import distributed
from pithtrain.modules.distributed import DistributedCfg, setup_distributed
from pithtrain.pipeline.dense_zero3 import DenseZero3DualPipeV, ShardedUnit
from tests.test_dualpipev import main

MODEL = "examples/pretrain_lm/qwen3.5-27b/config.json"


class Zero3Accounting:
    """
    Counts FSDP's own all-gathers and reduce-scatters during a DenseZero3DualPipeV step (so a
    gather or reduce FSDP issues behind the units' back is caught), the most units unsharded at
    once, and the units' gathers that were not prefetched or that FSDP's hooks had to do.
    """

    def __init__(self, model: DenseZero3DualPipeV) -> None:
        import torch.distributed.fsdp._fully_shard._fsdp_param_group as group_module

        self.model, self.units = model, model.units()
        self.events: list = []
        names = {id(group): unit.name for unit in self.units for group in unit.groups}
        all_gather, reduce = group_module.foreach_all_gather, group_module.foreach_reduce

        def counted_all_gather(fsdp_params, *args, **kwargs):
            self.events.append(("all_gather", self.owner(fsdp_params, names)))
            return all_gather(fsdp_params, *args, **kwargs)

        def counted_reduce(fsdp_params, *args, **kwargs):
            self.events.append(("reduce_scatter", self.owner(fsdp_params, names)))
            return reduce(fsdp_params, *args, **kwargs)

        group_module.foreach_all_gather = counted_all_gather
        group_module.foreach_reduce = counted_reduce
        self.peak = 0
        wait = ShardedUnit.wait

        def tracked_wait(unit):
            wait(unit)
            self.peak = max(self.peak, sum(u.is_unsharded() for u in self.units))

        ShardedUnit.wait = tracked_wait  # test process only

    def owner(self, fsdp_params, names) -> str:
        for unit in self.units:
            for group in unit.groups:
                if group.fsdp_params[0] is fsdp_params[0]:
                    return unit.name
        return "unknown"

    def start(self) -> None:
        self.events.clear()
        self.peak = 0
        for unit in self.units:
            unit.stats.clear()

    def check(self, num_microbatches: int, training: bool) -> None:
        per_unit = {}
        for kind, name in self.events:
            per_unit.setdefault(name, {"all_gather": 0, "reduce_scatter": 0})[kind] += 1
        # A unit is gathered for each forward and each backward of a micro-batch, except that one
        # a forward chunk ends with (a stage's last layer, or its epilog) stays gathered when the
        # next chunk is a backward that starts with it.
        tails = {id(u) for units in self.model.stage_units for u in (units.layers[-1], *units.epilog)}  # fmt: skip
        reduces = num_microbatches if training else 0
        for unit in self.units:
            got = per_unit.pop(unit.name, {"all_gather": 0, "reduce_scatter": 0})
            if not training:
                gathers = [num_microbatches]
            elif id(unit) in tails:
                gathers = range(num_microbatches, 2 * num_microbatches + 1)
            else:
                gathers = [2 * num_microbatches]
            assert got["all_gather"] in gathers and got["reduce_scatter"] == reduces, (distributed.rank, unit.name, got)  # fmt: skip
            assert unit.stats["hook_unshard"] == 0, (distributed.rank, unit.name, dict(unit.stats))
        assert not per_unit, (distributed.rank, per_unit)  # no collective outside the units
        # Every rank of the FSDP group issues the same collectives in the same order.
        gathered = [None] * torch.distributed.get_world_size(self.model.fsdp_group)
        torch.distributed.all_gather_object(gathered, self.events, group=self.model.fsdp_group)
        assert all(events == gathered[0] for events in gathered), (
            "collective order differs across ranks"
        )
        # Forward: the current unit and the prefetched next one; the epilog gathers norm and
        # lm_head together, so with the next prefetch at most three.
        assert self.peak <= 3, (distributed.rank, self.peak)
        late = sum(unit.stats["late_gather"] for unit in self.units)
        if distributed.rank == 0:
            print(
                f"[INFO] zero3 {'training' if training else 'forward-only'} step: {len(self.events)} collectives, "
                f"peak {self.peak} units unsharded, {late} gathers not prefetched",
                flush=True,
            )


class AccountedDenseZero3DualPipeV(DenseZero3DualPipeV):
    """
    DenseZero3DualPipeV whose every step is checked by Zero3Accounting.
    """

    def __init__(self, modules) -> None:
        super().__init__(modules)
        self.accounting = Zero3Accounting(self)

    def step(self, microbatches, *args, **kwargs):
        training = torch.is_grad_enabled()
        self.accounting.start()
        result = super().step(microbatches, *args, **kwargs)
        self.accounting.check(len(microbatches), training=training)
        return result


@record
def entry() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pp-size", type=int, required=True)
    parser.add_argument("--cp-size", type=int, default=1)
    parsed = parser.parse_args()

    cfg = SimpleNamespace()
    cfg.distributed = DistributedCfg()
    cfg.distributed.pipeline_parallel_size = parsed.pp_size
    cfg.distributed.expert_parallel_size = 1
    cfg.distributed.context_parallel_size = parsed.cp_size

    setup_distributed(cfg)
    main(MODEL, make_engine=AccountedDenseZero3DualPipeV)


if __name__ == "__main__":
    entry()
