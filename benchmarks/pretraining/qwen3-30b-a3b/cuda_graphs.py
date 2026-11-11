"""Seeded correctness or force-balanced throughput fixture with per-rank CUDA memory evidence."""

import argparse
import json
import os
import socket
import time
from functools import partial
from pathlib import Path

import torch

from pithtrain.contexts import distributed
from pithtrain.modules.training import (
    make_adamw_optimizer,
    make_constant_scheduler,
    make_wsd_scheduler,
)
from pithtrain.tasks import pretrain_lm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--mbs", type=int, default=1)
    parser.add_argument("--gbs", type=int, default=64)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--performance", action="store_true")
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--dataset", type=Path, default=Path("workspace/datasets/dclm-baseline/toktxt/qwen3")
    )
    parser.add_argument("--output", type=Path, default=Path("workspace/cuda-graphs/evidence"))
    args = parser.parse_args()

    cfg = pretrain_lm.PretrainLMCfg()
    cfg.dataset = args.dataset
    mesh = cfg.distributed
    mesh.pipeline_parallel_size = args.pp
    mesh.expert_parallel_size = args.ep
    mesh.context_parallel_size = args.cp
    training = cfg.training
    training.model = Path("examples/pretrain_lm/qwen3-30b-a3b")
    training.optimizer = make_adamw_optimizer
    training.moe_load_balance_type, training.moe_load_balance_coef = "global-batch", 1e-3
    training.micro_batch_size, training.global_batch_size = args.mbs, args.gbs
    training.sequence_length = args.seq
    training.fp8, training.benchmark, training.seed = args.fp8, args.performance, 1234
    training.max_steps = args.steps or (12 if args.performance else 32)
    # The same driver can measure the base revision, where this option does not exist.
    if args.graphs:
        training.cuda_graphs = True
    if args.performance:
        training.scheduler, training.lr = make_constant_scheduler, 1e-6
    else:
        training.scheduler = partial(
            make_wsd_scheduler, start_lr=1e-6, warmup_ratio=1.0, decay_ratio=0.0
        )
        training.lr = 1e-5

    def record(start, *, step=None, phase=None):
        stats = dict(
            rank=distributed.rank,
            world_size=torch.distributed.get_world_size(),
            hostname=socket.gethostname(),
            slurm_job=os.environ.get("SLURM_JOB_ID"),
            slurm_step=os.environ.get("SLURM_STEP_ID"),
            seconds=time.perf_counter() - start,
            peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),
            allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(),
            capacity=torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
            free_after_step=torch.cuda.mem_get_info()[0],
        )
        if step is None:
            stats["phase"] = phase
            suffix = f"startup-rank{distributed.rank}"
        else:
            stats["step"] = step
            suffix = f"rank{distributed.rank}"
            if args.graphs:
                from pithtrain.pipeline.cuda_graphs import graph_stats

                stats["graphs"] = graph_stats()
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / f"{args.tag}-{suffix}.jsonl").open("a") as output:
            output.write(json.dumps(stats) + "\n")

    original_setup = pretrain_lm.setup_training
    original_tune = pretrain_lm.tune_kernels
    original_step = pretrain_lm.train_step

    def measured_setup(*positional, **keywords):
        start = time.perf_counter()
        result = original_setup(*positional, **keywords)
        torch.cuda.synchronize()
        record(start, phase="setup_training")
        return result

    def measured_tune():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        original_tune()
        torch.cuda.synchronize()
        record(start, phase="tuning")

    def measured_step(cfg, dataset, step):
        torch.cuda.synchronize()
        start = time.perf_counter()
        original_step(cfg, dataset, step)
        torch.cuda.synchronize()
        record(start, step=step + 1)

    pretrain_lm.setup_training = measured_setup
    pretrain_lm.tune_kernels = measured_tune
    pretrain_lm.train_step = measured_step
    pretrain_lm.launch(cfg)


if __name__ == "__main__":
    main()
