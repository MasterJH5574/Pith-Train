"""Pretrain DeepSeek-V2-Lite with pp=2 / ep=4 — short benchmark run."""

from pathlib import Path

from pithtrain.tasks.pretrain_language_model import PretrainLanguageModelCfg, launch

cfg = PretrainLanguageModelCfg()

distributed = cfg.distributed
distributed.context_parallel_size = 1
distributed.pipeline_parallel_size = 2
distributed.expert_parallel_size = 4

training = cfg.training
training.model = Path("examples/pretrain_language_model/deepseek-v2-lite/config.json")
training.optimizer = "Adam"
training.scheduler = "CosineAnnealing"
training.max_lr = 4.2e-4
training.min_lr = 1.0e-5
training.warmup_steps = 128
training.max_steps = 12  # short — first few are warmup; step-time stabilizes after ~5
training.micro_batch_size = 1
training.global_batch_size = 64  # global_batch_size = micro_batch_size * num_chunks
training.sequence_length = 2048
training.dataset = Path("workspace/datasets/dclm-baseline/toktxt/deepseek-v2")
training.moe_load_balance_type = "sequence"
training.moe_load_balance_coef = 3e-3
training.fp8_training = "disabled"
training.save_interval = None  # disable checkpointing for benchmark
training.save_location = Path("workspace/checkpoints/deepseek-v2-lite")
training.nsys_start = 8
training.nsys_stop = 9

if __name__ == "__main__":
    launch(cfg)
