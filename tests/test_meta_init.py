"""
setup_training(weights_from_checkpoint=True) builds the model with its parameters on meta, shards it,
materializes it uninitialized and skips init_weights; load_checkpoint must then fill every parameter.
This checks that it ends in exactly the state of the default path (CPU build, init_weights, FSDP)
after the same load, on a 4-layer dense Qwen3.5 cut from examples/pretrain_lm/qwen3.5-27b:

- every model state tensor (full and local shard, placements), every buffer (the rotary caches) and,
  for a checkpoint with optimizer state, every optimizer state tensor are bit-identical;
- a checkpoint missing a model key fails the load, as does a key that does not map back to the model;
- save_checkpoint and iter_canonical_parameters refuse the uninitialized weights before the load.

CPU (two gloo ranks, no GPU; DenseZero3DualPipeV, which needs CUDA streams, is replaced by a holder):
    python -m pytest -q tests/test_meta_init.py
GPU (the real setup_training and DenseZero3DualPipeV on NCCL):
    torchrun --nproc-per-node=8 tests/test_meta_init.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

CONFIG = Path(__file__).resolve().parent.parent / "examples/pretrain_lm/qwen3.5-27b/config.json"
NUM_LAYERS = 4


def tiny_config(directory: Path) -> Path:
    config = json.loads(CONFIG.read_text())
    config.update(
        hidden_size=64,
        intermediate_size=96,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        vocab_size=320,
        num_hidden_layers=NUM_LAYERS,
        layer_types=config["layer_types"][:NUM_LAYERS],
        bos_token_id=None,
        eos_token_id=None,
    )
    path = directory / "config.json"
    path.write_text(json.dumps(config))
    return path


def training_cfg(model: Path):
    from pithtrain.modules.distributed import DistributedCfg
    from pithtrain.modules.training import (
        TrainingCfg,
        make_adamw_optimizer,
        make_constant_scheduler,
    )

    cfg = SimpleNamespace(training=TrainingCfg(), distributed=DistributedCfg())
    t = cfg.training
    t.model, t.sequence_length, t.lr, t.max_steps = model, 64, 1e-3, 10
    t.micro_batch_size, t.global_batch_size = 1, 2
    t.optimizer, t.scheduler = make_adamw_optimizer, make_constant_scheduler
    t.fp8 = False
    return cfg


def setup_cpu_mesh(rank: int, world_size: int) -> None:
    """setup_device_mesh on CPU: pp 1, cp 1, ep 1, dp over the world."""
    from pithtrain.contexts import distributed

    init = dist.init_device_mesh
    attn_mesh = init("cpu", (1, world_size, 1), mesh_dim_names=("pp", "dp", "cp"))
    expt_mesh = init("cpu", (1, world_size, 1), mesh_dim_names=("pp", "dp", "ep"))
    distributed.rank, distributed.world_size = rank, world_size
    distributed.device = torch.device("cpu")
    distributed.attn_mesh, distributed.expt_mesh = attn_mesh, expt_mesh
    distributed.pp_size, distributed.pp_rank, distributed.pp_group = 1, 0, attn_mesh.get_group("pp")
    distributed.cp_size, distributed.cp_rank, distributed.cp_group = 1, 0, attn_mesh.get_group("cp")
    distributed.ep_size, distributed.ep_rank, distributed.ep_group = 1, 0, expt_mesh.get_group("ep")
    distributed.dp_size, distributed.dp_rank = world_size, attn_mesh.get_local_rank("dp")


def snapshot():
    """Every model state tensor, buffer and optimizer state tensor, gathered, on the CPU."""
    from torch.distributed.tensor import DTensor

    from pithtrain.contexts import training

    def full(t):
        if isinstance(t, DTensor):
            return (
                "dtensor",
                tuple(t.placements),
                tuple(t.to_local().shape),
                t.full_tensor().cpu(),
            )
        return ("tensor", t.cpu().clone())

    model = training.model
    state = {k: full(v.detach()) for k, v in model.state_dict().items()}
    buffers = {k: full(v) for k, v in model.named_buffers()}
    names = {p: n for n, p in model.named_parameters()}
    optim = {}
    for opt in training.optimizers:
        for p, entry in opt.state.items():
            for k, v in entry.items():
                optim[names[p], k] = full(v) if isinstance(v, torch.Tensor) else v
    return state, buffers, optim


def assert_identical(a, b, what: str) -> None:
    assert a.keys() == b.keys(), "%s keys differ: %s" % (what, sorted(a.keys() ^ b.keys())[:5])
    for key in a:
        x, y = a[key], b[key]
        assert type(x) is type(y) and x[:-1] == y[:-1], "%s %s layout: %s vs %s" % (
            what,
            key,
            x[:-1],
            y[:-1],
        )
        if isinstance(x[-1], torch.Tensor):
            assert x[-1].dtype == y[-1].dtype and torch.equal(x[-1], y[-1]), "%s %s differs" % (
                what,
                key,
            )
        else:
            assert x[-1] == y[-1], "%s %s differs" % (what, key)


def build(cfg, weights_from_checkpoint: bool) -> None:
    from pithtrain.contexts import training
    from pithtrain.modules.training import setup_training

    training.model = training.optimizers = training.schedulers = None
    setup_training(cfg, weights_from_checkpoint=weights_from_checkpoint)
    assert training.weights_pending_load is weights_from_checkpoint


def train_one_step() -> None:
    """Move every parameter off its init and give the optimizer state, deterministically."""
    from pithtrain.contexts import training

    for i, p in enumerate(training.model.parameters()):
        local = p.to_local() if hasattr(p, "to_local") else p
        g = torch.full_like(p, 0.01 * (i + 1))
        p.grad = g
        with torch.no_grad():
            local.add_(0.5 + 0.001 * i)
    for opt in training.optimizers:
        opt.step()
        opt.zero_grad(set_to_none=True)


def save(root: Path, step: int, model_only: bool, drop: str | None = None) -> None:
    """A checkpoint as save_checkpoint writes it, or as an HF import writes it with model_only."""
    import torch.distributed.checkpoint as dcp

    from pithtrain.contexts import training
    from pithtrain.modules.checkpoint import CheckpointState

    state = CheckpointState(training.model, training.optimizers, training.schedulers, model_only)
    state_dict = state.state_dict()
    if drop is not None:
        del state_dict["model"][drop]
    dcp.save({"app": state_dict}, checkpoint_id=Path(root, "torch-dcp", "%08d" % step))


def check(root: Path, device: str) -> None:
    from pithtrain.contexts import training
    from pithtrain.modules import checkpoint
    from pithtrain.modules.checkpoint import (
        iter_canonical_parameters,
        load_checkpoint,
        save_checkpoint,
    )

    rank = dist.get_rank()
    if rank == 0:
        tiny_config(root)
    dist.barrier()
    cfg = training_cfg(root / "config.json")

    # The checkpoints: a model-only import (step 0) and a resume with optimizer state (step 1).
    build(cfg, weights_from_checkpoint=False)
    train_one_step()
    save(root, 0, model_only=True)
    save(root, 1, model_only=False)
    canonical = sorted(checkpoint.to_canonical_model(training.model.state_dict(), training.model))
    save(root, 2, model_only=True, drop=canonical[len(canonical) // 2])

    for step in (0, 1):
        build(cfg, weights_from_checkpoint=False)
        load_checkpoint(root, step)
        reference = snapshot()

        build(cfg, weights_from_checkpoint=True)
        for refused in (
            lambda: save_checkpoint(root, 9),
            lambda: next(iter_canonical_parameters()),
        ):
            try:
                refused()
            except AssertionError as error:
                assert "uninitialized" in str(error)
            else:
                raise AssertionError("used the uninitialized weights")
        load_checkpoint(root, step)
        assert not training.weights_pending_load
        result = snapshot()
        for what, a, b in zip(("model state", "buffer", "optimizer state"), reference, result):
            assert_identical(a, b, what)
        if rank == 0:
            print(
                "[%s] step %d: %d model tensors, %d buffers, %d optimizer entries identical"
                % (device, step, *map(len, result)),
                flush=True,
            )

    # A checkpoint missing a key: DCP refuses it.
    build(cfg, weights_from_checkpoint=True)
    try:
        load_checkpoint(root, 2)
    except BaseException as error:  # DCP's CheckpointException, on every rank, is a BaseException
        assert "Missing key" in str(error), error
    else:
        raise AssertionError("loaded a checkpoint missing a key")
    assert training.weights_pending_load

    # A key that does not map back to the model: the strict check refuses it.
    to_localized = checkpoint.to_localized_model

    def drop_one(canonical_state, model):
        result = to_localized(canonical_state, model)
        result.pop(sorted(result)[0])
        return result

    checkpoint.to_localized_model = drop_one
    try:
        load_checkpoint(root, 0)
    except Exception as error:
        assert "Strict checkpoint load" in str(error), error
    else:
        raise AssertionError("loaded with a model key left unfilled")
    finally:
        checkpoint.to_localized_model = to_localized
    if rank == 0:
        print("[%s] strict load refuses a missing key and an unmapped key" % device, flush=True)


def stub_gpu_imports() -> None:
    """
    pithtrain.kernels probes the GPU and loads GPU kernel libraries at import. No kernel runs in
    this test, so on a CPU-only host claim an SM100 and stand in for each library that fails to
    load (deep_gemm's extension, for one, needs a newer glibc than a bare host may have).
    """
    import importlib
    from types import ModuleType
    from unittest import mock

    arch = ModuleType("pithtrain.kernels.arch")
    arch.ARCH_MAJOR, arch.SM_COUNT, arch.NATIVE_E8M0 = 10, 148, True
    sys.modules["pithtrain.kernels.arch"] = arch
    for _ in range(50):
        try:
            importlib.import_module("pithtrain.modules.training")
            return
        except ImportError as error:
            # The top-level package of the library that failed, e.g. deep_gemm for deep_gemm._C:
            # the innermost installed package in the traceback, else the module the error names.
            import traceback

            frames = [f.filename for f in traceback.extract_tb(error.__traceback__)]
            installed = [f.split("site-packages/", 1)[1] for f in frames if "site-packages/" in f]
            failed = installed[-1].split("/")[0] if installed else (error.name or "").split(".")[0]
            if not failed or failed in ("pithtrain", "torch") or failed in sys.modules and isinstance(sys.modules[failed], mock.MagicMock):  # fmt: skip
                raise
        for name in [n for n in sys.modules if n == failed or n.startswith(failed + ".") or n.startswith("pithtrain")]:  # fmt: skip
            del sys.modules[name]
        sys.modules[failed] = mock.MagicMock(name=failed)
        sys.modules["pithtrain.kernels.arch"] = arch
        print("stubbed %s for the CPU-only import" % failed, flush=True)
    raise RuntimeError("too many GPU libraries to stub")


def cpu_worker(rank: int, world_size: int, root: str, port: int) -> None:
    if not torch.cuda.is_available():
        stub_gpu_imports()
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from pithtrain.modules import training as training_module
        from pithtrain.modules.logging import setup_stdout

        setup_stdout()
        setup_cpu_mesh(rank, world_size)

        class Holder(nn.Module):  # DenseZero3DualPipeV's parameter layout, without its CUDA streams
            def __init__(self, modules):
                super().__init__()
                self.module = modules

        training_module.DenseZero3DualPipeV = Holder
        check(Path(root), "cpu")
    finally:
        dist.destroy_process_group()


def test_meta_init_matches_default_path_cpu():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with tempfile.TemporaryDirectory() as root:
        mp.start_processes(cpu_worker, args=(2, root, port), nprocs=2, start_method="fork")


def gpu_main() -> None:
    from pithtrain.modules.distributed import setup_device_mesh
    from pithtrain.modules.logging import setup_stdout

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    setup_stdout()
    from pithtrain.modules.distributed import DistributedCfg

    setup_device_mesh(DistributedCfg(), local_rank)
    root = Path(tempfile.gettempdir(), "pithtrain_test_meta_init")
    if dist.get_rank() == 0:
        import shutil

        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
    dist.barrier()
    try:
        check(root, "cuda")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(gpu_main())
