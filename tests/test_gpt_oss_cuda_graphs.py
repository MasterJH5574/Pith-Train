"""GPT-OSS static attention replay with alternating windows and live backward state."""

import copy

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("replay", [False, True], ids=["routing", "replay-global-lb"])
def test_attention_replay_with_sinks_biases_and_updates(monkeypatch, fp8, replay):
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

    from pithtrain.contexts import distributed, training
    from pithtrain.models.gpt_oss import GptOssDecoderLayer, GptOssRotaryEmbedding
    from pithtrain.modules.load_balance import GlobalBatchLoadBalanceLoss
    from pithtrain.operators.linear import BF16Linear, FP8Linear
    from pithtrain.pipeline.cuda_graphs import (
        MAX_GRAPHS,
        capture_pending,
        graph_stats,
        live_graphs,
        release_graphs,
    )

    release_graphs()
    torch._dynamo.reset()
    monkeypatch.setattr(training, "fp8", fp8, raising=False)
    monkeypatch.setattr(training, "Linear", FP8Linear if fp8 else BF16Linear, raising=False)
    monkeypatch.setattr(distributed, "ep_size", 1, raising=False)
    config = GptOssConfig(
        hidden_size=256,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=128,
        max_position_embeddings=256,
        rope_scaling={"rope_type": "default", "rope_theta": 150000},
        attention_bias=True,
    )
    torch.manual_seed(1234)
    layers = torch.nn.ModuleList([GptOssDecoderLayer(config, i) for i in range(2)]).to(
        device="cuda", dtype=torch.bfloat16
    )
    # Identical initial weights isolate the effect of the attention window.
    layers[1].load_state_dict(layers[0].state_dict())
    # Routers use empty initialization until training setup fills them.
    for layer in layers:
        torch.nn.init.normal_(layer.mlp.router.weight, std=0.02)
        if replay:
            loss_fn = GlobalBatchLoadBalanceLoss(0.01)
            loss_fn.init_buffers(config.num_local_experts, torch.device("cuda"))
            layer.mlp.router.load_balance_loss_fn = loss_fn
    initial = copy.deepcopy(layers.state_dict())
    with torch.device("cuda"):
        rotary = GptOssRotaryEmbedding(config)(256)
    rotary = tuple(x.to(torch.bfloat16).unsqueeze(0) for x in rotary)
    inputs = [torch.randn(1, 256, 256, device="cuda", dtype=torch.bfloat16) for _ in range(12)]
    tangents = [torch.randn_like(inputs[0]) for _ in range(2)]
    route_tangent = torch.randn(256, 2, device="cuda", dtype=torch.float32)
    results = []
    try:
        for enabled in (False, True):
            monkeypatch.setattr(training, "cuda_graphs", enabled)
            layers.load_state_dict(initial)
            params = {
                name: p for name, p in layers.named_parameters() if ".mlp.experts." not in name
            }
            # Random output tangents are summed over tokens and three micro-batches.
            optimizer = torch.optim.SGD(params.values(), lr=1e-4)
            observed, counts = [], []
            # Stats sum a weak registry. Keep released wrappers alive during this measurement
            # so garbage collection cannot subtract an earlier precision's replay counters.
            prior_graphs = live_graphs()
            before = graph_stats()
            for step in range(4):
                if replay:
                    lb_state = [
                        (loss.count_buffer.clone(), loss.total_tokens.clone())
                        for layer in layers
                        for loss in (layer.mlp.router.load_balance_loss_fn,)
                    ]
                capture_pending()
                optimizer.zero_grad(set_to_none=True)
                if replay:
                    for layer, (count, total) in zip(layers, lb_state):
                        loss_fn = layer.mlp.router.load_balance_loss_fn
                        torch.testing.assert_close(loss_fn.count_buffer, count, rtol=0, atol=0)
                        torch.testing.assert_close(loss_fn.total_tokens, total, rtol=0, atol=0)
                        loss_fn.reset()
                xs = [x.clone().requires_grad_() for x in inputs[step * 3 : step * 3 + 3]]
                indices = [
                    (
                        torch.arange(256, device="cuda")[:, None] // 128
                        + step
                        + i
                        + torch.arange(2, device="cuda")
                    )
                    % config.num_local_experts
                    for i in range(3)
                ]
                # Six live forwards alternate full/sliding on the same compiled entry point.
                outputs = [
                    layer.forward_stage1_compute(x, rotary, replay_idx=idx if replay else None)
                    for x, idx in zip(xs, indices)
                    for layer in layers
                ]
                for out in outputs:
                    assert all(torch.isfinite(t).all() for t in out if t is not None), (
                        enabled,
                        step,
                    )
                if replay:
                    for i, out in enumerate(outputs):
                        torch.testing.assert_close(out[2], indices[i // 2], rtol=0, atol=0)
                    expected_counts = torch.bincount(
                        torch.cat(indices).flatten(), minlength=config.num_local_experts
                    )
                    for layer in layers:
                        loss_fn = layer.mlp.router.load_balance_loss_fn
                        torch.testing.assert_close(loss_fn.count_buffer, expected_counts.float())
                        assert loss_fn.total_tokens.item() == 3 * 256 * 2
                if step == 0:
                    assert not torch.equal(outputs[0][0][:, 128:], outputs[1][0][:, 128:])
                snapshots = [
                    tuple(t.detach().clone() if t is not None else None for t in out)
                    for out in outputs
                ]
                for out in reversed(outputs):
                    torch.autograd.backward((out[0], out[1], out[3]), (*tangents, route_tangent))
                # A later replay must not overwrite an earlier live output or saved activation.
                for out, saved in zip(outputs, snapshots):
                    for actual, expected in zip(out, saved):
                        if actual is not None:
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                observed.extend(t for out in snapshots for t in out if t is not None)
                observed.extend(x.grad.clone() for x in xs)
                for name, p in params.items():
                    assert p.grad is not None and torch.isfinite(p.grad).all(), name
                    # Sinks and every attention bias must contribute, not pass by being skipped.
                    if name.endswith(("sinks", "bias", "router.weight")):
                        assert torch.count_nonzero(p.grad) > 0, name
                    observed.append(p.grad.clone())
                optimizer.step()
                observed.extend(p.detach().clone() for p in params.values())
                counts.append(graph_stats()["instances"])
            if enabled:
                assert 4 <= counts[1] <= MAX_GRAPHS, counts
                assert counts[1:] == [counts[1]] * 3, counts
                stats = graph_stats()
                assert stats["forward_replays"] - before["forward_replays"] == 18, stats
                assert stats["backward_replays"] - before["backward_replays"] == 18, stats
            del prior_graphs
            results.append(observed)
        for reference, captured in zip(*results):
            if not reference.is_floating_point():
                torch.testing.assert_close(captured, reference, rtol=0, atol=0)
                continue
            # FA4 backward reductions are not bit reproducible. Compare every tensor, including
            # sink/bias gradients, with normalized squared error rather than a tiny-grad skip.
            numerator = (captured.float() - reference.float()).square().sum()
            denominator = captured.float().square().sum() + reference.float().square().sum()
            assert (numerator / denominator.clamp_min(1e-20)).item() < 1e-3
    finally:
        release_graphs()
        torch._dynamo.reset()


@pytest.mark.parametrize("recompute", [False, True])
def test_global_load_balance_keeps_each_forward_fraction(monkeypatch, recompute):
    from pithtrain.contexts import training
    from pithtrain.modules.load_balance import GlobalBatchLoadBalanceLoss
    from pithtrain.modules.recompute import norm_recompute

    monkeypatch.setattr(training, "fp8", False, raising=False)
    loss_fn = GlobalBatchLoadBalanceLoss(0.01)
    loss_fn.init_buffers(4, torch.device("cuda"))

    def loss(scores, idx):
        return loss_fn(scores, idx, 4, 2)

    def compute(scores, idx):
        # Qwen encloses the router in this additional selective-checkpoint context.
        return norm_recompute(loss, 4, scores, idx) if recompute else loss(scores, idx)

    compiled = torch.compile(compute, fullgraph=True)
    counts = torch.zeros(4, device="cuda")
    pending = []
    for i in range(3):
        scores = torch.randn(256, 4, device="cuda", requires_grad=True)
        idx = (
            torch.arange(256, device="cuda")[:, None] // 128 + i + torch.arange(2, device="cuda")
        ) % 4
        counts += torch.bincount(idx.flatten(), minlength=4)
        fraction = counts / ((i + 1) * 256 * 2)
        value = compiled(scores, idx)
        expected = 0.04 * torch.dot(fraction, scores.detach().mean(0))
        torch.testing.assert_close(value, expected)
        pending.append((value, scores, (0.04 / 256) * fraction.clone()))
    # Even reset before backward cannot change the coefficients from those forwards.
    loss_fn.reset()
    for value, scores, expected_grad in reversed(pending):
        value.backward()
        torch.testing.assert_close(scores.grad, expected_grad.expand_as(scores))
    torch._dynamo.reset()


def verify_distributed_router_replay(with_hook, model="gpt", executor=False):
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

    from pithtrain.contexts import distributed, training
    from pithtrain.models.gpt_oss import GptOssDecoderLayer, GptOssRotaryEmbedding
    from pithtrain.modules.load_balance import GlobalBatchLoadBalanceLoss, MoELoadBalanceLossTracker
    from pithtrain.operators.grouped_linear import GroupedLinear
    from pithtrain.operators.linear import BF16Linear
    from pithtrain.pipeline import cuda_graphs as cg
    from pithtrain.pipeline.execution import run_backward

    training.fp8, training.Linear = False, BF16Linear
    training.GroupedLinear = GroupedLinear
    rank = torch.distributed.get_rank()
    config = GptOssConfig(
        hidden_size=256,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        layer_types=["sliding_attention"],
        sliding_window=128,
        max_position_embeddings=256,
        rope_scaling={"rope_type": "default", "rope_theta": 150000},
        attention_bias=True,
    )
    torch.manual_seed(1234)
    layer_type, rotary_type = GptOssDecoderLayer, GptOssRotaryEmbedding
    if model == "qwen":
        from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

        from pithtrain.models.qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding

        config = Qwen3MoeConfig(
            hidden_size=256,
            intermediate_size=256,
            moe_intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            num_experts=4,
            num_experts_per_tok=2,
            num_hidden_layers=1,
            max_position_embeddings=256,
            rope_scaling={"rope_type": "default", "rope_theta": 150000},
            attention_bias=True,
        )
        layer_type, rotary_type = Qwen3MoeDecoderLayer, Qwen3MoeRotaryEmbedding
    layer = layer_type(config, 0).to(device="cuda", dtype=torch.bfloat16)
    router = layer.mlp.gate if model == "qwen" else layer.mlp.router
    torch.nn.init.normal_(router.weight, std=0.02)
    initial = copy.deepcopy(layer.state_dict())
    loss_fn = GlobalBatchLoadBalanceLoss(0.01, distributed.ep_group)
    loss_fn.init_buffers(4, torch.device("cuda"))
    router.load_balance_loss_fn = loss_fn
    with torch.device("cuda"):
        rotary = rotary_type(config)(256)
    rotary = tuple(x.to(torch.bfloat16).unsqueeze(0) for x in rotary)
    calls, routes, retained = [], [], []

    def replay(num_tokens, top_k):
        calls.append((training.current_microbatch, num_tokens, top_k))
        return routes[training.current_microbatch]

    def retain_router(module, args, outputs):
        retained.append((args[0], *outputs))

    router.router_replay = replay
    hook = router.register_forward_hook(retain_router) if with_hook else None
    params = {n: p for n, p in layer.named_parameters() if ".experts." not in n}
    results = []
    try:
        for enabled in (False, True):
            training.cuda_graphs = enabled
            layer.load_state_dict(initial)
            optimizer = torch.optim.SGD(params.values(), lr=1e-4)
            observed, counts, snapshots = [], [], []
            torch.manual_seed(5678 + rank)
            for step in range(5):
                before_calls = len(calls)
                before_hooks = len(retained)
                state = (loss_fn.count_buffer.clone(), loss_fn.total_tokens.clone())
                cg.capture_pending()
                assert len(calls) == before_calls
                assert len(retained) == before_hooks
                torch.testing.assert_close(loss_fn.count_buffer, state[0], rtol=0, atol=0)
                torch.testing.assert_close(loss_fn.total_tokens, state[1], rtol=0, atol=0)
                loss_fn.reset()
                optimizer.zero_grad(set_to_none=True)
                routes[:] = [
                    (
                        (
                            torch.arange(256, device="cuda")[:, None] // 128
                            + step
                            + i
                            + rank
                            + torch.arange(2, device="cuda")
                        )
                        % 4
                    ).to(torch.int32 if step % 2 else torch.int64)
                    for i in range(3)
                ]
                expected_counts = torch.zeros(4, device="cuda")
                xs, outputs = [], []
                with cg.graph_step([layer], max_leases=3, executor_outputs=executor):
                    for i in range(3):
                        training.current_microbatch = i
                        x = torch.randn(
                            1, 256, 256, device="cuda", dtype=torch.bfloat16
                        ).requires_grad_()
                        out = (
                            cg.pipeline_call(layer, layer.forward_stage1, x, rotary)
                            if executor
                            else layer.forward_stage1(x, rotary)
                        )
                        if with_hook:
                            assert len(retained) == before_hooks + i + 1
                            torch.testing.assert_close(retained[-1][1], routes[i], rtol=0, atol=0)
                        xs.append(x)
                        outputs.append((out[0], out[1], out[2].topk_weight))
                        if executor:
                            snapshots.append(((out[1],), (out[1].detach().clone(),)))
                            assert out[0].untyped_storage()._cdata not in cg.owners[layer].saved
                        expected_counts += torch.bincount(routes[i].long().flatten(), minlength=4)
                        global_counts = expected_counts.clone()
                        torch.distributed.all_reduce(global_counts, group=distributed.ep_group)
                        torch.testing.assert_close(
                            loss_fn.count_buffer, global_counts, rtol=0, atol=0
                        )
                        assert loss_fn.total_tokens.item() == (i + 1) * 256 * 2 * 2
                        if with_hook:
                            snapshots.append(
                                (retained[-1], tuple(t.detach().clone() for t in retained[-1]))
                            )
                        tracked = MoELoadBalanceLossTracker.losses[-1]
                        snapshots.append(((tracked,), (tracked.clone(),)))
                    assert calls[-3:] == [(i, 256, 2) for i in range(3)]
                    for out in reversed(outputs):
                        run_backward(
                            list(out), [torch.ones_like(t) for t in out], graph_owner=layer
                        )
                    assert len(calls) == before_calls + 3
                    assert len(retained) == before_hooks + (3 if with_hook else 0)
                for actual, saved in snapshots:
                    for a, b in zip(actual, saved):
                        torch.testing.assert_close(a, b, rtol=0, atol=0)
                observed.extend(t.detach().clone() for out in outputs for t in out)
                observed.extend(x.grad.clone() for x in xs)
                observed.extend(t.clone() for t in MoELoadBalanceLossTracker.losses)
                MoELoadBalanceLossTracker.losses.clear()
                for name, p in params.items():
                    assert p.grad is not None and torch.isfinite(p.grad).all(), name
                    if name.endswith(("router.weight", "router.bias", "gate.weight", "sinks")):
                        assert torch.count_nonzero(p.grad), name
                    observed.append(p.grad.clone())
                optimizer.step()
                counts.append(cg.graph_stats()["instances"])
            if enabled:
                stats = cg.graph_stats()
                print(f"DISTRIBUTED_ROUTER_GRAPH_RANK_{rank}={stats}", flush=True)
            results.append(observed)
        for a, b in zip(*results):
            error = (a.float() - b.float()).square().sum()
            scale = a.float().square().sum() + b.float().square().sum()
            assert (error / scale.clamp_min(1e-20)).item() < 1e-3
        print(f"DISTRIBUTED_ROUTER_NUMERICAL_PASS_RANK_{rank} model={model}", flush=True)
        assert stats["owned_instances"] > 0
        if with_hook:
            # Appending caller-visible tensors changes Dynamo's list-length guard on
            # every call. Each compiled parent is used once; the bounded variant set
            # must fall back safely, including all retained outputs checked above.
            assert len(cg.owners[layer].variants) == cg.MAX_GRAPHS
            assert stats["instances"] == cg.MAX_GRAPHS
            assert stats["forward_replays"] == stats["backward_replays"] == 0, stats
            assert stats["fallbacks"] == 30, stats
        else:
            assert stats["direct_saved_replays"] > 0
            assert stats["forward_replays"] == stats["backward_replays"] == 9, stats
        assert stats["live_saved_storages"] == 0
        assert bool(stats["pipeline_output_replays"]) == (executor and not with_hook)
        assert counts[2:] == [counts[2]] * 3
        print(f"DISTRIBUTED_ROUTER_PASS_RANK_{rank}", flush=True)
    finally:
        if hook is not None:
            hook.remove()
        cg.release_graphs()
        torch._dynamo.reset()


@pytest.mark.parametrize("with_hook", [False, True], ids=["owned", "retaining-hook"])
@pytest.mark.parametrize("executor", [False, True], ids=["direct", "executor"])
def test_distributed_router_replay(with_hook, executor):
    from pithtrain.modules.distributed import DistributedCfg
    from tests.utilities import launch

    cfg = DistributedCfg()
    cfg.expert_parallel_size = 2
    launch(cfg, verify_distributed_router_replay, with_hook, "gpt", executor)


@pytest.mark.parametrize("executor", [False, True], ids=["direct", "executor"])
def test_qwen_distributed_global_load_balance_replay(executor):
    from pithtrain.modules.distributed import DistributedCfg
    from tests.utilities import launch

    cfg = DistributedCfg()
    cfg.expert_parallel_size = 2
    launch(cfg, verify_distributed_router_replay, False, "qwen", executor)
