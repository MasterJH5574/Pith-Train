"""GPT-OSS static attention replay with alternating windows and live backward state."""

import copy

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
def test_attention_replay_with_sinks_biases_and_updates(monkeypatch, fp8):
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

    from pithtrain.contexts import distributed, training
    from pithtrain.models.gpt_oss import GptOssDecoderLayer, GptOssRotaryEmbedding
    from pithtrain.operators.linear import BF16Linear, FP8Linear
    from pithtrain.pipeline.cuda_graphs import (
        MAX_GRAPHS,
        capture_pending,
        graph_stats,
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
    initial = copy.deepcopy(layers.state_dict())
    with torch.device("cuda"):
        rotary = GptOssRotaryEmbedding(config)(256)
    rotary = tuple(x.to(torch.bfloat16).unsqueeze(0) for x in rotary)
    inputs = [torch.randn(1, 256, 256, device="cuda", dtype=torch.bfloat16) for _ in range(12)]
    tangents = [torch.randn_like(inputs[0]) for _ in range(2)]
    results = []
    try:
        for enabled in (False, True):
            monkeypatch.setattr(training, "cuda_graphs", enabled)
            layers.load_state_dict(initial)
            params = {name: p for name, p in layers.named_parameters() if ".mlp." not in name}
            # Random output tangents are summed over tokens and three micro-batches.
            optimizer = torch.optim.SGD(params.values(), lr=1e-4)
            observed, counts = [], []
            before = graph_stats()
            for step in range(4):
                capture_pending()
                optimizer.zero_grad(set_to_none=True)
                xs = [x.clone().requires_grad_() for x in inputs[step * 3 : step * 3 + 3]]
                # Six live forwards alternate full/sliding on the same compiled entry point.
                outputs = [layer.forward_stage1_compute(x, rotary) for x in xs for layer in layers]
                for out in outputs:
                    assert all(torch.isfinite(t).all() for t in out), (enabled, step)
                if step == 0:
                    assert not torch.equal(outputs[0][0][:, 128:], outputs[1][0][:, 128:])
                snapshots = [tuple(t.detach().clone() for t in out) for out in outputs]
                for out in reversed(outputs):
                    torch.autograd.backward(out, tangents)
                # A later replay must not overwrite an earlier live output or saved activation.
                for out, saved in zip(outputs, snapshots):
                    for actual, expected in zip(out, saved):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                observed.extend(t for out in snapshots for t in out)
                observed.extend(x.grad.clone() for x in xs)
                for name, p in params.items():
                    assert p.grad is not None and torch.isfinite(p.grad).all(), name
                    # Sinks and every attention bias must contribute, not pass by being skipped.
                    if name.endswith(("sinks", "bias")):
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
            results.append(observed)
        for reference, captured in zip(*results):
            # FA4 backward reductions are not bit reproducible. Compare every tensor, including
            # sink/bias gradients, with normalized squared error rather than a tiny-grad skip.
            numerator = (captured.float() - reference.float()).square().sum()
            denominator = captured.float().square().sum() + reference.float().square().sum()
            assert (numerator / denominator.clamp_min(1e-20)).item() < 1e-3
    finally:
        release_graphs()
        torch._dynamo.reset()
