# CUDA graphs

`TrainingCfg.cuda_graphs` (default off; Qwen3 MoE and GPT-OSS, BF16, dense micro-batches)
replays each layer's compiled stage 1 forward and backward from a CUDA graph.

Graphs save host launch time, so they speed up only steps where the host is the bottleneck: up to
~10% with 2048 or fewer tokens per micro-batch per rank, and a 3-5% slowdown from 4096 up, where
the boundary copies cost more than the host time saved. A larger micro-batch is the better lever:
4096 tokens without graphs beats 2048 with graphs by 20% (see Performance).

## What is captured

The compiled region of stage 1 is captured: input norm, attention (including the CP ring
attention's communication), post-attention norm, router scores and top-k. The capture sits below
AOTAutograd and above Inductor: `compile_stage` compiles the region with a backend whose inner
compile wraps each Inductor forward and backward callable in a `StaticGraph`
(`pithtrain/pipeline/cuda_graphs.py`).

Everything with dynamic shapes or host decisions stays eager: EP dispatch and combine, the
routed experts (stage 3), aggregation (stage 5, which runs eagerly ahead of the captured stage 1
when graphs are on), FSDP, the optimizer.

## Replay contract

Every replay copies the region's read inputs into the graph's scratch inputs, replays, copies
mutated inputs back (e.g. `main_grad`), and copies every output out to ordinary allocator
storage. Nothing a caller or autograd holds points into graph memory, so saved activations follow
the pipeline's real lifetimes and no address has to stay stable across replays, steps or FSDP
reshards. The price is the copy traffic.

A graph is prepared during the first (ordinary) step and captured at the start of the next step,
while the pipeline is drained. A call whose signature, stream or storage layout differs from the
capture runs the ordinary compiled code. At most `MAX_GRAPHS` regions are captured per process.

## Performance

8xH200, BF16, force-balanced routing, with the deferred EP dispatch; median of steps 2-12.
`graph_stats()` reports the bytes each replay copies across the boundary (parameters, other
inputs, `main_grad` in and out, outputs).

| Config | Graphs off | Graphs on | Boundary bytes / step |
| --- | ---: | ---: | ---: |
| Qwen3-30B-A3B PP1/EP8/CP1, seq 2048 (reference mesh) | 3.059 s | 2.832 s (-7.4%) | 324 GB |
| GPT-OSS-20B PP1/EP8, seq 2048 (reference mesh) | 2.237 s | 2.183 s (-2.4%) | |
| GPT-OSS-20B PP1/EP8, seq 8192 | 3.510 s | 3.618 s (+3.1%) | 221 GB |
| Qwen3-30B-A3B PP1/EP8/CP2, seq 8192 | 5.386 s | 5.583 s (+3.7%) | 473 GB |
| Qwen3-30B-A3B PP2/EP4/CP1, seq 4096 | 2.269 s | 2.389 s (+5.2%) | |
| GPT-OSS-20B PP2/EP4, seq 4096 | 3.985 s | 4.117 s (+3.4%) | |

Two runs per arm agree within 1%. What decides the sign is the number of tokens a rank processes
per micro-batch (sequence length x micro-batch size / CP): graphs win at 2048 and lose at 4096,
whatever the mesh. On the reference mesh (Qwen3 PP1/EP8/CP1), with the same 262k tokens per step:

| Tokens per micro-batch | Graphs off | Graphs on |
| ---: | ---: | ---: |
| 1024 (GBS 256) | 5.86 s | 5.26 s (-10.4%) |
| 2048 (GBS 128) | 3.10 s | 2.82 s (-8.9%) |
| 4096 (GBS 64) | 2.25 s | 2.36 s (+4.8%) |

Larger micro-batches are faster per token whether or not graphs are on: 4096 tokens without graphs
beats 2048 with graphs by 20%. The host stops mattering once a micro-batch carries enough work to
hide it. In Nsight, the GPU of Qwen3 PP1/EP8/CP1 at seq 2048 is fully idle for 34% of the step
waiting for the host to launch the next kernel; at PP1/EP8/CP2 seq 8192 the same idle is 5.6%,
almost all of it in the pipeline fill phase at the start of each step, and only 20 ms over the 5 s
overlapped phase. Capturing the fused stage 5 + stage 1
region instead of stage 1 alone moves 922 instead of 473 GB per step at seq 8192 and costs 176 ms
more, about 0.39 ms per GB of boundary copies. At that rate the copies cost 3.4% of the seq 8192
step and 4.1% of the seq 2048 step: with free copies, graphs would be worth about -0.5% at seq 8192
and about -11.5% at seq 2048. The copies, not the replay, decide the sign; about half of them
(weights and fp32 `main_grad`) do not change between replays of one layer.

## Comparison with Megatron-LM

Megatron-LM (MCore 685fb8b, TE graphs of `attn moe_router moe_preprocess`, the same scope as
here) on the same node, same Qwen3-30B-A3B meshes, real routing (Pith: force-balanced), median of
steps 6-15:

| Config | Pith off | Pith on | Megatron off | Megatron on |
| --- | ---: | ---: | ---: | ---: |
| PP1/EP8/CP1, seq 2048, GBS 128 | 3.130 s | 2.852 s (-8.9%) | 8.943 s | 5.519 s (-38%) |
| PP1/EP8/CP2, seq 8192, GBS 64 | 5.418 s | 5.565 s (+2.7%) | 11.037 s | 7.012 s (-36%) |

The difference is the baseline, not the graphs. In one profiled step at seq 2048 the GPU waits on
the host for 8.1 s in Megatron and 1.7 s in Pith, with a similar number of kernel launches (59k vs
65k): Megatron's eager modules cost far more host time per launch than Pith's compiled regions
(cpp_wrapper) and TVM-FFI kernel launches. Graphs remove about 21k (Megatron) and 29k (Pith)
launches per step, saving roughly 265 us vs 15 us of host time each. At seq 8192 Megatron's GPU
still waits on the host for 10.3 of 17.4 profiled seconds; Pith's waits 0.35 of 6.2. Megatron's
graphs also copy far less: one graph per layer and micro-batch, captured in schedule order into a
shared pool, returns outputs in place and copies only inputs (about 25 GB per step at seq 2048,
against 324 GB here).

## Why weights and main_grad are copied

About half the boundary bytes are layer weights and fp32 `main_grad`, which do not change between
a layer's replays. Reading them in place needs one graph per layer whose captured addresses stay
valid. They do not: FSDP re-gathers the unsharded weights into whatever blocks the allocator
returns after each reshard, and `main_grad` is freed after every step. Over four steps of Qwen3
PP1/EP8/CP1, each layer's weights and `main_grad` took three to four addresses (the 512 x 2048
k/v projections a new one almost every step), so a prototype that captured one graph per set of
bound addresses found only 1 of 192 sets repeating. Binding in place therefore requires FSDP to
keep its all-gather output buffers and `main_grad` allocated across steps; at seq 8192 it would
still be bounded by the ~0.5% above.
