# Megatron allocator comparison and Pith graph-pool diagnosis — 2026-11-16

The measured Megatron CUDA-graph path does not show Pith's repeated allocation/free stalls.
However, our first capacity-pressure intervention also shows that low free memory alone is not
sufficient to reproduce those stalls. The more specific suspect is private graph-pool reservation
and reuse, including a CP-rank asymmetry hidden by the earlier rank-0-only memory summary.

This report extends [the previous copy/overlap diagnosis](cuda-graphs-slowdown-diagnosis-20261116.md).
The observations here refine its causal interpretation; they do not claim that every remaining
millisecond has been attributed or that output lending has met the throughput target.

## Controlled capacity intervention

Same frozen Pith source, Qwen3-30B-A3B BF16, eight H200, PP1/EP8/CP2, sequence 8192,
micro-batch 1, global batch 64. Force-balanced routing, seeded fresh weights, same corpus and
instrumentation across arms. All four complete runs used node `fs-mbz-gpu-532`, step `3074761.12`
in array `3006977`. Each has 15 steps, measuring steps 10–15 using the maximum duration across
ranks per step and then the median. A final copying repeat was preempted after step 6 and excluded;
none of the completed arms was rerun.

| Mode | Median step s | Tokens/s | Peak allocated GiB | Peak reserved GiB | Allocation retries in timed window |
| --- | ---: | ---: | ---: | ---: | ---: |
| No CUDA graphs | 5.7235 | 91,603 | 96.34 | 115.89 | 0 |
| Graphs, copying | 5.9860 | 87,586 | 100.19 | 117.38 | 0 |
| Graphs, output lending | 5.9876 | 87,562 | 108.54 | 133.20 | 12, on rank 3 |
| Graphs, copying + unused 16 GiB | 5.9807 | 87,663 | 116.19 | 133.16 | 0 |

The unused reservation is one persistent CUDA uint8 tensor, allocated and initialized before
warmup. It does not change graph captures, replay counts, model computations or copy payloads.
Copying and copying+16GiB have identical input/output copy bytes per step. All three graph-enabled
measurements have stable graph/fallback counts in their timed windows.

The pressure arm reached about **49 MiB free device memory**, yet had **zero device allocations,
frees or retries** during the measured steps. Lending reached about **11 MiB free** and rank 3
performed 1,429 device allocations and 1,468 frees across the six measured steps. Device-free
samples are taken after each step; they are not the minimum of every instant inside the step.
Peak allocation and peak reservation are independent maxima, not simultaneous snapshots.

**Interpretation:** total occupancy alone did not reproduce the failure. The pressure tensor's
memory is live and predictable; graph-private inactive space and changing allocation lifetimes
can leave a different set of reusable blocks. The result favors investigating pool layout and
reuse over merely lowering a global memory threshold. It does not by itself prove which individual
allocation causes a retry, nor show that near-zero headroom is safe for arbitrary workloads.

Lending and copying were effectively tied in this run, both about 4.4% below graphs disabled.
Thus the allocation cycle is not an explanation for the entire graphs-versus-disabled regression:
the copying mode has no cycle and still loses throughput.

## CP-rank memory asymmetry

Reexamining all eight ranks of the previous matched lending trace gives:

| CP rank | Global ranks | Scheduled-pool live allocation GiB | Scheduled-pool reservation GiB | Reservation minus live allocation GiB |
| --- | --- | ---: | ---: | ---: |
| 0 | 0, 2, 4, 6 | 11.289 | 18.070 | 6.781 |
| 1 | 1, 3, 5, 7 | 11.289 | 23.820 | 12.531 |

Every rank with timed allocation retries across the three controlled lending runs is in the
second group: ranks 1 and 5, then 7, then 3. Each rank retains the same graph-pool live bytes,
but CP rank 1 reserves **5.75 GiB more**. The earlier report's rank-0 pool size understated the
reservation on the ranks that actually stalled.

The dense zigzag ring-attention forward implementation has a matching asymmetry: at CP=2,
the remote block on CP rank 1 computes a full-query output and merges it into newly allocated
FP32 accumulators; CP rank 0 computes only the back half and updates the existing accumulator.
The lending backend gives independent pools to forward slots, so temporary space required during
capture can remain reserved in each slot even after the temporary tensors die.

A fresh full-model snapshot now resolves that mapping exactly. All eight ranks have 73
forward pools, each with 158.360MiB live allocation. Each reserves 248MiB on CP rank 0 versus
328MiB on CP rank 1. The common backward pool reserves 400 versus 448MiB. Thus the difference is
`73 × 80MiB + 48MiB = 5.75GiB`. The extra capacity is in inactive blocks after the step, not
additional live output tensors. The snapshot is after step 3, when all 286 graph variants exist;
the run finishes eight steps without new variants. It ran as step `3075023.36` on node 800.

The 18.07/23.82GiB totals above, matching the previous report, cover scheduled pools. Including
the two original unscheduled graph pools gives **76 pools**, **11.476GiB live allocation** and
**18.578/24.453GiB reservation** on CP rank 0/1. Megatron's 8.69–8.78GiB measurement below includes
all of its graph pools.

A two-GPU isolated ring-attention capture reproduces the **80MiB per-forward-pool difference**:
134MiB reservation on CP rank 0 versus 214MiB on CP rank 1, both with 32.5MiB live outputs.
A task-local experiment changing the full-sequence merge to update its FP32 accumulator in place
reduces CP rank 1 to **150MiB**, while CP rank 0 stays at 134MiB. Eager and three graph replays
produce bit-identical outputs and LSE to the original implementation on both ranks. The paired
probe completes successfully as step `3075023.39` on node 532.

This directly identifies a removable **64MiB temporary allocation per isolated forward pool**.
Multiplying by 73 gives a potential 4.56GiB reduction, but that is an estimate: full-stage allocation
packing may differ, and no full-model throughput or backward-correctness claim is made for this
probe. It is not a production patch. The initial baseline-only probe retained live graphs during
process-group teardown and stalled after writing successful checks; only our own step `.32` was
cancelled. The paired probe resets graphs before teardown and exits cleanly.

The previous affected-rank trace also locates the allocator work on the CPU: of 195.59ms,
8.51ms occurs inside a graph-forward wrapper, 27.67ms inside expert backward, 27.03ms inside
dispatch backward, 38.60ms inside the forward epilogue, and 74.27ms inside the pipeline but outside
its finer NVTX ranges; the remaining 19.52ms is spread across other stages/prologues. These
intervals classify allocator API calls, not total stage overhead. They support interference with
surrounding allocations, rather than expensive execution inside `cudaGraphLaunch`. The 74.27ms
cannot be attributed to a particular operation from these NVTX labels alone.

## New Megatron comparison at PP1/EP8/CP2/S8192

Both nine-step Megatron runs completed on node `fs-mbz-gpu-800`, step `3075023.12` in the same
array, after preemption moved the allocation to a new numeric job. This matches Pith's model,
GPU type/count, PP/EP/CP, sequence length, micro-batch and global batch. It does **not** match
routing, corpus, kernels or scheduler: Megatron uses its normal router and Nemotron corpus,
Pith uses force-balanced routing and its existing fixture. These are diagnostic comparisons,
not a framework throughput ranking or a cross-framework correctness test. Neither installed
Megatron nor TE source was changed; task-local instrumentation checks the TE source hash.

All ranks finish with **zero allocation retries**, both graphs off and on. During profiled
step 6, each rank makes 0–3 CUDA allocations and zero frees. The worst allocation API total is
2.46ms off and 5.56ms on; the worst on interval during GPU inactivity is 4.45ms. Rank 0 makes
no allocation/free calls in either trace. This confirms that the earlier low allocator cost was
not confined to CP1/S4096.

Megatron captures 2,304 graphs into **one pool per rank**. That pool reserves 8.783GiB on CP rank 0
and 8.688GiB on CP rank 1, with 0.863GiB live allocation after capture. This snapshot is not the
peak of live graph activations during execution: capture-time ownership has deliberately been
released where the replay schedule permits reuse. At step 6 the maximum total peak allocation
across ranks is 98.03GiB off versus 93.46GiB on; maximum reservation is 112.27 versus 114.21GiB,
and minimum sampled free device memory is 21.75 versus 19.00GiB.

Rank-0 boundary-copy comparison, complete step 6:

| Category | Pith lending GPU ms | Megatron GPU ms | Megatron concurrent other GPU ms | Megatron without other GPU activity ms |
| --- | ---: | ---: | ---: | ---: |
| Forward inputs | 29.04 | 24.63 | 22.76 | 1.87 |
| Backward inputs | 78.49 | 20.27 | 18.04 | 2.22 |
| Forward outputs | 8.97 | 0 | 0 | 0 |
| Backward outputs / mutated operands | 85.22 | 0 | 0 | 0 |
| Total | **201.72** | **44.90** | **40.81** | **4.09** |

The 156.82ms total-copy difference comprises approximately 4.41ms forward-input difference,
58.22ms backward-input difference, and 94.19ms of Pith output/mutation copies absent from TE's
replay boundary. Graph boundaries cover different operations, so this subtraction locates costs
rather than proving that substituting TE would save exactly those milliseconds.

Pith moves **326.55GiB/step/rank** across graph boundaries; Megatron moves **64.5GiB** (37.5 forward,
27 backward). Megatron has 5,376 copy operations versus Pith's 3,072 batched copy-kernel launches.
These counts use different batching granularity, but they rule out a simple explanation that
Pith is slower merely because it submits more individual boundary-copy kernels. The payload
is about 5.1 times larger and the measured GPU copy duration about 4.5 times larger.

Pith's bytes split into 41.31GiB forward inputs, 121.36GiB backward inputs, 14.04GiB forward
outputs, and 149.84GiB backward outputs. The largest backward-output components are:

| Buffer copied out of Pith backward | Payload per step/rank GiB |
| --- | ---: |
| Expanded expert-combine input gradient, `(32768, 2048)` BF16 | 83.375 |
| FP32 dense-weight gradient buffers (main_grad mutations) | 54.000 |
| Hidden-state activation gradient, `(1, 4096, 2048)` BF16 | 12.000 |
| Router/normalization/probability gradients and remaining small buffers | 0.463 |

The first two account for 91.7% of backward-output bytes. Backward also copies the FP32 main_grad
buffers **into** the graph (54GiB), and BF16 dense projection weights (27GiB). Forward copies
those projection weights as well. Unlike TE's stable parameter binding, the generic Pith boundary
mirrors operands whose storage may change through FSDP. Eliminating those transfers requires
stable storage or schedule-aware binding, not merely replacing `copy_` with a faster API.

These byte counts come from captured tensor shapes/dtypes and steady-state replay frequencies.
The reconstructed totals match the independently recorded step-6 before/after copy counters
exactly. They are payload accounting, not per-tensor time estimates: tensors share batched copy
kernels, so their milliseconds cannot be directly assigned in proportion to their byte counts.

Megatron's 90.9% temporal copy overlap is mainly **HybridEP synchronization**, not useful compute:
30.34ms intersects EP synchronization kernels, 9.69ms EP transfers, 0.11ms NCCL, 0.61ms compute,
and 0.09ms other memory activity. Categories can overlap each other; their individual intersections
are not additive. Such waiting may itself depend on copies completing, so 4.09ms is an activity
classification, not a measured counterfactual copy penalty. Across all ranks total copies are
44.90–47.91ms and isolated copies 3.63–4.23ms. Pith's corresponding rank-0 intervals are 201.72ms
total and 165.22ms without concurrent GPU work.

CPU copy-submission spans total 124.87ms for Megatron, with 120.12ms concurrent GPU activity and
4.74ms during GPU idle. Pith's corresponding totals are 281.07ms, 271.76ms and 9.30ms. Thus the
large difference in summed host submission duration is mostly overlapped in **both** frameworks;
it is not an additional 156ms of exposed CPU latency. These NVTX durations exclude instrumentation
around the NVTX scope; the task-local Python counters include that overhead and are therefore
slightly larger. GPU payload timings are correlated to those same scopes.

For context, the disjoint rank-0 profile accounting is:

| Megatron wall-time category | Graphs off ms | Graphs on ms | Change ms |
| --- | ---: | ---: | ---: |
| Compute, including concurrent communication | 4379.20 | 4620.49 | +241.29 |
| Boundary copies after excluding overlap with compute | 0 | 44.29 | +44.29 |
| Communication only | 2676.53 | 3109.90 | +433.38 |
| Other memory | 52.15 | 51.70 | -0.45 |
| GPU inactive | 10341.28 | 1245.50 | -9095.77 |
| Whole profiled step | 17449.15 | 9071.88 | -8377.27 |

The dominant observed change is removal of GPU inactivity, despite more exposed communication
and slightly more compute activity. This supports a much larger launch/submission opportunity
in Megatron's baseline than in Pith's already-compiled path. Profiling exaggerates that gap:
short unprofiled steps 7–9 have median maximum-rank times of 12.768s off and 7.661s on, rather than
17.449s and 9.072s. Those three steps confirm the direction but are not a long throughput acceptance
run. These captures do not provide a hardware-counter diagnosis of the compute increase.

## Megatron evidence already available at PP2/EP4/CP1/S4096

Whole-step-6 Nsight analysis uses all streams on each rank, not only the compute stream.
These traces are Qwen BF16, eight H200, micro-batch 1, global batch 64. They use a different mesh
and sequence length from the Pith experiment above, so their absolute milliseconds are not a
matched-work comparison.

| Mode | CUDA allocation calls per rank | Maximum summed allocation API time per rank | cudaFree calls |
| --- | ---: | ---: | ---: |
| Megatron graphs disabled | 0–2 | 2.13 ms | 0 |
| Megatron graphs enabled | 0–3 | 2.01 ms | 0 |
| Older Pith CP1 graph backend, same nominal mesh/sequence | 0–2 | 0.99 ms | 0 |

Megatron's graph-enabled maximum allocator time coinciding with GPU inactivity is 1.78 ms per
rank. In contrast, the prior Pith CP2 lending trace has 571 allocation/free calls taking 195.59 ms
on its affected rank, including 114.10 ms during GPU inactivity. This is a measured difference,
not a fundamental property of CUDA graphs. The old Megatron runs did not record allocation-retry
counters, so no claim of zero retries is inferred from missing counters.

Megatron had approximately 20 GiB of free device memory in those runs. The older Pith CP1 backend
had approximately 2 GiB free and still did not show repeated allocation/free calls. This is another
reason to distinguish memory layout and lifetime from raw occupancy.

Megatron rank-0 boundary copies in that graph-enabled step:

| Category | GPU ms | Concurrent other GPU activity ms | Without other GPU activity ms |
| --- | ---: | ---: | ---: |
| Forward inputs | 9.234 | 6.028 | 3.206 |
| Backward incoming gradients | 9.288 | 4.265 | 5.024 |
| Forward outputs | 0 | 0 | 0 |
| Backward returned activation gradients | 0 | 0 | 0 |
| Total | 18.523 | 10.293 | 8.230 |

Of the 10.293ms concurrent interval, 7.391ms overlaps HybridEP synchronization kernels,
2.541ms EP transfer, 0.034ms NCCL, 0.314ms compute and 0.013ms other memory activity. Consequently,
55.6% temporal overlap must not be described as 55.6% hidden behind useful compute. CPU copy
submission totals 76.14ms, with 27.35ms during GPU inactivity; this is separate from GPU payload
time and must not be added to it. Totals already cover the full step on one rank.

## What Megatron does differently

The measured path uses Transformer Engine graphs, not Megatron's alternative local graph manager.
Pinned MCore commit `685fb8bd970185e9ac3283fe32a5d6667bf4b268` supplies pipeline execution order to
TE commit `44f31bbeaf99bd30a67b928000d4b980026debd2`. It expands that order for layerwise MoE overlap
and delayed weight gradients, enables graph-input/output-buffer reuse, and disables returned
parameter-gradient cloning because MCore's accumulation hooks consume those gradients.

TE captures forward and backward in execution order into one shared pool, releases ownership
when the corresponding capture-time consumer has finished, and keeps non-owning output aliases
for replay. It reuses input buffers only when the schedule says the earlier backward has consumed
them, and returns output/activation-gradient views directly. Inputs are copied only if their
addresses differ. See the pinned [TE implementation](https://github.com/NVIDIA/TransformerEngine/blob/44f31bbeaf99bd30a67b928000d4b980026debd2/transformer_engine/pytorch/graph.py)
and [MCore integration](https://github.com/NVIDIA/Megatron-LM/blob/685fb8bd970185e9ac3283fe32a5d6667bf4b268/megatron/core/transformer/cuda_graphs.py).

Pith already reuses a bounded set of runtime schedule slots. The gap is in how capture-time
allocation is planned across forward and backward, not simply the absence of a slot counter.
Our safe lending implementation isolates each forward slot's pool; it preserves live outputs
but also repeats temporary capacity. Backward still copies returned gradients and mutated
operands, including FP32 main_grad. Those output copies were approximately 85ms per step in
the latest Pith rank-0 trace, with almost no concurrent GPU work.

Blindly replacing the separate pool handles with one handle is unsafe. Capture B could reuse
capture A's temporary address for an output that remains live when A next replays. Correct sharing
requires a capture/replay order and last-consumer plan covering EP/CP communication, pipeline P2P,
and deferred weight-gradient work. TE's particular assumptions about when a preceding chunk's
P2P gradient has completed cannot be imported into DualPipeV without establishing the equivalent
ordering.

## Improvement direction

1. Test the identified in-place ring merge in the full model: it has removed 64MiB per pool in
   the isolated probe. Require forward/backward correctness, lower graph-pool reservation, fewer
   allocation retries, and a throughput comparison against graphs disabled. If it only improves
   memory but not time, do not count it as meeting the performance goal.
2. For a general solution, record DualPipeV static invocations and last consumers, capture in
   that order into a shared arena, and retain only outputs that must survive until their consumers.
   Validate against the no-graph baseline, tracking retries, peer waiting, memory and throughput.
3. Address backward output/main_grad copying separately. Eliminating allocator churn alone does
   not guarantee a win: the copying backend already has zero churn and is slower than no graphs.

The direct cost evidence is stronger than before, but there is not yet a demonstrated performance
fix from this investigation. No production code or installed Megatron/TE source was modified.

## Evidence and reproducibility

[Machine-readable evidence](cuda-graphs-megatron-memory-evidence-20261116.json) includes the
four complete Pith arms, all-rank allocator and phase counters, prior per-rank pool totals,
Megatron allocator/overlap queries and the older Pith CP1 control.

Task-worktree artifact directories are `workspace/cuda-graphs/megatron-memory-20261116`,
`workspace/cuda-graphs/megatron-memory-target-20261116` `workspace/cuda-graphs/pool-layout-20261116` and `workspace/cuda-graphs/ring-pool-reuse-probe-20261116`.
The cluster root is `/mnt/weka/home/hao.kang/pith-train/cuda-graphs-overhead`. Raw Nsight traces
remain on the cluster; analysis exports and scripts are local. Production-source SHA256:
`51fa25f1547f5522dcd11910794bf7c7f52eec5dfa64570ee028a49020a2e505`.
