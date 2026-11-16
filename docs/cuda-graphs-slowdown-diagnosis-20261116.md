# Where the output-lending copy savings went — 2026-11-16

**Follow-up:** the [Megatron comparison and capacity intervention](cuda-graphs-megatron-memory-20261116.md)
subsequently completed the pressure probe. High occupancy alone did not reproduce the stalls;
that report refines the pool-layout explanation and includes the previously missing CP-rank asymmetry.

The output-lending implementation removes substantial GPU copy traffic, but its retained graph
pools introduce allocation pressure and distributed stalls. In a controlled same-node profile,
copying falls by **110.30 ms**, while the whole step becomes **69.78 ms slower**. This report
accounts for that difference and separates observed timing from causal conclusions.

This is a diagnosis of Qwen3-30B-A3B BF16 on 8 H200, PP1/EP8/CP2, sequence 8192, micro-batch 1,
global batch 64. It is not a conclusion about every model or configuration. No production code
was changed for this investigation.

## Controlled experiment

All six throughput runs and both new traces ran sequentially on `fs-mbz-gpu-298`, inside numeric
job `3073368` of the authorized array `3006977`. Source was the immutable `candidate-a03` snapshot
of commit `065a19f`, production-source SHA256
`51fa25f1547f5522dcd11910794bf7c7f52eec5dfa64570ee028a49020a2e505`.

The three modes are graphs disabled; the same code with `CombineBufferPool.lend_outputs=False`
(copying); and unchanged output lending. Thus the copying/lending comparison isolates output
ownership rather than comparing unrelated branches. The copying arm is the schedule-aware
prototype immediately before output lending, not every historical CUDA-graph backend.

Order: off, copy, lend, lend, copy, off. Each run has 15 steps; the timed window is steps 10–15.
Per-step duration is the maximum over all eight ranks; each run contributes one median. All
variants have identical low-overhead phase events and allocator-counter instrumentation, with
no added synchronization. Routing is force-balanced; the seeded fresh-weight fixture and corpus
match the previous report. This is not a released-checkpoint convergence experiment.

| Mode | Run 1 median s | Run 2 median s | Tokens/s from mean of run medians | Throughput vs off | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Graphs disabled | 5.6756 | 5.6857 | 92,294 | baseline | 96.34 |
| Graphs with copying | 5.9477 | 5.9414 | 88,196 | -4.44% | 100.20 |
| Graphs with output lending | 6.0315 | 5.9516 | 87,505 | -5.19% | 108.54 |

Lending's 12 timed steps span 5.8703–6.3447 s, versus copying's 5.9352–5.9857 s. Lending sometimes
beats copying on an individual step, but does not beat graphs disabled. Both disabled runs and
both copying runs have zero device allocations, frees or allocation retries in the timed window.
Lending has 22 retries across ranks 1 and 5 in the first run and 12 on rank 7 in the second.
These are timed-window counts, not warmup counts. Every retry is inside the pipeline phase.
PyTorch defines an allocation retry as a failed allocation followed by cache reclamation and retry;
see its [memory_stats documentation](https://docs.pytorch.org/docs/stable/generated/torch.cuda.memory.memory_stats.html).

GPU event intervals locate the slowdown inside `DualPipeV.step`. Rank-0 pipeline medians are
5619.7/5628.3 ms off, 5890.0/5883.0 ms copying, and 5961.3/5883.9 ms lending. Scaling stays at
7.9 ms, the optimizer at 26.6 ms, and gradient clipping changes from about 11.7 to 14.6–15.4 ms.
These event intervals include stream waits: they are elapsed phase time, not pure kernel compute.
Longer CPU optimizer-call spans do not imply a slower optimizer kernel.

## Forward/backward input/output copies

The new paired Nsight traces cover the same step 6 and all ranks. Tables below use rank 0 unless
specified. Stage-invocation counts match on every rank; runtime correlation has zero missing
kernel launches. Both modes execute exactly 768 forward and 768 backward replays in the step,
with no new capture or fallback. Rank 0 has the same 76,649 compute kernels, 8,559 communication
kernels, and 3,072 batched boundary-copy kernels in both modes.

All times below are totals for **one complete step on one rank**, already including all layers
and micro-batches. Do not multiply them by the number of layers, micro-batches or ranks.

| Direction | Copying GPU ms | Lending GPU ms | Lending concurrent communication ms | Lending overlap % | Lending without other GPU work ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Forward inputs | 28.94 | 29.04 | 6.77 | 23.3% | 22.28 |
| Backward inputs | 130.02 | 78.49 | 29.35 | 37.4% | 49.14 |
| Forward outputs | 67.68 | 8.97 | 0.00 | 0.0% | 8.97 |
| Backward outputs | 85.39 | 85.22 | 0.39 | 0.5% | 84.83 |
| Total | **312.02** | **201.72** | **36.50** | **18.1%** | **165.22** |

Copying's total has 53.50 ms concurrent with communication and 258.53 ms without other GPU work.
Lending therefore removes 110.30 ms of total copy activity and 93.31 ms of copy activity without
concurrent work. There is no observed compute/copy overlap in either trace.

This is primarily less payload per batched copy, not fewer copy-kernel launches: logical boundary
traffic falls from 522.43 to 326.55 GiB/step/rank, while the batched call count stays unchanged.
The fused stage-5/next-stage-1 forward output category falls from 59.47 to 0.78 ms. Its remaining
logging scalar is copied, and the separate stage-1 path contributes the other 8.19 ms. Backward
output copying remains; this category also includes mutated operands copied back to their callers,
such as FP32 `main_grad`, and is not exclusively returned activation gradients.

CPU copy submission is separate from the GPU payload work:

| Direction | Copying CPU ms | Lending CPU ms | Lending concurrent GPU work ms | Lending during GPU idle ms |
| --- | ---: | ---: | ---: | ---: |
| Forward inputs | 100.55 | 98.18 | 90.69 | 7.50 |
| Backward inputs | 127.00 | 92.66 | 91.48 | 1.19 |
| Forward outputs | 83.54 | 38.90 | 38.53 | 0.37 |
| Backward outputs | 50.67 | 51.32 | 51.07 | 0.25 |
| Total | **361.77** | **281.07** | **271.76** | **9.30** |

CPU submission falls 80.71 ms, but its portion during GPU inactivity falls only 1.40 ms
(10.70 to 9.30). About 97% of CPU copy submission overlaps GPU work in both modes. These are
CPU execution/submission spans, not host-to-device payload transfers. They must not be added to
GPU copy duration. Concurrent activity can contend for resources, so measured overlap is not
assumed to be cost-free.

## What became slower: complete step accounting

The following wall-time categories are disjoint and sum exactly to the profiled step. Compute
gets priority over simultaneous communication, then copies, then other memory operations;
communication-only means no concurrent compute/copy/memory operation. NCCL activity can include
waiting for a peer, so communication-only time is not a measurement of network transfer speed.

| Rank-0 wall-time category | Copying ms | Lending ms | Change ms |
| --- | ---: | ---: | ---: |
| Compute, including concurrent communication | 4682.93 | 4691.66 | +8.73 |
| Boundary copies, including concurrent communication | 312.02 | 201.72 | -110.30 |
| Communication only | 547.02 | 761.26 | +214.24 |
| Other memory operations | 20.22 | 20.19 | -0.03 |
| No GPU activity | 812.33 | 769.47 | -42.86 |
| **Whole step** | **6374.52** | **6444.30** | **+69.78** |

Thus the copy saving is real, but a much larger increase in exposed communication outweighs it.
This is an accounting of the same complete traces, not a sum of unrelated microbenchmarks.
The matched middle window, `forward chunk 8 (phase1) backward chunk 7 (phase0)`, improves from
175.43 to 169.71 ms. Gains within a steady pipeline window can coexist with a slower whole step.

Rank 5 explains an important part of rank 0's communication increase. In the copying trace,
rank 5 has no CUDA allocation/free calls. In the lending trace:

| Rank-5 CPU API | Calls | Wall time ms | Concurrent GPU work ms | During GPU idle ms |
| --- | ---: | ---: | ---: | ---: |
| cudaMalloc | 215 | 105.23 | 50.90 | 54.33 |
| cudaFree | 356 | 90.36 | 30.59 | 59.77 |
| Total | **571** | **195.59** | **81.49** | **114.10** |

Only 41.7% of this API time overlaps GPU activity; 58.3% coincides with rank 5's GPU being idle.
Its whole-step GPU-idle time grows **214.09 ms**, from 740.76 to 954.85 ms. Its compute decreases
33.40 ms and copies decrease 115.17 ms, but that idle increase makes its measured step 64.46 ms
longer. Step boundaries differ slightly between ranks; these are not independent losses to add.

Using Nsight's common clock across processes, rank 0 spends **118.67 ms in communication-only
activity during rank 5's allocation/free calls**, another 5.99 ms idle, and 70.93 ms doing other
GPU work. Other ranks show 108–118 ms of communication-only activity during those same calls.
This is direct evidence of concurrent peer waiting consistent with propagation of the allocator
stall. It is not evidence that the interconnect itself became 214 ms slower.

The 114.10 ms directly inside allocation/free calls does not individually explain all 214.09 ms
of added rank-5 idle time. Roughly 100 ms remains outside those API intervals; changing CPU
scheduling/bookkeeping and profiler perturbation have not been separated causally for that part.
The full wall-time balance is closed, but it would overstate the evidence to assign every added
idle millisecond to the CUDA allocator.

## Why output lending causes this pressure

The correctness-safe implementation retains graph outputs and saved tensors in separate forward
slot pools until backward consumes them. Unlike copied scratch outputs, these pools cannot be
immediately reused by another live micro-batch. On rank 0, captured private-pool reservation rises
from **0.63 to 18.07 GiB**; allocated bytes inside those pools rise from zero to **11.29 GiB**.
Total peak reservation rises from about 117.1 to 133.2 GiB, and total peak allocation grows by
8.35 GiB versus copying. Pool measurements are components of the totals, not additional memory.

This provides a concrete mechanism for the retries: avoiding copies pins more memory, leaving
less reusable capacity for the dynamic portions of the pipeline. The graph-variant count grows
148 to 286, but replay and compute-kernel counts do not grow. The cost is chiefly storage lifetime
and its interaction with the allocator, rather than extra model arithmetic.

A same-node test adding an unused 16-GiB reservation to the copying mode was prepared to isolate
capacity pressure from graph ownership. A fresh disabled trace was also prepared. Neither ran:
two step-launch attempts were rejected because other work occupied the requested node. Retries
were stopped; no other agent's step was cancelled. No conclusion in this report relies on those
unexecuted probes. Consequently, total capacity versus private-pool fragmentation/lifetime has
not yet been isolated by an intervention.

## Kernel contention and the comparison with graphs disabled

The existing same-node disabled/lending trace pair was also reanalyzed by matching stage,
invocation and kernel ordinal. For the backward GEMM family
`nvjet_sm90_tss_128x256_64x4_2x1_v_badd_coopA_NTN`:

- 525 matched kernels without communication in either run have a median duration ratio of 1.008.
- 598 kernels that newly overlap communication have a median ratio of 1.821.
- Their concurrent communication is the forward EP combine (`stage4_f`), not CP ring attention.
- Across all 1,242 calls, this family's summed time rises from 125.53 to 161.95 ms.

A second GEMM family shows the same pattern: 1.022 with no communication in either run, versus
1.601 when communication newly overlaps it. This supports resource contention during changed
pipeline overlap, rather than a uniformly slower GEMM implementation. Nsight Systems does not
identify whether SM residency, memory bandwidth or cache contention dominates; no hardware-counter
or communication-serialization intervention was performed.

The fresh copying/lending pair corroborates that distinction. For the first GEMM family, 618
non-overlapping pairs are effectively unchanged (1.001), while 55 newly overlapping pairs take
1.722 times as long. Nevertheless, whole-step compute grows only 8.73 ms between these two
modes, so GEMM contention is not the main incremental regression from output lending.

Against graphs disabled, lending still has approximately 202 ms of boundary-copy work, 165 ms
without concurrent GPU work, plus allocation stalls and changed compute/communication overlap.
These explain concrete costs that can exceed launch savings. They are not additive independent
estimates of the unprofiled slowdown. The earlier disabled/lending Nsight pair reversed the
unprofiled ordering (6.346 versus 6.314 s profiled, against roughly 5.675 versus 5.944 s unprofiled).
Profiling perturbs the launch-heavy disabled path more, so that pair cannot provide an exact
causal budget for the real throughput difference. The new copying/lending pair agrees in sign
with the unprofiled repeats, but is still diagnostic rather than the throughput verdict.

## Assessment

The implementation successfully removes forward-output and saved-activation copy traffic.
It has not met the performance target. The strongest explanation for losing the new savings is
memory retained by the output-lifetime policy, causing allocator churn inside the pipeline and
peer waits. A second, distinct effect is increased kernel duration under changed communication
overlap; it matters relative to graphs disabled but is small in the latest copying/lending delta.
CPU copy submission is predominantly overlapped and is not the largest exposed cost here.

The next intervention should target the persistent graph-pool footprint and validate that retries
and peer waits disappear, before doing more CPU-copy optimization or extending a configuration
sweep. The prepared memory-pressure probe would help distinguish capacity from pool topology.
Returning more borrowed outputs without controlling their storage lifetime is unlikely to solve
the present regression. Correctness constraints on aliasing and asynchronous consumers still apply.

## Artifacts

The [machine-readable evidence](cuda-graphs-slowdown-diagnosis-evidence-20261116.json) contains
run summaries, all-rank profile accounting, CPU copy/allocator overlap, cross-rank intersections,
matched-kernel comparisons, source identity and export hashes.

Local artifact directory:
`workspace/cuda-graphs/output-leases-20261116/slowdown-diagnosis` in the task-5 worktree.
The same relative directory is under `/mnt/weka/home/hao.kang/pith-train/cuda-graphs-overhead`
on the cluster. Raw `.nsys-rep` files remain there. The completed trace prefix is
`qwen-s8192-copy-lend-diagnosis-try01-profile-{copy,lend}-node0`.

Analysis scripts: `compare_fresh.py`, `analyze_cpu_copies.py`, `analyze_cross_rank.py`,
`compare_chunks.py`, and `match_fresh_kernel_overlap.py`; these compose the existing
`analyze_pith_profile.py`, `analyze_overlap.py`, and `analyze_allocator.py` helpers.
The prior disabled/lending source trace prefix is
`qwen-pp1-ep8-cp2-s8192-a03-suite-try01-profile-{off,on}-node0` in the parent artifact directory.
