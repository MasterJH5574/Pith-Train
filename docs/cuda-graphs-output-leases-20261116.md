# Scheduled CUDA-graph output reuse — 2026-11-16

This is a bounded research implementation of directly returning graph-owned forward tensors.
The acceptance target is throughput versus CUDA graphs disabled, with correct training and usable
memory. It does not eliminate every graph boundary copy or implement Megatron's full integration.

## Implementation

The existing DualPipeV combine-buffer pool already knows when a combined activation is acquired
and when its corresponding aggregation backward has finished. The new implementation extends that
lifetime to graph-produced forward outputs and AOT saved activations. Backward binds those saved
allocations directly, avoiding the copy back into its input scratch space.

Each forward slot uses an independent CUDA graph allocation pool, so a replay for a different live
micro-batch cannot overwrite its outputs. FIFO retirement, one spare slot and stream events retain
the existing schedule's reuse ordering. This is deliberately conservative: sharing arenas safely
requires planning their lifetimes across the actual forward/backward schedule. Merely sharing one
pool is insufficient: a temporary address used by graph A can later become graph B's retained
output during capture, then A's replay can overwrite B while that output is still live. Keeping
B's tensor object alive does not retroactively change A's recorded addresses.

The public-output contract is explicit. Qwen's fused stage-5/next-stage-1 region declares hidden
states, residuals, routing indices and routing weights eligible for borrowing. Its load-balance
metric is still copied because logging retains it until the end of the step. Unknown public
outputs and outputs aliasing protected storage keep their copying behavior. Forward consumers
cannot reassign an earlier layer's outputs to a newer slot's lifetime.

Backward returned gradients still copy. In particular, the expert-combine gradient can remain in
use by asynchronous EP communication after its producer returns; eliminating that copy requires
extending ownership through the communication consumer. This experiment does not assume backward
return means that all consumers have finished.

The path is selected by the existing eligible Qwen BF16, CP>1, EP>1 training integration. Ordinary
calls and other models retain their existing output semantics. Reset clears output registrations
and the dynamically keyed backward input scratch buffers.

## Validation method

- GPU tests exercise two simultaneously live micro-batches, reverse backward order, parameter
  updates, chained layer lifetimes, retained logging metrics, and cleanup. Both lending and
  copying paths are covered.
- Distributed reference checks compare losses, every parameter gradient and updated parameters
  against the non-pipelined reference. Three steps exercise initial capture, discovery of direct
  backward inputs, and steady replay. Expected forward/backward replay counts are asserted.
- Full-model correctness uses real routing for 32 steps: two graph-disabled runs establish the
  run-to-run envelope, then the candidate is compared with the first baseline.
- Throughput uses force-balanced routing, four independent processes in off/on/on/off order on
  the same H200 node. Each process runs 15 steps; steps 10–15 are timed after capture stabilizes.
  Per-step duration is the maximum across all eight ranks. Each run contributes one median.
- Paired Nsight captures use step 6, all ranks, and the same source and configuration. GPU copy
  durations, overlap and whole-step activity are computed from the same traces. Profiled timings
  are diagnostic; unprofiled runs determine the throughput verdict.

Configuration: Qwen3-30B-A3B, BF16, 8 H200, PP1/EP8/CP2, sequence 8192, micro-batch 1,
global batch 64. The existing fixture starts from identical seeded fresh weights and the same
corpus; it is not a released-checkpoint convergence study.

Production behavior in the full suite is commit `065a19f` (frozen snapshot `candidate-a03`,
production-source SHA256 `51fa25f1547f5522dcd11910794bf7c7f52eec5dfa64570ee028a49020a2e505`).
The follow-up commit clears retained objects on reset; it does not change measured training replay.
Final formatting changes were checked for AST equality with the tested snapshot.

## Results

The repeated throughput comparison is negative: **4.53% lower throughput than graphs disabled**.
Forward output borrowing is feasible, but this version is not a performance improvement and should
not replace the existing backend in production.

| Independent run | Step seconds | Tokens/s | Peak allocated GiB | Peak reserved GiB |
| --- | ---: | ---: | ---: | ---: |

| perf-off0 | 5.6770 | 92,354 | 96.34 | 115.92 |
| perf-on0 | 5.9383 | 88,290 | 108.54 | 133.20 |
| perf-on1 | 5.9497 | 88,120 | 108.54 | 133.20 |
| perf-off1 | 5.6728 | 92,421 | 96.34 | 116.17 |

The median of the two run-level step medians is 5.6749 s off and 5.9440 s on, a 269.1 ms
increase. The two independently paired throughput changes are -4.40% and -4.65%. No timed step
adds a graph instance or fallback; source identities and all eight rank records were checked.

Peak allocated memory grows by 12.20 GiB; peak reserved memory grows by 17.0–17.3 GiB.
Each balanced candidate process logs 50 recovered CUDA allocation failures, while both baseline
processes log zero. These counts cover the entire process, including warmup; they are not a
measurement of allocation overhead in the timed window. The real-routing candidate logs 147
allocation failures but completes all 32 steps. Merely avoiding a fatal OOM is not an adequate
memory qualification for this version. Minimum device memory free after a step falls from
17.1 GiB in the balanced baselines to 43 MiB in the candidate; the real-routing candidate reaches
only 5 MiB free. These are post-step snapshots, not an estimate of the minimum within a step.

The 32-step correctness check passes. Cross-entropy mean absolute delta is 0.00016875 versus
0.000178125 between the two baselines (0.95x); load-balance delta is 0.00030119 versus 0.00047284
(0.64x). This supports absence of a detected regression over this window, not long-run convergence.

Rank-0 steady replay copies 162.67 GiB of input payload and 163.88 GiB of output payload per step, about 326.55
GiB total. The previous schedule prototype moved 522.43 GiB for the same geometry, so the boundary
traffic falls about 37.5%. This is a deterministic work-volume comparison; the older prototype's
throughput was measured on a different node and is not used as a controlled latency comparison.

## Profile analysis

Both traces use training step 6 on all eight ranks of the same node. Runtime correlation is
complete (zero unmatched kernel launches), stage-invocation counts match, and the candidate adds
exactly 768 forward and 768 backward replays without capture or fallback growth in the measured
step. Rank 0 is the primary timeline; rank 5 is additionally examined because it performs allocator
reclamation. The machine-readable [evidence](cuda-graphs-output-leases-evidence-20261116.json)
contains all ranks, trace hashes and paths.

### Remaining graph copies, rank 0, entire step

| Copy direction | GPU activity ms | Overlap with communication ms | Overlap fraction | Without concurrent GPU work ms |
| --- | ---: | ---: | ---: | ---: |
| Forward inputs | 28.85 | 6.63 | 23.0% | 22.23 |
| Backward inputs | 79.99 | 36.61 | 45.8% | 43.38 |
| Forward outputs | 8.98 | 0.00 | 0.0% | 8.98 |
| Backward outputs | 86.00 | 0.36 | 0.4% | 85.64 |
| Total | 203.83 | 43.60 | 21.4% | 160.23 |

Here “output copies” includes copying mutated operands back to their callers, such as FP32
`main_grad` accumulators, as well as returned tensors. The input rows likewise include more than
activation inputs. The 86 ms backward-output category is not exclusively the EP activation gradient.

There is no observed overlap between these copy kernels and compute kernels in this rank's trace;
the observed overlap is communication. Concurrency is measured, not assumed free of contention.
The fused stage-5/next-stage-1 forward output copies account for only **0.77 ms** of the 8.98 ms:
its large outputs are borrowed, while the logging scalar remains copied. The other 8.21 ms comes
from the separate stage-1 path. This is direct evidence that the intended forward-copy removal
works; backward output copying remains the largest uncovered copy category.

CPU `CG.copy_grouped` spans total 273.68 ms. Of that, 265.91 ms (97.2%) overlaps GPU activity,
leaving 7.77 ms during GPU inactivity. These are CPU submission spans, not host-to-device tensor
payload transfers. They must not be added to GPU copy duration or to enclosing graph-call spans.

### Allocation pressure, rank 5, the same step

The baseline has no `cudaMalloc` or `cudaFree` calls in the measured step. The candidate has:

| API | Calls | CPU wall time ms | During GPU activity ms | During GPU inactivity ms |
| --- | ---: | ---: | ---: | ---: |
| cudaMalloc | 215 | 90.18 | 44.63 | 45.55 |
| cudaFree | 355 | 76.07 | 24.00 | 52.07 |
| Total | 570 | 166.25 | 68.63 | 97.62 |

Thus 58.7% of the allocation/free API time coincides with this rank's GPU being idle. Rank 5's
whole-step GPU-inactive time grows from 663.09 to 799.66 ms. This identifies a concrete extra cost
associated with the memory pressure; it is not just a warning count. Other ranks can wait for
this rank, so their synchronization spans are not independent costs to sum with it.

The candidate's private graph pools reserve 18.07 GiB on rank 0, with 11.29 GiB allocated. The
previous copying schedule prototype's corresponding diagnostic was 0.63 GiB reserved and zero
allocated in those pools after conversion to non-owning scratch views. The new isolated arenas
protect outputs that remain live, but retain considerably more device memory. These pool figures
are components of the total allocation/reservation, not additional memory to add to the peaks.

### Whole-step accounting and matched stage samples

The following categories are disjoint. Compute includes time concurrent with communication;
copy includes time concurrent with communication; other memory activity is assigned before
communication-only time. Their sum equals the profiled step exactly.

| Rank-0 wall-time category | Graph off ms | Candidate ms | Change ms |
| --- | ---: | ---: | ---: |
| Compute | 4615.90 | 4671.74 | +55.84 |
| Graph boundary copies | 0.00 | 203.83 | +203.83 |
| Communication without compute/copy/memory | 973.59 | 754.75 | -218.85 |
| Other memory activity | 18.77 | 20.18 | +1.41 |
| No GPU activity | 737.41 | 663.70 | -73.71 |
| Total | 6345.68 | 6314.20 | -31.48 |

The same scheduler window, `forward chunk 8 (phase1) backward chunk 7 (phase0)`, is 173.90 ms
off and 170.40 ms on. These windows use matching CPU NVTX boundaries and clip GPU intervals in
wall time. Two matched stage invocations inside that window illustrate both directions:

| Stage invocation (zero-based occurrence) | Compute off/on ms | Copies off/on ms | GPU execution envelope off/on ms |
| --- | --- | --- | --- |
| Layer 35 stage 5 + layer 36 stage 1 forward, occurrence 7 | 1.073 / 1.121 | 0 / 0.039 | 1.970 / 1.683 |
| Layer 11 stage 5 + layer 12 stage 1 backward, occurrence 5 | 2.369 / 2.576 | 0 / 0.222 | 2.780 / 3.057 |

An execution envelope is the first-to-last GPU event attributed to that invocation; it is not an
independent critical-path cost. Across the fused backward invocations, the same two large GEMM
kernel families gain 36.42 and 24.76 ms with unchanged invocation counts; FlashAttention backward
gains 14.40 ms. Other kernels get faster, including an elementwise family by 35.20 ms. The total
fused-backward compute increase is 45.39 ms. This shows that the remaining change is not solely
copy duration. These measurements do not establish whether the kernel differences come from
cache behavior, contention or another scheduling consequence.

**Profiling changes the net result:** these traces favor the candidate by 31.48 ms, whereas both
unprofiled pairs lose approximately 269 ms per step. Therefore this table reconciles the traced
step, but cannot honestly be presented as an exact causal decomposition of the unprofiled loss.
In particular, do not add rank 0's 160.23 ms copy-only time to rank 5's 97.62 ms allocator-idle
time and call the sum the end-to-end regression: those ranks execute concurrently. What is
established is substantial remaining copy work, measurable allocator churn, changed kernel
execution times, and a repeatable unprofiled slowdown.

## Test results and scope

- Buffer/lifetime and import-layer checks: 14 passed, including metric persistence and cleanup.
- Generic/GPT-OSS single-GPU regression suite: 92 passed; six distributed cases initially skipped
  because they require two GPUs. All six then passed in the targeted two-GPU follow-up, including
  retained-hook and executor cases. Total targeted test count: **112 passed**.
- Native Qwen reference: PP1/EP8/CP2 on eight GPUs and PP2/EP2/CP2 on four GPUs, S128, three steps
  each, passed loss/gradient and updated canonical parameter checks. The fixture's existing BF16
  noise-floor exclusions remain in place. Final-step forward/backward replay counts are asserted.
- Full Qwen S8192 real-routing loss-envelope comparison: passed, as quantified above.
- Ruff checks and syntax checks passed; the final formatting-only changes preserve the tested AST.

This is not a claim of new GPT-OSS output lending, FP8 output lending, packed-sequence qualification,
or multi-node performance qualification. GPT-OSS tests cover regression of its existing backend.

## Conclusion and recommendation

There is no fundamental requirement to copy a CUDA graph's outputs. This implementation demonstrates
safe direct forward output reuse and direct backward saved-input access under the scheduler's
lifetime contract. It also demonstrates why that change alone is insufficient: the conservative
arena policy consumes nearly all remaining memory, while backward outputs and other inputs still
produce substantial uncovered copy work.

Do not promote this candidate. If work continues, the next bounded design should combine safe
shared-arena reuse based on DualPipeV's actual capture/replay order with gradient-output ownership
through the asynchronous EP consumer. The former addresses the memory cost introduced here; the
latter targets part of the remaining 86 ms backward-output copy category, which barely overlaps
other work in this trace. Removing its accumulator copy-back component additionally requires
stable gradient-buffer addressing compatible with FSDP. This profile does not isolate the timing
of each tensor inside that grouped-copy kernel. Retest against graphs disabled after each change. Merely enabling a shared
pool, reducing the live-buffer bound, or reporting savings versus another graph prototype would
not qualify the design.

This follows Megatron/Transformer Engine's ownership principle for forward outputs, but does not
match its complete schedule-aware graph-pool and backward-consumer integration. Borrowing outputs
is feasible; achieving both safe reuse and low memory overhead is the remaining integration work.

## Artifacts and reproducibility

All cluster launches used steps inside array **3006977**; no jobs were submitted. The full suite
ran on numeric job **3073368**, step **41**, node **fs-mbz-gpu-298**. It completed and released the
node. The benchmark, profile and validation source snapshots are immutable in the artifact root.

Local artifact root:
`/home/ruihangl/Workspace/agent-dev/.agent-dev/worktrees/5/workspace/cuda-graphs/output-leases-20261116`

Cluster artifact root:
`/mnt/weka/home/hao.kang/pith-train/cuda-graphs-overhead/workspace/cuda-graphs/output-leases-20261116`

The common prefix is `qwen-pp1-ep8-cp2-s8192-a03-suite-try01`. Open
`<prefix>-profile-off-node0.nsys-rep` and `<prefix>-profile-on-node0.nsys-rep` for the paired GUI
comparison. SQLite exports, matched stage data, decoded timelines, overlap summaries and allocator
summaries are beside them. `run_suite.sh`, `train_driver.py`, `profile_driver.py`,
`summarize_suite.py`, `analyze_allocator.py` and `build_evidence.py` record the executed method.
The timeline and interval parsers are the preserved `analyze_pith_profile.py` in the sibling
`schedule-integration-20261113` directory and `analyze_overlap.py` in `residual-overlap-20261116`.

