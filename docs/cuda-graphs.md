# CUDA graphs for Qwen3 training

Set `cfg.training.cuda_graphs = True` to opt into CUDA graph replay. The default is `False`;
use separate processes with otherwise identical configurations for comparisons. The implementation
uses the pinned PyTorch 2.13 Inductor/AOTAutograd interfaces.

The captured regions are Qwen3 MoE's compiled stage 1 and fused stage 5/stage 1, and their compiled
backwards. This includes norms (with the existing selective recomputation),
attention, router scores/top-k, and weighted expert aggregation. Global-batch load-balance
collectives inside stage 1 are part of its graph. Expert routing decisions remain data-dependent.
Dispatch metadata, expert computation, expert and pipeline communication, deferred expert weight
gradients, FSDP hooks/collectives, embeddings, loss, and optimizer updates execute normally.
Standalone stage 5 keeps its ordinary compiled path: its small aggregation does not justify
copying inputs and outputs around a separate CUDA graph.

The first complete step warms the compiled regions and prepares scratch inputs. Pending captures
run at the start of a later step, before the pipeline issues communication: capture synchronizes
the device and must not run in the middle of the overlapped schedule. Capture records operations
without an extra execution of stateful load-balance updates. Exclude every step whose capture count
increases, as well as JIT and kernel tuning, from steady-state throughput measurements.

Each compiled forward or backward owns one reusable scratch set and one private CUDA memory pool.
There are at most 12 such sets per process, shared across layers using the same compiled callable.
Every call refreshes current inputs, including weights, FP8 operands and accumulated gradients.
Mutations are copied back to the current caller's storage. Input aliases are returned against that
storage; other outputs, including saved activations, are copied into ordinary allocations. Those
allocations live until the corresponding micro-batch's backward, just as with graphs disabled.
There is no graph or persistent activation set per micro-batch, optimizer step, or routing pattern.

This ownership is necessary because FSDP frees and reallocates unsharded parameters and `main_grad`
between steps. In particular, a forward may pass through `main_grad` while it has zero-byte storage;
the graph does not read it until backward, after FSDP has allocated and zeroed it.

Other model families and packed micro-batches are rejected when graphs are requested. Inference
uses the ordinary compiled path. An instance captures only its first eligible shape/stride/scalar
signature on one compute stream; subsequent mismatches, writable input aliases, and calls beyond
the instance limit run normally. Capture errors are raised, rather than hiding a potentially
partially executed collective behind fallback.
Supported training keeps dense shapes and graph/fallback decisions consistent across ranks.
Rank-asymmetric changes around captured collectives are unsupported. With context parallelism,
the ring-attention communication inside stage 1 is captured as well; EP dispatch and pipeline
communication remain outside it.

Graphs release automatically before the framework's process-group teardown. Hosts that destroy
their own process groups must first call `pithtrain.pipeline.cuda_graphs.release_graphs()` on every
rank. NCCL requires this release order when graphs retain collective operations; see
[NVIDIA's teardown guidance](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/process-hang.html).

`pithtrain.pipeline.cuda_graphs.graph_stats()` reports captured instance count, forward/backward
replays, ordinary calls, and capture time. Peak allocated **and reserved** CUDA memory should be
measured separately during startup/capture and steady state on every rank. Input/output copies and
private pools have a cost; capture success alone does not establish a throughput benefit.

## Reproducing the comparisons

Use `benchmarks/pretraining/qwen3-30b-a3b/cuda_graphs.py` under an eight-GPU `torchrun` in the
repository environment. It uses seeded fresh weights (`1234`), AdamW, global-batch load balancing,
micro-batch size 1 and global batch size 64. It needs the Qwen3-tokenized DCLM corpus, but no released
checkpoint. The default 32-step correctness run uses real routing and linear LR warmup from
`1e-6` to `1e-5`, following the `validate-correctness` fixture.

Keep a copy of the driver outside the checkouts: the base revision does not contain this new
script. Use the same environment and invoke that copy from each checkout with `PYTHONPATH`
pointing at the selected source. For example, on already allocated compute:

```bash
PYTHONPATH="$PWD" OMP_NUM_THREADS=8 torchrun --standalone --nproc-per-node=8 \
    /path/to/cuda_graphs.py --pp 2 --ep 4 --cp 1 --fp8 --tag base0 > base0.log 2>&1
```

Run the base revision twice (`base0`, `base1`), then the candidate with `--graphs --tag feature`.
Use separate processes and distinct tags: evidence files append, so reusing a tag would mix runs.
The correctness skill's `scripts/compare.py base0.log base1.log feature.log` compares the loss
deltas to the base/base envelope; all three logs must contain all 32 steps.

For performance, add `--performance` to both arms. This selects force-balanced routing,
constant LR `1e-6`, and 12 steps. Repeat whole runs. Use a common steady-state window after all
capture, compilation and tuning; the measurements below use steps 6–12. The performance skill's
log comparator is a useful secondary check, but its default exclusion of only step 1 is
insufficient for graph capture in step 2.

The driver writes per-rank JSONL under `workspace/cuda-graphs/evidence` (override with `--output`).
For end-to-end time, take the maximum `seconds` across ranks for each step, then the median of the
steady-state steps. This includes the final GPU synchronization and training's logging/GC;
the repository's printed step time ends earlier. Tokens/sec is global batch × sequence length
divided by this synchronized time. Report the whole-run medians separately instead of treating
correlated steps as independent runs.

For memory, take the highest `peak_allocated` and `peak_reserved` across all ranks separately
for setup/tuning/capture and steady state. `capacity - peak_reserved` is an allocator gap,
not physical free memory: `free_after_step` measures CUDA's physical free bytes only at the
synchronized sampling boundary. Check both memory and graph count over repeated steps.

## Validation configuration

Validation uses one node with eight H200 GPUs (139.803 GiB usable capacity each), BF16 parameters
and the same batch sizes and recomputation in each enabled/disabled pair. `--fp8` selects the
existing FP8 linear backend. The baseline is `543b1c9bb495e8619b5db1516d85fd511d4cee3f`.
The measurements used a patch on that revision, identified by training-source SHA-256
`d7521f9a19dcf4b5a0daf5e1d441c244f7a277ad4c1fb2f38fb293369429d63c` (sorted relative
`pithtrain/**/*.py` paths, each followed by NUL, file contents, and NUL).

The proposed PP=1/EP=1/CP=1 BF16 starting case does not fit even with graphs disabled: the first
backward failed to allocate another 768 MiB with 135.33 GiB allocated by PyTorch and 436.81 MiB
free on the failing rank. Comparisons therefore start at EP=8; neither arm's workload is reduced
to conceal graph overhead. H200 is the measured target; Blackwell is not validated here.

## Measured results

The measurements below use PyTorch 2.13.0+cu130, CUDA 13.0, FlashAttention-4 4.0.0b28,
QuACK 0.6.5, DeepGEMM 2.6.1 and Transformers 5.22.1. Each comparison uses the same environment.
All timing rows use seven synchronized samples (steps 6–12) per run; two whole-run comparisons
are reported separately for configurations with a demonstrated gain.

| PP / EP / CP | Precision / sequence | Pair | Step seconds, base → graphs | Tokens/sec, base → graphs | Step reduction | Within-run SD, base / graphs |
| --- | --- | --- | --- | --- | --- | --- |
| 1 / 8 / 1 | BF16 / 2048 | 1 | 1.728 → 1.740 | 75,851 → 75,344 | **−0.67%** | 0.0138 / 0.0061 s |
| 2 / 4 / 1 | BF16 / 2048 | 1 | 1.764 → 1.838 | 74,307 → 71,327 | **−4.18%** | 0.0301 / 0.0153 s |
| 2 / 4 / 2 | BF16 / 2048 | 1 | 4.469 → 3.083 | 29,326 → 42,509 | 31.01% | 0.0715 / 0.0415 s |
| 2 / 4 / 2 | BF16 / 2048 | 2 | 4.391 → 2.963 | 29,853 → 44,229 | 32.50% | 0.0342 / 0.0589 s |
| 2 / 4 / 1 | FP8 / 2048 | 1 | 2.306 → 2.070 | 56,845 → 63,331 | 10.24% | 0.0132 / 0.0226 s |
| 2 / 4 / 1 | FP8 / 2048 | 2 | 2.332 → 2.106 | 56,212 → 62,251 | 9.70% | 0.0359 / 0.0258 s |
| 2 / 4 / 1 | BF16 / 4096 | 1 | 2.630 → 2.812 | 99,659 → 93,211 | **−6.92%** | 0.0090 / 0.0081 s |

Thus CP=2 improves throughput by 44.95–48.16%, and FP8 by 10.74–11.41%. These gains exceed
both the within-run timing variation and the variation between the repeated whole runs.
BF16 at CP=1 and sequence 4096 instead takes 6.92% more time (6.47% less throughput).
BF16 at CP=1 and sequence 2048 also has no demonstrated benefit: throughput changes by
−0.67% at PP1/EP8 and −4.01% at PP2/EP4. Keep graphs disabled for these configurations.

The correctness comparisons contain all 32 steps on all eight ranks. Ratios below are the
feature/base loss difference divided by the base/base envelope; the workflow's pass threshold
is 3×. These finite runs check regression, not long-run convergence.

| PP / EP / CP | Precision / sequence | Cross-entropy / load-balance ratio | Verdict |
| --- | --- | --- | --- |
| 1 / 8 / 1 | BF16 / 2048 | 0.59 / 0.46 | PASS |
| 2 / 4 / 1 | BF16 / 2048 | 0.53 / 0.51 | PASS |
| 2 / 4 / 2 | BF16 / 2048 | 0.23 / 0.16 | PASS |
| 2 / 4 / 1 | FP8 / 2048 | 0.46 / 0.65 | PASS |
| 2 / 4 / 1 | BF16 / 4096 | 0.79 / 0.72 | PASS |

Every rank holds four graphs from step 2 onward; final real-routing counts are 23,808 forward
and 23,808 backward replays at CP=2, and 11,904 of each for every other case. Input data and
optimizer weights change throughout these runs.

Memory below is the maximum across all eight ranks, separately for allocated and reserved
bytes. Startup includes setup, tuning, and steps 1–5 (including capture in step 2); steady state
is steps 6–12. The first reported performance pair supplies each row. Deltas compare identical
workloads and include graph scratch buffers, escaped outputs, and private pools.

| PP / EP / CP, precision, sequence | Phase | Allocated GiB, base → graphs | Added GiB / % | Reserved GiB, base → graphs | Added GiB / % |
| --- | --- | --- | --- | --- | --- |
| 1 / 8 / 1, BF16, 2048 | Startup/capture | 83.699 → 85.086 | 1.387 / 1.66% | 101.324 → 102.840 | 1.516 / 1.50% |
| 1 / 8 / 1, BF16, 2048 | Steady | 83.699 → 85.084 | 1.385 / 1.65% | 101.324 → 102.840 | 1.516 / 1.50% |
| 2 / 4 / 1, BF16, 2048 | Startup/capture | 80.401 → 81.779 | 1.379 / 1.71% | 95.650 → 97.393 | 1.742 / 1.82% |
| 2 / 4 / 1, BF16, 2048 | Steady | 80.399 → 81.781 | 1.381 / 1.72% | 95.650 → 97.393 | 1.742 / 1.82% |
| 2 / 4 / 2, BF16, 2048 | Startup/capture | 75.136 → 76.050 | 0.913 / 1.22% | 88.367 → 89.635 | 1.268 / 1.43% |
| 2 / 4 / 2, BF16, 2048 | Steady | 75.136 → 76.050 | 0.914 / 1.22% | 88.367 → 89.635 | 1.268 / 1.43% |
| 2 / 4 / 1, FP8, 2048 | Startup/capture | 82.298 → 83.556 | 1.258 / 1.53% | 95.650 → 97.207 | 1.557 / 1.63% |
| 2 / 4 / 1, FP8, 2048 | Steady | 82.298 → 83.556 | 1.258 / 1.53% | 95.650 → 97.207 | 1.557 / 1.63% |
| 2 / 4 / 1, BF16, 4096 | Startup/capture | 91.451 → 93.737 | 2.285 / 2.50% | 113.459 → 116.482 | 3.023 / 2.66% |
| 2 / 4 / 1, BF16, 4096 | Steady | 91.449 → 93.735 | 2.286 / 2.50% | 113.459 → 116.482 | 3.023 / 2.66% |

For example, CP=2 adds 981,087,744 allocated bytes and 1,361,051,648 reserved bytes in the
first pair; its repeat adds the same allocated bytes and 1,168,113,664 reserved bytes.
Both FP8 pairs add 1,350,466,048 allocated bytes and 1,671,430,144 reserved bytes.

Real routing has larger peaks because expert token counts vary. The following table uses
steps 6–32; baseline ranges span the two baseline runs. Sampled free memory is the lowest
observed across ranks at synchronized step boundaries, **not** physical free memory at the
allocation peak. Every baseline-fitting workload remains usable with graphs enabled.

| PP / EP / CP, precision, sequence | Baseline allocated / reserved GiB | Graph allocated / reserved GiB | Sampled free GiB, graphs |
| --- | --- | --- | --- |
| 1 / 8 / 1, BF16, 2048 | 83.803–83.818 / 103.736–104.357 | 85.194 / 105.041 | 28.970 |
| 2 / 4 / 1, BF16, 2048 | 80.857–80.864 / 98.441–98.793 | 82.225 / 100.293 | 34.257 |
| 2 / 4 / 2, BF16, 2048 | 75.390–75.392 / 88.691–88.863 | 76.314 / 90.875 | 43.108 |
| 2 / 4 / 1, FP8, 2048 | 83.836–83.841 / 100.596–100.904 | 85.101 / 102.080 | 32.470 |
| 2 / 4 / 1, BF16, 4096 | 92.756–92.757 / 120.098–120.406 | 95.038 / 123.418 | 11.132 |

End-of-step allocated and reserved memory are flat on every graph-enabled rank across these
performance windows. The FP8 baselines add a 2 MiB reservation on four ranks during the window.
With real routing, end-of-step allocated memory and graph count are unchanged from step 6 to
step 32 on every rank. The maximum per-rank reservation growth over that interval is
0.529–2.676 GiB with graphs and 0.529–2.859 GiB across the baseline runs. Dynamic expert shapes
and allocator caching therefore have not reached a complete plateau; there is no observed
growth in live graph buffers. No claim is made about arbitrarily long real-routing runs.

Startup times below are maxima across ranks for the first reported pair. Capture is the sum
of four graph captures on the slowest rank, measured separately within step 2. Setup, first-step
compilation, and tuning depend on cache state; these observed costs are not a startup speedup
claim. All are excluded from steady-state timing.

| PP / EP / CP, precision, sequence | Setup seconds, base → graphs | Step 1 seconds, base → graphs | Tuning seconds, base → graphs | Capture seconds, graphs |
| --- | --- | --- | --- | --- |
| 1 / 8 / 1, BF16, 2048 | 32.93 → 33.41 | 48.13 → 47.87 | 35.37 → 35.62 | 0.0238 |
| 2 / 4 / 1, BF16, 2048 | 29.55 → 29.41 | 66.04 → 65.73 | 38.98 → 37.97 | 0.0291 |
| 2 / 4 / 2, BF16, 2048 | 34.71 → 29.21 | 108.63 → 82.84 | 34.03 → 33.59 | 0.0265 |
| 2 / 4 / 1, FP8, 2048 | 31.06 → 33.47 | 43.02 → 44.25 | 0.79 → 2.43 | 0.0241 |
| 2 / 4 / 1, BF16, 4096 | 29.46 → 29.47 | 88.15 → 63.12 | 35.06 → 38.70 | 0.4110 |

All 20 focused checks pass on one H200:

```bash
pytest tests/test_cuda_graphs.py tests/test_import_layers.py \
    tests/kernels/test_flash_attn_memoize.py -q
```

They cover multiple live micro-batches, fresh input and weight storage,
successive optimizer updates, fp32 gradient accumulation, aliases, fallback, nested compilation's
grad mode, FA4 capture-stream handling, and NCCL teardown. The eight-GPU DualPipeV native reference
also checks outputs, all parameter gradients, and canonical parameters in BF16 and FP8 at
PP2/EP4/CP1, using eight layers and sequence length 128. Every rank reports four graphs and
80 forward plus 80 backward replays in each reference run.

To repeat the native reference, run the following twice, with `FP8=0` and `FP8=1`:

```bash
CUDA_GRAPHS=1 FP8=0 torchrun --standalone --nnodes=1 --nproc-per-node=8 \
    tests/test_dualpipev.py --pp-size 2 --ep-size 4 --cp-size 1 \
    --model examples/pretrain_lm/qwen3-30b-a3b/config.json
```

These use the integration test's existing numerical tolerances (including the FP8 expert
gradient tolerance); the small scratch-replay accumulation test additionally requires exact
agreement between graphs enabled and disabled. Ruff lint and formatting checks also pass.

The measured support is dense Qwen3-30B-A3B on H200 with the configurations above. The original
PP1/EP1 baseline OOM leaves that case unvalidated; other models, packed samples, Blackwell,
multi-node execution, and rank-asymmetric capture decisions are outside this validation.
