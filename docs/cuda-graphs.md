# CUDA graphs for Qwen3 training

Set `cfg.training.cuda_graphs = True` to opt into CUDA graph replay. The default is `False`;
use separate processes with otherwise identical configurations for comparisons. The implementation
uses the pinned PyTorch 2.13 Inductor/AOTAutograd interfaces.

The captured regions are Qwen3 MoE's compiled stage 1 and its compiled backward on every layer.
This includes norms (with the existing selective recomputation),
attention and router scores/top-k. Global-batch load-balance
collectives inside stage 1 are part of its graph. Expert routing decisions remain data-dependent.
Dispatch metadata, expert computation, expert and pipeline communication, deferred expert weight
gradients, FSDP hooks/collectives, embeddings, loss, and optimizer updates execute normally.
Stage 5 keeps its ordinary compiled path: its aggregation does not justify copying the large
expert-output buffers around a CUDA graph. The fused stage 5/stage 1 entry retains its FSDP scheduling
and calls ordinary compiled aggregation followed by captured stage 1. With graphs disabled or
grad mode off, it uses the original merged compiled forward/backward.

The first complete step warms the compiled regions and prepares scratch inputs. Pending captures
run at the start of a later step, before the pipeline issues communication: capture synchronizes
the device and must not run in the middle of the overlapped schedule. Capture records operations
without an extra execution of stateful load-balance updates. Exclude every step whose capture count
increases, as well as JIT and kernel tuning, from steady-state throughput measurements.

Each compiled forward or backward owns one reusable scratch set and one private CUDA memory pool.
There are at most 12 such sets per process, shared across layers using the same compiled callable.
Every call refreshes current inputs, including weights, FP8 operands and accumulated gradients.
Copies are grouped by dtype and dense-layout eligibility. Strided slices stay separate so they
do not force otherwise eligible tensors onto PyTorch's per-tensor foreach copy path. The groups
contain indices only; they add no persistent tensor buffers.
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
Override batch sizes with `--mbs` and `--gbs`. The driver also works with multi-node `torchrun`;
every evidence row records the global rank, world size, hostname, and Slurm job/step when present.

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

## Original single-node validation

The following historical measurements belong to the original implementation, later committed as
`18a73db5182c7129f7577fb702136f23853ce136`. It captured stage 1 and merged stage 5/1 (four graphs per
rank). The follow-up below narrows capture to stage 1 and groups copies; these historical numbers
do not validate that follow-up.

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

### Original measured results

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
multi-node execution, and rank-asymmetric capture decisions are outside this original validation.

## Longer sequences and two-node follow-up

The follow-up fixes two measured costs without removing replay safeguards. PyTorch's foreach
copy fast path requires compatible dtypes and dense layouts (see its
[fast-path restrictions](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/ForeachUtils.h)). A single mixed list of BF16, FP32,
integer and strided tensors fell back to many individual copies. Grouping by dtype and dense-layout
eligibility reduced an isolated, frequency-weighted copy diagnostic from 35,763 CUDA copy events
to 5,355 and from 269.0 ms to 224.0 ms. It replayed foreach-copy layouts observed in one training
step, including graph I/O and other foreach copies; these are diagnostic totals, not measured
whole-step kernel counts after the fix or an end-to-end speedup. Stage 5's large expert-output
tensors also increased copy traffic and persistent storage. Keeping aggregation outside capture
reduces those costs while retaining
captured stage-1 forward/backward on every layer. It applies uniformly across shapes and meshes.

The final training source is a patch on `18a73db5182c7129f7577fb702136f23853ce136`, identified by
SHA-256 `8244132de43938beb075fd3a2a266936bfdfca0df36894d54b7ae9123cb5e82e` using the recipe above.
The intermediate copy-grouping-only source is
`a147f077290ae15aa238bf2a11ff9581c96e708b7ebb38a6649518b49216663e`; its measurements remain separate.
The public driver, identical across comparison arms, has SHA-256
`5048846791715e61f3732586a46b255c26a348904e34bebdce5d791e45aaf750`.
The graph-disabled merged computation and call path are unchanged by both follow-up fixes.
Some first performance baselines use the intermediate `a147f077` source with graphs disabled;
the disabled methods and call path were checked for AST identity before reusing them. All new
32-step correctness triples use the final `8244132d` source with the flag off, off, then on.
Repeated performance baselines also use that final source. All arms use the same driver.

Two-node runs use 16 H200 GPUs with one eight-rank `torchrun` per node. PP is the outer mesh
axis, so PP=2 spans nodes; EP and CP groups stay within a node. All comparisons use the same
precision, batch sizes, recomputation and seed within each pair, and run sequentially. The
BF16 path recomputes the two hidden-size RMSNorm outputs; the FP8 path keeps the existing
non-recomputed behavior. Attention, GEMMs and custom operators retain their saved tensors.
The
[benchmark reference configuration](https://github.com/mlc-ai/pith-train-bench/tree/pith-train/scripts/qwen3_30b_a3b/seq4096_bf16/h200_2n8g/pp2_dp8_cp1_ep8)
supplies the PP2/EP8/CP1, S4096, MBS2/GBS512 shape. The reference-shaped timing pair uses the
seeded fresh-weight DCLM fixture above, not that repository's released-checkpoint/Megatron-data
training recipe. The main matrix uses GBS128 so its 32-step S8192 correctness run fits the
existing corpus shard; the GBS512 pair remains separate. MBS1/CP2 and S16384/CP4 cases explicitly
change both arms' configuration and are not directly comparable to the MBS2 matrix.

On already allocated compute, run the same command on each node, with a shared unique rendezvous
endpoint on the first node:

```bash
PYTHONPATH="$PWD" OMP_NUM_THREADS=8 torchrun --nnodes=2 --nproc-per-node=8 \
    --rdzv-backend=c10d --rdzv-id="$RUN_TAG" --rdzv-endpoint="$FIRST_NODE:29474" \
    /path/to/cuda_graphs.py --pp 2 --ep 8 --cp 1 --seq 4096 --mbs 2 --gbs 128 \
    --tag "$RUN_TAG"
```

Use three distinct tags for real-routing `base0`, `base1`, and feature (`--graphs`); add
`--performance` to both arms for throughput and repeat whole runs. The reference-shaped pair
uses `--gbs 512`; the EP4 case uses `--ep 4`; CP2 uses `--cp 2` at S4096 and S8192; the MBS1
CP2 diagnostic uses `--mbs 1`; the long-context case uses `--cp 4 --seq 16384 --mbs 1 --gbs 64`.
Single-node retests use `--nnodes=1`, MBS1/GBS64 and the original meshes. Keep distinct attempt tags
for retries, since the evidence files append. Check every global rank before aggregating results.

The final boundary passed 21 focused checks on H200, including exact outputs/gradients over successive
updates with multiple outstanding micro-batches, ordinary aggregation backward crossing capture,
and mixed dtype/layout copies. Native reference checks also passed at eight GPUs in BF16 and FP8
(PP2/EP4/CP1), and 16 GPUs in BF16 (PP2/EP8/CP2). Each native run checks two forward/backward
steps' losses and gradients, then canonical exports before and after manual weight mutation,
using the repository tolerances. The full-model 32-step runs exercise AdamW updates. Each rank reported
two graphs with 80 forward and 80 backward replays. These eight-layer/S128 native fixtures
complement full-model loss-envelope validation; they do not replace it.

The historical completed runs used 16 Qwen3 token shards. That shared corpus became unavailable
before the remaining single-node correctness baselines and performance repeats. Recovery uses a
separate task-owned copy of one Qwen3-tokenized DCLM shard (74,750,913 uint32 tokens; file SHA-256
`1377c75aec957ec03da2f038b6b237d3eca026d8b45d79adfd45ce547c669022`). New runs verify the ordered
shard names and content hashes before launch. The dataset identity, including dtype and loader
selection, is `a2ff9c70c8df769591fe26c09b3559b7b1e3a633b0c6eca67a10ad4b812de09e`.
Historical token bytes were not hashed and cannot be asserted identical to the recovered corpus.
Each new correctness triple and timing pair uses the recovered corpus in **all** arms; the four
previously completed single-node feature runs were rerun for that reason. Completed two-node
correctness triples remain separate. Independent timing pairs across the two corpora are labelled
as such; they are not repeat measurements of identical input samples.

All 11 selected full-model configurations pass the finite 32-step correctness check:
ten pass the original two-baseline envelope, and S4096 on one node passes the prescribed
third-baseline investigation described below. Its original INVESTIGATE verdict is retained.
The table below identifies each configuration; the reference-shaped GBS512 case is timing-only,
with real-routing correctness measured at GBS128. This does not establish real-routing correctness
or memory headroom at GBS512. Each final-source feature run uses two graphs per rank. The replay
counts are actual counts on every rank during 32 training steps with changing micro-batch inputs
and AdamW updates. They cover every stage-1 forward/backward call after the first warmup step.

Whole-run repeats preserve the gains in one-node CP2 (+48.92% to +51.77% throughput),
one-node FP8 (+16.14% to +16.74%), and two-node S4096/CP2/MBS1
(+26.37% to +26.81%). These are within-configuration gains; the two-node MBS2/CP1
baseline is faster in absolute tokens/sec than the MBS1/CP2 candidate. Larger micro-batches
and CP1 remain sensitive to graph copy overhead. Keep the default disabled and measure the
actual workload before opting in. The tables retain slowdowns and report the repeated runs
separately; within-run SD is not a confidence interval based on independent observations.

The three earlier BF16 regressions were retested against both the original graph implementation
and graphs disabled, with reversed run order in the second comparison. The repair improves
throughput over the original graphs by 4.05–4.91% at PP1/EP8/S2048, 4.30–8.07% at PP2/EP4/S2048,
and 3.77–3.91% at PP2/EP4/S4096. The table below reports each comparison separately.

The PP2/EP4/S2048 BF16 gain is consistent across the two comparisons (+4.25–4.78%).
The PP1/EP8 gain shrinks from 9.04% to 1.14%; the latter is comparable to its within-run spread,
so this is not a reliable speedup recommendation. S4096/CP1 remains slower by 3.30–5.77% despite
improving over the original graphs. The two-node reference-shaped pair remains slower by
4.21–4.41%, and S16384/CP4 by 1.72–2.48%. The other large BF16 cases have only one final pair;
their negative results do not support enabling graphs.

The resumed runs used the same package versions; Slurm selected available nodes from the
authorized allocation. Rank hostnames and step identities are recorded per run. Interrupted,
externally contended, and failed-launch attempts are preserved and excluded; successful retries
have distinct tags. Historical single-node feature runs with startup co-tenancy are also excluded
from the final correctness verdicts and replaced by the matched-corpus runs above.

### S4096 correctness investigation

The original one-node S4096 ratios were CE 2.115 and load balance 3.660: the latter triggered
the correctness skill's 3–5× investigation range. Before observing the third baseline, the
follow-up method was fixed to compare the mean of **all three** baseline-pair mean absolute
deltas with the mean of **all three** feature-to-baseline mean absolute deltas. It uses the same
3× threshold and retains the original result, rather than selecting a favorable baseline pair.
All four runs use identical source, driver, data, mesh, seed, LR schedule and 32-step length.

| Metric | Baseline MAE: 0/1, 0/2, 1/2 | Feature MAE: vs 0, vs 1, vs 2 | Expanded floor / signal | Ratio / verdict |
| --- | --- | --- | --- | --- |
| cross-entropy-loss | 8.125e-05, 0.00010625, 0.0001 | 0.000171875, 0.000140625, 9.0625e-05 | 9.5833333e-05 / 0.000134375 | 1.402 / PASS |
| load-balance-loss | 5.2e-05, 0.0001356875, 0.00010675 | 0.00019034375, 0.00016646875, 0.00010803125 | 9.8145833e-05 / 0.00015494792 | 1.579 / PASS |

Step-1 CE is 12.2760 for all three baselines and 12.2761 for graphs; load-balance values are
1.594292 / 1.594291 / 1.594289 versus 1.594306. These are logged finite-precision losses,
not bitwise equality claims. The independent raw-log audit reproduces both the original and
expanded verdicts. Native gradient/output checks and successive AdamW updates provide separate
coverage; this finite envelope does not establish long-run convergence.

### Configurations

| Case | Nodes / GPUs | PP / EP / CP | Precision | Sequence | MBS / GBS |
| --- | --- | --- | --- | --- | --- |
| 1n-s4096 | 1 / 8 | 2 / 4 / 1 | BF16 | 4096 | 1 / 64 |
| 1n-ep8-s2048 | 1 / 8 | 1 / 8 / 1 | BF16 | 2048 | 1 / 64 |
| 1n-pp2-s2048 | 1 / 8 | 2 / 4 / 1 | BF16 | 2048 | 1 / 64 |
| 2n-reference | 2 / 16 | 2 / 8 / 1 | BF16 | 4096 | 2 / 512 |
| 2n-ep8 | 2 / 16 | 2 / 8 / 1 | BF16 | 4096 | 2 / 128 |
| 2n-ep4 | 2 / 16 | 2 / 4 / 1 | BF16 | 4096 | 2 / 128 |
| 2n-cp2 | 2 / 16 | 2 / 8 / 2 | BF16 | 4096 | 2 / 128 |
| 2n-s8192 | 2 / 16 | 2 / 8 / 2 | BF16 | 8192 | 2 / 128 |
| 1n-cp2 | 1 / 8 | 2 / 4 / 2 | BF16 | 2048 | 1 / 64 |
| 1n-fp8 | 1 / 8 | 2 / 4 / 1 | FP8 | 2048 | 1 / 64 |
| 2n-cp2-mb1 | 2 / 16 | 2 / 8 / 2 | BF16 | 4096 | 1 / 128 |
| 2n-cp4-s16384 | 2 / 16 | 2 / 8 / 4 | BF16 | 16384 | 1 / 64 |

### Correctness

| Case | CE / LB envelope ratio | Verdict | F/B replays per rank, each direction |
| --- | --- | --- | --- |
| 1n-s4096 | 2.12 / 3.66 | INVESTIGATE | 11,904 |
| 1n-s4096, three-baseline follow-up | 1.40 / 1.58 | PASS; original result retained above | same feature run |
| 1n-ep8-s2048 | 1.30 / 0.95 | PASS | 11,904 |
| 1n-pp2-s2048 | 0.88 / 1.06 | PASS | 11,904 |
| 2n-reference | — | Timing shape only; real-routing validation uses GBS128 | — |
| 2n-ep8 | 0.75 / 1.10 | PASS | 5,952 |
| 2n-ep4 | 1.33 / 0.93 | PASS | 5,952 |
| 2n-cp2 | 2.00 / 1.75 | PASS | 11,904 |
| 2n-s8192 | 0.84 / 0.84 | PASS | 11,904 |
| 1n-cp2 | 1.06 / 0.92 | PASS | 23,808 |
| 1n-fp8 | 0.80 / 0.90 | PASS | 11,904 |
| 2n-cp2-mb1 | 0.68 / 0.42 | PASS | 23,808 |
| 2n-cp4-s16384 | 0.91 / 1.13 | PASS | 23,808 |

### Performance

Pair 1 uses the historical corpus; pair 2 uses the separately identified recovered corpus.
Each pair has matched inputs.

| Case | Whole-run pair | Seconds, off → graphs | Tokens/sec, off → graphs | Throughput change | Seconds saved | Within-run SD, off / graphs |
| --- | --- | --- | --- | --- | --- | --- |
| 1n-ep8-s2048 | 1 | 1.852 → 1.698 | 70,789 → 77,189 | +9.04% | +0.154 | 0.0632 / 0.0149 s |
| 1n-ep8-s2048 | 2 | 1.715 → 1.696 | 76,411 → 77,284 | +1.14% | +0.019 | 0.0295 / 0.0221 s |
| 1n-pp2-s2048 | 1 | 1.833 → 1.749 | 71,508 → 74,923 | +4.78% | +0.084 | 0.0282 / 0.0209 s |
| 1n-pp2-s2048 | 2 | 1.776 → 1.704 | 73,804 → 76,939 | +4.25% | +0.072 | 0.0199 / 0.0222 s |
| 1n-s4096 | 1 | 2.642 → 2.732 | 99,212 → 95,940 | -3.30% | -0.090 | 0.0446 / 0.0188 s |
| 1n-s4096 | 2 | 2.543 → 2.699 | 103,078 → 97,127 | -5.77% | -0.156 | 0.0147 / 0.0040 s |
| 1n-cp2 | 1 | 4.436 → 2.979 | 29,548 → 44,002 | +48.92% | +1.457 | 0.0576 / 0.0621 s |
| 1n-cp2 | 2 | 4.364 → 2.875 | 30,034 → 45,582 | +51.77% | +1.489 | 0.0545 / 0.0531 s |
| 1n-fp8 | 1 | 2.375 → 2.035 | 55,179 → 64,416 | +16.74% | +0.341 | 0.0259 / 0.0158 s |
| 1n-fp8 | 2 | 2.348 → 2.021 | 55,833 → 64,845 | +16.14% | +0.326 | 0.0186 / 0.0142 s |
| 2n-reference | 1 | 9.091 → 9.490 | 230,693 → 220,982 | -4.21% | -0.399 | 0.0203 / 0.0132 s |
| 2n-reference | 2 | 9.067 → 9.486 | 231,299 → 221,088 | -4.41% | -0.419 | 0.0114 / 0.0133 s |
| 2n-ep8 | 1 | 2.478 → 2.582 | 211,611 → 203,054 | -4.04% | -0.104 | 0.0163 / 0.0068 s |
| 2n-ep4 | 1 | 2.433 → 2.523 | 215,508 → 207,820 | -3.57% | -0.090 | 0.0167 / 0.0120 s |
| 2n-cp2 | 1 | 2.969 → 3.054 | 176,560 → 171,662 | -2.77% | -0.085 | 0.0379 / 0.0377 s |
| 2n-s8192 | 1 | 5.817 → 5.975 | 180,274 → 175,497 | -2.65% | -0.158 | 0.0562 / 0.0248 s |
| 2n-cp2-mb1 | 1 | 4.561 → 3.597 | 114,938 → 145,757 | +26.81% | +0.964 | 0.0487 / 0.0433 s |
| 2n-cp2-mb1 | 2 | 4.522 → 3.578 | 115,949 → 146,524 | +26.37% | +0.944 | 0.0307 / 0.0214 s |
| 2n-cp4-s16384 | 1 | 7.964 → 8.103 | 131,669 → 129,402 | -1.72% | -0.140 | 0.0661 / 0.0267 s |
| 2n-cp4-s16384 | 2 | 7.805 → 8.004 | 134,338 → 131,006 | -2.48% | -0.199 | 0.0155 / 0.0076 s |

### Repair comparison (original graphs versus final graphs)

| Case | Whole-run pair | Seconds, original → final | Final throughput change | Peak allocated GiB saved | Peak reserved GiB saved |
| --- | --- | --- | --- | --- | --- |
| 1n-ep8-s2048 | 1 | 1.767 → 1.698 | +4.05% | 0.582 | 0.318 |
| 1n-ep8-s2048 | 2 | 1.779 → 1.696 | +4.91% | 0.587 | 0.965 |
| 1n-pp2-s2048 | 1 | 1.825 → 1.749 | +4.30% | 0.582 | 0.684 |
| 1n-pp2-s2048 | 2 | 1.841 → 1.704 | +8.07% | 0.583 | 0.684 |
| 1n-s4096 | 1 | 2.839 → 2.732 | +3.91% | 0.949 | 0.990 |
| 1n-s4096 | 2 | 2.801 → 2.699 | +3.77% | 0.948 | 1.311 |

### Memory (first whole-run pair; maxima across all ranks)

| Case | Phase | Allocated GiB, off → graphs | Added allocated bytes / GiB / % | Reserved GiB, off → graphs | Added reserved bytes / GiB / % | Peak ranks allocated / reserved, off → graphs | Sampled physical free GiB, off → graphs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1n-ep8-s2048 | startup | 83.699 → 84.502 | +862,276,096 / +0.803 / +0.96% | 101.324 → 102.521 | +1,285,554,176 / +1.197 / +1.18% | 0 → 4 / 4 → 0 | 32.686 → 31.491 |
| 1n-ep8-s2048 | steady | 83.699 → 84.502 | +862,276,096 / +0.803 / +0.96% | 101.324 → 102.521 | +1,285,554,176 / +1.197 / +1.18% | 0 → 4 / 4 → 0 | 32.686 → 31.491 |
| 1n-pp2-s2048 | startup | 80.400 → 81.196 | +854,290,432 / +0.796 / +0.99% | 95.650 → 96.709 | +1,136,656,384 / +1.059 / +1.11% | 3 → 3 / 4 → 7 | 38.901 → 37.841 |
| 1n-pp2-s2048 | steady | 80.398 → 81.197 | +858,622,464 / +0.800 / +0.99% | 95.650 → 96.709 | +1,136,656,384 / +1.059 / +1.11% | 3 → 0 / 4 → 7 | 38.901 → 37.841 |
| 1n-s4096 | startup | 91.451 → 92.786 | +1,433,187,840 / +1.335 / +1.46% | 113.463 → 115.367 | +2,044,723,200 / +1.904 / +1.68% | 0 → 1 / 4 → 4 | 21.089 → 19.182 |
| 1n-s4096 | steady | 91.451 → 92.785 | +1,432,594,432 / +1.334 / +1.46% | 113.463 → 115.367 | +2,044,723,200 / +1.904 / +1.68% | 0 → 0 / 4 → 4 | 21.089 → 19.182 |
| 1n-cp2 | startup | 75.137 → 75.654 | +555,565,056 / +0.517 / +0.69% | 88.373 → 89.051 | +727,711,744 / +0.678 / +0.77% | 3 → 0 / 1 → 2 | 45.612 → 44.934 |
| 1n-cp2 | steady | 75.136 → 75.654 | +556,351,488 / +0.518 / +0.69% | 88.373 → 89.051 | +727,711,744 / +0.678 / +0.77% | 1 → 0 / 1 → 2 | 45.612 → 44.934 |
| 1n-fp8 | startup | 82.298 → 83.032 | +788,218,368 / +0.734 / +0.89% | 95.654 → 96.590 | +1,004,535,808 / +0.936 / +0.98% | 1 → 3 / 7 → 5 | 38.897 → 37.960 |
| 1n-fp8 | steady | 82.298 → 83.032 | +788,220,416 / +0.734 / +0.89% | 95.654 → 96.590 | +1,004,535,808 / +0.936 / +0.98% | 3 → 3 / 7 → 5 | 38.897 → 37.960 |
| 2n-reference | startup | 87.970 → 90.406 | +2,616,022,528 / +2.436 / +2.77% | 118.328 → 121.494 | +3,399,483,392 / +3.166 / +2.68% | 0 → 0 / 14 → 14 | 15.395 → 12.231 |
| 2n-reference | steady | 87.970 → 90.406 | +2,616,245,248 / +2.437 / +2.77% | 118.328 → 121.494 | +3,399,483,392 / +3.166 / +2.68% | 0 → 0 / 14 → 14 | 15.395 → 12.231 |
| 2n-ep8 | startup | 87.969 → 90.399 | +2,608,696,320 / +2.430 / +2.76% | 118.609 → 121.492 | +3,095,396,352 / +2.883 / +2.43% | 6 → 0 / 10 → 11 | 15.116 → 12.233 |
| 2n-ep8 | steady | 87.965 → 90.397 | +2,611,449,856 / +2.432 / +2.76% | 118.609 → 121.492 | +3,095,396,352 / +2.883 / +2.43% | 0 → 0 / 10 → 11 | 15.116 → 12.233 |
| 2n-ep4 | startup | 91.824 → 94.253 | +2,608,304,128 / +2.429 / +2.65% | 116.340 → 120.270 | +4,219,469,824 / +3.930 / +3.38% | 0 → 7 / 12 → 10 | 18.761 → 14.829 |
| 2n-ep4 | steady | 91.823 → 94.253 | +2,609,472,512 / +2.430 / +2.65% | 116.340 → 120.270 | +4,219,469,824 / +3.930 / +3.38% | 0 → 7 / 12 → 10 | 18.761 → 14.829 |
| 2n-cp2 | startup | 62.714 → 64.050 | +1,434,782,208 / +1.336 / +2.13% | 77.613 → 79.434 | +1,954,545,664 / +1.820 / +2.35% | 3 → 5 / 12 → 9 | 55.891 → 54.087 |
| 2n-cp2 | steady | 62.715 → 64.050 | +1,434,326,528 / +1.336 / +2.13% | 77.613 → 79.434 | +1,954,545,664 / +1.820 / +2.35% | 5 → 5 / 12 → 9 | 55.891 → 54.087 |
| 2n-s8192 | startup | 87.982 → 90.403 | +2,599,372,800 / +2.421 / +2.75% | 118.936 → 122.219 | +3,525,312,512 / +3.283 / +2.76% | 7 → 6 / 12 → 13 | 14.569 → 11.302 |
| 2n-s8192 | steady | 87.983 → 90.403 | +2,598,305,792 / +2.420 / +2.75% | 118.936 → 122.219 | +3,525,312,512 / +3.283 / +2.76% | 4 → 6 / 12 → 13 | 14.569 → 11.302 |
| 2n-cp2-mb1 | startup | 50.095 → 50.891 | +854,768,128 / +0.796 / +1.59% | 60.959 → 62.090 | +1,214,251,008 / +1.131 / +1.86% | 3 → 5 / 12 → 13 | 72.546 → 71.430 |
| 2n-cp2-mb1 | steady | 50.093 → 50.891 | +856,463,360 / +0.798 / +1.59% | 60.959 → 62.090 | +1,214,251,008 / +1.131 / +1.86% | 4 → 5 / 12 → 13 | 72.546 → 71.430 |
| 2n-cp4-s16384 | startup | 62.726 → 64.066 | +1,438,903,296 / +1.340 / +2.14% | 77.719 → 79.443 | +1,851,785,216 / +1.725 / +2.22% | 7 → 3 / 14 → 13 | 55.786 → 54.075 |
| 2n-cp4-s16384 | steady | 62.726 → 64.065 | +1,437,079,040 / +1.338 / +2.13% | 77.719 → 79.443 | +1,851,785,216 / +1.725 / +2.22% | 7 → 3 / 14 → 13 | 55.786 → 54.075 |

### Startup (first whole-run pair; seconds, maximum rank; cached environment)

| Case | Setup, off / graphs | Step 1, off / graphs | Tuning, off / graphs | Warmup steps 1–5, off / graphs | Graph capture within step 2 |
| --- | --- | --- | --- | --- | --- |
| 1n-ep8-s2048 | 34.14 / 33.67 | 76.00 / 50.65 | 35.98 / 35.51 | 84.38 / 58.72 | 0.016 |
| 1n-pp2-s2048 | 34.82 / 29.74 | 72.20 / 66.68 | 35.82 / 38.98 | 84.95 / 74.39 | 0.014 |
| 1n-s4096 | 30.67 / 34.71 | 66.48 / 63.06 | 41.48 / 38.22 | 83.55 / 74.61 | 0.016 |
| 1n-cp2 | 29.36 / 29.25 | 129.80 / 89.00 | 37.80 / 34.07 | 164.21 / 109.08 | 0.246 |
| 1n-fp8 | 33.20 / 31.64 | 70.96 / 43.04 | 5.47 / 1.90 | 81.14 / 51.65 | 0.016 |
| 2n-reference | 19.58 / 19.86 | 135.02 / 92.94 | 40.08 / 36.20 | 172.42 / 136.71 | 0.015 |
| 2n-ep8 | 19.91 / 21.10 | 88.74 / 87.83 | 36.12 / 35.69 | 99.14 / 98.62 | 0.027 |
| 2n-ep4 | 29.47 / 29.73 | 72.75 / 70.48 | 36.28 / 38.97 | 86.56 / 86.55 | 0.029 |
| 2n-cp2 | 20.14 / 19.95 | 145.45 / 105.33 | 36.43 / 40.50 | 158.41 / 118.42 | 0.021 |
| 2n-s8192 | 21.11 / 23.49 | 123.22 / 91.26 | 36.63 / 36.27 | 151.54 / 121.67 | 0.428 |
| 2n-cp2-mb1 | 19.76 / 19.86 | 134.27 / 89.65 | 39.67 / 36.36 | 169.14 / 123.05 | 0.047 |
| 2n-cp4-s16384 | 19.60 / 20.02 | 144.28 / 101.50 | 38.90 / 40.15 | 177.36 / 137.12 | 0.312 |

### Real-routing memory (steps 6–32)

| Case | Baseline peak allocated / reserved GiB (range over two runs) | Graph peak allocated / reserved GiB | Sampled physical free GiB, baseline range → graphs | Graph end allocated / reserved growth, step 6→32 GiB |
| --- | --- | --- | --- | --- |
| 1n-cp2 | 75.374–75.378 / 88.512–89.020 | 75.894 / 89.340 | 44.966–45.473 → 44.768 | +0.000 / +0.455 |
| 1n-ep8-s2048 | 83.998–84.004 / 104.182–104.416 | 84.811 / 105.051 | 29.595–29.829 → 28.962 | +0.000 / +3.152 |
| 1n-fp8 | 83.805–83.813 / 99.721–99.736 | 84.546 / 100.508 | 34.813–34.829 → 34.042 | +0.000 / +1.137 |
| 1n-pp2-s2048 | 80.773–80.773 / 98.195–98.205 | 81.565 / 99.256 | 36.346–36.356 → 35.294 | +0.000 / +1.273 |
| 1n-s4096 | 92.410–92.414 / 119.598–119.678 | 93.740 / 121.668 | 14.872–14.952 → 12.882 | +0.000 / +3.299 |
| 2n-cp2 | 63.556–63.568 / 98.242–99.572 | 64.926 / 101.182 | 33.948–35.278 → 32.339 | +0.000 / +15.775 |
| 2n-cp2-mb1 | 50.579–50.601 / 65.723–65.908 | 51.395 / 67.324 | 67.612–67.798 → 66.196 | +0.000 / +3.535 |
| 2n-cp4-s16384 | 64.042–64.045 / 94.359–94.945 | 65.392 / 97.387 | 38.575–39.161 → 36.134 | +0.000 / +17.064 |
| 2n-ep4 | 93.950–93.957 / 134.926–134.955 | 96.395 / 135.086 | 0.173–1.011 → 0.013 | +0.000 / +8.443 |
| 2n-ep8 | 88.982–89.033 / 133.688–133.709 | 91.481 / 133.508 | 0.054–0.606 → 0.407 | +0.000 / +12.537 |
| 2n-s8192 | 88.346–88.683 / 133.180–133.461 | 90.869 / 133.465 | 0.475–0.483 → 0.163 | +0.000 / +11.600 |

### Memory lifetime and scope

Startup memory includes setup, both tuning calls and steps 1–5, including capture in step 2.
Steady timing memory covers steps 6–12; real-routing memory covers steps 6–32. Allocated and
reserved maxima can occur on different ranks and steps. The peak-rank column reports each
separately. Physical free memory is sampled at synchronized boundaries, not at allocation peaks.
All GPUs have 139.803 GiB usable capacity. Raw byte deltas accompany GiB and percentages so
rounding does not hide the overhead.

Across all 20 balanced timing pairs, added peak allocation is 0.517–2.437 GiB (0.69–2.77%) and
added reservation is 0.551–3.930 GiB (0.54–3.38%). The explicit graph-capture interval is
0.014–0.428 seconds on the slowest rank; it excludes the much larger first-step compilation
and tuning costs listed above. These are observations in a cached environment, not cold-start
cost predictions. Per-rank JSONL and the evidence index retain both timing pairs' full memory
and startup measurements.

In every completed final graph-enabled performance run, the graph count and end-of-step allocated/reserved
bytes are flat over steps 6–12. The two-node S4096/CP1/MBS2 comparison adds 2.432 GiB allocated
at GBS128 and 2.437 GiB at GBS512: quadrupling the outstanding work does not quadruple graph
scratch. Ordinary saved activations still follow each micro-batch's lifetime. Real-routing
end-of-step live allocation is flat, but allocator reservations grow and recycle in both arms.
The full real-routing table and warning counts below are essential to judging capacity; balanced
timing alone understates this behavior. These finite runs do not prove a long-run reservation
plateau or long-run convergence.

The S4096/MBS2/CP1 real-routing runs completed with very little sampled physical free memory
in both arms; the smallest sample was about 13 MiB. Their live allocation remained below 97 GiB,
leaving allocator-held cache that can be reclaimed. Both arms logged allocation-failure warnings
and then completed all 32 steps. These are recovered allocator failures, not terminal OOMs;
completion does not make these configurations a recommendation with ample physical headroom.

| Case | Logged allocator warnings: base0 / base1 / graphs |
| --- | --- |
| 1n-cp2 | 0 / 0 / 0 |
| 1n-ep8-s2048 | 0 / 0 / 0 |
| 1n-fp8 | 0 / 0 / 0 |
| 1n-pp2-s2048 | 0 / 0 / 0 |
| 1n-s4096 | 0 / 0 / 0 |
| 2n-cp2 | 0 / 0 / 0 |
| 2n-cp2-mb1 | 0 / 0 / 0 |
| 2n-cp4-s16384 | 0 / 0 / 0 |
| 2n-ep4 | 2 / 2 / 5 |
| 2n-ep8 | 4 / 3 / 7 |
| 2n-s8192 | 8 / 6 / 9 |

Warnings count logged allocation-failure messages, not every internal allocator retry.
The third S4096 baseline also completed without a logged allocator warning.

The original PP1/EP1 baseline did not fit, so it remains unvalidated. The measured support is
dense Qwen3-30B-A3B on H200 at the listed meshes and batch sizes. Packed samples, other models,
Blackwell, rank-asymmetric capture decisions, and unbounded training duration remain outside
this validation. Ruff lint/format checks and the focused import-layer checks pass. No kernel
exports, architecture-selection rules, optimizer settings, routing policy, or recomputation
policy changed in this follow-up.
