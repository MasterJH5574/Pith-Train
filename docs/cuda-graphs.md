# CUDA graphs for training

## Consolidated investigation — 2026-11-14

**Research result: graph capture works for Qwen3 and GPT-OSS, but a general throughput improvement
has not been achieved. Do not enable the new schedule-buffer strategy in production on the basis
of this branch.** This section supersedes earlier candidate-selection conclusions below. The
main review branch retains the existing backend; a separate research branch preserves the schedule
experiments as individually reviewable commits. Inclusion there is not performance promotion.
The machine-readable companion, [cuda-graphs-evidence-20261114.json](cuda-graphs-evidence-20261114.json),
records the measured results, immutable source identities and trace provenance.
Review the existing backend on [agent-dev/task-4](https://github.com/MasterJH5574/Pith-Train/tree/agent-dev/task-4)
and the schedule experiments on [agent-dev/task-4-schedule-research](https://github.com/MasterJH5574/Pith-Train/tree/agent-dev/task-4-schedule-research).
The research series separates initial schedule binding (`7a9ca40`), FIFO reuse (`3585e2e`),
non-owning scratch outputs (`1d85bdd`) and observer guards (`e9067cd`). The common replay-lifetime
and captured-transfer changes are `c02c701`.

### Completed throughput sweep

All rows use H200, BF16 and micro-batch size 1. TPS is global tokens/second. Each arm has two
independent runs; entries are medians of run-level throughput over steady steps 6–15. Peak allocated
memory is the largest measured graph-on allocation, per rank. OOM attempts are excluded from paired
comparisons; the table contains the completed adjusted configurations.

| Model | Nodes | PP/EP/CP | Sequence | Global batch | Graph off TPS | Graph on TPS | Change | On peak GiB |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-OSS-20B | 1 | 1/8/1 | 4096 | 128 | 144,112 | 143,410 | -0.49% | 76.68 |
| GPT-OSS-20B | 1 | 1/4/1 | 4096 | 128 | 140,562 | 140,214 | -0.25% | 90.05 |
| GPT-OSS-20B | 1 | 2/4/1 | 4096 | 128 | 127,543 | 126,786 | -0.59% | 72.67 |
| GPT-OSS-20B | 1 | 1/8/1 | 8192 | 128 | 148,529 | 148,351 | -0.12% | 100.58 |
| Qwen3-30B-A3B | 1 | 1/8/2 | 4096 | 64 | 58,058 | 79,359 | +36.69% | 84.06 |
| Qwen3-30B-A3B | 1 | 1/4/2 | 4096 | 64 | 58,933 | 77,845 | +32.09% | 102.75 |
| Qwen3-30B-A3B | 1 | 2/4/1 | 4096 | 128 | 109,652 | 109,126 | -0.48% | 104.30 |
| Qwen3-30B-A3B | 1 | 1/8/2 | 8192 | 64 | 92,273 | 89,279 | -3.24% | 96.86 |
| Qwen3-30B-A3B | 1 | 1/8/2 | 8192 | 128 | 94,057 | 90,716 | -3.55% | 96.86 |
| Qwen3-30B-A3B | 1 | 1/8/4 | 16384 | 64 | 71,602 | 70,254 | -1.88% | 96.87 |
| GPT-OSS-20B | 2 | 2/8/1 | 8192 | 32 | 241,683 | 240,655 | -0.43% | 65.71 |
| Qwen3-30B-A3B | 2 | 2/8/2 | 8192 | 64 | 169,845 | 166,007 | -2.26% | 63.24 |
| Qwen3-30B-A3B | 2 | 2/8/2 | 8192 | 128 | 176,290 | 171,476 | -2.73% | 63.24 |
| Qwen3-30B-A3B | 2 | 2/8/4 | 16384 | 128 | 138,085 | 135,528 | -1.85% | 63.25 |

CP1 and CP>1 do **not** differ only in attention: CP1 uses experimental ownership/indirect
transfers (plus a benchmark helper for native publication in GPT); CP>1 uses shared scratch and
copies escaping activations and current operands. Compare graph on/off within a row, not backend
performance across rows. Both S4096 winners use the same CP2 scratch backend as the S8192 regression,
so switching backends cannot explain that sequence-length reversal. The final S8192 regression is
about 3%, not 30%. The two-node regressions are about 2–3%; scale alone is not the explanatory variable.

The percentage winners are also not necessarily the best absolute training mesh. The earlier Qwen
CP1 graph-off S4096 observation was about 108k TPS, above the CP2 winners' 78–79k TPS; its graph-on
partner did not complete, so it is not a qualified paired row. Configuration selection and the
benefit of graph capture are separate decisions.

### Whole-step evidence and the sequence-length reversal

The original matched Qwen PP1/EP8/CP2, global-batch-64 comparison fixes rank 0, training step 6,
layer/stage and micro-batch occurrence. Every GPU operation is correlated back to its CPU launch.
The summaries and timeline viewer retain whole-step, stage and individual-kernel views of the
same traces. Unprofiled timing is used for throughput conclusions.

| Sequence | Profiled step off/on | GPU inactive time off/on | Added graph input/output copy activity |
| --- | --- | --- | --- |
| 4096 | 5952.6 / 4249.4 ms | 1900.4 / 653.6 ms | 108.65 / 65.38 ms |
| 8192 | 6270.9 / 6154.9 ms | 685.4 / 677.9 ms | 177.98 / 100.47 ms |

At S4096, substantial host submission gaps disappear. At S8192, the longer GPU work already hides
much more host work; the remaining launch benefit is small, while copying scales with tensors.
This supports a headroom-dependent explanation, not a universal claim that CPU submission never
matters. Graph replay counts, stage counts, source identity and all-rank capture/fallback counters
match expected execution counts and show no evidence of skipped work in the speedup runs. This is evidence against a skipped-work artifact,
not a substitute for long real-routing correctness validation.

Profiling perturbs the result: the S8192 profile itself is slightly faster with graphs, whereas
unprofiled comparisons consistently lose roughly 2–3%. Therefore the 278 ms of copy activity is
**not** a proven additive explanation of the unprofiled slowdown. CPU dispatch waits can grow
when submission becomes faster and catches up with the GPU; those waits are not another independent
cost to add. GPU activity, CPU issuance and exposed critical-path time are kept separate.
In that same S8192 profile, compute union increases about 24.9 ms and copy activity adds 278.5 ms,
but communication-only time drops about 413.3 ms and idle time drops 7.5 ms; other memory activity
adds about 1.5 ms. These disjoint changes explain why this particular instrumented step is 116 ms
faster even though the unprofiled runs regress. They prevent attributing the whole unprofiled
regression to copies from an incompatible local measurement.

For S8192, the shared backend moves about 428 GiB per rank per step through graph boundary copies.
The captured ABI includes four FP32 projection-gradient accumulators totaling 72 MiB per backward;
copy-in plus copy-back over 768 backward replays accounts for 108 GiB of that traffic. This is a
byte-volume calculation, not an attribution of 108 GiB to a separately measured latency.

### Capture coverage and the stage-5 experiment

The original backend captures static stage 1 forward/backward: normalization, attention, router
scores/top-k and applicable load-balance computation. CP ring-attention communication is inside
that capture. GPT full/sliding attention, sinks and projection biases are covered by its separate
compiled variants. Dynamic dispatch metadata, EP communication, routed expert stage 3, deferred
expert weight gradients, FSDP orchestration and the optimizer remain outside.

For the original Qwen CP2 rank-0 profiles, graph computation covers 46.1% of forward and 49.6% of
backward layer-compute time at S4096; at S8192 it covers 53.1% and 56.2%. These denominators exclude
communication, graph copies, embedding/head and optimizer work. **Routed expert stage 3 is 0%
captured** in both directions. At S8192 it contributes approximately 418 ms forward and 1181 ms
backward in this profile. Router work is fused into stage 1, so there is no honest standalone
percentage for all “MoE work” without defining a finer attribution scheme.

Stage 5 is static enough to capture. We implemented a Qwen BF16 CP>1 pilot that captures the
existing fused stage-5/next-stage-1 callable, with bounded, schedule-owned combine inputs.
Capturability was not the blocker. The expanded AOT boundary exports a large backward gradient
and saved hidden tensor: at S8192 the initial prototype adds about 93.9 GiB of output traffic per
rank/step, principally 83.4 GiB of stage-5 input gradients plus 10.4 GiB of saved hidden states.
It also loses throughput; increased copies and reuse dependencies are candidate contributors.
The original attention boundary was reasonable; expanding it
without integrating the output and backward consumers makes the interface more expensive.

### Schedule-aware reuse: implementation, ablations and limits

The scheduler bounds live combine buffers using each chunk's layer count and DualPipeV pipeline
position. Each acquired buffer has a lease; backward retirement records a compute-stream event;
reuse on the communication stream waits for that event. The pool asserts that it drains after the
step. Graph variants bind the actual combine slot directly, avoiding an input staging copy.

The first-free policy immediately reused a just-retired backward slot, adding a forward-combine
dependency on that backward. The matched initial-prototype trace was 188.9 ms slower: compute union
fell 120.2 ms, copy activity rose 32.9 ms, communication-only time rose 133.8 ms, idle time rose
143.4 ms, and other memory activity fell about 0.9 ms. These disjoint categories reconcile the
profiled step delta; the ordering analysis and subsequent ablation support the dependency hypothesis.
They do not establish that every extra idle millisecond has the same cause.

FIFO retirement plus one spare slot recovered 1.6% throughput relative to that prototype, with
identical copy volumes, but remained 0.9% behind the existing graph backend. The spare tensor was
128 MiB; **total allocation grew by 8.1 GiB**, because more live graph variants retained their
scratch output allocations. Counting the spare tensor alone had understated the overhead.

We then adopted the relevant Transformer Engine ownership idea: keep non-owning views of temporary
outputs whose allocation belongs to the matching private graph pool. Inputs and external allocations
retain normal ownership; aliases and offsets are preserved, the graph keeps the pool alive, and
all escaping outputs are copied before another replay may overwrite scratch. This reduced peak
allocation from 121.8 to 100.2 GiB, with essentially unchanged throughput. The existing backend used
96.9 GiB. Saving memory did not turn the prototype into a performance winner.

Late hooks/callbacks can retain a supposedly private buffer. The final revision rejects saved-tensor
observers throughout scheduled execution, temporarily permits only the compiler's internal setup,
and rechecks model methods and callbacks at consumers. Only registered, unchanged force-balance
callbacks qualify. Unsupported callbacks present at step entry use the ordinary path; unsupported
callbacks added after scheduled execution begins raise before consuming pooled storage. This is deliberately a bounded
research integration, not a general observer-safe zero-copy API.

Final-source fresh comparisons, one run per arm on the same node, steps 6–15 maximum-rank medians:

| Sequence | Backend | Step seconds | TPS | Change vs existing graphs | Peak allocated / reserved GiB |
| ---: | --- | ---: | ---: | ---: | --- |
| 4096 | Graphs off | 4.4533 | 58,866 | -25.71% | 83.71 / 101.53 |
| 4096 | Existing graphs | 3.3082 | 79,240 | +0.00% | 84.07 / 101.75 |
| 4096 | Schedule prototype a05 | 3.3752 | 77,667 | -1.99% | 85.80 / 100.14 |
| 8192 | Graphs off | 5.7082 | 91,849 | +3.27% | 96.34 / 115.89 |
| 8192 | Existing graphs | 5.8948 | 88,941 | +0.00% | 96.86 / 116.75 |
| 8192 | Schedule prototype a05 | 5.9529 | 88,073 | -0.98% | 100.19 / 117.13 |

These final runs are a qualification screen, not repeated acceptance evidence. No production
performance promotion follows from a small single-run difference.

### Megatron comparison: measured copies and integration differences

We inspected and instrumented the installed implementations, rather than extrapolating from
Megatron's end-to-end speedup. The matched-mesh captures use Qwen, one eight-H200 node,
PP2/EP4/CP1, S4096, MBS1, global batch 64, real routing and step 6. They retain different corpora,
software stacks, kernels and pipeline layer partitions (Pith 24/24, Megatron 25/23), so absolute
cross-framework throughput is not a controlled ranking. This Pith core profile excludes the
sweep-only native publication helper; its CPU timings must not be assigned to those helper rows.

| Per-rank, per-step measurement | Pith core ownership/indirect backend | Megatron / Transformer Engine |
| --- | --- | --- |
| Replay-boundary device bytes | 39.33 GiB | 25.16–27.34 GiB |
| GPU replay-copy activity | 25.11–26.28 ms | 17.54–19.13 ms |
| CPU copy/publication NVTX spans | 70.09–89.86 ms | 63.84–78.48 ms |
| Additional pointer-table upload | 768 uploads, 1.5 MiB; 2.61–2.75 ms GPU | Not this mechanism |
| Pith weight-mirror refresh | 0.856 GiB; about 0.69–0.73 ms more D2D activity than off | Different parameter-storage integration |

Every Pith replay has exactly one captured input-copy and one output-copy kernel; the GPU
correlations match 384 forward plus 384 backward replays/rank. Table uploads match replay counts.
The graph-on minus graph-off D2D byte difference exactly equals the weight-mirror allocation on
every rank. Megatron's copy counts/bytes likewise match its instrumentation; no parameter-gradient
clone calls were recorded. CPU spans include instrumentation and are not interchangeable with GPU
time. Pith's public-output allocation and other wrapper checks are not all inside publication spans.

**The hypothesis that Pith's CPU buffer-copy time must be vastly larger than Megatron's is not
supported for this CP1 backend.** Issuance/publication spans are comparable. Pith copies more bytes,
but the extra GPU transfer activity is measured in milliseconds, not seconds. The earlier roughly
270 ms Pith copy result came from the CP2 shared backend and a different per-rank workload; it must
not be directly compared with Megatron's 18 ms as if the implementation and configuration matched.

In the matched CP1 rank-0 profile, Pith's step grows 3080.0 → 3161.8 ms. GPU inactive time grows
358.7 → 495.2 ms, while compute union is approximately unchanged and communication-only time falls.
The main-stream gap with the next launch not yet submitted grows 326.9 → 473.8 ms. Thus the instrumented owned path loses host-submission time; extra device copying alone does not
explain the 81.8 ms profiled delta. Graph-on has additional NVTX instrumentation, so this is a
measured location of lost time in the trace, not yet an attribution of the unprofiled regression
to one Python function or proof that all of this time is production wrapper overhead.

Megatron's earlier unprofiled comparisons improved roughly 40–44%. Its instrumented rank-0
profile has much larger idle/submission headroom (about 6.93 → 2.93 seconds of GPU inactivity),
whereas Pith already has compiled stages and comparatively little headroom in this configuration.
The profiler perturbs Megatron substantially; these numbers establish the different timeline shape,
not an unprofiled 4-second saving or an exact accounting of the 40% gain.

Megatron captures attention, router and MoE preprocessing with Transformer Engine graph helpers.
Its schedule-aware storage keeps saved tensors alive until their consumers and releases allocation
ownership when safe; returned outputs can detach graph-backed storage instead of being copied.
Pith's FSDP reallocations, deferred gradients and DualPipeV consumers require different lifetime
contracts. We aligned ownership slots, direct saved inputs, stable weight mirrors, captured
transfers and temporary-pool ownership where those contracts were proven. We did not replace
DualPipeV or the expert stack with Megatron's. Matching the API or capture region alone cannot
supply identical memory lifetimes or identical remaining host-launch headroom.

### Validation, disposition and next work

Final research source a05 passed 113 single-GPU checks. Six distributed checks skipped on that
one-GPU allocation were then run separately on two H200s and all passed (119 checks in total).
Coverage includes replay/mutation/alias lifetimes, public outputs, current weights, graph transfers,
GPT full/sliding attention with sinks/biases, real and replayed routing, and distributed global
load-balance gradients. The new schedule-buffer/observer unit run also passed all 11 checks.

Cold-cache eight-layer native reference comparisons passed for Qwen PP1/EP8/CP2, Qwen PP2/EP4/CP2,
and GPT PP1/EP8/CP1, at sequence 128/BF16. These cover forward/backward and manual parameter mutation;
they are not a 32-step full-model optimizer acceptance run. Final S4096/S8192 throughput screens
use the full Qwen model and force-balanced routing. Stable capture/fallback counts and drained
buffer pools were checked over steady steps. The auxiliary-stream test synchronizes before graph
reset; it does not prove teardown with outstanding replay. No SM100 runtime validation was done.

Ruff lint and format checks passed for changed Python files. An independent read-only review found
and prompted the late-observer fix, then found no remaining concrete blocker for the bounded
research branch. One intermediate revision failed import layering and was corrected by moving the
callback registry into contexts; this was not a numerical failure. The CP1 matched profile logged
allocator OOM/reclaim attempts during startup but completed on all ranks, so its startup memory
headroom remains a concern.

We did not spend another long real-routing acceptance matrix on a candidate that failed the
performance screen. Historical correctness runs below must not be read as final-a05 full-model
qualification. The broader objective of repeatable speedups without correctness regressions remains
unmet; passing these bounded checks is not a claim of universal correctness or production readiness.

The new matched-mesh Megatron comparison is Qwen-only; it does not establish the cause of GPT's
small throughput differences.

The implementation uses generic replay machinery for Qwen and GPT-OSS, not a Qwen-only graph
runtime. GPT remains CP1-only, matching its existing model restriction. DeepSeek V2-Lite has a
source-level portability assessment below, not runtime support: dense-layer output lifetimes,
shared-expert gradients, MLA and its CP ring need their own contracts/tests. No fundamental
impossibility has been established for DeepSeek or for a future faster graph integration.

The performance objective is **not met broadly**. The strong S4096 CP2 gains are repeatable and
consistent with the same profiles' launch gaps; the other completed meshes show near-neutral
performance or modest regressions. The new schedule integration demonstrates feasible bounded
reuse and fixes a real memory-retention problem, but has not justified replacing the faster
existing backend. Do not hardcode graph use universally on this evidence.

Stop the broad sweeps and do not keep tuning the rejected stage-5 boundary. If development resumes,
the next bounded experiment should eliminate a measured boundary cost by integrating both its
producer and its backward consumer, then prove that the saved latency reaches the critical path.
For CP1, first isolate the newly measured host submission gap in the same matched trace; for CP2,
prioritize saved tensors and FP32 gradient accumulator copies rather than adding more capture.
Require repeated unprofiled gains, base/base/candidate real-routing optimizer validation, and
startup plus steady allocated/reserved memory before promotion. Another general graph API rewrite
or a wider configuration sweep is not warranted by these findings.

## Implementation reference and earlier experiment records

Set `cfg.training.cuda_graphs = True` to opt into CUDA graph replay. The default is `False`;
use separate processes with otherwise identical configurations for comparisons. The implementation
uses the pinned PyTorch 2.13 Inductor/AOTAutograd interfaces.

Qwen3 MoE captures its compiled stage 1 and its compiled backward on every layer.
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

BF16 CP1 DualPipeV training additionally uses explicit layer and training-step ownership. Each
layer refreshes detached weight mirrors after its FSDP all-gather once per step; repeated addresses
or tensor versions never authorize skipping that refresh. Owned state stays on its initial compute
stream; foreign-stream calls execute ordinarily without refreshing mirrors or preparing owned
graphs. A foreign backward records its actual input allocations on the consuming stream before
the callable can clear arguments or replace tensor storage. Retirement also records an event before
private graph storage can be reused; that event alone does not protect retained caller allocations.
Backward writes the actual retained
FP32 `main_grad`, preserving the original in-place accumulation order. FSDP zeros it at the first
backward and reduces it once at step end; a forward-only step does not reduce old gradients again.

Each owned compiled forward variant lazily reserves at most `2 * PP` saved-activation slots per
layer and output-lifetime mode, with a corresponding backward graph per slot. There are at most
three modes: fresh public outputs, executor-authorized activation borrowing, and borrowing with
a retained private producer input. AOT's private saved-tensor tail may borrow
graph storage. With EP greater than one, built-in executor calls may also borrow the normalized
stage-1 activation: dispatch materializes separate token storage before returning. This requires
unchanged model methods and consumers, no observing hooks or tensor modes, and a saved tensor
through which backward can identify the slot. Direct model calls retain fresh outputs. Residuals,
routing results, tracked loss values, input aliases, and intermediate bases used to reconstruct
public aliases keep ordinary escaping storage. In particular, a stage-5 callback may retain a
residual after the corresponding backward and slot reuse. Installed saved-tensor hooks use the
copying path. Slots retire after the whole layer backward succeeds; cross-stream fallback establishes
an event before reuse, and exceptions leave outstanding slots quarantined. Each forward slot
has its own capture pool: sharing pools would let an earlier graph's temporary overwrite a later
graph's live saved tensor. Compatible input scratch can still be shared. Backward reads supported
private saved views directly; mutable, negative, conjugated, and unmatched views retain copying.
There are at most 12 compiled variants per owner, and no cache keyed by step or routing pattern.

At an adjacent built-in aggregation/stage-1 boundary, a fresh unexposed input may additionally
retain its forward staging copy in the slot. Backward reads that copy directly, removing a
second transfer of the same input. Ordinary and boundary inputs remain copied. Producer class
and instance methods, operand types, modes, fresh storage and the consumer's observers are
checked explicitly; a router callback that replaces the compute entry invalidates authorization
for that invocation even if the replacement later restores it. AOT still receives the original
saved input alias and performs its normal version checks. The retained scratch is never shared
with another slot or with ordinary calls, and public residuals remain fresh. This costs one
additional hidden-state buffer per participating forward slot; it is bounded by the same pool
limit and has not yet demonstrated a throughput benefit.

A further experimental BF16 CP1 path captures the remaining dense transfers around computation.
It publishes current source and fresh destination addresses through a device table, using at most
128 event-protected pinned upload buffers per replay stream (256 int64 entries each). Dynamic
CUDA tensors record that stream; input/weight freshness, output lifetimes, mutation versions and
logical-view fallbacks remain explicit. Output addresses are bound after capture, and the planning
capture never executes stateful computation. The kernels follow the CuTe SM90/TIRx SM100 operator
boundary. The latest candidate passes 121 focused H200 checks, including stream ordering, pending
upload reuse, transfers above 8 GiB and native metadata-guard fallback, plus import/lint/format
checks. Eight-layer GPT and Qwen BF16 native checks pass (two forward/backward steps and manual
weight mutation, not optimizer training). Initial full-model, one-node PP2/EP4/CP1 BF16 S4096
MBS1/GBS64 timing pairs still lose throughput: GPT is 1.28% slower and Qwen 1.86% slower than
graphs disabled. These are single candidate-selection pairs, not repeated acceptance evidence.
Qwen's steady peak reserved memory reaches 134.30 GiB, leaving only about 189 MiB of sampled
physical free memory; short-run stability does not establish headroom or an unlimited-run plateau.
The implementation remains unqualified for performance, and the SM100 port is uncompiled and
untested.

Eligible concrete inputs use the pinned PyTorch native metadata checker. It retains metadata,
not tensor storage; observing modes and stricter native misses use the original Python signature
comparison. An outer TorchFunction scope restores state if a native metadata query throws.
Storage, writable-alias, direct-save, owner/epoch, current-weight and stream guards remain live.
Expected metadata for graph-owned saves is cached; caller views and retained producer aliases
are checked on each invocation. Reduced helper cost has not established a training speedup.

This ownership path is experimental: the tests below do not establish a throughput benefit.
FP8, CP greater than one, and standalone compiled calls retain the shared copying path. In that
path, each compiled forward or backward owns one reusable scratch set and a private CUDA memory pool.
There are at most 12 such sets per process, shared across layers using the same compiled callable.
Every call refreshes current inputs, including weights, FP8 operands and accumulated gradients.
Under AOT's no-grad execution, aligned dense copies with mixed dtypes share one foreach launch
through physical int32 views. These views preserve bits and version counters; only the static graph
side is cached. Current offsets, pointer alignment and logical conjugate/negative flags are checked
before each copy. Homogeneous copies, unsupported layouts/alignment, logical views and grad-enabled
calls keep the grouped dtype/layout path. Strided slices stay separate in that fallback. Cached
views refer to existing storage and are released with the graphs; they add no tensor storage.
Mutations are copied back to the current caller's storage. Input aliases are returned against that
storage; other outputs, including saved activations, are copied into ordinary allocations. Those
allocations live until the corresponding micro-batch's backward, just as with graphs disabled.
When AOT returns the same output view more than once, those returns share one escaping allocation
within that invocation. Storage identity, offset, shape, strides, dtype, device and logical
conjugate/negative flags must match; distinct replays still receive fresh escaping storage.
Distinct view objects keep independent shape/stride metadata, while true repeated objects retain
their identity within the invocation.
This shared copying path has no graph or persistent activation set per micro-batch, optimizer
step, or routing pattern. The owned path instead has the fixed slot bound described above.

This ownership is necessary because FSDP frees and reallocates unsharded parameters and `main_grad`
between steps. In particular, a forward may pass through `main_grad` while it has zero-byte storage;
the graph does not read it until backward, after FSDP has allocated and zeroed it.
Scratch placeholders that are neither read nor mutated retain their metadata and distinct storage
identity but release their backing allocation after preparation. Writable inputs retain storage.

GPT-OSS captures its existing `forward_stage1_compute` and backward: input norm, full or sliding
attention (including learnable sinks and projection biases), residual addition, and post-attention
norm, followed by router projection, top-k, softmax and optional load-balance operations. Route
replay lookup, load-balance tracking, dispatch preparation and aggregation remain outside capture.
Global-batch load balancing saves each forward's fraction vector for backward, so later live
micro-batches and counter resets cannot change its gradient coefficients. Full and
sliding attention specialize separately; graph counts depend on the compiled variants rather than
a model-specific constant. GPT-OSS retains its existing CP=1 restriction.

The pinned compiler also needs a first-launch metadata repair: an Inductor C++ wrapper can reuse
a Triton autotuner while its descriptive kernel name points to another binary with a different
argument layout. The lazy resolution helper runs the kernel once and republishes metadata from
that actual launcher before C++ loads it. A compilation future is resolved once; if restoring an
autotuner rebuilt its candidates but retained a cached fast launcher, initialization selects a
candidate again before deriving its binary metadata. This process-local repair does not change the installed
environment or add Python work to steady graph replay. Compile-time autotuning is outside its scope.
Compiled stage setup enables Dynamo's module-hook guards, which the pinned default otherwise
omits. The step-level guard-skip optimization is disabled while router hooks are installed, so
hook addition/removal and changes to their captured Python state remain observable.

Model families other than Qwen3 MoE and GPT-OSS, and packed micro-batches, are rejected when graphs
are requested. GPT-OSS-20B and GPT-OSS-120B share this implementation; testing one does not validate
the other's memory, correctness, or throughput. Inference
uses the ordinary compiled path. An instance captures only its first eligible shape/stride/scalar
signature on one compute stream; subsequent mismatches, writable input aliases, and calls beyond
the instance limit run normally. Capture errors are raised, rather than hiding a potentially
partially executed collective behind fallback.
Supported training keeps dense shapes and graph/fallback decisions consistent across ranks.
Rank-asymmetric changes around captured collectives are unsupported. With Qwen context parallelism,
the ring-attention communication inside stage 1 is captured as well; EP dispatch and pipeline
communication remain outside it.

Graphs release automatically before the framework's process-group teardown. Hosts that destroy
their own process groups must first call `pithtrain.pipeline.cuda_graphs.release_graphs()` on every
rank. NCCL requires this release order when graphs retain collective operations; see
[NVIDIA's teardown guidance](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/process-hang.html).

`pithtrain.pipeline.cuda_graphs.graph_stats()` reports captured instance count, forward/backward
replays, ordinary calls, capture time, owned instances, slot/live-save counts, direct saved-input
replays, actual executor-output slot/replay counts, and weight-mirror bytes/refreshes. Peak allocated **and reserved** CUDA memory should be
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

## GPT-OSS-20B follow-up

The shared comparison driver accepts `--model gpt-oss-20b`, selecting the GPT-OSS configuration
and GPT-OSS-tokenized DCLM corpus. Its Qwen defaults are unchanged. For the full 24-layer model,
the bounded comparison uses one eight-H200 node, PP2/EP4/CP1, BF16, sequence 4096, MBS1/GBS64:

```bash
OMP_NUM_THREADS=8 torchrun --standalone --nproc-per-node=8 \
    benchmarks/pretraining/qwen3-30b-a3b/cuda_graphs.py --model gpt-oss-20b \
    --pp 2 --ep 4 --cp 1 --seq 4096 --mbs 1 --gbs 64 --tag gptoss-base0
```

Use distinct tags for the second graph-disabled baseline and the graph-enabled arm (`--graphs`).
Correctness uses 32 real-routing steps and the loss-envelope method above. A matching timing pair
adds `--performance` to both arms (12 force-balanced steps); compare synchronized max-rank medians
after capture/tuning, and repeat whole runs before claiming a gain. No released checkpoint is
needed. Verify at least 2048 corpus samples before launch.

The small native reference uses the existing eight-layer/8192-vocabulary GPT-OSS fixture and
sequence 256, so the 128-token sliding window has an effect. Run both `FP8=0` and `FP8=1`:

```bash
CUDA_GRAPHS=1 FP8=0 SEQUENCE_LENGTH=256 torchrun --standalone --nproc-per-node=4 \
    tests/test_dualpipev.py --pp-size 2 --ep-size 2 --cp-size 1 \
    --model examples/pretrain_lm/gpt-oss-20b/config.json
```

It compares complete outputs, losses and parameter gradients, reports attention sink/bias
gradients without skipping small values, and checks the actual per-rank replay counts. Its two
forward/backward steps and subsequent manual weight mutation check canonical export; they are
not optimizer-step tests. `tests/test_gpt_oss_cuda_graphs.py` separately compares four SGD updates
with changing inputs, full/sliding attention, sink/bias gradients and six live forward calls.
The existing shared-backend tests cover aliases, mutable inputs, replaced weight storage and
`main_grad`. Neither these small fixtures nor a full-model BF16 run establishes GPT-OSS-120B or
full-model FP8 correctness/performance.

Both four-H200 native checks pass at this fixture, without geometry changes beyond the test's
existing eight-layer/8192-vocabulary reduction:

| Precision | Largest output normalized difference | Largest parameter-gradient normalized difference | Graphs per rank | Forward / backward replays per rank |
| --- | --- | --- | --- | --- |
| BF16 | 0.00002228 | 0.000783 | 4 | 80 / 80 on all four ranks |
| FP8 | 0.00055231 | 0.005321 | 4 | 40 / 40 on ranks 0–1; 120 / 120 on ranks 2–3 |

FP8 uses GPT-OSS's existing edge-light layer placement; the unequal replay counts match the
number of layers hosted on each rank. All captured attention sinks and projection biases are
compared. The native test retains its existing noise-floor skip for tiny non-attention,
non-expert gradients (including router bias and post-attention norm weights); the focused test
skips no static-stage gradients.

The full-model BF16 off/off/on comparison passes over all 32 real-routing steps:

| Metric | Off/off mean absolute delta | Off/on mean absolute delta | Ratio |
| --- | --- | --- | --- |
| Cross-entropy | 0.00248125 | 0.00300625 | 1.212 |
| Load balance | 0.00252109 | 0.00475275 | 1.885 |

Both ratios are below the prescribed 3× threshold. An allocation change placed the first baseline
on a different H200 node from the other two arms; source, environment, configuration and corpus
identities match across all three arms. Every graph-enabled rank captures four graphs at step 2,
retains four through step 32, and records 5952 forward plus 5952 backward replays: 192 of each per step after
warmup. These runs use AdamW updates and fresh weights, unlike the native export tests.

| Real-routing arm | Startup peak allocated / reserved GiB | Steps 6–32 peak allocated / reserved GiB | Minimum steady sampled physical free GiB |
| --- | --- | --- | --- |
| Off, first baseline | 66.472 / 102.127 | 66.887 / 83.732 | 50.755 |
| Off, second baseline | 66.449 / 82.350 | 66.888 / 83.287 | 51.202 |
| On | 68.336 / 83.932 | 68.775 / 87.232 | 47.257 |

Values are maxima over all eight ranks; startup includes setup, tuning and steps 1–5. Physical
free memory is sampled at synchronized boundaries, not at allocation peaks. End-of-step allocated
bytes are constant on every rank over steps 6–32. Maximum per-rank reservation growth is
4.725 / 3.305 / 3.301 GiB for the two baselines and graph arm, respectively. No allocation-failure
or OOM messages were logged on any rank. These finite runs establish neither long-run convergence
nor an allocator plateau.

Two complete force-balanced timing pairs use the same full-model configuration, 12 steps per
arm, on the same H200 node. Max-rank synchronized whole-step medians over steps 6–12 show a small
repeatable **slowdown**, with no workload adjustment:

| Pair | Off → on seconds | Off / on sample SD seconds | Step-time change | Throughput change |
| --- | --- | --- | --- | --- |
| 1 | 2.190692 → 2.247856 | 0.010671 / 0.006520 | +2.61% | −2.54% |
| 2 | 2.176307 → 2.247366 | 0.015791 / 0.004644 | +3.27% | −3.16% |

The generic comparison script's printed step times exclude only step 1; those are secondary
measurements. The table uses synchronized whole steps after tuning/capture. Each graph-enabled
timing run ends with four graphs and 2112 forward/2112 backward replays on every rank.

| Timing memory, ranges across the two pairs | Off allocated / reserved GiB | On allocated / reserved GiB | Off → on minimum sampled physical free GiB |
| --- | --- | --- | --- |
| Startup | 65.939 / 77.180 | 67.817 / 79.512–80.023 | 57.310 → 54.466–54.978 |
| Steps 6–12 | 65.929 / 77.180 | 67.806–67.808 / 79.512–80.023 | 57.310 → 54.466–54.978 |

End-of-step allocated and reserved bytes are flat on every rank throughout both timing windows;
there are no logged allocation failures. Setup takes 27.50–28.67 s, first-step compilation/warmup
60.19–71.24 s, and summed tuning calls 26.49–26.73 s across these four runs. Maximum summed
graph-capture time per rank is 0.0353 / 0.0362 s in the two graph arms; their complete step 2 takes
2.565 / 2.962 s. These startup observations use existing caches and do not establish a cold-start
benefit. This coverage is GPT-OSS-20B on H200 with PyTorch 2.13.0+cu130, FA4 4.0.0b28 and
QuACK 0.6.5; it does not establish a benefit for GPT-OSS or results for the untested configurations
listed above. Graphs remain opt-in.

## Replay allocation follow-up

Profiling the shared backend found repeated residual outputs and unread forward `main_grad`
placeholders. The backend now copies each exact repeated output view once and releases backing
storage for placeholders that are neither read nor mutated. Distinct view objects keep independent
metadata, and every invocation still receives fresh escaping storage. Current weights, inputs and
accumulated gradients are refreshed on every replay; pointer equality is not a freshness check.

The following diagnostics use H200, BF16, fresh weights, force-balanced routing, five warmup
steps and one Nsight Systems step with graph-node detail. The original backend is
`bbc7dd069175b490ed3f2f98e1c546d87a29a9b6`; the revised training-source SHA-256 is
`6a8e2f7c4f8a468c708725f006e13b8970f05a1704f218aacd896c735dca1bd4`.

| Model, GPUs, PP/EP/CP, S, MBS/GBS | Original → revised copy GPU ms/rank | Output-copy reduction ms/rank |
| --- | --- | --- |
| GPT-OSS-20B, 8, 2/4/1, 4096, 1/64 | 85.425–88.133 → 83.424–86.318 | 1.638–1.809 |
| Qwen3-30B-A3B, 8, 2/4/1, 4096, 1/64 | 177.467–181.823 → 175.529–179.784 | 2.347–2.592 |
| Qwen3-30B-A3B, 16, 2/8/1, 4096, 2/512 | 579.299–600.541 → 563.273–589.754 | 9.161–10.222 |

These are summed GPU activities, not critical-path savings or throughput measurements. The
original host signature/storage checks consume 14.975–16.047, 36.035–38.293 and 76.028–78.736 ms
per rank respectively. Most input copies and host checks remain. Major attention and GEMM launch
counts match the graph-disabled path; no extra launches were found in those families. Communication
overlap varies by rank and does not establish a uniform communication-delay explanation. Replacing
grouped copies with individual copies did not reduce GPU copy time in the isolated GPT diagnostic
and increased host enqueue cost, so the grouped-copy implementation is retained.

All 27 focused checks pass, including repeated-output identity, distinct-view metadata mutation,
non-dense offsets, multiple retained replays, negative/conjugate values, sink/bias gradients and
optimizer updates. Final-source native references also pass for GPT BF16/FP8 at PP2/EP2/CP1,
S256 on four GPUs; Qwen BF16/FP8 at PP2/EP4/CP1, S128 on eight GPUs; and Qwen BF16 at
PP2/EP8/CP2, S128 on sixteen GPUs. They use eight layers, MBS3 and 20 local chunks; GPT retains
its 8192-vocabulary fixture, while Qwen uses 151936. These native checks use two F/B steps and
manual weight mutation for canonical export, with the existing tolerances and noise-floor skips.
They do not replace full-model optimizer training or performance measurements.

The allocation cleanup is **not an established throughput fix**. Its 32-step real-routing
comparisons pass on one H200 node at PP2/EP4/CP1, BF16, S4096, MBS1/GBS64: cross-entropy/load-balance
envelope ratios are 0.827/0.738 for GPT-OSS-20B and 0.793/0.687 for Qwen3-30B-A3B. The two-node
correctness comparisons and the seven-configuration repeated timing matrix remain incomplete.
Earlier allocation-prototype timing still showed approximately 3% lower GPT throughput than
graphs disabled. Reduced scratch memory and successful replay do not establish a speedup.

A subsequent BF16 experiment computed weight-gradient contributions in scratch and added them
to the current accumulator, avoiding its input copy. Single unprofiled diagnostic pairs at the
same one-node configuration measured 2.169555 → 2.252499 seconds for GPT and 2.558142 → 2.690930
seconds for Qwen (graphs disabled → experiment): throughput fell 3.68% and 4.93%, respectively.
These are max-rank medians over steps 6–12 of 12-step force-balanced runs, with matching source,
inputs and physical GPUs within each pair. They are not repeated performance acceptance evidence.
That experiment is excluded from this implementation; the backend retains ordinary gradient
accumulation and all input/output ownership safeguards. The remaining slowdown is unresolved.

## Bounded ownership experiment

The BF16 CP1 ownership path described above was tested at training-source SHA-256
`dd3b7be9b4b4de36972347fcc9511907d85cf9b7cb0337ba745c805549c2ccc4`.
It is **not a demonstrated throughput fix**. Forty focused tests pass, including optimizer
updates, replaced weights/accumulators, concurrent live forwards, private saved views,
public/intermediate aliases, hooks, stream fallback, and failed-backward quarantine. Ruff,
formatting, import-layer and FA4 memoization checks pass.

Both BF16 native references pass: GPT-OSS-20B on four H200s at PP2/EP2/CP1, S256, and Qwen3-30B-A3B
on eight H200s at PP2/EP4/CP1, S128. Both use eight layers, MBS3 and 20 chunks; GPT uses the existing
8192-token vocabulary fixture and Qwen uses 151936. Maximum parameter-gradient differences are
0.000749 and 0.001807 respectively. GPT output and sink/bias checks pass. Every rank captures
20 graphs, uses ten forward slots, records 80 forward/backward/direct-saved-input replays, and
ends with no live saved tensors. These remain two-F/B/manual-weight-mutation checks, not
optimizer-step training comparisons.

Initial full-model timing uses one eight-H200 node, PP2/EP4/CP1, BF16, S4096, MBS1/GBS64,
12 force-balanced steps, and the synchronized max-rank median over steps 6–12. Each enabled/
disabled pair uses identical source, corpus, configuration, packages and physical GPUs.

| Model | Disabled → owned replay seconds | Throughput change | Graphs / forward slots per rank |
| --- | --- | --- | --- |
| GPT-OSS-20B | 2.169768 → 2.194690 | −1.14% | 60 / 30 |
| Qwen3-30B-A3B | 2.558101 → 2.601955 | −1.69% | 120 / 60 |

These are single candidate-selection pairs; no repeatable gain is claimed. Every rank keeps
constant graph/slot counts and zero live saves at step boundaries. Final replay counts are
2112 F/B/direct for GPT and 4224 for Qwen, with one mirror refresh per layer per step.

| Model | Steady allocated GiB, disabled → replay | Reserved GiB, disabled → replay | Minimum sampled free GiB, disabled → replay |
| --- | --- | --- | --- |
| GPT-OSS-20B | 65.929 → 69.304 | 77.184 → 82.957 | 57.306 → 51.524 |
| Qwen3-30B-A3B | 91.446 → 96.560 | 113.510 → 119.830 | 21.042 → 14.690 |

Startup allocated maxima are 65.939 → 69.313 GiB for GPT and 91.446 → 96.560 GiB for Qwen;
startup reservation maxima equal the table. Maximum summed capture time is 0.452/0.562 seconds
for GPT/Qwen, excluded from timing. No rank logs an allocation failure or shows end-of-step
allocated/reserved growth over the seven-step timing window. This does not establish a
real-routing allocator plateau or long-run memory safety.

A matching GPT step-six trace measures 26.894–27.282 ms/rank of remaining copy-kernel activity,
versus 85.425–88.133 ms in the historical original trace. Actual per-instance replay deltas give
34.12994 GiB/rank of logical payload: 21.46875 GiB into scratch and 12.66119 GiB out. These are
payload and GPU-activity measurements, not DRAM traffic or additive wall-time savings. The four
large backward inputs are the original layer input, a saved public residual, and two tangents.
Nested dispatch/leaf CPU ranges are deduplicated in the revised analysis. Profile instrumentation
changes the on/off timing relationship; only unprofiled runs can support throughput claims.

Two deliberately unsafe fixed-workload diagnostics do not validate implementation changes.
Borrowing public outputs saves 10.268 ms against its diagnostic control but violates retained
output/hook lifetimes. Omitting the remaining activation/tangent copies atop that mechanism saves
another 13.242 ms in a separate matched experiment, while using stale values and changing losses.
Those effects must not be added or treated as correct training speedups. Bypassing all replay
guards on the ownership backend changes 2.196050 to 2.196154 seconds, with no measured benefit.
Production retains the guards, current input copies, and fresh public outputs.

For this earlier ownership experiment, public-save shadow mapping and per-slot input retention
were unimplemented hypotheses. The later retained-input path described above still has no
demonstrated throughput gain. A shadow mapping must establish immutability and preserve alias/mutation
semantics; a pointer or version alone does not detect every external write. Eliminating incoming
tangent or fresh-output transfers further requires a producer/consumer buffer-lifetime contract
or validated dynamic operand rebinding. Neither capability is established by these experiments.
The current source has not completed the four full-model real-routing comparisons, all five
native configurations, or the seven repeated timing configurations required for the performance
follow-up. Earlier-source FP8/CP and full-model correctness results do not validate this ownership
path. Throughput acceptance remains unmet; no final performance-qualified change is established.

### Later transfer and host-batching screen

A later task-only prototype combines indirect forward/backward operands with native metadata and
pointer-publication batching. The production source was
`a36313198b59b01a11c0403023d298b0c5e368949dfd60f8e5e82eaf1700e29a`;
compiler/native helper injections were recorded separately and are not adopted production code.
On one eight-H200 node at PP2/EP4/CP1 BF16, S4096 MBS1/GBS64, each arm ran 25 force-balanced
steps with identical post-warmup GC policy. Maximum-rank synchronized medians use steps 6–25.

| Model | Disabled seconds | Indirect F/B seconds | Indirect F/B + host batching seconds | Combined throughput vs disabled |
| --- | ---: | ---: | ---: | ---: |
| GPT-OSS-20B | 2.106438346 | 2.113323366 | 2.110077239 | −0.172453% |
| Qwen3-30B-A3B | 2.485667735 | 2.506811552 | 2.507393310 | −0.866461% |

These single screening triplets establish no repeatable gain. Every measured combined replay
uses the intended native checks/publication, with no recapture or fallback growth. End-of-step
allocated/reserved memory is flat over this finite window; no long-run plateau is claimed.
The prototype passes 34 focused cases, ten observer/version/stream/global-load-balance cases,
and the two BF16 eight-layer native fixtures above (two F/B steps plus manual weight mutation,
not full-model optimizer correctness). Passing those checks does not qualify its performance.
The combined prototype is rejected; the full performance acceptance matrix remains incomplete.
Host spans and GPU-copy activity do not identify an additive, recoverable wall-time budget.

## DeepSeek V2-Lite portability assessment

DeepSeek remains rejected by the setup allowlist. Source inspection has not established a
fundamental capture failure for dense CP1 training; it does not establish runtime safety either.
Its existing `forward_stage1_compute` is a candidate for the same explicit decorator integration.
No pipeline schedule change or model registry is needed.

| Source and entry | Concrete difference and required check |
| --- | --- |
| `models/deepseek_v2.py`, `DeepSeekV2DecoderLayer.forward_stage1_compute` | The first layer is dense (`first_k_dense_replace=1`); it returns before shared experts and routing. Later MoE layers compute shared experts and router results inside stage 1. Warm and count both static variants. The dense MLP and routed experts stay in stage 3. |
| `models/deepseek_v2.py`, `DeepSeekV2DecoderLayer.forward_stage1`; `pipeline/execution.py`, `stage1_f` | Dense layers return `routing=None`, bypassing EP dispatch's materialization even when EP is greater than one. Their normalized output feeds the dense MLP directly. Keep that output in ordinary escaping storage until its separate consumer/lifetime contract is established; Qwen/GPT's dispatch-output borrowing rule is insufficient. |
| `models/deepseek_v2.py`, `DeepSeekV2DecoderLayer.forward_stage5` | The existing comment records that merging aggregation with the next stage 1 changes forward bits. Retain this separate boundary. |
| `models/deepseek_v2.py`, `DeepSeekV2Attention.forward` | CP1 decompresses MLA K/V before FA4. CP>1 calls the separate MLA ring, passing `kv_b_proj.weight` directly and optional quantized operands. Qwen's ring validation does not cover this path. |
| `operators/fp8_swiglu_mlp.py`, `fp8_swiglu_mlp` and its backward | Shared experts add saved gate/up activations and transposed FP8 operands to capture. The fused path requires a multiple of 128 rows. Check all shared-projection gradients, current weights and saved-activation lifetimes for several outstanding micro-batches. |
| `operators/ring_attention.py`, `mla_zigzag_ring_fwd`, `mla_zigzag_ring_bwd` | Shape/rank-controlled loops issue batched P2P operations and wait for completion. No tensor-value extraction was found in these paths. Native capture must still verify collective ordering, buffer reuse and warmup of every FA4 signature. |
| `operators/ring_attention.py`, `mla_zigzag_ring_setup_context` | Backward holds four nondifferentiable FP8 operands on its context, alongside saved BF16 tensors. Check that AOT exposes the operands needed for replay and that updates refresh them. Nondifferentiability does not prove safe storage lifetime. |
| `modules/wgrad_accumulation.py`, `_attach_main_grads` | MLA ring backward recomputes decompression and returns a direct FP32 `kv_b` weight gradient. FSDP merges it with `main_grad` before freeing storage. Check gradient magnitude as well as direction to detect missing accumulation or CP reduction. |

The setup allowlist and ordinary stage-1 decorator are current integration gaps. Packed input is
an existing hard restriction of graph-enabled DualPipeV; DeepSeek additionally rejects packed
CP>1 because its MLA ring has no document boundaries. Shared-expert capture, MLA ring capture,
FP8 backward lifetimes, scratch memory, and throughput are **untested risks**, not demonstrated
failures. They are not covered by GPT-OSS results.

The smallest next validation is one actual-geometry dense layer and one MoE layer at CP1,
sequence 256/MBS1, first BF16 then FP8. Compare ordinary and captured outputs, input gradients and
all shared-expert gradients over live micro-batches and replaced/updated weights, with bounded
graph counts. Start with ordinary escaping public outputs, without the experimental executor
output or retained-producer-input loans. Then use the eight-layer PP2/EP2/CP1 native fixture for
FSDP storage lifetimes and shared-expert accumulation. Any later borrowing must separately prove
the dense and MoE producer/consumer contracts, including observing hooks and method replacement.
Before enabling CP, add a separate two-GPU CP2 MLA replay check at sequence 512/MBS1 (256 local
tokens and 128-token half-blocks), using the unsharded dense and FP8 references in
`tests/operators/test_ring_attention.py`. No DeepSeek model CUDA-graph run or performance claim accompanies
this assessment.
