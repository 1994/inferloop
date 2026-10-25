# Default serving performance gap — 2026-10-24

Status: open investigation. The initial snapshot below is based on main
`94fde96` after PR #46. Follow-up work adds opt-in diagnostics and experiment
validation; no faster automatic default has been qualified.

The default path improved over the previous default, but it has not demonstrated
competitive performance against vLLM. The objective is one fast automatic default:
users should not need to choose an execution profile, disable vision, shrink
context, or search a configuration matrix to obtain reasonable throughput.

## Continuation audit after integrating main

PR #47 now includes main `62417eb` (merge `19b7df7`). Main changed prefix
fingerprinting and output handling as well as CUDA build boundaries; new
experiments must bind the new release binary rather than reuse old timings.

The [resource audit](artifacts/default-serving-resource-audit-2026-10-24.json)
recomputes the final accepted serving plan from every original prefix0 report
and server log. The first preparation/tuning model load is not the serving load.

| Original pair | Baseline slots / physical KV tokens | Candidate slots / physical KV tokens |
| --- | --- | --- |
| 0 | 4 / 20,512 | 7 / 35,744 |
| 1 | 7 / 38,592 | 4 / 24,576 |
| 2 | 4 / 24,576 | 8 / 40,192 |
| 3 | 7 / 35,072 | 7 / 35,072 |
| 4 | 7 / 35,072 | 7 / 35,072 |

The first three pairs do not isolate attention arithmetic at fixed resources.
The service's `kv_cache.pool_bytes` represents logical slot capacity, not
measured physical GPU allocation. Matching CLI memory fractions and context
limits did not prevent different capture/replan outcomes. The old five-pair
statistical result remains `not_demonstrated`; neither dropping the mismatched
pairs nor attributing their TPOT changes to FA4 is valid.

The benchmark tools now retain the final accepted plan and can require identical
scheduler settings, execution geometry, logical KV capacity, physical KV tokens,
quote and envelope for a declared native experiment. Actual allocation is retained
but not required to match byte for byte. Frozen pair counts are exact, and a
predeclared balanced sequence can use AB/BA/BA/AB/AB/BA instead of the old gate's
hardcoded alternating order. These are measurement changes, not serving
optimizations; old profiles and evidence are unchanged.

The optimization order also needs correction. The historical
[qualification evidence](artifacts/canonical-target-verification-2026-10-24.json)
records that precise-prefix protection recovered 14 of 15 selected known
counterexamples, compared with 2 of 15 before the protection. Removing that
protection must first survive a reject-only counterexample screen. Such a
selected screen cannot qualify quality, even when it finds no regression.
The original complete 1,024-case quality gate remains required for a retained
candidate.

Canonical-prefix attention accounted for about 1.202 of 17.876 seconds of
instrumented GPU graph time. Making only that component four times faster would
save about 0.902 seconds (5.0% of graph time), assuming other work stays fixed.
This is an opportunity estimate, not a serving speedup claim. Token trajectories,
MTP acceptance, admission and batching can change the end-to-end result.

An existing desktop GPU context alone does not establish active interference.
Future experiments on the user's idle desktop must record its known background
processes and activity, keep the resource plan comparable, and stop on environment
or resource-contract failure. The old zero-other-process confirmation remains
blocked with zero samples; it cannot be relabeled as this new experiment.

## Small counterexample screen: prefix0 rejected

A fresh screen on integrated source `0827d67` used one release binary and the
union of all 15 original prefix counterexamples and eight later V11 regressions:
21 unique cases in their original suite order, with fixed batches of 16 then 5.
The original prompt tokens, scorer, EOS and 512-token cap were retained. Both
arms used normal automatic execution/resources with vision enabled.

Baseline completed all 21 cases with 19 correct. On the same first 16 cases,
baseline was correct on 14 and prefix0 on seven: five zero-shot and two eight-shot
gross regressions, no gains. The candidate stopped at that batch boundary under
the preregistered reject-only rule; the last five selected cases were not run.
Four failures reached the 512-token cap without a final answer; three ended at
EOS with a wrong numeric answer. Independent decoding and scoring reproduced
all results.

The two serving captures had exactly the same plan: 12 slots, 54,464 physical KV
tokens, capacity 8192, prompt width 1024, quote 6,181,388,816 bytes, allocated
6,166,990,136 bytes and envelope 6,181,624,556 bytes. All four preparation/serving
captures succeeded on their first attempt. Thus different resident plans do not
explain these observed quality regressions. The idle desktop remained present;
238 startup/execution environment samples were retained without telemetry errors.

Under the frozen gross-regression contract, retain the 2048-position guard and
do not promote prefix0 or launch its six-pair confirmation. This selected screen
is not a representative accuracy estimate, a full 1,024-case result, or a
performance measurement. It shows specific failures and rejects the old
per-item compatibility contract; it cannot by itself rule out FA4 under a
different, prospectively specified task-quality contract.

The [screen evidence](artifacts/default-serving-prefix0-screen-2026-10-24.json)
binds source, binary, selection, all attempted outcomes, raw file hashes and the
unattempted cases. No prior result or gate was overwritten.

## Task accuracy and per-item compatibility are different contracts

The current gate limits old-correct/new-wrong outcomes without offsetting
old-wrong/new-correct gains. Its one-sided CP95 upper bound must stay below 2%
overall and in each 512-case stratum. Five losses in a stratum exceed that bound
even if more than five other answers improve. This tests preservation of the
previous success set as well as task quality. The historical V10 result, for
example, had 968 correct in both arms and 16 losses/16 gains, yet failed the
gross-regression contract.

Upstream projects separate these concerns rather than define one universal
accuracy tolerance:

- [vLLM v0.31.0 GSM8K tests](https://github.com/vllm-project/vllm/blob/v0.31.0/tests/evals/gsm8k/test_gsm8k_correctness.py)
  check aggregate accuracy against a model-specific threshold minus tolerance.
  Its [Qwen3.5 NVFP4/MTP configuration](https://github.com/vllm-project/vllm/blob/v0.31.0/tests/evals/gsm8k/configs/Qwen3.5-397B-A17B-NVFP4-DEP2-MTP.yaml)
  uses 0.88 minus 0.03, with 1,319 questions and a 12,000-token cap. These settings
  cannot be transferred to our model or 512-token contract.
- [SGLang's contribution guide](https://docs.sglang.io/docs/developer_guide/contribution_guide)
  describes its 200-example GSM8K command as a sanity check and warns of
  batching/non-determinism variation. Its
  [decode-cache regression test](https://github.com/sgl-project/sglang/blob/main/test/registered/disaggregation/test_disaggregation_decode_radix_cache.py)
  uses a task-score floor and a net score-drop bound; this is a particular
  feature test, not a project-wide accuracy allowance.
- This does not remove numeric checks:
  [SGLang's nightly precision suite](https://docs.sglang.io/docs/references/nightly_precision_regression)
  also compares intermediate tensors against a baseline, using a default
  relative-difference threshold of 1e-3 in that suite.

Before changing policy, use a small representative selection made independently
of model outcomes and report net accuracy, losses, gains, truncation and output
length separately. Keep the current formal gates and defaults unchanged.
A future task-quality non-inferiority contract would need its margin and
statistics frozen before qualification; it must not rewrite the rejected
counterexample result or select a threshold after observing candidate scores.

## Representative task-quality diagnosis

A separate outcome-independent sample selected 32 zero-shot and 32 eight-shot
cases by a frozen SHA256 ordering of IDs, then retained the original suite order
in four batches of sixteen. Both arms completed all 64 cases without a
score-dependent stopping rule. This is sampled from the previously used corpus,
not an unseen holdout.

| Stratum | Baseline correct | Prefix0 correct | Losses | Gains |
| --- | --- | --- | --- | --- |
| Zero-shot | 29 / 32 | 30 / 32 | 0 | 1 |
| Eight-shot | 30 / 32 | 28 / 32 | 2 | 0 |
| Overall | 59 / 64 | 58 / 64 | 2 | 1 |

The net observed change is minus one correct answer, or -1.5625 percentage
points. Both arms have three length-limited outputs and four missing final
answers; all remain counted. The eight-shot losses include a completed wrong
answer and a new 512-token truncation. The zero-shot gain avoids a baseline
truncation. Output length therefore remains part of task quality, not a reason
to exclude an unfavorable result.

The same release binary, original prompt tokens, scorer, EOS and 512-token cap
were used. Both arms copied the same completed tuning seed and ended with the
same tuning hash. Before/after serving plans were identical: twelve slots,
54,464 target KV tokens, prompt width 1024 and capacity 8192. Both preparation
and serving captures succeeded without replanning. External environment
sampling completed without errors while desktop programs remained open.

This small diagnostic neither proves task-quality equivalence nor establishes a
population-level decline. It does show why the historical-counterexample screen
must not be used as an aggregate accuracy estimate. A formal non-inferiority
margin needs a separate, prospectively specified contract and enough evidence;
the existing gross-regression gate and 2048-position default guard remain
unchanged. Do not expand immediately to the full suite just to seek a passing
score. First establish a worthwhile service-performance effect or diagnose the
specific numerical/output changes.

The [representative-screen evidence](artifacts/default-serving-representative-screen-2026-10-24.json)
retains all 64 paired outcomes, source/build and selection identities, complete
serving-plan readbacks and raw evidence hashes. No performance result is claimed.

## Recorded measurements

RTX 5090 on 192.168.1.45, Qwen3.8-27B-NVFP4. C16 means sixteen requests, each
with 4,096 input tokens and 256 fixed output tokens.

| Study | C16 wall time | Scope |
| --- | --- | --- |
| Historical vLLM | 12.031 s | Text only, memory share 0.88, context 32K, step budget 2048, MTP2 |
| Historical native fast profile | 12.272 s | Matched historical setup; 2.001% more wall time |
| Current automatic native path | 18.534–18.582 s | Vision enabled, equal-budget native before/after study at share 0.85, context 262K, step budget 1024 |
| Current true default preview | 17.206 s | No execution/resource flags, share 0.9; one measured trial, descriptive only |

The arithmetic increase from historical vLLM to the current fixed-budget native
study is 54.0–54.4%. This is a cross-study reference, **not a measured matched
current-native/vLLM gap**. Historical arrivals used a simultaneous barrier;
current arrivals were ordered 10 ms apart. The input artifact hashes differ.
Historical native disabled autotuning and explicitly selected the execution path;
current native uses normal tuning and automatic selection. The current quality
guard keeps canonical F32 attention through the first 2,048 prompt positions.
These differences prevent assigning the entire timing difference to any one cause.

The same-budget native before/after result remains valid: C16 wall time fell
22.90–23.01 to 18.53–18.58 s, short-request wall time 4.967 to 2.642 s, and 32K
TTFT 20.58–20.60 to 5.10–5.12 s. Improvement over the old default does not establish
vLLM parity. The frozen fixed-budget study contains an inherited V11 scope-label
typo; its separate metadata correction binds the actual V12 binary and sources.

## Confirmed observations and unresolved causes

- The historical native run captured sixteen slots and 71,200 target KV tokens.
  Current fixed-budget process C1 captured seven slots and 35,072 KV tokens.
  Seven is an observed funded width under that study's 0.85 budget, not a
  hardcoded default and not a hardware concurrency limit. Usable memory, vision,
  execution storage and external desktop allocations all affect funding.
- Current C1 C16 median request TTFT is about 8.34 s and median TPOT about 23.6 ms.
  Historical vLLM process B1 recorded about 5.48 s and 14.2 ms respectively.
  These are unmatched diagnostic signals, not a causal decomposition.
- Current C1 measured GPU utilization about 98%. This does not establish useful
  compute efficiency or identify which GPU stage dominates.
- A 262K admission ceiling does not mean 262K KV storage is eagerly allocated
  for every request. Pool startup capacity in the current record is 8192.
- Both runtime paths already use BF16 bulk attention and MTP2. Remaining
  performance work cannot be reduced to exposing these flags.
- The cost attributable to reduced pool width, the 1K prompt rung, the F32 quality
  guard, state copies, draft execution and target verification is still unquantified.
  No specific kernel has yet been established as the dominant remaining bottleneck.

The original complete 1,024-case regression suite remains qualified: both arms
968 correct, two regressions and two gains; all original 2% CP95 gross-regression
gates pass. The control corpus was retained from the previous study, not an unseen
holdout. All 40 capped failures per arm remain counted. This issue does not
reinterpret the rejected earlier candidates as passing.

## Initial allocation diagnostic

An unchanged-main run with no execution/resource arguments and the existing
opt-in JSONL host profiler completed on the same host. It funded **11 slots**,
46,048 target KV tokens, prompt width 1024, slot capacity 8192, with the normal
0.9 memory-share policy. This confirms that the earlier seven-slot observation
must not be generalized to every default run.

Allocator-adjusted available-memory decreases in the final accepted capture were approximately:

| Stage | Incremental allocation |
| --- | ---: |
| Target pool | 3.20 GiB |
| Draft pool | 0.20 GiB |
| Target/draft prompt worker | 1.43 GiB |
| Per-slot homes | 0.16 GiB |

The target-pool figure includes useful recurrent state and KV pages as well as
execution storage; it is not a measurement of waste. Prompt scratch is included
when the worker is constructed. These stage counters identify where to inspect,
but do not yet justify removing any allocation or changing precision.

The instrumented warmup and measured C16 trials completed all 4,096 output tokens
in 19.145 s and 19.009 s respectively. This is one process with synchronous host
profiling enabled, so its timing is diagnostic and does not replace the qualified
uninstrumented performance studies. No performance or quality fix was made.
Startup includes an accepted initial width-11 capture, a later width-12 attempt
rejected for exceeding its unchanged envelope, then the accepted final width-11
plan. All attempts are retained. Full stage counters, source contract and raw
hashes are included in the manifest.

## Investigation and acceptance

- [ ] Account for model/vision residency, target and draft pools, KV pages,
  prompt worker, shared scratch, per-slot homes and graph/driver overhead.
  Preserve raw stage counters, failed captures and automatic replans.
- [ ] Separate observed allocation costs from quote/accounting conservatism.
  Reduce repeated or oversized storage while retaining physical budget checks,
  ownership and request progress. Do not enlarge the safety guard or engine
  budget merely to report a larger pool.
- [ ] Measure achieved batching, admission wait, prefill, draft, target verify,
  accepted-prefix commit and MTP acceptance. Attribute GPU time before optimizing
  arithmetic or removing the F32 quality guard.
- [ ] Keep a single automatic user-facing default, vision availability and long
  request support. Hardware/available-memory adaptation should be internal.
- [ ] Compare each proposed fix against unchanged main using fresh balanced
  processes, fixed inputs/outputs and equal resource envelopes. Report actual
  funded slots/KV and complete telemetry; include C16, short requests, 32K TTFT,
  long-to-short recovery and constrained requests.
- [ ] Complete a fresh controlled vLLM comparison. Historical data may be shown
  as a reference with its conditions, but must not be relabeled as current parity.
  State any unavoidable context, batching or memory-policy differences.
- [ ] Preserve the original full quality suite, scorer, EOS policy, token cap and
  gates. Use targeted diagnosis first and run full qualification for a candidate
  that has already demonstrated a meaningful performance improvement.
- [ ] Publish qualified source/binary hashes, raw evidence and remaining gaps.
  A tracking document alone does not close this issue.

Supplemental vLLM attempts already preserved: V2 reached readiness but the C16
client could not create threads within the unchanged 256-task guard. V3 capped
CPU library threads, then startup rejected context 32,784 because it required
1.71 GiB KV while only 1.47 GiB was available. Neither produced valid timing
samples. These failures do not measure vLLM throughput.

## Evidence and implementation entry points

The [evidence manifest](artifacts/default-serving-performance-gap-2026-10-24.json)
contains immutable raw file hashes and study locations on 192.168.1.45.
The prior [qualification artifact](artifacts/canonical-target-verification-2026-10-24.json)
records the merged default and rejected attempts.

Allocation and plan selection start in
[CLI CUDA loading](../../../crates/service/cli/src/backend/cuda.rs),
[whole-pool budgeting](../../../crates/backend/cuda/src/loading/budget.rs),
[pool capture and execution](../../../crates/backend/cuda/src/executor/execution.rs),
and [opt-in execution profiling](../../../crates/backend/cuda/src/executor/profiling.rs).
