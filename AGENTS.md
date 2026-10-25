# Engineering rules

These instructions apply throughout inferloop. Read any narrower `AGENTS.md` before
editing its directory. Preserve unrelated work in the checkout.

## Performance first

- Performance is a primary acceptance criterion for every change, including features,
  refactors, fixes, dependencies, build settings, and defaults. Assess regression risk
  before implementation; preserving behavior alone does not establish performance safety.
  Correctness, numerical contracts, and memory safety remain mandatory.
- Consider end-to-end throughput, TTFT/TPOT and tail latency, host/device memory peaks,
  allocations, copies, transfers, synchronization, and startup cost as applicable.
  Include indirect effects on batching, concurrency, graph replay, and resource capacity.
- For changes that may affect execution cost, compare representative release workloads
  against the baseline on the affected paths and configurations. Report regressions per
  case; an aggregate speedup must not hide a slower workload or reduced concurrency.
- Address unexpected regressions before merging. Any intentional trade-off must quantify
  the cost, identify affected workloads, and explain the benefit. If measurement is
  unavailable, report the risk and missing evidence rather than claim no regression.
- Every PR must state its performance impact and the evidence or reasoning behind it.
  Changes with no execution effect, such as prose-only edits, may give a brief explanation
  instead of running benchmarks. Use the existing performance gates and baseline contract.

## Architecture

- inferloop is a GPU inference engine for CUDA and Metal. CPU operators are independent
  test references, not a production execution backend. Unsupported capabilities must
  return an explicit error rather than silently select a fallback.
- Keep crates under their existing responsibility groups. Put implementation in focused
  modules; keep `lib.rs` a small facade with explicit exports. Do not introduce cycles,
  service dependencies in implementation crates, or concrete backend dependencies in runtime.
- Model providers and `model/recipes` own topology, weight semantics, position encoding,
  and draft graphs. The compiler validates and lowers the supplied graph; runtime uses
  the backend's bound execution graph instead of reconstructing it from `ModelIr`.
- Scheduler owns queueing, fairness, and packing policy; state owns logical allocation,
  KV, and prefix lifetimes; runtime coordinates admission, dispatch, completion, and
  cancellation; backends own physical buffers, kernel selection, and device execution.
- Extend shared IR/SPI and kernel contracts at their owning layer. Keep model names,
  GPU marketing names, and algorithm-specific branches out of common scheduling code.
  Select kernels by declared capabilities, shapes, layouts, and numeric requirements.
- Reuse existing contracts and tools. Add a trait, crate, feature, or dependency only
  for a concrete need; do not create parallel implementations of existing ownership or policy.

For changes across layers, check [code layout](docs/architecture/layout.md) and
[backend contracts](docs/architecture/backends.md).

## Rust and API design

- Encode distinct IDs, units, modes, and valid states in types. Use newtypes and enums
  where plain integers or boolean mode arguments hide meaning. Validate invariants at
  construction; do not expose fields that let callers bypass them.
- Borrow read-only inputs as `&str`, `&[T]`, or `&T`; use `&mut T` for mutation and owned
  `T` for retention or consumption. Avoid `&String`, `&Vec<T>`, and borrowed inputs that
  are immediately cloned into storage.
- Resolve borrow conflicts by shortening borrows, splitting fields, or moving values
  with `mem::take`, `mem::replace`, or `Option::take`. Clone only for an intentional
  copy, snapshot, or shared lifetime; keep `Arc` and locks tied to real sharing needs.
- Keep visibility minimal and exports explicit. Prefer concrete types; introduce
  generics or `dyn Trait` for actual substitution or heterogeneous storage. Do not add
  public APIs or product features solely to reach private code from tests.
- Derive common traits when their semantics fit. Equality, ordering, and hashing must
  agree; float ordering and tie-breaking must be explicit and deterministic. Keep
  secrets out of `Debug`, `Display`, and logs.
- Use standard conversion traits and `as_` / `to_` / `into_` naming. Use checked
  arithmetic and fallible conversions for external dimensions, offsets, byte counts,
  and capacities; validate before indexing or allocating.
- Use `Option` for expected absence and the project's existing `Result`/error types
  for actionable failures. Preserve error codes, causes, and useful context; do not
  introduce another application error framework or flatten typed errors into strings.
- Do not add production `unwrap`, `expect`, `panic!`, `todo!`, `unimplemented!`,
  `dbg!`, or ignored fallible results, including in `main`. Review existing narrowly
  justified invariant exceptions before changing their recovery or cleanup semantics.
- Use clear loops or short iterator chains. Import item paths with `use`; paths of
  three or more segments must not be embedded in expressions or signatures. Avoid
  glob exports, speculative abstractions, and macros where ordinary Rust suffices.
- Inherit workspace versions, MSRV, edition, and lints. Keep intentional version pins
  and `Cargo.lock`. Follow the numeric-literal policy: non-exempt values become named,
  documented constants. Keep function length at most 100 and cognitive complexity at
  most 25; fix the design instead of raising thresholds or broadly suppressing lints.
- Document public contracts: ownership, preconditions, errors, side effects, and safety
  obligations. Rustdoc examples should run and propagate errors with `?`. Comments
  explain non-obvious reasons; source files are not design journals or progress logs.
- Use existing structured `tracing` events for operational logs. Preserve CLI and
  protocol stdout; do not replace structured diagnostics with `println!` or `dbg!`.

## Resource and request safety

- Every buffer, page, credit, ticket, and reader must have a clear owner, handoff, and
  release condition. Bind RAII guards to named variables; provide explicit fallible
  close operations when needed, and keep destructors non-panicking.
- Validate identities, ranges, capacity, duplicate states, and cursor transitions
  before publishing a batch. Stage fallible work before mutating ownership; failed
  reservations must leave allocator cursors and logical/physical state consistent.
- CPU return, cancellation, timeout, disconnect, and ticket drop do not prove device
  completion. Retain buffers, pinned memory, scratch, page leases, and credits until
  the relevant device fences, resource acknowledgements, and CPU readers finish.
- Check owner, index, and generation on handles. Reject stale, foreign, or duplicate
  acknowledgements; retire exhausted generations instead of reusing an identity.
- Share only committed full prefix blocks; use copy-on-write for shared mutable tails.
  Prefix reuse must match model, revision, precision, layout, and readout compatibility.
  Cache eviction releases its own references, not active requests' ownership.
- Keep preparation, admission, resource waits, GPU flight, CPU output, and delivery
  separately accounted. A slow reader must not stall unrelated requests; cancellation
  and shutdown must remain serviceable when bulk queues are full.
- Quiesce admission and dispatch, drain device work, resource commands, and readers,
  then checkpoint. Restore only after validating schema, weights, program, backend,
  provider identities, cursors, and payload; never restore native tickets. If resources
  cannot safely drain, isolate the engine instead of reporting idle or reusing memory.

For lifetime changes, follow [request lifecycle](docs/architecture/request-lifecycle.md),
[CPU ownership](docs/architecture/cpu-runtime.md), and [KV ownership](docs/architecture/kv-manager.md).

## Concurrency, unsafe, and device boundaries

- Preserve actor and device-owner boundaries. Use existing bounded worker pools;
  do not add a runtime, a thread per request, unbounded queues, or unbounded task spawning.
  Keep blocking work off asynchronous service threads and propagate cancellation.
- Keep locks short; do not hold them across `.await` or external callbacks. Treat lock
  poisoning through the component's failure protocol, not unchecked `into_inner` recovery.
  Require `Send`/`Sync` where the real thread contract needs them.
- Prevent lost wakeups: arm notification, recheck work/epochs, then park. Polling and
  spinning need a budget. Callbacks should publish reserved state and wake the owner,
  not allocate or run scheduling logic.
- `unsafe_code` is denied by default. Keep necessary platform/device FFI in approved,
  minimal boundaries behind safe APIs. Each unsafe block/impl needs a concrete `SAFETY:`
  argument covering bounds, initialization, aliasing, lifetime, and synchronization.
- Check FFI ABI, alignment, nullability, allocation/free ownership, and thread/stream
  affinity. Do not bypass lifetimes with `transmute`, add `unsafe impl Send` to silence
  an error, or let a panic unwind through an incompatible FFI boundary.
- Host Release/Acquire ordering does not replace DMA or device synchronization. Metal
  shared storage still needs completion/reader discipline. Keep isolated device-build
  workspaces and their toolchains outside the stable host dependency graph.

## Performance and numerical correctness

- Profile the actual bottleneck before changing a hot path. Reuse prepared buffers,
  reserve known capacity, batch work, and keep per-tick/per-token allocation, deep
  copies, formatting, serialization, and lock traffic out of steady-state execution.
- Preserve device residency and graph replay where applicable. New host readbacks,
  stream synchronization, graph recapture, or allocation in decode need a demonstrated
  reason and measured whole-path cost. Do not mechanically change allocators, hashers,
  container types, or cache policy.
- Resource quotes must cover the execution path actually selected, including temporary
  workspace, retention, host mirrors, and recapture. Reusing a cached shell does not
  make its retained memory free; verify peak and concurrent ownership, not just tensor size.
- Preserve dtype, accumulation, rounding, masking, tie-breaking, seed, and output
  semantics unless changing them is explicitly in scope. Do not loosen numeric
  tolerances to make an optimization pass or substitute another head's output.
- Validate optimized kernels against an independent reference, including tails, padding,
  masks, empty/all-masked cases where supported, finite outputs, and state transitions.
  Validate full requested output and resource cleanup, not just an initial token prefix.
- Performance evidence requires comparable release builds, actual effective settings,
  identical workload/model/hardware conditions, warmup, repeated samples, and declared
  statistics. Separate operator, model, and serving results; report regressions per case.
  A faster microbenchmark is not proof of a faster service.

Use the [performance baseline contract](docs/plans/performance/baseline.md) for comparisons.

## Tests and delivery

- Reproduce bugs with failing tests before fixing them. Test observable behavior,
  properties, and failure/cleanup paths rather than mirrored branches or constants.
  Preserve API, wire format, checkpoint, and error compatibility unless the task changes them.
- Put test bodies in the owning crate's `tests/`; mount private unit tests through the
  existing `#[cfg(test)]`/`#[path]` pattern. Register targets and update the test inventory
  when collection changes. Do not put new test bodies in `src/` or silently lose cases.
- Cover relevant feature/target combinations. Use deterministic inputs, bounded waits,
  isolated ports, and reliable cleanup. A selected hardware suite must fail for missing
  devices/fixtures; zero collected or fully ignored tests are not successful acceptance.
- Use independent scalar references for numeric checks. Host tests and CUDA compilation
  without devices cannot establish GPU support or throughput. Use Miri for applicable
  host unsafe abstractions; it does not validate device execution.
- Use repository entry points instead of blanket `--all-features`, which can pull CUDA
  Toolkit requirements into host checks. Toolchain and command details live in
  [quality gates](docs/guides/quality-gates.md) and the [Makefile](Makefile).

| Change | Required validation |
|---|---|
| Rust | `cargo fmt --all --check` and `make check-rust` |
| Documentation only | Local links, `git diff --check`, and `python3 tools/check/layout.py` |
| Tooling / dependencies / MSRV / Linux | Corresponding `make check-tools`, `check-security`, `check-msrv`, or `check-linux` |
| CUDA | `make test-cuda-hosted`; `make test-cuda` for device behavior |
| Metal | `make test-metal` for device behavior |
| Attention numerics / performance | `make check-attention` with the required reference assets and hardware |

Report what changed, its performance impact, what ran, and the observed results.
Distinguish measured outcomes from assumptions, and compiled, executed, ignored, and
unavailable checks. Keep credentials, full models, generated build output,
and temporary runs out of commits; retain reproducible accepted benchmark evidence in
its established location.
