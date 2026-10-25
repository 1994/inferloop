# CUDA nonblocking comparison reproduction

This is comparison evidence, not a certified stable serving baseline. See the [review](../../cuda-nonblocking-2026-10-25.md) and
[machine summary](../cuda-nonblocking-2026-10-25.json). Some original process ratios exceed the declared
10% drift tolerance; the MTP2 batch4 TTFT interval still crosses the latency limit after
all ten permitted pairs. Do not discard slow pairs or treat missing EOS TPOT as zero.

The runners are snapshots of the experiment scripts. They use the persistent directory
`/home/r/benchmarks/inferloop/cuda-nonblocking-v1` on host `45`. Its `inputs/` contains the original profile/input files and the frozen `tuning.json` (SHA256
`9d89a4e2333ad5903789eb06bf3e8a86f3276b189888fa5f85a6a59fbbd7e2f6`).
The table is retained there, rather than copied into this source archive. Its existing
environment selector fixes tuning for comparison; it does not activate async execution.

Before reproducing, verify clean source checkouts and executable hashes against
`paired-plan.json`. Baseline is `4746d89`, measured candidate is `212ebac`.
The hard-coded repository/model paths belong to this device window; inspect them before
running elsewhere. Both binaries were built with:

```sh
cargo build --locked --release -p infer-cli --no-default-features --features cuda
```

The runner invokes the repository's bounded `safe-run.sh`, then `serve-compare.py`, and
checks complete tokens, finish reasons, effective configuration and per-trial cache reuse.
Use fresh tags; existing reports must never be overwritten. Run sequentially, without
other builds or GPU benchmarks:

```sh
python3 reproduce-pairs.py --pairs 5 --profiles 2b-mtp0 27b-mtp0 27b-mtp2 --tag NEW
python3 reproduce-pairs.py --pairs 5 --start 5 --profiles 27b-mtp2 --tag NEW
python3 reproduce-pairs.py --pairs 5 --profiles 27b-mtp0-private-decode 27b-mtp2-private-decode --tag NEW-PRIVATE
```

The second line follows the predeclared extension after MTP2 batch4 TTFT remained
inconclusive. The supplemental profiles contain only `hot_long`: one warmup followed by
three measured repeats, with newline padding before the question and natural EOS.
The earlier four-case supplemental profiles were superseded before any supplemental run;
they are retained in the raw directory and are not mixed into the reported matrix.

Recompute the summaries with the original tags, or substitute the new tags:

```sh
python3 summarize.py --root /home/r/benchmarks/inferloop/cuda-nonblocking-v1 --tag v3 --pairs 5 --profiles 2b-mtp0 27b-mtp0 --output /tmp/cuda-small-summary.json
python3 summarize.py --root /home/r/benchmarks/inferloop/cuda-nonblocking-v1 --tag v3 --pairs 10 --profiles 27b-mtp2 --output /tmp/cuda-mtp2-summary.json
python3 summarize.py --root /home/r/benchmarks/inferloop/cuda-nonblocking-v1 --tag v3-private-decode --pairs 5 --profiles 27b-mtp0-private-decode 27b-mtp2-private-decode --output /tmp/cuda-private-summary.json
```

The sibling JSON indexes every raw report/telemetry hash and all per-case ratios,
intervals, unavailable metrics, finite-sample p95 counts, startup and sampled memory.
Raw reports contain the full requested output and effective build/model/hardware identity.
Committed input JSON uses compact whitespace; normalized workload hashes and all token values match the original files. Profile bytes are unchanged. The committed input/profile files preserve the experiment; the old historical baseline
profiles were not modified. These samples do not establish continuous-load SLOs, production
P99, cross-GPU performance or full configured sixteen-sequence memory capacity.
