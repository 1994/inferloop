#!/bin/sh
# One fail-fast gate used locally and by CI. Never suppress failed checks.
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

rust_checks() {
    python3 -m unittest discover -s tools/package -p 'test_*.py'
    python3 -m unittest discover -s tools/attention -p 'test_attention_gate.py'
    # Collect every bench module: the review found one of them was collected nowhere.
    python3 -m unittest discover -s tools/bench -p 'test_*.py'
    cargo fmt --manifest-path tools/bench/attention/Cargo.toml --check
    cargo fmt --manifest-path tools/bench/containers/Cargo.toml --check
    cargo clippy --locked --manifest-path tools/bench/containers/Cargo.toml --all-targets -- -D warnings
    # Collect every check module rather than naming one, for the same reason as tools/bench.
    python3 -m unittest discover -s tools/check -p 'test_*.py'
    python3 tools/check/layout.py
    python3 tools/check/policy.py
    # The test-migration ratchet: test bodies only leave src/, and no new consumer of the CPU
    # test executors appears while E1 proceeds.
    python3 tools/check/test-inventory.py
    cargo fmt --all --check
    # Isolated production checks prevent test features from merging into the CLI.
    cargo clippy --locked -p infer-cli --no-default-features --all-targets -- -D warnings
    cargo clippy --locked -p infer-ir --no-default-features --all-targets -- -D warnings
    cargo clippy --locked --all-targets -- -D warnings
    # Hosted CI has no CUDA Toolkit for this job. Excluding the CUDA package alone is not
    # enough: --all-features also enables it through infer-cli/cuda.
    # CUDA builds belong to the package matrix, header-only checks to check-cuda-hosted and
    # device checks to check-cuda.
    # Every registered golden records what produced it, so it can be regenerated rather than trusted.
    python3 tools/check/golden-provenance.py
    # All embedded device artifacts, including Rust SIMT, share one CPU-only gate.
    # Linux additionally validates the frozen FA4 ELF/host ABI dependencies.
    python3 tools/cuda/kernels.py verify
    # Each workspace boundary states the dependency or toolchain isolation it exists for.
    python3 tools/check/workspaces.py
    # A signed baseline's numbers have to follow from the evidence it froze, not merely hash to it.
    python3 tools/bench/verify-baseline.py
    # Production dependency graphs, not just the host CLI's: each target resolves its own feature
    # set, and the plan asks for build edges, duplicates and lockfile identity to be recorded.
    python3 tools/check/release-dependencies.py
    cargo clippy --locked -p infer-backend-cuda --no-default-features --all-targets -- -D warnings
    cargo clippy --locked --workspace --exclude infer-backend-cuda --all-targets -- -D warnings
    test_checks
    RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --exclude infer-backend-cuda --no-deps
    cargo build --locked --release -p infer-cli --no-default-features
    cpu_checks
}

# The workspace test entries, the one place they are listed: `make test` runs these and nothing
# else, while `make check-rust` runs them after linting.
test_checks() {
    # Every test entry reports what it collected, ran, ignored and failed; an entry that only
    # compiles says `compiled` instead of leaving the reader to infer it from a long log.
    python3 tools/check/test-report.py --entry cli -- \
        cargo test --locked -p infer-cli --no-default-features
    python3 tools/check/test-report.py --entry ir -- \
        cargo test --locked -p infer-ir --no-default-features
    # The entire CUDA library is gated off here; this checks isolation, not test execution.
    # Its pure host cases run with the rest of the suite in cuda_hosted_checks.
    python3 tools/check/test-report.py --entry cuda-no-default -- \
        cargo check --locked -p infer-backend-cuda --no-default-features --all-targets
    python3 tools/check/test-report.py --entry workspace -- \
        cargo test --locked --workspace --exclude infer-backend-cuda
}

cpu_checks() {
    cargo fmt -p infer-cpu-bench --check
    CARGO_TARGET_DIR=target/cpu-bench cargo clippy --locked -p infer-cpu-bench --all-targets -- -D warnings
    mkdir -p artifacts/cpu-completion
    CARGO_TARGET_DIR=target/cpu-bench cargo run --locked --release -p infer-cpu-bench > artifacts/cpu-completion/allocation.json
}

tool_checks() {
    uv tool run --from ruff==0.15.7 ruff check tools
    uv tool run --from ruff==0.15.7 ruff format --check tools
    go run github.com/rhysd/actionlint/cmd/actionlint@v1.7.7 -shellcheck= .github/workflows/*.yml
}

security_checks() {
    # The only accepted advisory is documented with its removal condition in deny.toml.
    cargo deny --locked check --deny warnings
    cargo audit --deny warnings --ignore RUSTSEC-2024-0436
    go run github.com/zricethezav/gitleaks/v8@v8.24.3 dir . --no-banner --redact=100 \
        --ignore-gitleaks-allow --exit-code 1
    if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        go run github.com/zricethezav/gitleaks/v8@v8.24.3 git . --no-banner --redact=100 \
            --ignore-gitleaks-allow --exit-code 1
    fi
}

# bindgen cannot find the host C standard headers by itself, and the path is host-specific.
# `package.py::native_env` performs the same discovery for packaging; the gate repeats it in shell
# so `make test-cuda-hosted` also works from a plain developer shell, not only through packaging.
cuda_bindgen_env() {
    include=$(gcc -print-file-name=include 2>/dev/null || true)
    if [ -n "$include" ] && [ -f "$include/stddef.h" ]; then
        case "${BINDGEN_EXTRA_CLANG_ARGS:-}" in
            *"$include"*) ;;
            *)
                BINDGEN_EXTRA_CLANG_ARGS="${BINDGEN_EXTRA_CLANG_ARGS:+$BINDGEN_EXTRA_CLANG_ARGS }-isystem $include"
                export BINDGEN_EXTRA_CLANG_ARGS
                ;;
        esac
    fi
}

# Headless CUDA half. Needs the Toolkit headers and libclang to compile, but never opens a device,
# so it runs on a hosted runner. This is the only check that compiles the cuda-gated `#[cfg(test)]`
# modules: without it a test-only break stays invisible to every other gate.
cuda_hosted_checks() {
    cuda_bindgen_env
    python3 tools/cuda/kernels.py verify
    cargo clippy --locked -p infer-backend-cuda --features cuda --all-targets -- -D warnings
    cargo clippy --locked -p infer-cli --features cuda --all-targets -- -D warnings
    # Release profile the baseline contract requires, reporting what it collected. The suite
    # branches on `cfg!(debug_assertions)` where the two differ.
    python3 tools/check/test-report.py --entry cuda -- \
        cargo test --locked --release -p infer-backend-cuda --features cuda
    python3 tools/check/test-report.py --entry cli-cuda -- \
        cargo test --locked -p infer-cli --features cuda
    RUSTDOCFLAGS="-D warnings" cargo doc --locked -p infer-backend-cuda --features cuda --no-deps
    # Compile the device example without running it: the kernels need a GPU on the next step.
    cargo build --locked --release -p infer-backend-cuda --features cuda --example cuda-kernels
}

# Device acceptance. Requires a driver plus a `tileiras` that supports the installed device.
cuda_device_checks() {
    cuda_bindgen_env
    cargo run --locked --release -p infer-backend-cuda --features cuda --example cuda-kernels
}

cuda_checks() {
    cuda_hosted_checks
    cuda_device_checks
}

metal_checks() {
    # Explicit backend selection makes missing GPU support fail instead of skipping.
    cargo build --locked --release -p infer-cli --no-default-features
    python3 tools/check/metal.py
    python3 tools/validation/cpu-open-loop.py
}

msrv_checks() {
    CARGO_TARGET_DIR=target/msrv cargo +1.90.0 check --locked -p infer-cli --no-default-features
    CARGO_TARGET_DIR=target/msrv cargo +1.90.0 check --locked --workspace --exclude infer-backend-cuda --all-targets
    CARGO_TARGET_DIR=target/cpu-bench-msrv cargo +1.90.0 check --locked --manifest-path tools/bench/cpu/Cargo.toml --all-targets
}

linux_checks() {
    if [ "$(uname -s)" = Linux ]; then
        cargo clippy --locked -p infer-core --all-targets -- -D warnings
        cargo test --locked -p infer-core placement:: -- --test-threads=1
    else
        # Compile Linux-only FFI and tests on development hosts; runtime tests belong to Linux CI.
        CARGO_TARGET_DIR=target/linux-check cargo clippy --locked -p infer-core --all-targets --target x86_64-unknown-linux-gnu -- -D warnings
    fi
}

linux_numa_checks() {
    if [ "$(uname -s)" != Linux ]; then
        echo 'NUMA acceptance requires a Linux host with memory-policy syscalls enabled' >&2
        exit 1
    fi
    python3 tools/check/test-report.py --entry numa -- \
        cargo test --locked --release -p infer-core placement::native::numa:: -- \
        --ignored --test-threads=1
}

case "${1:-all}" in
    all)
        linux_checks
        rust_checks
        tool_checks
        security_checks
        msrv_checks
        if [ "$(uname -s)" = Darwin ]; then metal_checks; fi
        ;;
    rust) rust_checks ;;
    test) test_checks ;;
    tools) tool_checks ;;
    security) security_checks ;;
    metal) metal_checks ;;
    cuda) cuda_checks ;;
    cuda-hosted) cuda_hosted_checks ;;
    cpu) cpu_checks ;;
    msrv) msrv_checks ;;
    linux) linux_checks ;;
    linux-numa) linux_numa_checks ;;
    attention) bash tools/bench/safe-run.sh --memory-gib 16 bash tools/bench/attention.sh ;;
    *) echo "usage: $0 [all|rust|test|tools|security|metal|cuda|cuda-hosted|cpu|msrv|linux|linux-numa|attention]" >&2; exit 2 ;;
esac
