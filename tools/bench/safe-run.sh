#!/usr/bin/env bash
# All descendants, including CUDA JIT compilers, inherit the same memory cgroup.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
memory_gib=${INFER_BENCH_MEMORY_GIB:-32}
time_limit_seconds=1800
while (( $# )); do
    case "$1" in
        --memory-gib) memory_gib=${2:-}; shift "$(( $# >= 2 ? 2 : 1 ))" ;;
        --time-limit-seconds) time_limit_seconds=${2:-}; shift "$(( $# >= 2 ? 2 : 1 ))" ;;
        --) shift; break ;;
        *) break ;;
    esac
done
if [[ ! "$time_limit_seconds" =~ ^[1-9][0-9]{1,3}$ ]] || (( time_limit_seconds < 11 || time_limit_seconds > 1800 )); then
    echo "Time limit must be 11..1800 seconds, including ten seconds for cleanup." >&2
    exit 2
fi
runtime_seconds=$((time_limit_seconds - 10))
if (( $# == 0 )); then
    echo "Usage: bash tools/bench/safe-run.sh [--memory-gib 2..32] [--time-limit-seconds 11..1800] COMMAND [ARGS...]" >&2
    exit 2
fi
# Smaller checks may opt into a lower ceiling; never exceed the default 32 GiB.
if [[ ! "$memory_gib" =~ ^([2-9]|[12][0-9]|3[0-2])$ ]]; then
    echo "Memory ceiling must be an integer from 2 to 32." >&2
    exit 2
fi
high_gib=$((memory_gib * 7 / 8))
cd -- "$root"
mkdir -p artifacts
exec 9>artifacts/benchmark-safe.lock
if ! flock -n 9; then
    echo "Another protected benchmark is running; refusing to queue." >&2
    exit 1
fi
# Leave at least 16 GiB available beyond the selected job ceiling at admission.
available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
if (( available_kib < (memory_gib + 16) * 1024 * 1024 )); then
    echo "Need $((memory_gib + 16)) GiB available host memory before starting this ${memory_gib} GiB job." >&2
    exit 1
fi
unit="inference-benchmark-$$"
echo "Protected unit: ${unit}.service; stop with: systemctl --user stop ${unit}.service" >&2
systemd-run --user --wait --pipe --collect --unit="$unit" \
    --working-directory="$root" \
    --property=MemoryAccounting=yes --property=MemoryHigh="${high_gib}G" \
    --property=MemoryMax="${memory_gib}G" --property=MemorySwapMax=0 \
    --property=OOMPolicy=kill \
    --property=KillMode=control-group --property=TimeoutStopSec=10s \
    --property=KillSignal=SIGINT --property=RuntimeMaxSec="${runtime_seconds}s" --property=TasksMax=256 \
    /usr/bin/env MAX_JOBS=1 NVCC_THREADS=1 FLASHINFER_NVCC_THREADS=1 \
    TORCHINDUCTOR_COMPILE_THREADS=1 CARGO_BUILD_JOBS=1 \
    /usr/bin/python3 tools/bench/check_limits.py -- "$@"
