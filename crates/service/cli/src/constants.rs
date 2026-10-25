//! Named CLI defaults, byte-size units and device limits shared across subcommands.

#[cfg(native_backend)]
use std::time::Duration;
/// One mebibyte in bytes, for budgets held as `usize`.
pub const MIB: usize = 1024 * 1024;
/// One mebibyte in bytes, for byte budgets held as `u64`.
pub const MIB_U64: u64 = MIB as u64;
/// One gibibyte in bytes, for device and model size budgets.
pub const GIB_U64: u64 = 1024 * MIB_U64;
/// Fraction of device memory the engine may use when `--gpu-memory-utilization` is defaulted.
pub const DEFAULT_GPU_MEMORY_UTILIZATION: f64 = 0.9;
/// Host-memory budget used when `--host-memory-mib` is omitted on the CPU test backend.
/// Native device backends derive their own budget from the device instead.
/// Default temporary weight-staging budget in MiB for upload and package inspection.
pub const DEFAULT_STAGING_MIB: usize = 4;
/// Maximum accepted host JSON document size for requests and golden files.
pub const MAX_JSON_FILE_BYTES: u64 = 64 * MIB_U64;
/// Default absolute tolerance for numeric comparisons.
pub const DEFAULT_ATOL: f64 = 1e-5;
/// Default relative tolerance for numeric comparisons.
pub const DEFAULT_RTOL: f64 = 1e-5;
/// Default request count for the benchmark command.
pub const DEFAULT_BENCHMARK_REQUESTS: usize = 32;
/// Default input sequence length in tokens for each benchmark request.
pub const DEFAULT_BENCHMARK_INPUT_TOKENS: usize = 16;
/// Default output length in tokens for each benchmark request.
pub const DEFAULT_BENCHMARK_OUTPUT_TOKENS: usize = 16;
/// Default time-to-first-token service-level objective in microseconds.
pub const DEFAULT_TTFT_SLO_US: u64 = 1_000_000;
/// Default time-per-output-token service-level objective in microseconds.
pub const DEFAULT_TPOT_SLO_US: u64 = 100_000;
/// Default tolerated p99 latency regression fraction for `compare`.
pub const DEFAULT_MAX_P99_REGRESSION: f64 = 0.05;
/// Engine ticks a CLI owner loop waits for its submitted workload to go idle.
#[cfg(native_backend)]
pub const MAX_IDLE_TICKS: u64 = 1_000_000;
/// Default context length in tokens for the model memory estimate.
pub const DEFAULT_CONTEXT_TOKENS: usize = 4096;
/// Default device memory in GiB for the model memory estimate.
pub const DEFAULT_DEVICE_MEMORY_GIB: u64 = 32;
/// Default KV page size in tokens when a caller does not override it.
#[cfg(metal_backend)]
pub const DEFAULT_KV_PAGE_TOKENS: usize = 16;
/// Default maximum token rows in one native device prefill chunk.
#[cfg(metal_backend)]
pub const DEFAULT_PREFILL_CHUNK_TOKENS: usize = 32;
/// Pause between polls while a device step is still in flight.
#[cfg(native_backend)]
pub const DEVICE_POLL_INTERVAL: Duration = Duration::from_micros(100);

/// Default port for the local inference HTTP server.
pub const DEFAULT_LISTEN_PORT: u16 = 8080;
