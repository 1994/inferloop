//! Benchmark quality contract.
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};

/// Number of divisions in the percentile rank scale used to resolve a latency rank to a sample index.
const PERCENTILE_SCALE: usize = 100;
/// Microseconds in one second, converting elapsed microseconds to seconds for rate metrics.
const MICROS_PER_SECOND: f64 = 1e6;
/// Median (50th percentile) end-to-end latency rank.
const P50_RANK: usize = 50;
/// 95th percentile end-to-end latency rank.
const P95_RANK: usize = 95;
/// 99th percentile end-to-end latency rank.
const P99_RANK: usize = 99;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequestMeasurement {
    pub ttft_us: Option<u64>,
    pub max_tpot_us: Option<u64>,
    pub e2e_us: u64,
    pub output_tokens: usize,
    pub successful: bool,
    /// Prompt tokens restored from a reused prefix instead of being prefilled.
    #[serde(default)]
    pub cached_tokens: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub backend: String,
    pub workload_fingerprint: String,
    pub ttft_slo_us: u64,
    pub tpot_slo_us: u64,
    pub elapsed_us: u64,
    pub requests: usize,
    pub successful: usize,
    pub output_tokens: usize,
    pub requests_per_second: f64,
    pub tokens_per_second: f64,
    pub slo_goodput: f64,
    pub p50_e2e_us: u64,
    pub p95_e2e_us: u64,
    pub p99_e2e_us: u64,
}
/// # Errors
/// Returns an invalid-input error for empty samples, invalid elapsed time, or inconsistent request measurements.
#[expect(
    clippy::cast_precision_loss,
    reason = "Floating-point statistics, normalization, and deterministic sampling intentionally convert bounded counts to floating-point values"
)]
pub fn benchmark(
    backend: String,
    workload_fingerprint: String,
    samples: &[RequestMeasurement],
    elapsed_us: u64,
    ttft_slo: u64,
    tpot_slo: u64,
) -> Result<BenchmarkReport> {
    if samples.is_empty() || elapsed_us == 0 {
        return Err(Error::invalid(
            "benchmark requires samples and elapsed time",
        ));
    }
    let mut latency: Vec<_> = samples.iter().map(|s| s.e2e_us).collect();
    latency.sort_unstable();
    let percentile = |p: usize| {
        latency[(latency.len() * p)
            .div_ceil(PERCENTILE_SCALE)
            .saturating_sub(1)]
    };
    let successful = samples.iter().filter(|s| s.successful).count();
    let good = samples
        .iter()
        .filter(|s| {
            s.successful
                && s.ttft_us.is_none_or(|t| t <= ttft_slo)
                && s.max_tpot_us.is_none_or(|t| t <= tpot_slo)
        })
        .count();
    let output_tokens = samples
        .iter()
        .filter(|s| s.successful)
        .map(|s| s.output_tokens)
        .sum();
    let seconds = elapsed_us as f64 / MICROS_PER_SECOND;
    Ok(BenchmarkReport {
        backend,
        workload_fingerprint,
        ttft_slo_us: ttft_slo,
        tpot_slo_us: tpot_slo,
        elapsed_us,
        requests: samples.len(),
        successful,
        output_tokens,
        requests_per_second: successful as f64 / seconds,
        tokens_per_second: output_tokens as f64 / seconds,
        slo_goodput: good as f64 / seconds,
        p50_e2e_us: percentile(P50_RANK),
        p95_e2e_us: percentile(P95_RANK),
        p99_e2e_us: percentile(P99_RANK),
    })
}
