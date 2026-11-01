//! One-pair rejection, AB search and independently captured BA confirmation.
/// Fresh AB search followed by separately recaptured BA confirmation.
pub(super) const SEARCH_TRIALS: usize = 2;
use super::Shape;
use crate::device::CudaDevice;
use crate::tuning::nvfp4_lt::Trial;
use crate::tuning::resident_gemm::{MeasurementOrder, PairedTimes};
use crate::tuning::{MAX_REPS, MIN_CONFIRM_SPEEDUP, MIN_REPS, REP_MS, WARMUP_MS, median};
use cutile::bench::{BenchOptions, do_bench_paired};
use cutile::error::tensor_error;
use cutile::prelude::*;
use infer_cuda_kernels::tile::runtime::Graph;
use std::time::Duration;
/// Stop gross losers before the normal sample count.
const CANARY_MAX_RATIO: f32 = 1.10;
pub(super) fn confirm(
    device: &CudaDevice,
    a: &Graph,
    b: &Graph,
    fresh_a: &Graph,
    fresh_b: &Graph,
    shape: Shape,
) -> Result<Option<Vec<Trial>>, DeviceError> {
    #[cfg(test)]
    super::CONFIRMATIONS.fetch_add(1, super::Ordering::Relaxed);
    let canary = BenchOptions {
        warmup: Duration::ZERO,
        rep: Duration::ZERO,
        min_reps: 1,
        max_reps: 1,
        clear_l2: true,
    };
    let canary = measure(device, a, b, &canary, false)?;
    if canary.candidate_median_ms > canary.baseline_median_ms * CANARY_MAX_RATIO {
        tracing::info!(target:"infer::tuning",?shape,"vendor FP4 canary rejected");
        return Ok(None);
    }
    let mut trials = Vec::new();
    for reverse in [false, true] {
        let (left, right) = if reverse { (fresh_a, fresh_b) } else { (a, b) };
        let cold = one(device, left, right, shape, true, reverse)?;
        if cold.speedup < MIN_CONFIRM_SPEEDUP {
            return Ok(None);
        }
        let warm = one(device, left, right, shape, false, reverse)?;
        if warm.speedup < MIN_CONFIRM_SPEEDUP {
            return Ok(None);
        }
        trials.push(Trial {
            order: if reverse {
                MeasurementOrder::CandidateFirst
            } else {
                MeasurementOrder::BaselineFirst
            },
            cold,
            warm,
        });
    }
    Ok(Some(trials))
}
fn one(
    device: &CudaDevice,
    a: &Graph,
    b: &Graph,
    shape: Shape,
    clear_l2: bool,
    reverse: bool,
) -> Result<PairedTimes, DeviceError> {
    let options = BenchOptions {
        warmup: Duration::from_millis(WARMUP_MS),
        rep: Duration::from_millis(REP_MS),
        min_reps: MIN_REPS,
        max_reps: MAX_REPS,
        clear_l2,
    };
    let times = measure(device, a, b, &options, reverse)?;
    tracing::info!(target:"infer::tuning",?shape,clear_l2,reverse,
        baseline_ms=times.baseline_median_ms,candidate_ms=times.candidate_median_ms,
        "vendor FP4 paired confirmation");
    #[cfg(test)]
    super::tests::evidence(
        &serde_json::json!({"phase":"vendor_fp4_confirm","m":shape.m,
        "n":shape.n,"k":shape.k,"clear_l2":clear_l2,"reverse":reverse,
        "baseline_ms":times.baseline_median_ms,"candidate_ms":times.candidate_median_ms,
        "speedup":times.speedup}),
    );
    Ok(times)
}
fn measure(
    device: &CudaDevice,
    a: &Graph,
    b: &Graph,
    options: &BenchOptions,
    reverse: bool,
) -> Result<PairedTimes, DeviceError> {
    let (first, second) = if reverse { (b, a) } else { (a, b) };
    let (x, y) = do_bench_paired(
        &device.stream,
        options,
        |_| {
            first
                .launch()
                .sync_on(&device.stream)
                .map_err(|e| tensor_error(&e.to_string()))
        },
        |_| {
            second
                .launch()
                .sync_on(&device.stream)
                .map_err(|e| tensor_error(&e.to_string()))
        },
    )?;
    let (a, b) = if reverse { (y, x) } else { (x, y) };
    let baseline_median_ms = a.median_ms();
    let candidate_median_ms = b.median_ms();
    let baseline_ms = a.times_ms().to_vec();
    let candidate_ms = b.times_ms().to_vec();
    if !baseline_median_ms.is_finite()
        || !candidate_median_ms.is_finite()
        || candidate_median_ms <= 0.0
        || !median(&baseline_ms).is_finite()
    {
        return Err(super::error("vendor FP4 timing is not finite"));
    }
    Ok(PairedTimes {
        baseline_ms,
        candidate_ms,
        baseline_median_ms,
        candidate_median_ms,
        speedup: baseline_median_ms / candidate_median_ms,
    })
}
