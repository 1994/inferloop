//! Reproducible paired measurements of the synchronous dense launch path.
use crate::target::CudaTarget;
use crate::{
    device::{CudaDevice, device_error},
    strategy::{LinearStrategy, LinearTiling},
};
use cutile::error::Error as CutileError;
use cutile::error::tensor_error;
use cutile::half::bf16;
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    prelude::*,
};
use infer_core::{Error, Result};
use serde::Serialize;
use std::result::Result as StdResult;
use std::{sync::Arc, time::Duration};

/// Synthetic matrices stay below this many elements.
const MAX_BASELINE_ELEMENTS: usize = 150_000_000;
/// Deterministic sample generators for the input vector and the weight matrix.
const VECTOR_SAMPLE_SEED: usize = 13;
/// Deterministic sample generators for the input vector and the weight matrix.
const WEIGHT_SAMPLE_SEED: usize = 7;
/// Original launch path, kept as the measurement baseline.
const BASELINE_TILE_ROWS: usize = 4;
/// Original launch path, kept as the measurement baseline.
const BASELINE_TILE_COLUMNS: usize = 128;
/// Untimed warmup wall clock per measurement.
const WARMUP_MS: u64 = 100;
/// Timed-repetition wall clock budget per measurement.
const REP_MS: u64 = 500;
/// Timed-repetition count bounds for stable medians.
const MIN_REPS: usize = 30;
/// Timed-repetition count bounds for stable medians.
const MAX_REPS: usize = 100;
/// Relative plus absolute slack against the CPU reference.
const VERIFY_TOLERANCE: f32 = 1e-4;
/// Deterministic sample value bounds: modulo, center and scale.
const SAMPLE_MODULUS: usize = 31;
/// Deterministic sample value bounds: modulo, center and scale.
const SAMPLE_CENTER: i16 = 15;
/// Deterministic sample value bounds: modulo, center and scale.
const SAMPLE_SCALE: f32 = 16.0;

#[derive(Serialize)]
pub struct Trial {
    pub tiling: LinearTiling,
    pub baseline_ms: Vec<f32>,
    pub candidate_ms: Vec<f32>,
    pub baseline_median_ms: f32,
    pub candidate_median_ms: f32,
    pub speedup: f32,
}

#[derive(Serialize)]
pub struct Baseline {
    pub schema_version: u32,
    pub debug_assertions: bool,
    pub gpu: String,
    pub target: CudaTarget,
    pub measurement: &'static str,
    pub dtype: &'static str,
    pub rows: usize,
    pub columns: usize,
    pub warmup_ms: u64,
    pub clear_l2: bool,
    pub trials: Vec<Trial>,
}

/// Measures synthetic BF16 weights at supplied model dimensions.
///
/// Outputs are checked against an independent CPU dot product before timing.
/// Includes output allocation, initialization and synchronous submission; this
/// is a launch-path baseline, not isolated kernel time or model throughput.
/// # Errors
/// Rejects excessive allocation, invalid results or device failures.
pub fn dense_baseline(
    device: &CudaDevice,
    rows: usize,
    columns: usize,
    strategy: &dyn LinearStrategy,
) -> Result<Baseline> {
    let count = rows
        .checked_mul(columns)
        .filter(|n| *n <= MAX_BASELINE_ELEMENTS)
        .ok_or_else(|| Error::invalid("baseline matrix exceeds element budget"))?;
    if rows == 0 || columns == 0 {
        return Err(Error::invalid("baseline dimensions must be positive"));
    }
    let vector: Vec<f32> = (0..columns)
        .map(|i| sample(i, VECTOR_SAMPLE_SEED))
        .collect();
    let values: Vec<bf16> = (0..count)
        .map(|i| bf16::from_f32(sample(i, WEIGHT_SAMPLE_SEED)))
        .collect();
    let expected: Vec<f32> = values
        .chunks_exact(columns)
        .map(|row| row.iter().zip(&vector).map(|(w, x)| w.to_f32() * x).sum())
        .collect();
    let x = device.upload(vector, &[columns])?;
    let w = device.upload(values, &[rows, columns])?;
    let baseline = LinearTiling::new(BASELINE_TILE_ROWS, BASELINE_TILE_COLUMNS)?;
    let candidates = strategy.candidates(columns);
    if candidates.is_empty() {
        return Err(Error::invalid("empty tuning search"));
    }
    let options = BenchOptions {
        warmup: Duration::from_millis(WARMUP_MS),
        rep: Duration::from_millis(REP_MS),
        min_reps: MIN_REPS,
        max_reps: MAX_REPS,
        clear_l2: true,
    };
    verify(device, &x, &w, baseline, &expected)?;
    let mut trials = Vec::new();
    for candidate in candidates {
        verify(device, &x, &w, candidate, &expected)?;
        let (a, b) = do_bench_paired(
            &device.stream,
            &options,
            |_| launch(device, &x, &w, baseline),
            |_| launch(device, &x, &w, candidate),
        )
        .map_err(device_error)?;
        trials.push(Trial {
            tiling: candidate,
            baseline_ms: a.times_ms().to_vec(),
            candidate_ms: b.times_ms().to_vec(),
            baseline_median_ms: a.median_ms(),
            candidate_median_ms: b.median_ms(),
            speedup: a.median_ms() / b.median_ms(),
        });
    }
    Ok(Baseline {
        schema_version: 2,
        debug_assertions: cfg!(debug_assertions),
        gpu: device.name()?,
        target: device.target().clone(),
        measurement: "CUDA events: synchronous launch including allocation and zeroing; synthetic weights",
        dtype: "BF16 weights, F32 input and accumulation",
        rows,
        columns,
        warmup_ms: WARMUP_MS,
        clear_l2: true,
        trials,
    })
}

fn sample(i: usize, multiplier: usize) -> f32 {
    let value =
        i16::try_from((i % SAMPLE_MODULUS) * multiplier % SAMPLE_MODULUS).unwrap_or_default();
    f32::from(value - SAMPLE_CENTER) / SAMPLE_SCALE
}

fn launch(
    device: &CudaDevice,
    x: &Arc<Tensor<f32>>,
    w: &Arc<Tensor<bf16>>,
    tiling: LinearTiling,
) -> StdResult<(), CutileError> {
    device
        .matvec_tiled(x.clone(), w.clone(), tiling)
        .map(|_| ())
        .map_err(|error| tensor_error(&error.to_string()))
}

fn verify(
    device: &CudaDevice,
    x: &Arc<Tensor<f32>>,
    w: &Arc<Tensor<bf16>>,
    tiling: LinearTiling,
    expected: &[f32],
) -> Result<()> {
    let out = device.matvec_tiled(x.clone(), w.clone(), tiling)?;
    let actual = device.read(&out)?;
    if actual.len() != expected.len()
        || actual.iter().zip(expected).any(|(a, e)| {
            !a.is_finite() || (a - e).abs() > e.abs().mul_add(VERIFY_TOLERANCE, VERIFY_TOLERANCE)
        })
    {
        return Err(Error::invariant(
            "CUDA baseline failed CPU numerical reference",
        ));
    }
    Ok(())
}
