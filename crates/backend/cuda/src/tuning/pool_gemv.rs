//! Measured crossover between the row-independent F32 GEMV and the blocked Dense GEMM.
//!
//! Both arms use the actual Dense BF16 weight, actual F32 input layout and the production
//! kernels. The comparison keeps GEMV's F32 accumulation order; it never casts activations to
//! BF16 to make the candidate win. A choice is cached only with a full weight proof and both
//! cold/warm paired orders.
use super::{MIN_CONFIRM_SPEEDUP, REP_MS, WARMUP_MS, update_table};
use crate::constants::{CB_DECODE_SLOTS, SMALL_GEMM_TILE};
use crate::device::{CudaDevice, device_error};
use crate::mlp::ProjectionWeight;
use crate::resident::pool_gemv::kernels as gemv_kernels;
use crate::resident::small_gemm::kernels as blocked_kernels;
use crate::strategy::LinearTiling;
use crate::tuning::resident_gemm::{ResidentTuningSession, SearchFamily};
use cutile::bench::{BenchOptions, do_bench_paired};
use cutile::error::tensor_error;
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Kernel identity for the crossover evidence; change when either arm's arithmetic changes.
const KERNEL_ID: &str = concat!("pool-gemv-vs-blocked-v1/", env!("CARGO_PKG_VERSION"));
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;
const INPUT_SCALE: f32 = 97.0;
const INPUT_CENTER: f32 = 125.0;
const INPUT_MODULUS: usize = 251;
const MATCH_TOLERANCE: f32 = 5e-4;
/// Microseconds per millisecond for the benchmark medians.
const MICROSECONDS_PER_MILLISECOND: f32 = 1000.0;
const REL_TOLERANCE: f64 = 1.0e-3;
const ABS_TOLERANCE: f64 = 1.0e-9;
const CACHE_STATES: [bool; 2] = [true, false];

/// The qualified dispatch choice for one Dense projection geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Choice {
    Gemv,
    Blocked,
}

/// One cached crossover decision with raw paired evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub device: String,
    pub driver: u32,
    pub driver_build: String,
    pub hardware: String,
    pub kernel: String,
    pub kernel_fingerprint: String,
    pub rows: usize,
    pub columns: usize,
    pub outputs: usize,
    pub weight_proof: String,
    pub choice: String,
    pub exact: bool,
    pub cold_gemv_us: f64,
    pub cold_blocked_us: f64,
    pub warm_gemv_us: f64,
    pub warm_blocked_us: f64,
    #[serde(default)]
    pub cold_ab_blocked_samples: Vec<f32>,
    #[serde(default)]
    pub cold_ab_gemv_samples: Vec<f32>,
    #[serde(default)]
    pub cold_ba_blocked_samples: Vec<f32>,
    #[serde(default)]
    pub cold_ba_gemv_samples: Vec<f32>,
    #[serde(default)]
    pub warm_ab_blocked_samples: Vec<f32>,
    #[serde(default)]
    pub warm_ab_gemv_samples: Vec<f32>,
    #[serde(default)]
    pub warm_ba_blocked_samples: Vec<f32>,
    #[serde(default)]
    pub warm_ba_gemv_samples: Vec<f32>,
    pub gemv_faster: bool,
}

struct Key<'a> {
    device: &'a str,
    driver: u32,
    driver_build: &'a str,
    hardware: &'a str,
    fingerprint: &'a str,
    rows: usize,
    columns: usize,
    outputs: usize,
    proof: &'a str,
}

impl Entry {
    fn matches(&self, key: &Key<'_>) -> bool {
        self.device == key.device
            && self.driver == key.driver
            && !self.driver_build.is_empty()
            && !self.hardware.is_empty()
            && self.kernel == KERNEL_ID
            && self.kernel_fingerprint == key.fingerprint
            && self.rows == key.rows
            && self.columns == key.columns
            && self.outputs == key.outputs
            && self.weight_proof == key.proof
            && (self.choice == "gemv" || self.choice == "blocked")
            && self.gemv_faster == (self.choice == "gemv")
            && self.cold_gemv_us.is_finite()
            && self.cold_blocked_us.is_finite()
            && self.warm_gemv_us.is_finite()
            && self.warm_blocked_us.is_finite()
            && self.cold_gemv_us > 0.0
            && self.cold_blocked_us > 0.0
            && self.warm_gemv_us > 0.0
            && self.warm_blocked_us > 0.0
            && self.valid_evidence()
            && samples_ok(&self.cold_ab_blocked_samples)
            && samples_ok(&self.cold_ab_gemv_samples)
            && samples_ok(&self.cold_ba_blocked_samples)
            && samples_ok(&self.cold_ba_gemv_samples)
            && samples_ok(&self.warm_ab_blocked_samples)
            && samples_ok(&self.warm_ab_gemv_samples)
            && samples_ok(&self.warm_ba_blocked_samples)
            && samples_ok(&self.warm_ba_gemv_samples)
    }

    fn valid_evidence(&self) -> bool {
        let Some(cold_ab) = pair_medians(&self.cold_ab_blocked_samples, &self.cold_ab_gemv_samples)
        else {
            return false;
        };
        let Some(cold_ba) = pair_medians(&self.cold_ba_blocked_samples, &self.cold_ba_gemv_samples)
        else {
            return false;
        };
        let Some(warm_ab) = pair_medians(&self.warm_ab_blocked_samples, &self.warm_ab_gemv_samples)
        else {
            return false;
        };
        let Some(warm_ba) = pair_medians(&self.warm_ba_blocked_samples, &self.warm_ba_gemv_samples)
        else {
            return false;
        };
        let cold_base =
            f64::midpoint(cold_ab.0, cold_ba.0) * f64::from(MICROSECONDS_PER_MILLISECOND);
        let cold_trial =
            f64::midpoint(cold_ab.1, cold_ba.1) * f64::from(MICROSECONDS_PER_MILLISECOND);
        let warm_base =
            f64::midpoint(warm_ab.0, warm_ba.0) * f64::from(MICROSECONDS_PER_MILLISECOND);
        let warm_trial =
            f64::midpoint(warm_ab.1, warm_ba.1) * f64::from(MICROSECONDS_PER_MILLISECOND);
        if !close64(cold_base, self.cold_blocked_us)
            || !close64(cold_trial, self.cold_gemv_us)
            || !close64(warm_base, self.warm_blocked_us)
            || !close64(warm_trial, self.warm_gemv_us)
        {
            return false;
        }
        let qualified = [cold_ab, cold_ba, warm_ab, warm_ba]
            .into_iter()
            .all(|(base, trial)| base / trial >= f64::from(MIN_CONFIRM_SPEEDUP));
        qualified == (self.choice == "gemv") && self.gemv_faster == qualified
    }

    fn geometry_matches(&self, key: &Key<'_>) -> bool {
        self.device == key.device
            && self.driver == key.driver
            && !self.driver_build.is_empty()
            && !self.hardware.is_empty()
            && self.kernel == KERNEL_ID
            && self.kernel_fingerprint == key.fingerprint
            && self.rows == key.rows
            && self.columns == key.columns
            && self.outputs == key.outputs
            && (self.choice == "gemv" || self.choice == "blocked")
            && self.valid_evidence()
    }
}

/// Select GEMV or the existing blocked fallback from actual-weight paired evidence.
pub(crate) fn select(
    device: &CudaDevice,
    weight: &ProjectionWeight,
    shape: (usize, usize, usize),
    tile: LinearTiling,
    autotune: bool,
    session: &ResidentTuningSession,
) -> Result<Choice> {
    let (rows, columns, outputs) = shape;
    let profile = device.profile()?;
    let ProjectionWeight::Dense(dense) = weight else {
        return Ok(Choice::Blocked);
    };
    if rows == 0 || columns == 0 || outputs == 0 {
        return Ok(Choice::Blocked);
    }
    let proof = weight_proof(device, dense)?;
    let identity = profile.identity();
    let driver = CudaDevice::driver_version()?;
    let driver_build = super::driver_build().unwrap_or_default();
    let hardware = serde_json::to_string(profile).map_err(device_error)?;
    let fingerprint = format!(
        "{}:kernel={}:geometry={rows}x{columns}x{outputs}",
        super::compiler_fingerprint(),
        kernel_fingerprint(device, rows, columns, outputs, tile)?
    );
    let key = Key {
        device: &identity,
        driver,
        driver_build: &driver_build,
        hardware: &hardware,
        fingerprint: &fingerprint,
        rows,
        columns,
        outputs,
        proof: &proof,
    };
    if let Some(choice) = cached(&key)? {
        session.plan_gemm_search(
            SearchFamily::PoolDense,
            &[[rows, columns, outputs]],
            CACHE_STATES.len() * 2,
        )?;
        session.complete_gemm_search(SearchFamily::PoolDense, [rows, columns, outputs])?;
        return Ok(choice);
    }
    if let Some(choice) =
        reuse_geometry(device, dense, (rows, columns, outputs), tile, &key, session)?
    {
        return Ok(choice);
    }
    if !autotune
        || !session.permits_gemm_search(
            SearchFamily::PoolDense,
            [rows, columns, outputs],
            CACHE_STATES.len() * 2,
        )?
    {
        return Ok(legacy_choice(rows));
    }
    let result = measure(
        device,
        dense,
        rows,
        columns,
        outputs,
        tile,
        &proof,
        &identity,
        driver,
        &driver_build,
        &hardware,
        &fingerprint,
    );
    match result {
        Ok(entry) => {
            session.complete_gemm_search(SearchFamily::PoolDense, [rows, columns, outputs])?;
            update_table(super::table_path().as_deref(), |table| {
                table.pool_gemv.retain(|old| {
                    !(old.device == entry.device
                        && old.driver == entry.driver
                        && old.driver_build == entry.driver_build
                        && old.hardware == entry.hardware
                        && old.kernel == entry.kernel
                        && old.kernel_fingerprint == entry.kernel_fingerprint
                        && old.rows == entry.rows
                        && old.columns == entry.columns
                        && old.outputs == entry.outputs
                        && old.weight_proof == entry.weight_proof)
                });
                table.pool_gemv.push(entry.clone());
            });
            Ok(if entry.choice == "gemv" {
                Choice::Gemv
            } else {
                Choice::Blocked
            })
        }
        Err(error) => {
            tracing::warn!(target:"infer::tuning", rows, columns, outputs, %error,
                "pool GEMV crossover measurement failed; using legacy fallback");
            Ok(legacy_choice(rows))
        }
    }
}

const fn legacy_choice(rows: usize) -> Choice {
    if rows <= CB_DECODE_SLOTS {
        Choice::Gemv
    } else {
        Choice::Blocked
    }
}

fn cached(key: &Key<'_>) -> Result<Option<Choice>> {
    let Some(path) = super::table_path() else {
        return Ok(None);
    };
    if !path.is_file() {
        return Ok(None);
    }
    let table = super::TilingTable::load(&path)?;
    Ok(table.pool_gemv.iter().find_map(|entry| {
        (entry.matches(key)
            && entry.driver_build == key.driver_build
            && entry.hardware == key.hardware)
            .then_some(if entry.choice == "gemv" {
                Choice::Gemv
            } else {
                Choice::Blocked
            })
    }))
}

fn cached_geometry(key: &Key<'_>) -> Result<Option<Choice>> {
    let Some(path) = super::table_path() else {
        return Ok(None);
    };
    if !path.is_file() {
        return Ok(None);
    }
    let table = super::TilingTable::load(&path)?;
    Ok(table.pool_gemv.iter().find_map(|entry| {
        (entry.geometry_matches(key)
            && entry.driver_build == key.driver_build
            && entry.hardware == key.hardware)
            .then_some(if entry.choice == "gemv" {
                Choice::Gemv
            } else {
                Choice::Blocked
            })
    }))
}

fn reuse_geometry(
    device: &CudaDevice,
    weight: &Arc<Tensor<bf16>>,
    shape: (usize, usize, usize),
    tile: LinearTiling,
    key: &Key<'_>,
    session: &ResidentTuningSession,
) -> Result<Option<Choice>> {
    let (rows, columns, outputs) = shape;
    let Some(choice) = cached_geometry(key)? else {
        return Ok(None);
    };
    if choice != Choice::Blocked
        && !verify_choice(device, weight, rows, columns, outputs, tile, choice)?
    {
        return Ok(None);
    }
    session.plan_gemm_search(
        SearchFamily::PoolDense,
        &[[rows, columns, outputs]],
        CACHE_STATES.len() * 2,
    )?;
    session.complete_gemm_search(SearchFamily::PoolDense, [rows, columns, outputs])?;
    Ok(Some(choice))
}

fn verify_choice(
    device: &CudaDevice,
    weight: &Arc<Tensor<bf16>>,
    rows: usize,
    columns: usize,
    outputs: usize,
    tile: LinearTiling,
    choice: Choice,
) -> Result<bool> {
    let chosen = run_choice(device, weight, rows, columns, outputs, tile, choice)?;
    let baseline = run_choice(
        device,
        weight,
        rows,
        columns,
        outputs,
        tile,
        Choice::Blocked,
    )?;
    Ok(chosen.host.len() == baseline.host.len()
        && chosen
            .host
            .iter()
            .zip(&baseline.host)
            .all(|(a, b)| close(*a, *b)))
}

fn kernel_fingerprint(
    device: &CudaDevice,
    rows: usize,
    columns: usize,
    outputs: usize,
    tile: LinearTiling,
) -> Result<String> {
    let input = api::meta::<f32>(&[rows, columns])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let weight = api::meta::<bf16>(&[outputs, columns])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut output = api::meta::<f32>(&[rows, outputs])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let gemv = gemv_kernels::dense((&mut output).partition([1, tile.rows()]), &input, &weight)
        .generics(vec![
            bf16::DTYPE.as_str().into(),
            tile.rows().to_string(),
            tile.columns().to_string(),
            columns.to_string(),
        ])
        .l2_cache_key_on(&device.stream)
        .map_err(device_error)?;
    let blocked = blocked_kernels::dense((&mut output).partition(SMALL_GEMM_TILE), &input, &weight)
        .generics(vec![bf16::DTYPE.as_str().into(), columns.to_string()])
        .l2_cache_key_on(&device.stream)
        .map_err(device_error)?;
    Ok(format!("{gemv}|{blocked}"))
}

fn weight_proof(device: &CudaDevice, weight: &Arc<Tensor<bf16>>) -> Result<String> {
    let values = device.read(weight)?;
    let mut hash = FNV_OFFSET;
    for value in values {
        for byte in value.to_bits().to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    Ok(format!("{hash:016x}"))
}

struct Run {
    graph: CudaGraph<()>,
    host: Vec<f32>,
}

fn run_choice(
    device: &CudaDevice,
    weight: &Arc<Tensor<bf16>>,
    rows: usize,
    columns: usize,
    outputs: usize,
    tile: LinearTiling,
    choice: Choice,
) -> Result<Run> {
    let input: Vec<f32> = (0..rows * columns)
        .map(|i| {
            (f32::from(u8::try_from(i % INPUT_MODULUS).unwrap_or(0)) - INPUT_CENTER) / INPUT_SCALE
        })
        .collect();
    let x = device.upload(input, &[rows, columns])?;
    let mut output = api::zeros::<f32>(&[rows, outputs])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        let recorded = match choice {
            Choice::Gemv => scope.record(
                gemv_kernels::dense((&mut output).partition([1, tile.rows()]), &x, weight)
                    .generics(vec![
                        bf16::DTYPE.as_str().into(),
                        tile.rows().to_string(),
                        tile.columns().to_string(),
                        columns.to_string(),
                    ]),
            ),
            Choice::Blocked => scope.record(
                blocked_kernels::dense((&mut output).partition(SMALL_GEMM_TILE), &x, weight)
                    .generics(vec![bf16::DTYPE.as_str().into(), columns.to_string()]),
            ),
        };
        recorded.map(drop)
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let host = device.read(&Arc::new(output))?;
    Ok(Run { graph, host })
}

#[expect(
    clippy::too_many_arguments,
    reason = "The measurement binds one actual weight and its full production geometry"
)]
fn measure(
    device: &CudaDevice,
    weight: &Arc<Tensor<bf16>>,
    rows: usize,
    columns: usize,
    outputs: usize,
    tile: LinearTiling,
    proof: &str,
    identity: &str,
    driver: u32,
    driver_build: &str,
    hardware: &str,
    fingerprint: &str,
) -> Result<Entry> {
    let gemv = run_choice(device, weight, rows, columns, outputs, tile, Choice::Gemv)?;
    let blocked = run_choice(
        device,
        weight,
        rows,
        columns,
        outputs,
        tile,
        Choice::Blocked,
    )?;
    if gemv.host.len() != blocked.host.len()
        || gemv
            .host
            .iter()
            .zip(&blocked.host)
            .any(|(a, b)| !close(*a, *b))
    {
        return Err(Error::invalid(
            "pool GEMV output differs beyond F32 tolerance",
        ));
    }
    let mut entry = Entry {
        device: identity.to_owned(),
        driver,
        driver_build: driver_build.to_owned(),
        hardware: hardware.to_owned(),
        kernel: KERNEL_ID.to_owned(),
        kernel_fingerprint: fingerprint.to_owned(),
        rows,
        columns,
        outputs,
        weight_proof: proof.to_owned(),
        choice: "blocked".into(),
        exact: false,
        cold_gemv_us: 0.0,
        cold_blocked_us: 0.0,
        warm_gemv_us: 0.0,
        warm_blocked_us: 0.0,
        cold_ab_blocked_samples: Vec::new(),
        cold_ab_gemv_samples: Vec::new(),
        cold_ba_blocked_samples: Vec::new(),
        cold_ba_gemv_samples: Vec::new(),
        warm_ab_blocked_samples: Vec::new(),
        warm_ab_gemv_samples: Vec::new(),
        warm_ba_blocked_samples: Vec::new(),
        warm_ba_gemv_samples: Vec::new(),
        gemv_faster: false,
    };
    let mut qualified = true;
    for cache in CACHE_STATES {
        let options = BenchOptions {
            warmup: Duration::from_millis(WARMUP_MS),
            rep: Duration::from_millis(REP_MS),
            min_reps: super::MIN_REPS,
            max_reps: super::MAX_REPS,
            clear_l2: cache,
        };
        let (base_ab, base_ab_samples, cand_ab, cand_ab_samples) =
            bench_pair(device, &blocked.graph, &gemv.graph, &options)?;
        let (cand_ba, cand_ba_samples, base_ba, base_ba_samples) =
            bench_pair(device, &gemv.graph, &blocked.graph, &options)?;
        let base = f32::midpoint(base_ab, base_ba);
        let candidate = f32::midpoint(cand_ab, cand_ba);
        if cache {
            entry.cold_blocked_us = f64::from(base * MICROSECONDS_PER_MILLISECOND);
            entry.cold_gemv_us = f64::from(candidate * MICROSECONDS_PER_MILLISECOND);
            entry.cold_ab_blocked_samples = base_ab_samples;
            entry.cold_ab_gemv_samples = cand_ab_samples;
            entry.cold_ba_blocked_samples = base_ba_samples;
            entry.cold_ba_gemv_samples = cand_ba_samples;
        } else {
            entry.warm_blocked_us = f64::from(base * MICROSECONDS_PER_MILLISECOND);
            entry.warm_gemv_us = f64::from(candidate * MICROSECONDS_PER_MILLISECOND);
            entry.warm_ab_blocked_samples = base_ab_samples;
            entry.warm_ab_gemv_samples = cand_ab_samples;
            entry.warm_ba_blocked_samples = base_ba_samples;
            entry.warm_ba_gemv_samples = cand_ba_samples;
        }
        // Both paired orders must clear the confirmation threshold. Averaging first would let
        // one strong order hide a regression in the other.
        let ab_ratio = base_ab / cand_ab;
        let ba_ratio = base_ba / cand_ba;
        if ab_ratio < MIN_CONFIRM_SPEEDUP || ba_ratio < MIN_CONFIRM_SPEEDUP {
            qualified = false;
        }
    }
    entry.choice = if qualified { "gemv" } else { "blocked" }.into();
    entry.gemv_faster = qualified;
    Ok(entry)
}

fn bench_pair(
    device: &CudaDevice,
    first: &CudaGraph<()>,
    second: &CudaGraph<()>,
    options: &BenchOptions,
) -> Result<(f32, Vec<f32>, f32, Vec<f32>)> {
    do_bench_paired(
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
    )
    .map(|(a, b)| {
        (
            a.median_ms(),
            a.times_ms().to_vec(),
            b.median_ms(),
            b.times_ms().to_vec(),
        )
    })
    .map_err(|e| device_error(e.to_string()))
}

fn pair_medians(base: &[f32], trial: &[f32]) -> Option<(f64, f64)> {
    if base.len() < 2 || trial.len() < 2 {
        return None;
    }
    let mut base = base.to_vec();
    let mut trial = trial.to_vec();
    if !base.iter().all(|v| v.is_finite() && *v > 0.0)
        || !trial.iter().all(|v| v.is_finite() && *v > 0.0)
    {
        return None;
    }
    base.sort_by(f32::total_cmp);
    trial.sort_by(f32::total_cmp);
    Some((f64::from(median(&base)), f64::from(median(&trial))))
}

const fn median(sorted: &[f32]) -> f32 {
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        f32::midpoint(sorted[middle - 1], sorted[middle])
    } else {
        sorted[middle]
    }
}

fn close64(a: f64, b: f64) -> bool {
    (a - b).abs() <= b.abs().mul_add(REL_TOLERANCE, ABS_TOLERANCE)
}

fn samples_ok(samples: &[f32]) -> bool {
    !samples.is_empty()
        && samples
            .iter()
            .all(|sample| sample.is_finite() && *sample > 0.0)
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= b.abs().mul_add(MATCH_TOLERANCE, MATCH_TOLERANCE)
}

#[cfg(test)]
#[path = "../../tests/unit/pool_gemv_tuning.rs"]
mod tests;
