//! Exact-output, quantization-inclusive autotuning of the fused prompt `SwiGLU` tile.
//!
//! The tuner runs at model load with the actual gate/up checkpoint weights. It first compares the
//! complete quantized code and scale tensors against the qualified 64x64 baseline, then measures
//! the two launch paths with the quantization epilogue included. A candidate is cached only when
//! that exact comparison and both paired orders confirm it.
use super::{MIN_CONFIRM_SPEEDUP, REP_MS, WARMUP_MS, update_table};
use crate::constants::NVFP4_GROUP_SIZE;
use crate::constants::NVFP4_QUANT_CODES_TILE;
use crate::constants::NVFP4_QUANT_SCALES_TILE;
use crate::device::{CudaDevice, DeviceProfile, device_error};
use crate::mlp::ProjectionWeight;
use crate::resident::nvfp4_gemm::kernels;
use crate::resident::nvfp4_gemm::workspace::{SwigluTile, swiglu_candidates};
use crate::tuning::resident_gemm::QUANTIZATION_ROWS;
use crate::tuning::resident_gemm::{ResidentTuningSession, SearchFamily};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::bench::{BenchOptions, do_bench_paired};
use cutile::error::tensor_error;
use cutile::prelude::*;
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

const REL_TOLERANCE: f64 = 1.0e-3;
const ABS_TOLERANCE: f64 = 1.0e-9;
const SEARCH_RANK: usize = 3;
const CACHE_STATES: usize = 2;
const MEASUREMENT_ARMS: usize = 2;

/// Kernel identity in the persistent table. Change when the packed kernel's arithmetic changes.
const KERNEL_ID: &str = concat!("packed-swiglu-v1/", env!("CARGO_PKG_VERSION"));
/// Fixed overhead for one shape's bookkeeping and launch setup.
/// Deterministic activation fixture bounds.
const INPUT_MODULUS: usize = 251;
const INPUT_SCALE: f32 = 97.0;
const INPUT_CENTER: f32 = 125.0;
/// Microseconds per millisecond for the benchmark API's millisecond medians.
const MICROSECONDS_PER_MILLISECOND: f32 = 1000.0;
/// FNV-1a constants for the loaded weight-scale proof.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// One measured tile decision, tied to device, kernel and exact geometry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub device: String,
    pub kernel: String,
    pub rows: usize,
    pub columns: usize,
    pub hidden: usize,
    #[serde(default)]
    pub weight_proof: String,
    pub driver_build: String,
    pub hardware: String,
    pub kernel_fingerprint: String,
    pub input_scale: f32,
    pub down_scale: f32,
    pub tile_rows: usize,
    pub tile_columns: usize,
    pub exact: bool,
    pub real_weights: bool,
    pub baseline_us: f64,
    pub candidate_us: f64,
    pub cold_base_us: f64,
    pub cold_trial_us: f64,
    pub warm_base_us: f64,
    pub warm_trial_us: f64,
    #[serde(default)]
    pub cold_ab_base_samples: Vec<f32>,
    #[serde(default)]
    pub cold_ab_trial_samples: Vec<f32>,
    #[serde(default)]
    pub cold_ba_base_samples: Vec<f32>,
    #[serde(default)]
    pub cold_ba_trial_samples: Vec<f32>,
    #[serde(default)]
    pub warm_ab_base_samples: Vec<f32>,
    #[serde(default)]
    pub warm_ab_trial_samples: Vec<f32>,
    #[serde(default)]
    pub warm_ba_base_samples: Vec<f32>,
    #[serde(default)]
    pub warm_ba_trial_samples: Vec<f32>,
    pub ab_base_us: f64,
    pub ab_trial_us: f64,
    pub ba_base_us: f64,
    pub ba_trial_us: f64,
    pub speedup: f64,
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

pub(crate) struct EntryKey<'a> {
    pub(crate) device: &'a str,
    pub(crate) weight_proof: &'a str,
    pub(crate) driver_build: &'a str,
    pub(crate) hardware: &'a str,
    pub(crate) kernel_fingerprint: &'a str,
    pub(crate) scales: (f32, f32),
    pub(crate) rows: usize,
    pub(crate) columns: usize,
    pub(crate) hidden: usize,
}

impl Entry {
    pub(crate) fn matches(&self, key: &EntryKey<'_>) -> bool {
        self.device == key.device
            && self.kernel == KERNEL_ID
            && !key.weight_proof.is_empty()
            && self.weight_proof == key.weight_proof
            && self.driver_build == key.driver_build
            && self.hardware == key.hardware
            && self.kernel_fingerprint == key.kernel_fingerprint
            && self.input_scale.to_bits() == key.scales.0.to_bits()
            && self.down_scale.to_bits() == key.scales.1.to_bits()
            && self.rows == key.rows
            && self.columns == key.columns
            && self.hidden == key.hidden
            && self.exact
            && self.real_weights
            && self.speedup >= 1.0
            && self.baseline_us.is_finite()
            && self.candidate_us.is_finite()
            && self.ab_base_us.is_finite()
            && self.ab_trial_us.is_finite()
            && self.ba_base_us.is_finite()
            && self.ba_trial_us.is_finite()
            && self.baseline_us > 0.0
            && self.candidate_us > 0.0
            && self.cold_base_us.is_finite()
            && self.cold_trial_us.is_finite()
            && self.warm_base_us.is_finite()
            && self.warm_trial_us.is_finite()
            && self.cold_base_us > 0.0
            && self.cold_trial_us > 0.0
            && self.warm_base_us > 0.0
            && self.warm_trial_us > 0.0
            && samples_ok(&self.cold_ab_base_samples)
            && samples_ok(&self.cold_ab_trial_samples)
            && samples_ok(&self.cold_ba_base_samples)
            && samples_ok(&self.cold_ba_trial_samples)
            && samples_ok(&self.warm_ab_base_samples)
            && samples_ok(&self.warm_ab_trial_samples)
            && samples_ok(&self.warm_ba_base_samples)
            && samples_ok(&self.warm_ba_trial_samples)
            && self.ab_base_us > 0.0
            && self.ab_trial_us > 0.0
            && self.ba_base_us > 0.0
            && self.ba_trial_us > 0.0
            && self.valid_evidence()
    }

    fn valid_evidence(&self) -> bool {
        let Some(cold_ab) = pair_medians(&self.cold_ab_base_samples, &self.cold_ab_trial_samples)
        else {
            return false;
        };
        let Some(cold_ba) = pair_medians(&self.cold_ba_base_samples, &self.cold_ba_trial_samples)
        else {
            return false;
        };
        let Some(warm_ab) = pair_medians(&self.warm_ab_base_samples, &self.warm_ab_trial_samples)
        else {
            return false;
        };
        let Some(warm_ba) = pair_medians(&self.warm_ba_base_samples, &self.warm_ba_trial_samples)
        else {
            return false;
        };
        let scale = f64::from(MICROSECONDS_PER_MILLISECOND);
        let cold_base = f64::midpoint(cold_ab.0, cold_ba.0) * scale;
        let cold_trial = f64::midpoint(cold_ab.1, cold_ba.1) * scale;
        let warm_base = f64::midpoint(warm_ab.0, warm_ba.0) * scale;
        let warm_trial = f64::midpoint(warm_ab.1, warm_ba.1) * scale;
        if !close64(cold_base, self.cold_base_us)
            || !close64(cold_trial, self.cold_trial_us)
            || !close64(warm_base, self.warm_base_us)
            || !close64(warm_trial, self.warm_trial_us)
            || !close64(warm_base, self.baseline_us)
            || !close64(warm_trial, self.candidate_us)
            || !close64(warm_ab.0 * scale, self.ab_base_us)
            || !close64(warm_ab.1 * scale, self.ab_trial_us)
            || !close64(warm_ba.0 * scale, self.ba_base_us)
            || !close64(warm_ba.1 * scale, self.ba_trial_us)
        {
            return false;
        }
        if self.tile_rows == SwigluTile::FALLBACK.rows
            && self.tile_columns == SwigluTile::FALLBACK.columns
        {
            return true;
        }
        let ratios: Vec<f64> = [cold_ab, cold_ba, warm_ab, warm_ba]
            .into_iter()
            .map(|(base, trial)| base / trial)
            .collect();
        if !ratios
            .iter()
            .all(|ratio| *ratio >= f64::from(MIN_CONFIRM_SPEEDUP))
        {
            return false;
        }
        let min = ratios.into_iter().fold(f64::INFINITY, f64::min);
        close64(min, self.speedup)
    }
}

pub(crate) fn complete_cache(
    session: &ResidentTuningSession,
    shape: [usize; SEARCH_RANK],
    candidates: usize,
) -> Result<()> {
    session.plan_gemm_search(
        SearchFamily::FusedSwiGLU,
        &[shape],
        candidates * CACHE_STATES * MEASUREMENT_ARMS,
    )?;
    session.complete_gemm_search(SearchFamily::FusedSwiGLU, shape)
}

/// Select a proven tile, or measure candidates with the loaded weights.
/// # Errors
/// Returns a pricing/load error when the weights cannot be identified or a candidate search
/// cannot be started. Candidate-local launch failures are logged and the conservative tile wins.
///
/// `autotune == false` still uses cached evidence, but never measures a missing shape.
/// Returns the conservative tile for a missing/failed search and records that fact in the table
/// when persistence is available.
pub(crate) fn select(
    device: &CudaDevice,
    profile: &DeviceProfile,
    gate: &ProjectionWeight,
    up: &ProjectionWeight,
    contract: Contract,
    autotune: bool,
    session: &ResidentTuningSession,
) -> Result<SwigluTile> {
    let (rows, columns, hidden) = contract.shape;
    let scales = contract.scales;
    let candidates = swiglu_candidates(rows, columns, hidden);
    let Some(fallback) = candidates.first().copied() else {
        return Err(Error::invalid("no legal fused SwiGLU tile"));
    };
    let identity = format!(
        "{}|driver={}",
        profile.identity(),
        CudaDevice::driver_version()?
    );
    let driver_build = super::driver_build().unwrap_or_default();
    let hardware = serde_json::to_string(profile).map_err(device_error)?;
    let fingerprint = format!(
        "{}:kernel={}",
        super::compiler_fingerprint(),
        kernel_fingerprint(device, gate, up, rows, columns, hidden)?
    );
    let proof = weight_proof(device, gate, up, scales)?;
    let cached_context = CachedContext {
        identity: &identity,
        weight_proof: &proof,
        driver_build: &driver_build,
        hardware: &hardware,
        fingerprint: &fingerprint,
        shape: contract.shape,
        scales,
        candidates: &candidates,
    };
    if let Some(tile) = verified_cached(
        device,
        gate,
        up,
        contract.shape,
        scales,
        fallback,
        cached_context,
    ) {
        complete_cache(session, [rows, columns, hidden], candidates.len())?;
        return Ok(tile);
    }
    if !autotune {
        return Ok(fallback);
    }
    if !session.permits_gemm_search(
        SearchFamily::FusedSwiGLU,
        [rows, columns, hidden],
        candidates.len() * CACHE_STATES * MEASUREMENT_ARMS,
    )? {
        return Ok(fallback);
    }
    match measure(device, gate, up, contract, &candidates) {
        Ok(selected) if selected.complete => {
            complete_cache(session, [rows, columns, hidden], candidates.len())?;
            persist(&cached_context, &selected);
            Ok(selected.tile)
        }
        Ok(_) => {
            tracing::warn!(
                target: "infer::tuning",
                rows, columns, hidden,
                "fused SwiGLU search budget exhausted; keeping uncovered fallback"
            );
            Ok(fallback)
        }
        Err(error) => {
            tracing::warn!(
                target: "infer::tuning",
                rows, columns, hidden, %error,
                "fused SwiGLU search failed; retaining 64x64 fallback"
            );
            Ok(fallback)
        }
    }
}

fn cached(context: &CachedContext<'_>) -> Result<Option<SwigluTile>> {
    let Some(path) = super::table_path() else {
        return Ok(None);
    };
    if !path.is_file() {
        return Ok(None);
    }
    let table = super::TilingTable::load(&path)?;
    Ok(table
        .swiglu
        .iter()
        .find_map(|entry| {
            entry
                .matches(&EntryKey {
                    device: context.identity,
                    weight_proof: context.weight_proof,
                    driver_build: context.driver_build,
                    hardware: context.hardware,
                    kernel_fingerprint: context.fingerprint,
                    scales: context.scales,
                    rows: context.shape.0,
                    columns: context.shape.1,
                    hidden: context.shape.2,
                })
                .then_some(SwigluTile {
                    rows: entry.tile_rows,
                    columns: entry.tile_columns,
                })
        })
        .filter(|tile| context.candidates.contains(tile)))
}

fn persist(context: &CachedContext<'_>, selected: &Candidate) {
    let (rows, columns, hidden) = context.shape;
    let entry = Entry {
        device: context.identity.to_owned(),
        kernel: KERNEL_ID.to_owned(),
        rows,
        columns,
        hidden,
        weight_proof: context.weight_proof.to_owned(),
        driver_build: context.driver_build.to_owned(),
        hardware: context.hardware.to_owned(),
        kernel_fingerprint: context.fingerprint.to_owned(),
        input_scale: context.scales.0,
        down_scale: context.scales.1,
        tile_rows: selected.tile.rows,
        tile_columns: selected.tile.columns,
        exact: true,
        real_weights: true,
        baseline_us: f64::from(selected.warm.base),
        candidate_us: f64::from(selected.warm.trial),
        cold_base_us: f64::from(selected.cold.base),
        cold_trial_us: f64::from(selected.cold.trial),
        warm_base_us: f64::from(selected.warm.base),
        warm_trial_us: f64::from(selected.warm.trial),
        cold_ab_base_samples: selected.cold.ab_base_samples.clone(),
        cold_ab_trial_samples: selected.cold.ab_trial_samples.clone(),
        cold_ba_base_samples: selected.cold.ba_base_samples.clone(),
        cold_ba_trial_samples: selected.cold.ba_trial_samples.clone(),
        warm_ab_base_samples: selected.warm.ab_base_samples.clone(),
        warm_ab_trial_samples: selected.warm.ab_trial_samples.clone(),
        warm_ba_base_samples: selected.warm.ba_base_samples.clone(),
        warm_ba_trial_samples: selected.warm.ba_trial_samples.clone(),
        ab_base_us: f64::from(selected.warm.ab_base),
        ab_trial_us: f64::from(selected.warm.ab_trial),
        ba_base_us: f64::from(selected.warm.ba_base),
        ba_trial_us: f64::from(selected.warm.ba_trial),
        speedup: f64::from(selected.speedup),
    };
    update_table(super::table_path().as_deref(), |table| {
        table.swiglu.retain(|old| {
            !(old.device == entry.device
                && old.kernel == entry.kernel
                && old.rows == entry.rows
                && old.columns == entry.columns
                && old.hidden == entry.hidden
                && old.weight_proof == entry.weight_proof
                && old.driver_build == entry.driver_build
                && old.hardware == entry.hardware
                && old.kernel_fingerprint == entry.kernel_fingerprint
                && old.input_scale.to_bits() == entry.input_scale.to_bits()
                && old.down_scale.to_bits() == entry.down_scale.to_bits())
        });
        table.swiglu.push(entry);
    });
}

#[derive(Clone, Copy)]
struct CachedContext<'a> {
    identity: &'a str,
    weight_proof: &'a str,
    driver_build: &'a str,
    hardware: &'a str,
    fingerprint: &'a str,
    shape: (usize, usize, usize),
    scales: (f32, f32),
    candidates: &'a [SwigluTile],
}

fn verified_cached(
    device: &CudaDevice,
    gate: &ProjectionWeight,
    up: &ProjectionWeight,
    shape: (usize, usize, usize),
    scales: (f32, f32),
    fallback: SwigluTile,
    context: CachedContext<'_>,
) -> Option<SwigluTile> {
    let (rows, columns, hidden) = shape;
    let tile = match cached(&context) {
        Ok(tile) => tile?,
        Err(error) => {
            tracing::warn!(target: "infer::tuning", %error, "invalid SwiGLU tuning cache; remeasuring");
            return None;
        }
    };
    match verify_cached(device, gate, up, shape, scales, fallback, tile) {
        Ok(true) => Some(tile),
        Ok(false) => {
            tracing::warn!(
                target: "infer::tuning",
                rows, columns, hidden, ?tile,
                "cached SwiGLU choice failed actual-weight exact-output verification"
            );
            None
        }
        Err(error) => {
            tracing::warn!(
                target: "infer::tuning", rows, columns, hidden, %error,
                "cached SwiGLU choice verification failed"
            );
            None
        }
    }
}

fn verify_cached(
    device: &CudaDevice,
    gate: &ProjectionWeight,
    up: &ProjectionWeight,
    shape: (usize, usize, usize),
    scales: (f32, f32),
    fallback: SwigluTile,
    tile: SwigluTile,
) -> Result<bool> {
    let baseline = run_for(device, gate, up, Contract { shape, scales }, fallback)?;
    let candidate = run_for(device, gate, up, Contract { shape, scales }, tile)?;
    Ok(baseline.bytes == candidate.bytes)
}

fn kernel_fingerprint(
    device: &CudaDevice,
    gate: &ProjectionWeight,
    up: &ProjectionWeight,
    rows: usize,
    columns: usize,
    hidden: usize,
) -> Result<String> {
    let (
        ProjectionWeight::Fp4(gate_codes, gate_scales, _),
        ProjectionWeight::Fp4(up_codes, up_scales, _),
    ) = (gate, up)
    else {
        return Err(Error::invalid("fused SwiGLU requires FP4 gate/up weights"));
    };
    let input = api::meta::<f32>(&[rows, columns])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut q = api::meta::<f4e2m1fnx2>(&[rows, columns / 2])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut qs = api::meta::<f8e4m3fn>(&[rows, columns / NVFP4_GROUP_SIZE])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut identities = vec![
        kernels::quantize_rows(
            (&mut q).partition([QUANTIZATION_ROWS, NVFP4_QUANT_CODES_TILE[1]]),
            (&mut qs).partition([QUANTIZATION_ROWS, NVFP4_QUANT_SCALES_TILE[1]]),
            &input,
            1.0,
        )
        .generics(vec![columns.to_string()])
        .l2_cache_key_on(&device.stream)
        .map_err(device_error)?,
    ];
    for tile in swiglu_candidates(rows, columns, hidden) {
        let mut codes = api::meta::<f4e2m1fnx2>(&[rows, hidden / 2])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let mut scales = api::meta::<f8e4m3fn>(&[rows, hidden / NVFP4_GROUP_SIZE])
            .sync_on(&device.stream)
            .map_err(device_error)?;
        identities.push(
            kernels::packed_swiglu(
                (&mut codes).partition([tile.rows, tile.columns / 2]),
                (&mut scales).partition([tile.rows, tile.columns / NVFP4_GROUP_SIZE]),
                &q,
                &qs,
                gate_codes,
                gate_scales,
                up_codes,
                up_scales,
                1.0,
                1.0,
                1.0,
            )
            .generics(
                [
                    columns,
                    tile.rows,
                    tile.columns,
                    tile.columns / 2,
                    tile.columns / NVFP4_GROUP_SIZE,
                ]
                .iter()
                .map(ToString::to_string)
                .collect(),
            )
            .l2_cache_key_on(&device.stream)
            .map_err(device_error)?,
        );
    }
    serde_json::to_string(&identities).map_err(device_error)
}

fn weight_proof(
    device: &CudaDevice,
    gate: &ProjectionWeight,
    up: &ProjectionWeight,
    scales: (f32, f32),
) -> Result<String> {
    let (
        ProjectionWeight::Fp4(gate_codes, gate_scales, gate_global),
        ProjectionWeight::Fp4(up_codes, up_scales, up_global),
    ) = (gate, up)
    else {
        return Err(Error::invalid("fused SwiGLU requires FP4 gate/up weights"));
    };
    let mut hash = FNV_OFFSET;
    let mut absorb = |bytes: &[u8]| {
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    };
    for tensor in [gate_codes, up_codes] {
        for dimension in tensor.shape() {
            absorb(&dimension.to_le_bytes());
        }
    }
    for scales in [gate_scales, up_scales] {
        let bytes: Vec<u8> = device
            .read(scales)?
            .into_iter()
            .map(|scale| scale.0)
            .collect();
        absorb(&bytes);
    }
    absorb(&gate_global.to_bits().to_le_bytes());
    absorb(&up_global.to_bits().to_le_bytes());
    absorb(&scales.0.to_bits().to_le_bytes());
    absorb(&scales.1.to_bits().to_le_bytes());
    Ok(format!("{hash:016x}"))
}

struct Evidence {
    base: f32,
    trial: f32,
    ab_base: f32,
    ab_trial: f32,
    ba_base: f32,
    ba_trial: f32,
    ab_base_samples: Vec<f32>,
    ab_trial_samples: Vec<f32>,
    ba_base_samples: Vec<f32>,
    ba_trial_samples: Vec<f32>,
}

#[derive(Clone, Copy)]
pub(crate) struct Contract {
    pub(crate) shape: (usize, usize, usize),
    pub(crate) scales: (f32, f32),
}

struct Candidate {
    tile: SwigluTile,
    cold: Evidence,
    warm: Evidence,
    speedup: f32,
    complete: bool,
}

fn measure(
    device: &CudaDevice,
    gate: &ProjectionWeight,
    up: &ProjectionWeight,
    contract: Contract,
    candidates: &[SwigluTile],
) -> Result<Candidate> {
    let (rows, columns, hidden) = contract.shape;
    let fallback = candidates[0];
    let baseline = run_for(device, gate, up, contract, fallback)?;
    let mut best = Candidate {
        tile: fallback,
        cold: Evidence {
            base: 0.0,
            trial: 0.0,
            ab_base: 0.0,
            ab_trial: 0.0,
            ba_base: 0.0,
            ba_trial: 0.0,
            ab_base_samples: Vec::new(),
            ab_trial_samples: Vec::new(),
            ba_base_samples: Vec::new(),
            ba_trial_samples: Vec::new(),
        },
        warm: Evidence {
            base: 0.0,
            trial: 0.0,
            ab_base: 0.0,
            ab_trial: 0.0,
            ba_base: 0.0,
            ba_trial: 0.0,
            ab_base_samples: Vec::new(),
            ab_trial_samples: Vec::new(),
            ba_base_samples: Vec::new(),
            ba_trial_samples: Vec::new(),
        },
        speedup: 1.0,
        complete: false,
    };
    let mut inspected = 0;
    let mut failed = false;
    for &tile in candidates {
        inspected += 1;
        let mut cold = None;
        let mut warm = None;
        let mut speedup = f32::INFINITY;
        let mut accepted = true;
        for cache in [true, false] {
            let options = BenchOptions {
                warmup: Duration::from_millis(WARMUP_MS),
                rep: Duration::from_millis(REP_MS),
                min_reps: super::MIN_REPS,
                max_reps: super::MAX_REPS,
                clear_l2: cache,
            };
            let (evidence, _) =
                match evaluate_tile(device, &baseline, gate, up, contract, &options, tile) {
                    Ok(value) => value,
                    Err(error) => {
                        tracing::warn!(target:"infer::tuning", rows, columns, hidden, ?tile, %error,
                            "fused SwiGLU candidate rejected");
                        accepted = false;
                        failed = true;
                        break;
                    }
                };
            let ab = evidence.ab_base / evidence.ab_trial;
            let ba = evidence.ba_base / evidence.ba_trial;
            if tile != fallback && (ab < MIN_CONFIRM_SPEEDUP || ba < MIN_CONFIRM_SPEEDUP) {
                accepted = false;
                break;
            }
            if tile != fallback {
                speedup = speedup.min(ab.min(ba));
            }
            if cache {
                cold = Some(evidence);
            } else {
                warm = Some(evidence);
            }
        }
        if !accepted {
            continue;
        }
        let (Some(cold), Some(warm)) = (cold, warm) else {
            continue;
        };
        if tile == fallback {
            best.cold = cold;
            best.warm = warm;
        } else if speedup > best.speedup {
            best = Candidate {
                tile,
                cold,
                warm,
                speedup,
                complete: false,
            };
        }
    }
    best.complete = inspected == candidates.len() && !failed;
    if best.cold.base == 0.0 || best.warm.base == 0.0 {
        return Err(Error::invariant("fused SwiGLU baseline evidence missing"));
    }
    Ok(best)
}

fn evaluate_tile(
    device: &CudaDevice,
    baseline: &Run,
    gate: &ProjectionWeight,
    up: &ProjectionWeight,
    contract: Contract,
    options: &BenchOptions,
    tile: SwigluTile,
) -> Result<(Evidence, f32)> {
    let candidate = run_for(device, gate, up, contract, tile)?;
    if candidate.bytes != baseline.bytes {
        return Err(Error::invalid(
            "fused SwiGLU candidate failed exact-output proof",
        ));
    }
    let (base_ab, base_ab_samples, cand_ab, cand_ab_samples) =
        bench_pair(device, &baseline.graph, &candidate.graph, options)?;
    let (cand_ba, cand_ba_samples, base_ba, base_ba_samples) =
        bench_pair(device, &candidate.graph, &baseline.graph, options)?;
    let base = f32::midpoint(base_ab, base_ba);
    let candidate = f32::midpoint(cand_ab, cand_ba);
    let evidence = Evidence {
        base: base * MICROSECONDS_PER_MILLISECOND,
        trial: candidate * MICROSECONDS_PER_MILLISECOND,
        ab_base: base_ab * MICROSECONDS_PER_MILLISECOND,
        ab_trial: cand_ab * MICROSECONDS_PER_MILLISECOND,
        ba_base: base_ba * MICROSECONDS_PER_MILLISECOND,
        ba_trial: cand_ba * MICROSECONDS_PER_MILLISECOND,
        ab_base_samples: base_ab_samples,
        ab_trial_samples: cand_ab_samples,
        ba_base_samples: base_ba_samples,
        ba_trial_samples: cand_ba_samples,
    };
    Ok((evidence, base / candidate))
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

fn run_for(
    device: &CudaDevice,
    gate: &ProjectionWeight,
    up: &ProjectionWeight,
    contract: Contract,
    tile: SwigluTile,
) -> Result<Run> {
    let (rows, columns, hidden) = contract.shape;
    let (
        ProjectionWeight::Fp4(gate_codes, gate_scales, gate_global),
        ProjectionWeight::Fp4(up_codes, up_scales, up_global),
    ) = (gate, up)
    else {
        return Err(Error::invalid("fused SwiGLU requires FP4 gate/up weights"));
    };
    run_once(
        device,
        tile,
        rows,
        columns,
        hidden,
        gate_codes,
        gate_scales,
        up_codes,
        up_scales,
        *gate_global,
        *up_global,
        contract.scales,
    )
}

/// The captured graph owns the recorded tensor storage: cutile's `Storage::retain(recording)`
/// installs a lease and cuda-async's graph retains it across replays. The host tensor handles may
/// therefore drop; this is the installed dependency contract, not an assumption about `()` output.
struct Run {
    graph: CudaGraph<()>,
    bytes: (Vec<u8>, Vec<u8>),
}

#[expect(
    clippy::too_many_arguments,
    reason = "The tuner binds both checkpoint projections and their exact fixture geometry"
)]
fn run_once(
    device: &CudaDevice,
    tile: SwigluTile,
    rows: usize,
    columns: usize,
    hidden: usize,
    gate_codes: &Tensor<f4e2m1fnx2>,
    gate_scales: &Tensor<f8e4m3fn>,
    up_codes: &Tensor<f4e2m1fnx2>,
    up_scales: &Tensor<f8e4m3fn>,
    gate_global: f32,
    up_global: f32,
    scales: (f32, f32),
) -> Result<Run> {
    let input: Vec<f32> = (0..rows * columns)
        .map(|i| {
            (f32::from(u8::try_from(i % INPUT_MODULUS).unwrap_or(0)) - INPUT_CENTER) / INPUT_SCALE
        })
        .collect();
    let x = device.upload(input, &[rows, columns])?;
    let mut q = api::zeros::<f4e2m1fnx2>(&[rows, columns / 2])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut qs = api::zeros::<f8e4m3fn>(&[rows, columns / NVFP4_GROUP_SIZE])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut codes = api::zeros::<f4e2m1fnx2>(&[rows, hidden / 2])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut output_scales = api::zeros::<f8e4m3fn>(&[rows, hidden / NVFP4_GROUP_SIZE])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        scope.record(
            kernels::quantize_rows(
                (&mut q).partition([QUANTIZATION_ROWS, NVFP4_QUANT_CODES_TILE[1]]),
                (&mut qs).partition([QUANTIZATION_ROWS, NVFP4_QUANT_SCALES_TILE[1]]),
                &x,
                scales.0,
            )
            .generics(vec![columns.to_string()]),
        )?;
        scope
            .record(
                kernels::packed_swiglu(
                    (&mut codes).partition([tile.rows, tile.columns / 2]),
                    (&mut output_scales).partition([tile.rows, tile.columns / NVFP4_GROUP_SIZE]),
                    &q,
                    &qs,
                    gate_codes,
                    gate_scales,
                    up_codes,
                    up_scales,
                    1.0 / (gate_global * scales.0),
                    1.0 / (up_global * scales.0),
                    scales.1,
                )
                .generics(
                    [
                        columns,
                        tile.rows,
                        tile.columns,
                        tile.columns / 2,
                        tile.columns / NVFP4_GROUP_SIZE,
                    ]
                    .iter()
                    .map(ToString::to_string)
                    .collect(),
                ),
            )
            .map(drop)
    })
    .map_err(device_error)?;
    graph
        .launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let code_bytes: Vec<u8> = device
        .read(&Arc::new(codes))?
        .into_iter()
        .map(f4e2m1fnx2::to_bits)
        .collect();
    let scale_bytes: Vec<u8> = device
        .read(&Arc::new(output_scales))?
        .into_iter()
        .map(|code| code.0)
        .collect();
    Ok(Run {
        graph,
        bytes: (code_bytes, scale_bytes),
    })
}
