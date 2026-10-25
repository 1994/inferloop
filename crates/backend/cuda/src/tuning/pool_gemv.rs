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
    pub kernel: String,
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
    pub gemv_faster: bool,
}

impl Entry {
    fn matches(
        &self,
        identity: &str,
        driver: u32,
        rows: usize,
        columns: usize,
        outputs: usize,
        proof: &str,
    ) -> bool {
        self.device == identity
            && self.driver == driver
            && self.kernel == KERNEL_ID
            && self.rows == rows
            && self.columns == columns
            && self.outputs == outputs
            && self.weight_proof == proof
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
    if let Some(choice) = cached(&identity, driver, rows, columns, outputs, &proof)? {
        return Ok(choice);
    }
    if !autotune
        || !session.permits_gemm_search(
            SearchFamily::Nvfp4,
            [rows, columns, outputs],
            CACHE_STATES.len() * 2,
        )?
    {
        return Ok(legacy_choice(rows));
    }
    let result = measure(
        device, dense, rows, columns, outputs, tile, &proof, &identity, driver,
    );
    let _ = session.complete_gemm_search(SearchFamily::Nvfp4, [rows, columns, outputs]);
    match result {
        Ok(entry) => {
            update_table(super::table_path().as_deref(), |table| {
                table.pool_gemv.retain(|old| {
                    !(old.device == entry.device
                        && old.rows == entry.rows
                        && old.columns == entry.columns
                        && old.outputs == entry.outputs)
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

fn cached(
    identity: &str,
    driver: u32,
    rows: usize,
    columns: usize,
    outputs: usize,
    proof: &str,
) -> Result<Option<Choice>> {
    let Some(path) = super::table_path() else {
        return Ok(None);
    };
    if !path.is_file() {
        return Ok(None);
    }
    let table = super::TilingTable::load(&path)?;
    Ok(table.pool_gemv.iter().find_map(|entry| {
        entry
            .matches(identity, driver, rows, columns, outputs, proof)
            .then_some(if entry.choice == "gemv" {
                Choice::Gemv
            } else {
                Choice::Blocked
            })
    }))
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
        kernel: KERNEL_ID.to_owned(),
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
        gemv_faster: false,
    };
    for cache in CACHE_STATES {
        let options = BenchOptions {
            warmup: Duration::from_millis(WARMUP_MS),
            rep: Duration::from_millis(REP_MS),
            min_reps: super::MIN_REPS,
            max_reps: super::MAX_REPS,
            clear_l2: cache,
        };
        let (base_ab, cand_ab) = bench_pair(device, &blocked.graph, &gemv.graph, &options)?;
        let (cand_ba, base_ba) = bench_pair(device, &gemv.graph, &blocked.graph, &options)?;
        let base = f32::midpoint(base_ab, base_ba);
        let candidate = f32::midpoint(cand_ab, cand_ba);
        if cache {
            entry.cold_blocked_us = f64::from(base * MICROSECONDS_PER_MILLISECOND);
            entry.cold_gemv_us = f64::from(candidate * MICROSECONDS_PER_MILLISECOND);
        } else {
            entry.warm_blocked_us = f64::from(base * MICROSECONDS_PER_MILLISECOND);
            entry.warm_gemv_us = f64::from(candidate * MICROSECONDS_PER_MILLISECOND);
        }
        // Both paired orders must clear the confirmation threshold. Averaging first would let
        // one strong order hide a regression in the other.
        let ab_ratio = base_ab / cand_ab;
        let ba_ratio = base_ba / cand_ba;
        if ab_ratio < MIN_CONFIRM_SPEEDUP || ba_ratio < MIN_CONFIRM_SPEEDUP {
            entry.choice = "blocked".into();
            entry.gemv_faster = false;
            return Ok(entry);
        }
    }
    if entry.cold_blocked_us / entry.cold_gemv_us >= f64::from(MIN_CONFIRM_SPEEDUP)
        && entry.warm_blocked_us / entry.warm_gemv_us >= f64::from(MIN_CONFIRM_SPEEDUP)
    {
        entry.choice = "gemv".into();
        entry.gemv_faster = true;
    }
    Ok(entry)
}

fn bench_pair(
    device: &CudaDevice,
    first: &CudaGraph<()>,
    second: &CudaGraph<()>,
    options: &BenchOptions,
) -> Result<(f32, f32)> {
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
    .map(|(a, b)| (a.median_ms(), b.median_ms()))
    .map_err(|e| device_error(e.to_string()))
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= b.abs().mul_add(MATCH_TOLERANCE, MATCH_TOLERANCE)
}

#[cfg(test)]
#[path = "../../tests/unit/pool_gemv_tuning.rs"]
mod tests;
