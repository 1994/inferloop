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
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::bench::{BenchOptions, do_bench_paired};
use cutile::error::tensor_error;
use cutile::prelude::*;
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Kernel identity in the persistent table. Change when the packed kernel's arithmetic changes.
const KERNEL_ID: &str = "packed-swiglu-v1";
/// Per-candidate wall clock allowance used to bound one shape's search.
const CANDIDATE_BUDGET: Duration = Duration::from_millis(2_000);
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
    pub tile_rows: usize,
    pub tile_columns: usize,
    pub exact: bool,
    pub real_weights: bool,
    pub baseline_us: f64,
    pub candidate_us: f64,
    pub speedup: f64,
}

impl Entry {
    pub(crate) fn matches(
        &self,
        device: &str,
        proof: &str,
        rows: usize,
        columns: usize,
        hidden: usize,
    ) -> bool {
        self.device == device
            && self.kernel == KERNEL_ID
            && !proof.is_empty()
            && self.weight_proof == proof
            && self.rows == rows
            && self.columns == columns
            && self.hidden == hidden
            && self.exact
            && self.real_weights
            && self.speedup >= 1.0
    }
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
    shape: (usize, usize, usize),
    autotune: bool,
) -> Result<SwigluTile> {
    let (rows, columns, hidden) = shape;
    let candidates = swiglu_candidates(rows, columns, hidden);
    let Some(fallback) = candidates.first().copied() else {
        return Err(Error::invalid("no legal fused SwiGLU tile"));
    };
    let identity = profile.identity();
    let proof = weight_proof(device, gate, up)?;
    let cached = match cached(&identity, &proof, rows, columns, hidden, &candidates) {
        Ok(tile) => tile,
        Err(error) => {
            tracing::warn!(target: "infer::tuning", %error, "invalid SwiGLU tuning cache; remeasuring");
            None
        }
    };
    if let Some(tile) = cached {
        return Ok(tile);
    }
    if !autotune {
        return Ok(fallback);
    }
    match measure(device, gate, up, rows, columns, hidden, &candidates) {
        Ok(selected) => {
            persist(&identity, &proof, shape, &selected);
            Ok(selected.tile)
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

fn cached(
    identity: &str,
    proof: &str,
    rows: usize,
    columns: usize,
    hidden: usize,
    candidates: &[SwigluTile],
) -> Result<Option<SwigluTile>> {
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
                .matches(identity, proof, rows, columns, hidden)
                .then_some(SwigluTile {
                    rows: entry.tile_rows,
                    columns: entry.tile_columns,
                })
        })
        .filter(|tile| candidates.contains(tile)))
}

fn persist(
    identity: &str,
    proof: &str,
    (rows, columns, hidden): (usize, usize, usize),
    selected: &Candidate,
) {
    let entry = Entry {
        device: identity.to_owned(),
        kernel: KERNEL_ID.to_owned(),
        rows,
        columns,
        hidden,
        weight_proof: proof.to_owned(),
        tile_rows: selected.tile.rows,
        tile_columns: selected.tile.columns,
        exact: true,
        real_weights: true,
        baseline_us: f64::from(selected.base_micros),
        candidate_us: f64::from(selected.trial_micros),
        speedup: f64::from(selected.speedup),
    };
    update_table(super::table_path().as_deref(), |table| {
        table.swiglu.retain(|old| {
            !(old.device == entry.device
                && old.kernel == entry.kernel
                && old.rows == entry.rows
                && old.columns == entry.columns
                && old.hidden == entry.hidden)
        });
        table.swiglu.push(entry);
    });
}

fn weight_proof(
    device: &CudaDevice,
    gate: &ProjectionWeight,
    up: &ProjectionWeight,
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
    Ok(format!("{hash:016x}"))
}

pub(crate) fn budget_for(candidates: usize) -> Duration {
    CANDIDATE_BUDGET * u32::try_from(candidates.max(1)).unwrap_or(u32::MAX)
}

struct Candidate {
    tile: SwigluTile,
    base_micros: f32,
    trial_micros: f32,
    speedup: f32,
}

fn measure(
    device: &CudaDevice,
    gate: &ProjectionWeight,
    up: &ProjectionWeight,
    rows: usize,
    columns: usize,
    hidden: usize,
    candidates: &[SwigluTile],
) -> Result<Candidate> {
    let fallback = candidates[0];
    let baseline = run_for(device, gate, up, (rows, columns, hidden), fallback)?;
    let options = BenchOptions {
        warmup: Duration::from_millis(WARMUP_MS),
        rep: Duration::from_millis(REP_MS),
        min_reps: super::MIN_REPS,
        max_reps: super::MAX_REPS,
        clear_l2: true,
    };
    let deadline = Instant::now() + budget_for(candidates.len());
    let mut best = Candidate {
        tile: fallback,
        base_micros: 0.0,
        trial_micros: 0.0,
        speedup: 1.0,
    };
    for &tile in candidates {
        if Instant::now() >= deadline {
            break;
        }
        match evaluate_tile(
            device,
            &baseline,
            gate,
            up,
            (rows, columns, hidden),
            &options,
            tile,
        ) {
            Ok((base, cand, speedup)) => {
                if tile == fallback {
                    best.base_micros = base;
                    best.trial_micros = cand;
                } else if speedup >= MIN_CONFIRM_SPEEDUP {
                    best = Candidate {
                        tile,
                        base_micros: base,
                        trial_micros: cand,
                        speedup,
                    };
                }
            }
            Err(error) => {
                tracing::warn!(target:"infer::tuning", rows, columns, hidden, ?tile, %error,
                    "fused SwiGLU candidate rejected");
            }
        }
    }
    if best.base_micros == 0.0 {
        // Even a no-win shape must keep the measured baseline identity for audits.
        let (base, cand) = bench_pair(device, &baseline.graph, &baseline.graph, &options)?;
        best.base_micros = base * MICROSECONDS_PER_MILLISECOND;
        best.trial_micros = cand * MICROSECONDS_PER_MILLISECOND;
    }
    Ok(best)
}

fn evaluate_tile(
    device: &CudaDevice,
    baseline: &Run,
    gate: &ProjectionWeight,
    up: &ProjectionWeight,
    (rows, columns, hidden): (usize, usize, usize),
    options: &BenchOptions,
    tile: SwigluTile,
) -> Result<(f32, f32, f32)> {
    let candidate = run_for(device, gate, up, (rows, columns, hidden), tile)?;
    if candidate.bytes != baseline.bytes {
        return Err(Error::invalid(
            "fused SwiGLU candidate failed exact-output proof",
        ));
    }
    let (base_ab, cand_ab) = bench_pair(device, &baseline.graph, &candidate.graph, options)?;
    let (cand_ba, base_ba) = bench_pair(device, &candidate.graph, &baseline.graph, options)?;
    let base = f32::midpoint(base_ab, base_ba);
    let candidate = f32::midpoint(cand_ab, cand_ba);
    let base_micros = base * MICROSECONDS_PER_MILLISECOND;
    let trial_micros = candidate * MICROSECONDS_PER_MILLISECOND;
    Ok((base_micros, trial_micros, base / candidate))
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

fn run_for(
    device: &CudaDevice,
    gate: &ProjectionWeight,
    up: &ProjectionWeight,
    (rows, columns, hidden): (usize, usize, usize),
    tile: SwigluTile,
) -> Result<Run> {
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
    )
}

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
    kernels::quantize_rows(
        (&mut q).partition([QUANTIZATION_ROWS, NVFP4_QUANT_CODES_TILE[1]]),
        (&mut qs).partition([QUANTIZATION_ROWS, NVFP4_QUANT_SCALES_TILE[1]]),
        &x,
        1.0,
    )
    .generics(vec![columns.to_string()])
    .sync_on(&device.stream)
    .map_err(device_error)?;
    let mut codes = api::zeros::<f4e2m1fnx2>(&[rows, hidden / 2])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let mut scales = api::zeros::<f8e4m3fn>(&[rows, hidden / NVFP4_GROUP_SIZE])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        scope
            .record(
                kernels::packed_swiglu(
                    (&mut codes).partition([tile.rows, tile.columns / 2]),
                    (&mut scales).partition([tile.rows, tile.columns / NVFP4_GROUP_SIZE]),
                    &q,
                    &qs,
                    gate_codes,
                    gate_scales,
                    up_codes,
                    up_scales,
                    1.0 / (gate_global * 1.0),
                    1.0 / (up_global * 1.0),
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
        .read(&Arc::new(scales))?
        .into_iter()
        .map(|code| code.0)
        .collect();
    Ok(Run {
        graph,
        bytes: (code_bytes, scale_bytes),
    })
}
