//! Actual-checkpoint down-projection split candidate driver (device-only, ignored).

use super::{kernels, projection_split_policy, split_scratch_bytes};
use crate::device::CudaDevice;
use crate::loading::{LoadOptions, LoadedModel};
use crate::mlp::ProjectionWeight;
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::bench::{BenchOptions, do_bench_paired};
use cutile::error::tensor_error;
use cutile::prelude::*;
use infer_core::{Error, ModelId};
use sha2::{Digest, Sha256};
use std::env::{self, var};
use std::error::Error as StdError;
use std::fs;
use std::sync::Arc;
use std::time::Duration;

const ROWS: usize = 48;

struct Down {
    codes: Arc<Tensor<f4e2m1fnx2>>,
    scales: Arc<Tensor<f8e4m3fn>>,
    global: f32,
    outputs: usize,
    columns: usize,
}

fn choose_down(loaded: &LoadedModel) -> Result<Down, Box<dyn StdError>> {
    let mut chosen = None;
    for weight in loaded.weights().projections.values() {
        let ProjectionWeight::Fp4(codes, scales, global) = weight else {
            continue;
        };
        let outputs = usize::try_from(codes.shape()[0]).map_err(|_| Error::invalid("rows"))?;
        let columns = usize::try_from(codes.shape()[1])
            .map_err(|_| Error::invalid("packed columns"))?
            .checked_mul(2)
            .ok_or_else(|| Error::invalid("columns"))?;
        if outputs < columns {
            let elements = outputs.saturating_mul(columns);
            if chosen
                .as_ref()
                .is_none_or(|(_, _, _, _, _, best)| elements > *best)
            {
                chosen = Some((
                    codes.clone(),
                    scales.clone(),
                    *global,
                    outputs,
                    columns,
                    elements,
                ));
            }
        }
    }
    let Some((codes, scales, global, outputs, columns, _)) = chosen else {
        return Err("no down-shaped FP4 projection".into());
    };
    Ok(Down {
        codes,
        scales,
        global,
        outputs,
        columns,
    })
}

struct Graphs {
    exact: Tensor<f32>,
    split: Tensor<f32>,
    exact_graph: CudaGraph<()>,
    split_graph: CudaGraph<()>,
    candidate: super::SplitCandidate,
}

fn record(
    device: &CudaDevice,
    down: &Down,
    input_bytes: &[u8],
    sms: usize,
) -> Result<Graphs, Box<dyn StdError>> {
    let candidate =
        projection_split_policy(ROWS, down.outputs, down.columns, sms, 256, 32 * 1024 * 1024)?
            .ok_or_else(|| Error::invalid("down split candidate rejected"))?;
    if candidate.scratch_bytes != split_scratch_bytes(ROWS, down.outputs, candidate.splits)? {
        return Err("split scratch quote mismatch".into());
    }
    let code_bytes = down.columns / 2;
    let scale_columns = down.columns / 16;
    let input_codes = device.upload(
        input_bytes
            .iter()
            .copied()
            .map(f4e2m1fnx2::from_bits)
            .collect::<Vec<_>>(),
        &[ROWS, code_bytes],
    )?;
    let input_scale_bits: Vec<u8> = (0..ROWS * scale_columns)
        .map(|i| u8::try_from(48 + i % 16).unwrap())
        .collect();
    let input_scales = device.upload(
        input_scale_bits
            .iter()
            .copied()
            .map(f8e4m3fn)
            .collect::<Vec<_>>(),
        &[ROWS, scale_columns],
    )?;
    let mut exact = api::zeros::<f32>(&[ROWS, down.outputs]).sync_on(&device.stream)?;
    let mut split = api::zeros::<f32>(&[ROWS, down.outputs]).sync_on(&device.stream)?;
    let mut partials =
        api::zeros::<f32>(&[candidate.splits, ROWS, down.outputs]).sync_on(&device.stream)?;
    let exact_graph = CudaGraph::scope(&device.stream, |scope| {
        scope.record(
            kernels::packed(
                (&mut exact).partition([64, 64]),
                &input_codes,
                &input_scales,
                &down.codes,
                &down.scales,
                down.global.recip(),
            )
            .generics(vec![
                down.columns.to_string(),
                ROWS.to_string(),
                down.outputs.to_string(),
            ]),
        )?;
        Ok(())
    })?;
    let split_graph = CudaGraph::scope(&device.stream, |scope| {
        scope.record(
            kernels::packed_split(
                (&mut partials).partition([1, 64, 64]),
                &input_codes,
                &input_scales,
                &down.codes,
                &down.scales,
            )
            .generics(vec![
                down.columns.to_string(),
                ROWS.to_string(),
                down.outputs.to_string(),
                candidate.splits.to_string(),
            ]),
        )?;
        scope.record(
            kernels::sum_splits(
                (&mut split).partition([64, 64]),
                &partials,
                down.global.recip(),
            )
            .generics(vec![
                ROWS.to_string(),
                down.outputs.to_string(),
                candidate.splits.to_string(),
            ]),
        )?;
        Ok(())
    })?;
    Ok(Graphs {
        exact,
        split,
        exact_graph,
        split_graph,
        candidate,
    })
}

fn scalar_reference(
    down: &Down,
    input_bytes: &[u8],
    input_scale_bits: &[u8],
    weight_bits: &[u8],
    weight_scale_bits: &[u8],
    actual: &[f32],
) -> f64 {
    let values = [0.0_f64, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let f4 = |code: u8| values[usize::from(code & 7)] * if code & 8 == 0 { 1.0 } else { -1.0 };
    let code_bytes = down.columns / 2;
    let scale_columns = down.columns / 16;
    let mut worst = 0.0_f64;
    for (row, column) in [
        (0usize, 0usize),
        (ROWS / 2, down.outputs / 3),
        (ROWS - 1, down.outputs - 1),
    ] {
        let mut sum = 0.0_f64;
        for k in 0..down.columns {
            let input = input_bytes[row * code_bytes + k / 2];
            let weight = weight_bits[column * code_bytes + k / 2];
            let input_code = if k % 2 == 0 { input & 15 } else { input >> 4 };
            let weight_code = if k % 2 == 0 { weight & 15 } else { weight >> 4 };
            let left = f4(input_code)
                * f64::from(fp8_scalar(input_scale_bits[row * scale_columns + k / 16]));
            let right = f4(weight_code)
                * f64::from(fp8_scalar(
                    weight_scale_bits[column * scale_columns + k / 16],
                ));
            sum = left.mul_add(right, sum);
        }
        let expected = sum * f64::from(down.global).recip();
        let got = f64::from(actual[row * down.outputs + column]);
        worst = worst.max((got - expected).abs());
    }
    worst
}

fn fp8_scalar(code: u8) -> f32 {
    let exponent = code >> 3;
    let mantissa = f32::from(code & 7);
    if exponent == 0 {
        mantissa / 512.0
    } else {
        (1.0 + mantissa / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    }
}

struct Measurement {
    down: Down,
    graphs: Graphs,
    input_bytes: Vec<u8>,
    path: String,
    sms: usize,
    baseline_ms: f32,
    candidate_ms: f32,
}

fn validate(device: &CudaDevice, m: Measurement) -> Result<(), Box<dyn StdError>> {
    let Measurement {
        down,
        graphs,
        input_bytes,
        path,
        sms,
        baseline_ms,
        candidate_ms,
    } = m;
    let Graphs {
        exact,
        split,
        candidate,
        ..
    } = graphs;
    let exact = exact.to_host_vec().sync_on(&device.stream)?;
    let split = split.to_host_vec().sync_on(&device.stream)?;
    let mut diff_sq = 0.0_f64;
    let mut base_sq = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for (&a, &b) in exact.iter().zip(&split) {
        let a = f64::from(a);
        let b = f64::from(b);
        diff_sq = f64::mul_add(a - b, a - b, diff_sq);
        base_sq = f64::mul_add(a, a, base_sq);
        max_abs = max_abs.max((a - b).abs());
    }
    let relative_l2 = (diff_sq / base_sq.max(1.0e-30)).sqrt();
    let weight_bits: Vec<u8> = (&down.codes)
        .to_host_vec()
        .sync_on(&device.stream)?
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let weight_scale_bits: Vec<u8> = (&down.scales)
        .to_host_vec()
        .sync_on(&device.stream)?
        .iter()
        .map(|v| v.0)
        .collect();
    let scale_columns = down.columns / 16;
    let input_scale_bits: Vec<u8> = (0..ROWS * scale_columns)
        .map(|i| u8::try_from(48 + i % 16).unwrap())
        .collect();
    let subset_reference_max_abs = scalar_reference(
        &down,
        &input_bytes,
        &input_scale_bits,
        &weight_bits,
        &weight_scale_bits,
        &split,
    );
    let speedup = baseline_ms / candidate_ms;
    if speedup <= 1.0 {
        return Err("down split candidate was not faster than exact".into());
    }
    let binary = env::current_exe()?;
    println!(
        "{}",
        serde_json::json!({
            "actual_checkpoint_down_split_candidate": true,
            "model": path,
            "shape": [ROWS, down.outputs, down.columns],
            "splits": candidate.splits,
            "tile": candidate.tile,
            "scratch_bytes": candidate.scratch_bytes,
            "baseline_median_ms": baseline_ms,
            "candidate_median_ms": candidate_ms,
            "speedup": speedup,
            "relative_l2_vs_exact": relative_l2,
            "max_abs_vs_exact": max_abs,
            "subset_reference_max_abs": subset_reference_max_abs,
            "weight_codes_sha256": format!("{:x}", Sha256::digest(&weight_bits)),
            "weight_scales_sha256": format!("{:x}", Sha256::digest(&weight_scale_bits)),
            "binary_sha256": format!("{:x}", Sha256::digest(fs::read(binary)?)),
            "device_sms": sms,
            "device": format!("{:?}", device.profile()?.identity()),
        })
    );
    if !relative_l2.is_finite() || relative_l2 > 1.0e-4 {
        return Err("split candidate relative L2 exceeded 1e-4".into());
    }
    if !subset_reference_max_abs.is_finite() || subset_reference_max_abs > 1.0e-4 {
        return Err("split candidate subset reference exceeded 1e-4".into());
    }
    Ok(())
}

#[test]
#[ignore = "real 27B checkpoint, CUDA, release and protected runner required"]
fn real_checkpoint_down_split_candidate_measurement() -> Result<(), Box<dyn StdError>> {
    if cfg!(debug_assertions) {
        return Err("requires cargo test --release".into());
    }
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let loaded = LoadedModel::open(
        device.clone(),
        &path,
        ModelId::new(1)?,
        LoadOptions {
            autotune: false,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    let down = choose_down(&loaded)?;
    let code_bytes = down.columns / 2;
    let input_bytes: Vec<u8> = (0..ROWS * code_bytes)
        .map(|i| u8::try_from((i * 5 + i / 7) % 256).unwrap())
        .collect();
    let sms = usize::try_from(device.profile()?.multiprocessors)
        .map_err(|_| Error::invalid("SM count"))?;
    let graphs = record(&device, &down, &input_bytes, sms)?;
    let options = BenchOptions {
        warmup: Duration::from_millis(50),
        rep: Duration::from_millis(200),
        min_reps: 10,
        max_reps: 100,
        clear_l2: true,
    };
    let (baseline, candidate) = do_bench_paired(
        &device.stream,
        &options,
        |_| {
            graphs
                .exact_graph
                .launch()
                .sync_on(&device.stream)
                .map_err(|e| tensor_error(&e.to_string()))
        },
        |_| {
            graphs
                .split_graph
                .launch()
                .sync_on(&device.stream)
                .map_err(|e| tensor_error(&e.to_string()))
        },
    )?;
    validate(
        &device,
        Measurement {
            down,
            graphs,
            input_bytes,
            path,
            sms,
            baseline_ms: baseline.median_ms(),
            candidate_ms: candidate.median_ms(),
        },
    )
}
