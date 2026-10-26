//! Actual-checkpoint down-projection split candidate driver (device-only, ignored).

use super::{kernels, projection_split_policy, split_scratch_bytes};
use crate::device::CudaDevice;
use crate::loading::{LoadOptions, LoadedModel};
use crate::mlp::ProjectionWeight;
use crate::tuning::{VerifySplitEntry, resolve_table, table_path};
use cuda_core::{f4e2m1fnx2, f8e4m3fn};
use cutile::bench::{BenchOptions, do_bench_paired};
use cutile::error::tensor_error;
use cutile::prelude::*;
use infer_core::{Error, ModelId, TensorId};
use infer_ir::TensorOp;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env::{self, var};
use std::error::Error as StdError;
use std::fs;
use std::sync::Arc;
use std::time::Duration;

const ROWS: usize = 48;

struct Down {
    weight_id: TensorId,
    codes: Arc<Tensor<f4e2m1fnx2>>,
    scales: Arc<Tensor<f8e4m3fn>>,
    global: f32,
    outputs: usize,
    columns: usize,
}

struct Layer {
    codes: Arc<Tensor<f4e2m1fnx2>>,
    scales: Arc<Tensor<f8e4m3fn>>,
    global: f32,
    input_scale: f32,
}

fn target_layers(
    loaded: &LoadedModel,
    down: &Down,
) -> Result<(Vec<Layer>, usize), Box<dyn StdError>> {
    let mut seen = BTreeSet::new();
    let mut layers = Vec::new();
    for node in &loaded.graph().nodes {
        if node.op != TensorOp::Linear {
            continue;
        }
        let Some(weight) = loaded.weights().projections.get(&node.inputs[1]) else {
            continue;
        };
        let ProjectionWeight::Fp4(codes, scales, global) = weight else {
            continue;
        };
        if codes.shape()
            != [
                i32::try_from(down.outputs)?,
                i32::try_from(down.columns / 2)?,
            ]
        {
            continue;
        }
        let address = codes.device_pointer().cu_deviceptr();
        if !seen.insert(address) {
            continue;
        }
        let input_scale = *loaded
            .weights()
            .input_scales
            .get(&node.inputs[1])
            .ok_or_else(|| Error::invalid("layer activation scale missing"))?;
        layers.push(Layer {
            codes: codes.clone(),
            scales: scales.clone(),
            global: *global,
            input_scale,
        });
    }
    let bytes = down.outputs * (down.columns / 2 + down.columns / 16);
    Ok((layers, bytes))
}

fn choose_down(loaded: &LoadedModel) -> Result<Down, Box<dyn StdError>> {
    let mut chosen = None;
    for (&weight_id, weight) in &loaded.weights().projections {
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
                .is_none_or(|(_, _, _, _, _, _, best)| elements > *best)
            {
                chosen = Some((
                    weight_id,
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
    let Some((weight_id, codes, scales, global, outputs, columns, _)) = chosen else {
        return Err("no down-shaped FP4 projection".into());
    };
    Ok(Down {
        weight_id,
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
    let candidate = projection_split_policy(
        ROWS,
        down.outputs,
        down.columns,
        sms,
        [64, 64],
        256,
        32 * 1024 * 1024,
    )?
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
                candidate.tile[0].to_string(),
                candidate.tile[1].to_string(),
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
                candidate.tile[0].to_string(),
                candidate.tile[1].to_string(),
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
    publish_entry(device, &down, &candidate, speedup, &weight_bits)?;
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

fn publish_entry(
    device: &CudaDevice,
    down: &Down,
    candidate: &super::SplitCandidate,
    speedup: f32,
    weight_bits: &[u8],
) -> Result<(), Box<dyn StdError>> {
    let path = table_path().ok_or_else(|| Error::invalid("verify split table path"))?;
    let mut table = resolve_table(Some(&path))?.unwrap_or_default();
    let hardware = device.profile()?.identity();
    table.verify_split.retain(|old| {
        old.hardware != hardware
            || old.mode != "nvfp4"
            || old.rows != ROWS
            || old.outputs != down.outputs
            || old.columns != down.columns
    });
    table.verify_split.push(VerifySplitEntry {
        hardware,
        mode: "nvfp4".to_owned(),
        rows: ROWS,
        outputs: down.outputs,
        columns: down.columns,
        splits: candidate.splits,
        scratch_bytes: candidate.scratch_bytes,
        kernel_schema: "nvfp4-packed-split-v1".to_owned(),
        speedup,
        reference: format!(
            "real_checkpoint_down_split_candidate_measurement:{:x}",
            Sha256::digest(weight_bits)
        ),
    });
    fs::write(
        &path,
        serde_json::to_string_pretty(&table)
            .map_err(|e| Error::invalid(format!("verify split table: {e}")))?,
    )
    .map_err(|e| Error::invalid(format!("verify split table write: {e}")))?;
    Ok(())
}

struct Measure<'a> {
    exact_graph: CudaGraph<()>,
    split_graph: CudaGraph<()>,
    exact: Vec<Tensor<f32>>,
    split: Vec<Tensor<f32>>,
    path: &'a str,
    down: &'a Down,
    scratch_bytes: u64,
    split_plans: usize,
    copies: usize,
    l2_bytes: usize,
    weight_bytes: usize,
    addresses: Vec<u64>,
}

fn measure_captured(device: &CudaDevice, m: Measure<'_>) -> Result<(), Box<dyn StdError>> {
    let Measure {
        exact_graph,
        split_graph,
        exact,
        split,
        path,
        down,
        scratch_bytes,
        split_plans,
        copies,
        l2_bytes,
        weight_bytes,
        addresses,
    } = m;
    let mut diff = 0.0_f64;
    let mut base = 0.0_f64;
    for (exact, split) in exact.into_iter().zip(split) {
        let exact_host = exact.to_host_vec().sync_on(&device.stream)?;
        let split_host = split.to_host_vec().sync_on(&device.stream)?;
        for (&a, &b) in exact_host.iter().zip(&split_host) {
            let a = f64::from(a);
            let b = f64::from(b);
            diff = f64::mul_add(a - b, a - b, diff);
            base = f64::mul_add(a, a, base);
        }
    }
    let relative_l2 = (diff / base.max(1.0e-30)).sqrt();
    if !relative_l2.is_finite() || relative_l2 > 1.0e-4 {
        return Err(format!("split output relative L2 {relative_l2}").into());
    }
    let mut best_speedup = 0.0_f32;
    for (phase, clear_l2) in [("cold", true), ("warm", false)] {
        best_speedup = best_speedup.max(measure_phase(
            device,
            &exact_graph,
            &split_graph,
            phase,
            clear_l2,
            copies,
        )?);
    }
    println!(
        "{}",
        serde_json::json!({
            "kind":"captured_workspace_nvfp4_candidate", "model":path,
            "weight":format!("{:?}", down.weight_id),
            "shape":[ROWS, down.outputs, down.columns],
            "scratch_bytes":scratch_bytes, "split_plan_count":split_plans,
            "layers":copies, "weight_bytes_each":weight_bytes,
            "working_set_bytes":copies*weight_bytes, "l2_bytes":l2_bytes,
            "distinct_weight_addresses":addresses,
            "relative_l2_vs_exact":relative_l2,
            "best_phase_speedup":best_speedup,
            "qualification":"CANDIDATE_NOT_PRODUCTION"
        })
    );
    Ok(())
}

fn measure_phase(
    device: &CudaDevice,
    exact_graph: &CudaGraph<()>,
    split_graph: &CudaGraph<()>,
    phase: &str,
    clear_l2: bool,
    copies: usize,
) -> Result<f32, Box<dyn StdError>> {
    let options = BenchOptions {
        warmup: Duration::from_millis(50),
        rep: Duration::from_millis(200),
        min_reps: 10,
        max_reps: 100,
        clear_l2,
    };
    let mut best = 0.0_f32;
    for order in ["AB", "BA"] {
        let (baseline, candidate) = if order == "AB" {
            do_bench_paired(
                &device.stream,
                &options,
                |_| {
                    exact_graph
                        .launch()
                        .sync_on(&device.stream)
                        .map_err(|e| tensor_error(&e.to_string()))
                },
                |_| {
                    split_graph
                        .launch()
                        .sync_on(&device.stream)
                        .map_err(|e| tensor_error(&e.to_string()))
                },
            )?
        } else {
            let (candidate, baseline) = do_bench_paired(
                &device.stream,
                &options,
                |_| {
                    split_graph
                        .launch()
                        .sync_on(&device.stream)
                        .map_err(|e| tensor_error(&e.to_string()))
                },
                |_| {
                    exact_graph
                        .launch()
                        .sync_on(&device.stream)
                        .map_err(|e| tensor_error(&e.to_string()))
                },
            )?;
            (baseline, candidate)
        };
        let speedup = baseline.median_ms() / candidate.median_ms();
        best = best.max(speedup);
        let copies_f32 = f32::from(u16::try_from(copies).unwrap_or(u16::MAX));
        println!(
            "{}",
            serde_json::json!({
                "kind":"captured_workspace_nvfp4_phase", "phase":phase, "order":order,
                "unit":"layer_set_cycle", "layers":copies,
                "baseline_ms":baseline.median_ms(), "candidate_ms":candidate.median_ms(),
                "speedup":speedup, "baseline_raw_ms":baseline.times_ms(),
                "candidate_raw_ms":candidate.times_ms(), "reps":baseline.reps(),
                "per_projection_baseline_ms":baseline.median_ms()/copies_f32,
                "per_projection_candidate_ms":candidate.median_ms()/copies_f32,
                "meets_1_02":speedup >= 1.02,
            })
        );
    }
    Ok(best)
}

fn capture_rotation(
    device: &CudaDevice,
    workspace: &mut super::Workspace,
    layers: &[Layer],
    input: &Tensor<f32>,
    outputs: &mut [Tensor<f32>],
    columns: usize,
) -> Result<CudaGraph<()>, Box<dyn StdError>> {
    let view = input.view(&[ROWS, columns])?;
    workspace.begin_quant_capture();
    let graph = CudaGraph::scope(&device.stream, |scope| {
        for (layer, output) in layers.iter().zip(outputs) {
            workspace.record_input(
                scope,
                &ProjectionWeight::Fp4(layer.codes.clone(), layer.scales.clone(), layer.global),
                layer.input_scale,
                &view,
                output,
                columns,
                Some(input.device_pointer().cu_deviceptr()),
            )?;
        }
        Ok(())
    });
    workspace.end_quant_capture();
    Ok(graph?)
}

type RotationBuffers = (Arc<Tensor<f32>>, Vec<Tensor<f32>>, Vec<Tensor<f32>>);

fn rotation_buffers(
    device: &CudaDevice,
    down: &Down,
    copies: usize,
) -> Result<RotationBuffers, Box<dyn StdError>> {
    let input = device.upload(
        (0..ROWS * down.columns)
            .map(|i| (f32::from(u16::try_from(i % 211).unwrap()) - 105.0) / 37.0)
            .collect::<Vec<_>>(),
        &[ROWS, down.columns],
    )?;
    let mut exact = Vec::with_capacity(copies);
    let mut split = Vec::with_capacity(copies);
    for _ in 0..copies {
        exact.push(api::zeros::<f32>(&[ROWS, down.outputs]).sync_on(&device.stream)?);
        split.push(api::zeros::<f32>(&[ROWS, down.outputs]).sync_on(&device.stream)?);
    }
    Ok((input, exact, split))
}

#[test]
#[ignore = "real 27B checkpoint, CUDA, release and protected runner required"]
fn real_checkpoint_down_split_candidate_measurement() -> Result<(), Box<dyn StdError>> {
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

#[test]
#[ignore = "real 27B checkpoint, CUDA, release and protected runner required"]
fn captured_workspace_nvfp4_real_weight_split() -> Result<(), Box<dyn StdError>> {
    use super::Workspace;
    use super::workspace::{
        reset_split_records, split_records, with_disabled_split, with_forced_split,
        with_forced_splits,
    };
    use crate::resident::ProgramGeometry;

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
    let geometry = ProgramGeometry {
        verification: ROWS,
        prompts: [0, 0],
    };
    let mut exact_workspace = with_disabled_split(|| {
        Workspace::program(&device, loaded.graph(), loaded.weights(), geometry)
    })?;
    let forced_s = var("INFER_CANDIDATE_S")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut split_workspace = with_forced_split(|| {
        if forced_s == 0 {
            Workspace::program(&device, loaded.graph(), loaded.weights(), geometry)
        } else {
            with_forced_splits(forced_s, || {
                Workspace::program(&device, loaded.graph(), loaded.weights(), geometry)
            })
        }
    })?;
    assert_eq!(exact_workspace.split_plan_count(), 0);
    reset_split_records();
    assert!(split_workspace.split_plan_count() > 0);
    let (layers, weight_bytes) = target_layers(&loaded, &down)?;
    let copies = layers.len();
    let l2 = usize::try_from(device.profile()?.l2_cache_bytes).unwrap_or(0);
    let (input, mut exact, mut split) = rotation_buffers(&device, &down, copies)?;
    reset_split_records();
    let exact_graph = capture_rotation(
        &device,
        &mut exact_workspace,
        &layers,
        &input,
        &mut exact,
        down.columns,
    )?;
    assert_eq!(split_records(), 0);
    reset_split_records();
    let split_graph = with_forced_split(|| {
        capture_rotation(
            &device,
            &mut split_workspace,
            &layers,
            &input,
            &mut split,
            down.columns,
        )
    })?;
    assert_eq!(split_records(), copies);
    exact_graph.launch().sync_on(&device.stream)?;
    split_graph.launch().sync_on(&device.stream)?;
    println!(
        "{}",
        serde_json::json!({"kind":"captured_workspace_prepared","prepared":exact_workspace.prepared_configs()})
    );
    measure_captured(
        &device,
        Measure {
            exact_graph,
            split_graph,
            exact,
            split,
            path: &path,
            down: &down,
            scratch_bytes: split_workspace.split_scratch_bytes(),
            split_plans: split_workspace.split_plan_count(),
            copies,
            l2_bytes: l2,
            weight_bytes,
            addresses: layers
                .iter()
                .map(|layer| layer.codes.device_pointer().cu_deviceptr())
                .collect(),
        },
    )?;
    Ok(())
}

#[test]
#[ignore = "real 27B checkpoint, CUDA, release and protected runner required"]
fn captured_workspace_nvfp4_real_weight_prefetch_probe() -> Result<(), Box<dyn StdError>> {
    use super::Workspace;
    use super::workspace::{with_disabled_split, with_forced_loop};
    use crate::resident::ProgramGeometry;

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
    let (layers, weight_bytes) = target_layers(&loaded, &down)?;
    let geometry = ProgramGeometry {
        verification: ROWS,
        prompts: [0, 0],
    };
    let mut plain = with_disabled_split(|| {
        Workspace::program(&device, loaded.graph(), loaded.weights(), geometry)
    })?;
    let mut prefetch = with_disabled_split(|| {
        with_forced_loop(2, || {
            Workspace::program(&device, loaded.graph(), loaded.weights(), geometry)
        })
    })?;
    let copies = layers.len();
    let l2 = usize::try_from(device.profile()?.l2_cache_bytes).unwrap_or(0);
    let (input, mut plain_out, mut fast_out) = rotation_buffers(&device, &down, copies)?;
    let plain_graph = capture_rotation(
        &device,
        &mut plain,
        &layers,
        &input,
        &mut plain_out,
        down.columns,
    )?;
    let fast_graph = capture_rotation(
        &device,
        &mut prefetch,
        &layers,
        &input,
        &mut fast_out,
        down.columns,
    )?;
    plain_graph.launch().sync_on(&device.stream)?;
    fast_graph.launch().sync_on(&device.stream)?;
    let plain_host = plain_out
        .into_iter()
        .map(|tensor| tensor.to_host_vec().sync_on(&device.stream))
        .collect::<Result<Vec<_>, _>>()?;
    let fast_host = fast_out
        .into_iter()
        .map(|tensor| tensor.to_host_vec().sync_on(&device.stream))
        .collect::<Result<Vec<_>, _>>()?;
    if plain_host != fast_host {
        return Err("prefetch changed output bits".into());
    }
    let mut best = 0.0_f32;
    for (phase, clear) in [("cold", true), ("warm", false)] {
        best = best.max(measure_phase(
            &device,
            &plain_graph,
            &fast_graph,
            phase,
            clear,
            copies,
        )?);
    }
    println!(
        "{}",
        serde_json::json!({
            "kind":"captured_workspace_nvfp4_prefetch", "layers":copies,
            "working_set_bytes":copies*weight_bytes, "l2_bytes":l2,
            "bitwise_equal":true, "best_phase_speedup":best,
            "qualification":"CANDIDATE_NOT_PRODUCTION"
        })
    );
    Ok(())
}

#[test]
#[ignore = "real 27B checkpoint, CUDA, release and protected runner required"]
fn prefetch8_autotune_production_selection() -> Result<(), Box<dyn StdError>> {
    use super::Workspace;
    use crate::resident::ProgramGeometry;
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let loaded = LoadedModel::open(
        device.clone(),
        &path,
        ModelId::new(1)?,
        LoadOptions {
            autotune: true,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    let workspace = Workspace::program(
        &device,
        loaded.graph(),
        loaded.weights(),
        ProgramGeometry {
            verification: ROWS,
            prompts: [0, 0],
        },
    )?;
    let prepared = workspace.prepared_configs();
    let entries = prepared
        .as_array()
        .ok_or_else(|| Error::invalid("prepared config array"))?;
    let exact = entries
        .iter()
        .find(|entry| {
            entry["mode"] == "nvfp4"
                && entry["m"] == 48
                && entry["n"] == 5120
                && entry["k"] == 17408
        })
        .ok_or_else(|| Error::invalid("missing exact prepared geometry"))?;
    let config = exact["prepared_config"]
        .as_str()
        .ok_or_else(|| Error::invalid("prepared config string"))?;
    if exact["weight_count"] != 56 || !config.contains("Prefetch8") {
        return Err(format!("autotune did not select exact Prefetch8: {prepared}").into());
    }
    println!(
        "{}",
        serde_json::json!({
            "kind":"prefetch8_autotune_selected", "prepared":prepared,
            "qualification":"production_loader_selected_candidate"
        })
    );
    Ok(())
}
