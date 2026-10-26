//! Actual-checkpoint verification pair context and Dense precision/latency driver.
use super::{LoadOptions, LoadedModel};
use crate::device::CudaDevice;
use crate::mlp::ProjectionWeight;
use crate::resident::{GdnPairContext, gemm, pair_counts, reset_pair_counts, small_gemm_kernels};
use cutile::bench::{BenchOptions, do_bench_paired};
use cutile::error::tensor_error;
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::{ModelId, TensorId};
use infer_ir::TensorOp;
use serde_json::json;
use std::collections::BTreeMap;
use std::env::var;
use std::error::Error;
use std::fs::read;
use std::sync::Arc;
use std::time::Duration;

fn tokens() -> Result<Vec<u32>, Box<dyn Error>> {
    let path = var("INFER_TEST_TOKENS")?;
    Ok(serde_json::from_slice(&read(path)?)?)
}

fn actual_pairs(loaded: &LoadedModel) -> Vec<(TensorId, TensorId)> {
    let mut producer = BTreeMap::new();
    for (index, node) in loaded.graph().nodes.iter().enumerate() {
        for id in &node.outputs {
            producer.insert(*id, index);
        }
    }
    let mut pairs = Vec::new();
    for node in &loaded.graph().nodes {
        if !matches!(node.op, TensorOp::Delta { .. }) || node.inputs.len() <= 2 {
            continue;
        }
        let (Some(&beta), Some(&alpha)) =
            (producer.get(&node.inputs[1]), producer.get(&node.inputs[2]))
        else {
            continue;
        };
        let (b, a) = (&loaded.graph().nodes[beta], &loaded.graph().nodes[alpha]);
        if b.op != TensorOp::Linear
            || a.op != TensorOp::Linear
            || b.outputs.len() != 1
            || a.outputs.len() != 1
        {
            continue;
        }
        let (Some(ProjectionWeight::Dense(wb)), Some(ProjectionWeight::Dense(wa))) = (
            loaded.weights().projections.get(&b.inputs[1]),
            loaded.weights().projections.get(&a.inputs[1]),
        ) else {
            continue;
        };
        if wb.shape() == wa.shape()
            && wb.shape().len() == 2
            && (1..=64).contains(&usize::try_from(wb.shape()[0]).unwrap_or(0))
        {
            pairs.push((b.inputs[1], a.inputs[1]));
        }
    }
    pairs
}

#[test]
#[ignore = "requires CUDA/checkpoint/frozen tokens; actual verify pair dispatch and full output"]
fn checkpoint_verify_pair_context_preserves_existing_dense_path() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let path = var("INFER_TEST_MODEL")?;
    let tokens = tokens()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 64,
            verification_width: 3,
            autotune: false,
            mtp_depth: 0,
            ..LoadOptions::default()
        },
    )?;
    reset_pair_counts();
    let mut program = loaded.sequence((tokens.len() + 64).next_power_of_two())?;
    let verify = pair_counts(GdnPairContext::Verify);
    let prompt = pair_counts(GdnPairContext::Prompt);
    eprintln!(
        "{}",
        json!({"verify_pair_events":{"attempts":verify.0,"records":verify.1},
        "prompt_pair_events":{"attempts":prompt.0,"records":prompt.1}})
    );
    assert!(verify.0 > 0, "actual verify capture saw no GDN pair");
    assert_eq!(
        verify.1, 0,
        "verify pair bypassed the existing Dense precision path"
    );
    assert_eq!(
        prompt.1, 0,
        "sm120 N48 prompt pair bypassed role-selected N16"
    );
    let mut elements = 0usize;
    let mut finite = 0usize;
    for (chunk, chunk_tokens) in tokens.chunks(64).enumerate() {
        let actual = program.prefill_batch(chunk_tokens, chunk * 64, true)?;
        assert_eq!(actual.len(), chunk_tokens.len());
        for (hidden, logits) in actual {
            assert!(!hidden.is_empty(), "prefill hidden is empty");
            elements += hidden.len() + logits.len();
            finite += hidden
                .iter()
                .chain(&logits)
                .filter(|value| value.is_finite())
                .count();
        }
    }
    let position = tokens.len();
    let batch: Vec<_> = tokens
        .iter()
        .copied()
        .cycle()
        .take(program.batch_width())
        .collect();
    let verified = program.step_batch(&batch, position)?;
    assert_eq!(verified.len(), batch.len());
    for (hidden, logits) in verified {
        assert!(
            !hidden.is_empty() && !logits.is_empty(),
            "verify row output is empty"
        );
        elements += hidden.len() + logits.len();
        finite += hidden
            .iter()
            .chain(&logits)
            .filter(|value| value.is_finite())
            .count();
    }
    assert_eq!(finite, elements, "verify full output is nonfinite");
    eprintln!(
        "{}",
        json!({"verify_full_output_elements":elements,"finite":finite,"batch_width":program.batch_width()})
    );
    Ok(())
}

type DensePair = (Arc<Tensor<bf16>>, Arc<Tensor<bf16>>);

fn pick_dense(loaded: &LoadedModel) -> Result<DensePair, Box<dyn Error>> {
    for (first, second) in actual_pairs(loaded) {
        let (Some(ProjectionWeight::Dense(wb)), Some(ProjectionWeight::Dense(wa))) = (
            loaded.weights().projections.get(&first),
            loaded.weights().projections.get(&second),
        ) else {
            continue;
        };
        return Ok((wb.clone(), wa.clone()));
    }
    Err("actual checkpoint exposed no Dense GDN pair".into())
}

#[test]
#[ignore = "requires CUDA/checkpoint; actual Dense verify baseline versus BF16 pair, rows 32/48/64"]
fn checkpoint_dense_verify_pair_precision_and_latency() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let path = var("INFER_TEST_MODEL")?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 64,
            autotune: false,
            mtp_depth: 0,
            ..LoadOptions::default()
        },
    )?;
    let (weight_b, weight_a) = pick_dense(&loaded)?;
    let columns = usize::try_from(weight_b.shape()[1])?;
    let outputs = usize::try_from(weight_b.shape()[0])?;
    assert_eq!(outputs, 48);
    for rows in [32usize, 48, 64] {
        let (differing, max_abs, timings) =
            dense_pair_row(&loaded, &weight_b, &weight_a, (rows, columns, outputs))?;
        eprintln!(
            "{}",
            json!({"dense_verify_pair":{"rows":rows,"columns":columns,"outputs":outputs,
            "baseline":"two_component_small_gemm","candidate":"single_component_dense_pair",
            "differing_bits":differing,"max_abs":max_abs,"timings":timings}})
        );
        assert!(
            differing > 0,
            "single-component pair unexpectedly bit-equal; re-audit precision claim"
        );
    }
    Ok(())
}

fn dense_pair_row(
    loaded: &LoadedModel,
    weight_b: &Tensor<bf16>,
    weight_a: &Tensor<bf16>,
    (rows, columns, outputs): (usize, usize, usize),
) -> Result<(usize, f32, Vec<serde_json::Value>), Box<dyn Error>> {
    let device = loaded.device();
    let input_values = (0..rows * columns)
        .map(|i| (f32::from(u16::try_from(i % 251).unwrap()) - 125.0) / 37.0)
        .collect::<Vec<_>>();
    let input = device.upload(input_values, &[rows, columns])?;
    let mut baseline_b = api::zeros::<f32>(&[rows, outputs]).sync_on(&device.stream)?;
    let mut baseline_a = api::zeros::<f32>(&[rows, outputs]).sync_on(&device.stream)?;
    let mut pair_b = api::zeros::<f32>(&[rows, outputs]).sync_on(&device.stream)?;
    let mut pair_a = api::zeros::<f32>(&[rows, outputs]).sync_on(&device.stream)?;
    let baseline = CudaGraph::scope(&device.stream, |scope| {
        scope.record(
            small_gemm_kernels::dense((&mut baseline_b).partition([16, 32]), &input, weight_b)
                .generics(vec![bf16::DTYPE.as_str().into(), columns.to_string()]),
        )?;
        scope.record(
            small_gemm_kernels::dense((&mut baseline_a).partition([16, 32]), &input, weight_a)
                .generics(vec![bf16::DTYPE.as_str().into(), columns.to_string()]),
        )?;
        Ok(())
    })?;
    let pair = CudaGraph::scope(&device.stream, |scope| {
        scope.record(
            gemm::dense_pair(
                (&mut pair_b).partition([32, 64]),
                (&mut pair_a).partition([32, 64]),
                &input,
                weight_b,
                weight_a,
            )
            .generics(vec![columns.to_string()]),
        )?;
        Ok(())
    })?;
    baseline.launch().sync_on(&device.stream)?;
    pair.launch().sync_on(&device.stream)?;
    let baseline_b = device.read_borrowed(&baseline_b)?;
    let baseline_a = device.read_borrowed(&baseline_a)?;
    let pair_b = device.read_borrowed(&pair_b)?;
    let pair_a = device.read_borrowed(&pair_a)?;
    for (name, values) in [
        ("baseline_b", &baseline_b),
        ("baseline_a", &baseline_a),
        ("pair_b", &pair_b),
        ("pair_a", &pair_a),
    ] {
        assert_eq!(values.len(), rows * outputs, "{name} length");
        assert!(
            values.iter().all(|value| value.is_finite()),
            "{name} finite"
        );
    }
    let differing = baseline_b
        .iter()
        .zip(&pair_b)
        .chain(baseline_a.iter().zip(&pair_a))
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let max_abs = baseline_b
        .iter()
        .zip(&pair_b)
        .chain(baseline_a.iter().zip(&pair_a))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let timings = pair_timings(device, &baseline, &pair)?;
    Ok((differing, max_abs, timings))
}

fn pair_timings(
    device: &CudaDevice,
    baseline: &CudaGraph<()>,
    pair: &CudaGraph<()>,
) -> Result<Vec<serde_json::Value>, Box<dyn Error>> {
    let mut timings = Vec::new();
    for cold in [true, false] {
        let options = BenchOptions {
            warmup: Duration::from_millis(20),
            rep: Duration::from_millis(120),
            min_reps: 5,
            max_reps: 100,
            clear_l2: cold,
        };
        let (ab_base, ab_pair) = do_bench_paired(
            &device.stream,
            &options,
            |_| {
                baseline
                    .launch()
                    .sync_on(&device.stream)
                    .map_err(|error| tensor_error(&error.to_string()))
            },
            |_| {
                pair.launch()
                    .sync_on(&device.stream)
                    .map_err(|error| tensor_error(&error.to_string()))
            },
        )?;
        let (ba_pair, ba_base) = do_bench_paired(
            &device.stream,
            &options,
            |_| {
                pair.launch()
                    .sync_on(&device.stream)
                    .map_err(|error| tensor_error(&error.to_string()))
            },
            |_| {
                baseline
                    .launch()
                    .sync_on(&device.stream)
                    .map_err(|error| tensor_error(&error.to_string()))
            },
        )?;
        timings.push(json!({"cold":cold,"base_ab_ms":ab_base.median_ms(),"pair_ab_ms":ab_pair.median_ms(),"pair_ba_ms":ba_pair.median_ms(),"base_ba_ms":ba_base.median_ms()}));
    }
    Ok(timings)
}
