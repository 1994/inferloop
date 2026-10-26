//! Real GDN weights must retain verify's two-term activation arithmetic.
use super::gdn_ba_pairs;
use crate::device::CudaDevice;
use crate::loading::{LoadOptions, LoadedModel};
use crate::mlp::ProjectionWeight;
use crate::resident::capture::error;
use crate::resident::{ActivationArena, prefill_projection, small_gemm};
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::ModelId;
use infer_ir::TensorNode;
use std::env::var;
use std::error::Error;
use std::sync::Arc;

fn check_pair(
    loaded: &LoadedModel,
    arena: &mut ActivationArena,
    first: &TensorNode,
    second: &TensorNode,
    rows: usize,
) -> Result<usize, Box<dyn Error>> {
    let device = loaded.device();
    let (Some(ProjectionWeight::Dense(wb)), Some(ProjectionWeight::Dense(wa))) = (
        loaded.weights().projections.get(&first.inputs[1]),
        loaded.weights().projections.get(&second.inputs[1]),
    ) else {
        return Err("structural pair lost its dense weights".into());
    };
    let columns = usize::try_from(wb.shape()[1])?;
    let outputs = usize::try_from(wb.shape()[0])?;
    let input: Vec<f32> = (0..rows * columns)
        .map(|i| (f32::from(u16::try_from(i * 17 % 251).unwrap()) - 125.0) / 37.0)
        .collect();
    let input_slot = arena.slot(first.inputs[0])?;
    arena.buffers[input_slot] = Some(
        Arc::try_unwrap(device.upload(input, &[rows * columns])?)
            .map_err(|_| "input upload unexpectedly shared")?,
    );
    let mut base_b = api::zeros::<f32>(&[rows, outputs]).sync_on(&device.stream)?;
    let mut base_a = api::zeros::<f32>(&[rows, outputs]).sync_on(&device.stream)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        {
            let input = arena.get(first.inputs[0]).map_err(error)?;
            let input = input.view(&[rows, columns])?;
            scope.record(
                small_gemm::kernels::dense((&mut base_b).partition([16, 32]), &input, wb)
                    .generics(vec![bf16::DTYPE.as_str().into(), columns.to_string()]),
            )?;
            scope.record(
                small_gemm::kernels::dense((&mut base_a).partition([16, 32]), &input, wa)
                    .generics(vec![bf16::DTYPE.as_str().into(), columns.to_string()]),
            )?;
        }
        let before = prefill_projection::verify_pair_captures();
        assert!(!prefill_projection::with_verify_pairs(false, || {
            prefill_projection::record_pair(
                scope,
                first,
                second,
                arena,
                (loaded.weights(), device.profile().map_err(error)?),
                prefill_projection::PairContext::Verify,
            )
        })?);
        assert!(prefill_projection::record_pair(
            scope,
            first,
            second,
            arena,
            (loaded.weights(), device.profile().map_err(error)?),
            prefill_projection::PairContext::Verify,
        )?);
        assert_eq!(prefill_projection::verify_pair_captures(), before + 1);
        Ok(())
    })?;
    graph.launch().sync_on(&device.stream)?;
    let baseline_b = device.read_borrowed(&base_b)?;
    let baseline_a = device.read_borrowed(&base_a)?;
    let paired_b = device.read_borrowed(arena.get(first.outputs[0])?)?;
    let paired_a = device.read_borrowed(arena.get(second.outputs[0])?)?;
    let differences = baseline_b
        .iter()
        .zip(&paired_b)
        .chain(baseline_a.iter().zip(&paired_a))
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert_eq!(
        differences, 0,
        "verify pair changed two-term precision at rows={rows}, layer={:?}",
        first.layer
    );
    assert!(
        paired_b
            .iter()
            .chain(&paired_a)
            .all(|value| value.is_finite())
    );
    Ok(paired_b.len() + paired_a.len())
}

#[test]
#[ignore = "requires CUDA and checkpoint; actual structural recorder and full verify outputs"]
fn real_gdn_pairs_preserve_verify_precision() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        var("INFER_TEST_MODEL")?,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 64,
            verification_width: 3,
            autotune: false,
            mtp_depth: 0,
            ..LoadOptions::default()
        },
    )?;
    let pairs = gdn_ba_pairs(loaded.graph(), loaded.weights());
    assert_eq!(pairs.len(), 48, "checkpoint structural GDN pair inventory");
    for rows in [48, 47] {
        let mut arena = ActivationArena::new_batched(
            loaded.device(),
            loaded.graph(),
            rows,
            usize::try_from(loaded.device().profile()?.arena_budget_bytes())?,
        )?;
        let mut elements = 0;
        for (&first, &second) in &pairs {
            elements += check_pair(
                &loaded,
                &mut arena,
                &loaded.graph().nodes[first],
                &loaded.graph().nodes[second],
                rows,
            )?;
        }
        eprintln!(
            "{}",
            serde_json::json!({"gdn_verify_pairs":pairs.len(),"rows":rows,"full_bitwise_elements":elements})
        );
    }
    Ok(())
}

use crate::resident::slot_batch::SlotPool;
use cutile::bench::{BenchOptions, do_bench_paired};
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use std::collections::BTreeMap;
use std::time::Duration;

struct PairWork {
    graph: CudaGraph<()>,
    _outputs: Vec<(Tensor<f32>, Tensor<f32>)>,
    _input: Arc<Tensor<f32>>,
}

fn pair_work(
    loaded: &LoadedModel,
    pairs: &BTreeMap<usize, usize>,
    paired: bool,
) -> Result<PairWork, Box<dyn Error>> {
    let rows = 48;
    let device = loaded.device();
    let first = &loaded.graph().nodes[*pairs.keys().next().ok_or("no GDN pairs")?];
    let Some(ProjectionWeight::Dense(weight)) = loaded.weights().projections.get(&first.inputs[1])
    else {
        return Err("GDN weight format changed".into());
    };
    let columns = usize::try_from(weight.shape()[1])?;
    let input = device.upload(
        (0..rows * columns)
            .map(|i| (f32::from(u16::try_from(i * 17 % 251).unwrap()) - 125.0) / 37.0)
            .collect(),
        &[rows, columns],
    )?;
    let mut outputs = Vec::new();
    for &index in pairs.keys() {
        let node = &loaded.graph().nodes[index];
        let Some(ProjectionWeight::Dense(weight)) =
            loaded.weights().projections.get(&node.inputs[1])
        else {
            return Err("GDN weight format changed".into());
        };
        assert_eq!(usize::try_from(weight.shape()[1])?, columns);
        let count = usize::try_from(weight.shape()[0])?;
        outputs.push((
            api::zeros::<f32>(&[rows, count]).sync_on(&device.stream)?,
            api::zeros::<f32>(&[rows, count]).sync_on(&device.stream)?,
        ));
    }
    let graph = CudaGraph::scope(&device.stream, |scope| {
        for ((&first, &second), (out_b, out_a)) in pairs.iter().zip(&mut outputs) {
            let (first, second) = (&loaded.graph().nodes[first], &loaded.graph().nodes[second]);
            let (Some(ProjectionWeight::Dense(wb)), Some(ProjectionWeight::Dense(wa))) = (
                loaded.weights().projections.get(&first.inputs[1]),
                loaded.weights().projections.get(&second.inputs[1]),
            ) else {
                return Err(tensor_error("GDN pair format changed").into());
            };
            if paired {
                scope.record(
                    small_gemm::kernels::dense_pair(
                        out_b.partition([16, 32]),
                        out_a.partition([16, 32]),
                        &input,
                        wb,
                        wa,
                    )
                    .generics(vec![columns.to_string()]),
                )?;
            } else {
                scope.record(
                    small_gemm::kernels::dense(out_b.partition([16, 32]), &input, wb)
                        .generics(vec![bf16::DTYPE.as_str().into(), columns.to_string()]),
                )?;
                scope.record(
                    small_gemm::kernels::dense(out_a.partition([16, 32]), &input, wa)
                        .generics(vec![bf16::DTYPE.as_str().into(), columns.to_string()]),
                )?;
            }
        }
        Ok(())
    })?;
    Ok(PairWork {
        graph,
        _outputs: outputs,
        _input: input,
    })
}

#[test]
#[ignore = "requires CUDA and checkpoint; all real GDN weights, paired ABBA cold/warm timing"]
fn real_gdn_verify_pair_latency() -> Result<(), Box<dyn Error>> {
    if cfg!(debug_assertions) {
        return Err("latency requires release".into());
    }
    let compile = CompileOptions::default();
    assert!(!compile.device_debug && compile.opt_level.is_none_or(|level| level == 3));
    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        var("INFER_TEST_MODEL")?,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 64,
            verification_width: 3,
            autotune: false,
            mtp_depth: 0,
            ..LoadOptions::default()
        },
    )?;
    let pairs = gdn_ba_pairs(loaded.graph(), loaded.weights());
    assert_eq!(pairs.len(), 48);
    let baseline = pair_work(&loaded, &pairs, false)?;
    let candidate = pair_work(&loaded, &pairs, true)?;
    let device = loaded.device();
    for cold in [true, false] {
        let options = BenchOptions {
            warmup: Duration::from_millis(50),
            rep: Duration::from_millis(180),
            min_reps: 20,
            max_reps: 100,
            clear_l2: cold,
        };
        for reverse in [false, true] {
            let (first, second) = if reverse {
                (&candidate, &baseline)
            } else {
                (&baseline, &candidate)
            };
            let (a, b) = do_bench_paired(
                &device.stream,
                &options,
                |_| {
                    first
                        .graph
                        .launch()
                        .sync_on(&device.stream)
                        .map_err(|e| tensor_error(&e.to_string()))
                },
                |_| {
                    second
                        .graph
                        .launch()
                        .sync_on(&device.stream)
                        .map_err(|e| tensor_error(&e.to_string()))
                },
            )?;
            let (old, new) = if reverse { (b, a) } else { (a, b) };
            eprintln!(
                "{}",
                serde_json::json!({
                    "gdn_verify_pair_latency":true, "pairs":pairs.len(), "rows":48,
                    "baseline":"two separate existing two-term MMA kernels",
                    "candidate":"one pair kernel with the same per-output accumulator order",
                    "cold_l2":cold, "reverse":reverse, "baseline_ms":old.median_ms(),
                    "candidate_ms":new.median_ms(), "speedup":old.median_ms()/new.median_ms(),
                    "baseline_raw_ms":old.times_ms(), "candidate_raw_ms":new.times_ms(),
                    "baseline_launches":96, "candidate_launches":48, "service_acceptance":false
                })
            );
        }
    }
    Ok(())
}

fn bits_equal(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "nonfinite output at {index}"
        );
        assert_eq!(a.to_bits(), b.to_bits(), "output bits at {index}");
    }
}

fn pool_states_equal(
    loaded: &LoadedModel,
    baseline: &SlotPool,
    candidate: &SlotPool,
    slots: usize,
) -> Result<usize, Box<dyn Error>> {
    let mut elements = 0;
    for slot in 0..slots {
        let (a, af) = baseline.test_states(slot);
        let (b, bf) = candidate.test_states(slot);
        assert_eq!(a.len(), b.len());
        for (id, parts) in a {
            assert_eq!(parts.len(), b[id].len());
            for (a, b) in parts.iter().zip(&b[id]) {
                let a = loaded.device().read_borrowed(a)?;
                let b = loaded.device().read_borrowed(b)?;
                bits_equal(&a, &b);
                elements += a.len();
            }
        }
        assert_eq!(af.len(), bf.len());
        for (id, (ak, av)) in af {
            let (bk, bv) = &bf[id];
            for (a, b) in [(ak, bk), (av, bv)] {
                let a = loaded.device().read_borrowed(a)?;
                let b = loaded.device().read_borrowed(b)?;
                assert_eq!(a.len(), b.len());
                assert!(a.iter().zip(&b).all(|(a, b)| a.0 == b.0), "whole FP8 cache");
                elements += a.len();
            }
        }
    }
    Ok(elements)
}

fn pool_round(
    loaded: &LoadedModel,
    baseline: &mut SlotPool,
    candidate: &mut SlotPool,
    active: &[usize],
    round: usize,
) -> Result<usize, Box<dyn Error>> {
    assert_eq!(baseline.cursors(), candidate.cursors());
    let lanes: Vec<_> = active
        .iter()
        .flat_map(|&slot| {
            let position = baseline.cursors()[slot];
            (0..3).map(move |offset| {
                (
                    slot,
                    offset,
                    u32::try_from(17 + round * 3 + slot * 7 + offset).unwrap(),
                    position + offset,
                    position + offset,
                )
            })
        })
        .collect();
    let a = baseline.run_verify(loaded.device(), &lanes)?;
    let b = candidate.run_verify(loaded.device(), &lanes)?;
    assert_eq!(a.len(), b.len());
    let mut elements = 0;
    for (a, b) in a.iter().zip(&b) {
        bits_equal(&a.0, &b.0);
        bits_equal(&a.1, &b.1);
        elements += a.0.len() + a.1.len();
    }
    for &slot in active {
        if round == 1 && slot == active[0] {
            baseline.discard(slot);
            candidate.discard(slot);
        } else {
            baseline.commit_verify(loaded.device(), slot, slot % 3)?;
            candidate.commit_verify(loaded.device(), slot, slot % 3)?;
        }
    }
    Ok(elements)
}

#[test]
#[ignore = "requires CUDA and MTP checkpoint; actual pooled dispatch, full outputs, rollback and states"]
fn real_pooled_gdn_pair_dispatch_and_full_outputs() -> Result<(), Box<dyn Error>> {
    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        var("INFER_TEST_MODEL")?,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 64,
            verification_width: 3,
            autotune: false,
            mtp_depth: 2,
            ..LoadOptions::default()
        },
    )?;
    let seed = loaded.sequence_pooled(64)?;
    for (slots, paged) in [(2, false), (16, false), (16, true)] {
        let kv_tokens = paged.then_some(slots * 64);
        let before = prefill_projection::verify_pair_captures();
        let mut baseline = prefill_projection::with_verify_pairs(false, || {
            loaded.slot_pool(slots, 64, kv_tokens)
        })?;
        assert_eq!(prefill_projection::verify_pair_captures(), before);
        let mut candidate = loaded.slot_pool(slots, 64, kv_tokens)?;
        let records = prefill_projection::verify_pair_captures() - before;
        assert_eq!(
            records,
            if slots == 16 { 48 } else { 0 },
            "actual pooled verify pair dispatch"
        );
        for slot in 0..slots {
            assert_eq!(baseline.take_free(), Some(slot));
            assert_eq!(candidate.take_free(), Some(slot));
            assert!(baseline.assign(slot, 64, loaded.device())?);
            assert!(candidate.assign(slot, 64, loaded.device())?);
            baseline.bind(slot, seed.states(), seed.fp8_states(), 0, loaded.device())?;
            candidate.bind(slot, seed.states(), seed.fp8_states(), 0, loaded.device())?;
        }
        let all: Vec<_> = (0..slots).collect();
        let mut elements = 0;
        for (round, active) in [all.clone(), vec![0, slots - 1], all].iter().enumerate() {
            elements += pool_round(&loaded, &mut baseline, &mut candidate, active, round)?;
        }
        let states = pool_states_equal(&loaded, &baseline, &candidate, slots)?;
        assert_eq!(baseline.cursors(), candidate.cursors());
        for slot in 0..slots {
            baseline.release(slot);
            candidate.release(slot);
        }
        let recycled = baseline.take_free().ok_or("released slot not reusable")?;
        assert_eq!(candidate.take_free(), Some(recycled));
        assert!(baseline.assign(recycled, 64, loaded.device())?);
        assert!(candidate.assign(recycled, 64, loaded.device())?);
        baseline.bind(
            recycled,
            seed.states(),
            seed.fp8_states(),
            0,
            loaded.device(),
        )?;
        candidate.bind(
            recycled,
            seed.states(),
            seed.fp8_states(),
            0,
            loaded.device(),
        )?;
        elements += pool_round(&loaded, &mut baseline, &mut candidate, &[recycled], 3)?;
        eprintln!(
            "{}",
            serde_json::json!({"actual_pooled_gdn":true,"slots":slots,"paged":paged,"verify_rows":slots*3,
            "pair_captures":records,"full_output_elements":elements,"whole_state_elements":states,
            "sparse_discard_and_partial_commit":true,"release_and_reuse":true})
        );
    }
    Ok(())
}
