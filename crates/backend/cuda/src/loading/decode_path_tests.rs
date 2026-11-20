//! Full checkpoint census of private/pooled arithmetic with a common forced trajectory.
//! Original GEMV is a diagnostic control; the experimental private lowering must match pool.
use super::support::{agreement, differences, seed_prefix, state_census_slot};
use super::{LoadOptions, LoadedModel};
use crate::tuning::table_path;
use crate::{
    device::CudaDevice,
    resident::{
        DeviceProgram, pool_f32_capture_records, slot_batch::SlotPool, with_pool_f32,
        with_precise_dense,
    },
};
use infer_core::{Error, ModelId, Result};
use serde_json::{Value, json};
use std::env::var;
use std::fs::read;
use std::path::PathBuf;

fn state_census(
    loaded: &LoadedModel,
    program: &DeviceProgram,
    pool: &SlotPool,
) -> Result<(usize, Vec<Value>)> {
    state_census_slot(loaded, program, pool, 0)
}
fn common_prefix(
    loaded: &LoadedModel,
    original: &mut DeviceProgram,
    aligned: &mut DeviceProgram,
    tokens: &[u32],
) -> Result<Vec<f32>> {
    let control = original.prefill_batch(&tokens[..179], 0, true)?;
    let precise = aligned.prefill_batch(&tokens[..179], 0, true)?;
    assert_eq!(control.len(), precise.len());
    let mut changed = 0;
    for (control, precise) in control.iter().zip(&precise) {
        changed += differences(&agreement(
            &control.0,
            &precise.0,
            "prefill arithmetic change",
        ));
    }
    seed_prefix(loaded, aligned, original)?;
    println!(
        "{}",
        json!({"private_pool_common_prefix":true,"original_prefill_hidden_differences":changed,"control_state_seeded_from_aligned":true,"F32_state":true,"FP8_KV":true})
    );
    Ok(precise
        .last()
        .ok_or_else(|| Error::invariant("prompt logits"))?
        .1
        .clone())
}

fn greedy(logits: &[f32]) -> Result<u32> {
    assert_ne!(logits.len(), 0);
    let mut best = 0;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    u32::try_from(best).map_err(|e| Error::invalid(e.to_string()))
}
#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL, frozen INFER_TEST_TOKENS/table; full private/pool arithmetic diagnostic, safe-run"]
fn checkpoint_private_pool_first_step_arithmetic_census() -> Result<()> {
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens_path = var("INFER_TEST_TOKENS").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens: Vec<u32> =
        serde_json::from_slice(&read(tokens_path).map_err(|e| Error::invalid(e.to_string()))?)
            .map_err(|e| Error::invalid(e.to_string()))?;
    assert_eq!(tokens.len(), 511);
    let cache_path = table_path().ok_or_else(|| Error::invalid("frozen table required"))?;
    let cache = read(&cache_path).map_err(|e| Error::invalid(e.to_string()))?;
    CudaDevice::enable_kernel_cache()?;
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::new(1)?,
        LoadOptions {
            prefill_width: 256,
            autotune: false,
            chunked_recurrent: true,
            mtp_depth: 0,
            ..LoadOptions::default()
        },
    )?;
    loaded.weights.narrow_prefill_width = 0;
    let mut original = with_precise_dense(false, || loaded.sequence(1024))?;
    let mut aligned = with_precise_dense(true, || loaded.sequence(1024))?;
    let mut pool = with_precise_dense(true, || loaded.slot_pool(4, 1024, None))?;
    let logits = common_prefix(&loaded, &mut original, &mut aligned, &tokens)?;
    let mut token = greedy(&logits)?;
    assert_eq!(pool.take_free(), Some(0));
    pool.bind(
        0,
        aligned.states(),
        aligned.fp8_states(),
        aligned.position(),
        loaded.device(),
    )?;
    let (initial, details) = state_census(&loaded, &aligned, &pool)?;
    assert_eq!(initial, 0, "initial state {details:?}");
    let (control_initial, details) = state_census(&loaded, &original, &pool)?;
    assert_eq!(control_initial, 0, "initial control state {details:?}");
    let mut aligned_differences = 0;
    let mut control_differences = 0;
    for step in 0..8 {
        let position = aligned.position();
        assert_eq!(original.position(), position);
        let original_output = original.step(token, position, position, None, true)?;
        let output = aligned.step(token, position, position, None, true)?;
        let pooled = pool.run_lanes(loaded.device(), &[(0, token, position, position)])?;
        assert_eq!(pooled.len(), 1);
        let reference = &pooled[0];
        let pooled_hidden = pool.test_hidden(loaded.device(), 0, 1)?;
        let hidden = agreement(&output.0, &pooled_hidden, "aligned hidden");
        let logits = agreement(&output.1, &reference.1, "aligned full vocabulary");
        let old_hidden = agreement(&original_output.0, &pooled_hidden, "original GEMV hidden");
        let old_logits = agreement(
            &original_output.1,
            &reference.1,
            "original GEMV full vocabulary",
        );
        aligned_differences += differences(&hidden) + differences(&logits);
        control_differences += differences(&old_hidden) + differences(&old_logits);
        let (state, state_details) = if step == 0 || step == 7 {
            state_census(&loaded, &aligned, &pool)?
        } else {
            (0, Vec::new())
        };
        aligned_differences += state;
        let (old_state, old_state_details) = if step == 0 {
            state_census(&loaded, &original, &pool)?
        } else {
            (0, Vec::new())
        };
        println!(
            "{}",
            json!({"private_pool_arithmetic_step":step,"position":position,"input_token":token,"aligned_hidden":hidden,"aligned_logits":logits,"original_hidden":old_hidden,"original_logits":old_logits,"aligned_state_differences":state,"aligned_state_details":state_details,"original_state_differences":old_state,"original_state_details":old_state_details,"MTP":0,"pool_width":4,"capacity":1024})
        );
        token = greedy(&output.1)?;
    }
    println!(
        "{}",
        json!({"private_pool_arithmetic_summary":true,"aligned_differences":aligned_differences,"original_control_output_differences":control_differences,"F32_state":true,"FP8_KV_complete":true,"steps":8})
    );
    assert_eq!(
        cache,
        read(cache_path).map_err(|e| Error::invalid(e.to_string()))?
    );
    // A census, not a gate: the numeric contract judges predictions, and private and pooled
    // tensor-core arithmetic need not agree bit for bit.
    Ok(())
}

struct PoolCheckpoint {
    controls: Vec<DeviceProgram>,
    pool: SlotPool,
    loaded: LoadedModel,
    next: Vec<u32>,
    vocabulary: u32,
    cache_path: PathBuf,
    cache: Vec<u8>,
}
fn pool_checkpoint() -> Result<PoolCheckpoint> {
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens_path = var("INFER_TEST_TOKENS").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens: Vec<u32> =
        serde_json::from_slice(&read(tokens_path).map_err(|e| Error::invalid(e.to_string()))?)
            .map_err(|e| Error::invalid(e.to_string()))?;
    assert_eq!(tokens.len(), 511);
    let cache_path = table_path().ok_or_else(|| Error::invalid("frozen table required"))?;
    let cache = read(&cache_path).map_err(|e| Error::invalid(e.to_string()))?;
    CudaDevice::enable_kernel_cache()?;
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::new(1)?,
        LoadOptions {
            prefill_width: 256,
            autotune: false,
            chunked_recurrent: true,
            mtp_depth: 0,
            ..LoadOptions::default()
        },
    )?;
    loaded.weights.narrow_prefill_width = 0;
    let mut first = with_pool_f32(false, || loaded.sequence_pooled(1024))?;
    let prompt = first.prefill_batch(&tokens[..179], 0, true)?;
    let logits = &prompt
        .last()
        .ok_or_else(|| Error::invariant("prompt output"))?
        .1;
    let vocabulary = u32::try_from(logits.len()).map_err(|e| Error::invalid(e.to_string()))?;
    let initial = greedy(logits)?;
    let mut controls = vec![first];
    // Other controls only replay private decode, so avoid unused wide prompt/verify captures.
    loaded.weights.prefill_width = 0;
    for _ in 1..4 {
        let mut program = with_pool_f32(false, || loaded.sequence_pooled(1024))?;
        seed_prefix(&loaded, &controls[0], &mut program)?;
        controls.push(program);
    }
    let before = pool_f32_capture_records();
    let mut pool = with_pool_f32(true, || loaded.slot_pool(4, 1024, None))?;
    let records = pool_f32_capture_records().saturating_sub(before);
    assert!(
        records >= 96,
        "F32 pool capture was not used for all real small projections: {records}"
    );
    println!(
        "{}",
        json!({"pool_four_capture_records":records,"expected_small_projections":96})
    );
    for (slot, program) in controls.iter().enumerate() {
        assert_eq!(pool.take_free(), Some(slot));
        pool.bind(
            slot,
            program.states(),
            program.fp8_states(),
            program.position(),
            loaded.device(),
        )?;
        let (count, details) = state_census_slot(&loaded, program, &pool, slot)?;
        assert_eq!(count, 0, "initial slot {slot}: {details:?}");
    }
    let next: Vec<u32> = (0..4)
        .map(|slot| (initial + u32::try_from(slot * 11).expect("four slots")) % vocabulary)
        .collect();
    Ok(PoolCheckpoint {
        controls,
        pool,
        loaded,
        next,
        vocabulary,
        cache_path,
        cache,
    })
}

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL/frozen tokens/table, safe-run; four distinct trajectories, inactive/rebind, complete private F32/pool equality"]
fn checkpoint_pool_four_slots_private_f32_census() -> Result<()> {
    let PoolCheckpoint {
        mut controls,
        mut pool,
        loaded,
        mut next,
        vocabulary,
        cache_path,
        cache,
    } = pool_checkpoint()?;
    let mut changed = 0;
    let mut total = 0;
    let patterns = [
        vec![0, 1, 2, 3],
        vec![0, 2],
        vec![1, 3],
        vec![3],
        vec![0, 1, 3],
        vec![0, 1, 2, 3],
        vec![1, 2],
        vec![0, 1, 2, 3],
    ];
    for (step, active) in patterns.iter().enumerate() {
        if step == 5 {
            pool.release(2);
            assert_eq!(pool.take_free(), Some(2));
            pool.bind(
                2,
                controls[2].states(),
                controls[2].fp8_states(),
                controls[2].position(),
                loaded.device(),
            )?;
        }
        let lanes: Vec<_> = active
            .iter()
            .map(|&slot| {
                (
                    slot,
                    next[slot],
                    controls[slot].position(),
                    controls[slot].position(),
                )
            })
            .collect();
        let private = lanes
            .iter()
            .map(|&(slot, token, position, _)| {
                controls[slot].step(token, position, position, None, true)
            })
            .collect::<Result<Vec<_>>>()?;
        let pooled = pool.run_lanes(loaded.device(), &lanes)?;
        assert_eq!(pooled.len(), active.len());
        for ((&slot, original), output) in active.iter().zip(&private).zip(&pooled) {
            let hidden = agreement(
                &original.0,
                &pool.test_hidden(
                    loaded.device(),
                    slot,
                    active.iter().max().map_or(0, |top| top + 1),
                )?,
                "pool four hidden",
            );
            let logits = agreement(&original.1, &output.1, "pool four vocabulary");
            changed += differences(&hidden) + differences(&logits);
            total += original.0.len() + original.1.len();
            println!(
                "{}",
                json!({"pool_four_model_step":step,"slot":slot,"hidden":hidden,"logits":logits})
            );
            next[slot] =
                (greedy(&original.1)? + u32::try_from(slot * 3).expect("four slots")) % vocabulary;
        }
        if step == 0 || step + 1 == patterns.len() {
            for (slot, program) in controls.iter().enumerate() {
                let (count, details) = state_census_slot(&loaded, program, &pool, slot)?;
                changed += count;
                println!(
                    "{}",
                    json!({"pool_four_model_state":step,"slot":slot,"different":count,"details":details})
                );
            }
        }
    }
    println!(
        "{}",
        json!({"pool_four_model_summary":true,"steps":8,"output_elements":total,"different":changed,"F32_state":true,"FP8_KV_complete":true,"private_arithmetic_unchanged":true})
    );
    assert_eq!(
        cache,
        read(cache_path).map_err(|e| Error::invalid(e.to_string()))?
    );
    assert_eq!(
        changed, 0,
        "pool four slots must match original private F32 computation"
    );
    Ok(())
}
