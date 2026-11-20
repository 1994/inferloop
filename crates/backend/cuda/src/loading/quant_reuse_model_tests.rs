//! Same-weight whole-model gate for adjacent shared-input projection quantization reuse.
use super::support::states;
use super::{LoadOptions, LoadedModel};
use crate::resident::nvfp4_gemm::QuantPairPlan;
use crate::tuning::table_path;
use crate::{
    device::CudaDevice,
    resident::{BatchOutput, DeviceProgram, nvfp4_gemm::with_quant_reuse_disabled},
};
use infer_core::{Error, ModelId, Result};
use std::env::var;
use std::fs::read;

pub(super) fn bits(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite() && expected.is_finite(),
            "{label} nonfinite at {index}"
        );
        assert_eq!(actual.to_bits(), expected.to_bits(), "{label} at {index}");
    }
}

pub(super) fn outputs(actual: &BatchOutput, expected: &BatchOutput) {
    assert_eq!(actual.len(), expected.len());
    assert_ne!(actual.len(), 0, "batch output is empty");
    for (actual, expected) in actual.iter().zip(expected) {
        bits(&actual.0, &expected.0, "hidden");
        bits(&actual.1, &expected.1, "logits");
    }
}

pub(super) fn program_states(
    loaded: &LoadedModel,
    actual: &DeviceProgram,
    expected: &DeviceProgram,
) -> Result<()> {
    assert_eq!(actual.position(), expected.position());
    states(
        loaded.device(),
        actual.states(),
        expected.states(),
        actual.fp8_states(),
        expected.fp8_states(),
    )
}

pub(super) fn prompt() -> Result<Vec<u32>> {
    let path = var("INFER_TEST_TOKENS").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens: Vec<u32> =
        serde_json::from_slice(&read(path).map_err(|e| Error::invalid(e.to_string()))?)
            .map_err(|e| Error::invalid(e.to_string()))?;
    if tokens.len() != 511 {
        return Err(Error::invalid(
            "quant reuse gate requires frozen 511-token prompt",
        ));
    }
    Ok(tokens)
}

pub(super) fn greedy(logits: &[f32]) -> Result<u32> {
    let mut best = 0usize;
    for index in 1..logits.len() {
        if logits[index] > logits[best] {
            best = index;
        }
    }
    u32::try_from(best).map_err(|e| Error::invalid(e.to_string()))
}

fn pooled(
    loaded: &LoadedModel,
    actual: &DeviceProgram,
    expected: &DeviceProgram,
    tokens: &[u32],
) -> Result<()> {
    let mut shared = loaded.slot_pool(4, 128, None)?;
    let mut separate = with_quant_reuse_disabled(|| loaded.slot_pool(4, 128, None))?;
    assert!(
        shared.quant_reuse_stats().1 > 0,
        "pooled capture must reuse input"
    );
    assert_eq!(separate.quant_reuse_stats().1, 0);
    for slot in 0..4 {
        assert_eq!(shared.take_free(), Some(slot));
        assert_eq!(separate.take_free(), Some(slot));
        shared.bind(
            slot,
            actual.states(),
            actual.fp8_states(),
            actual.position(),
            loaded.device(),
        )?;
        separate.bind(
            slot,
            expected.states(),
            expected.fp8_states(),
            expected.position(),
            loaded.device(),
        )?;
    }
    for active in [vec![0, 1, 2, 3], vec![0, 2], vec![1, 3], vec![0, 1, 2, 3]] {
        let lanes: Vec<_> = active
            .iter()
            .map(|&slot| {
                let position = shared.cursors()[slot];
                (
                    slot,
                    tokens[(position + slot) % tokens.len()],
                    position,
                    position,
                )
            })
            .collect();
        assert_eq!(shared.cursors(), separate.cursors());
        let a = shared.run_lanes(loaded.device(), &lanes)?;
        let b = separate.run_lanes(loaded.device(), &lanes)?;
        // Read active logits, then compare every slot state including inactive lanes.
        outputs(&a, &b);
        for slot in 0..4 {
            let (a, af) = shared.test_states(slot);
            let (b, bf) = separate.test_states(slot);
            states(loaded.device(), a, b, af, bf)?;
        }
    }
    shared.release(2);
    separate.release(2);
    assert_eq!(shared.take_free(), Some(2));
    assert_eq!(separate.take_free(), Some(2));
    shared.bind(
        2,
        actual.states(),
        actual.fp8_states(),
        actual.position(),
        loaded.device(),
    )?;
    separate.bind(
        2,
        expected.states(),
        expected.fp8_states(),
        expected.position(),
        loaded.device(),
    )?;
    let position = actual.position();
    outputs(
        &shared.run_lanes(loaded.device(), &[(2, tokens[0], position, position)])?,
        &separate.run_lanes(loaded.device(), &[(2, tokens[0], position, position)])?,
    );
    for slot in 0..4 {
        let (a, af) = shared.test_states(slot);
        let (b, bf) = separate.test_states(slot);
        states(loaded.device(), a, b, af, bf)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL and frozen 511-token INFER_TEST_TOKENS; safe-run"]
fn shared_quantization_preserves_checkpoint_model_bits() -> Result<()> {
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens = prompt()?;
    let cache_path = table_path().ok_or_else(|| Error::invalid("frozen tuning table required"))?;
    let cache = read(&cache_path).map_err(|e| Error::invalid(e.to_string()))?;
    CudaDevice::enable_kernel_cache()?;
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::new(1)?,
        LoadOptions {
            prefill_width: 256,
            verification_width: 3,
            mtp_depth: 0,
            autotune: false,
            chunked_recurrent: true,
            ..LoadOptions::default()
        },
    )?;
    let plan = QuantPairPlan::new(&loaded.graph, &loaded.weights);
    let planned = (0..loaded.graph.nodes.len())
        .filter(|&index| plan.role(index).is_some())
        .count();
    eprintln!(
        "{}",
        serde_json::json!({"quant_reuse_planned_nodes":planned})
    );
    for width in [64, 256] {
        loaded.weights.prefill_width = width;
        loaded.weights.narrow_prefill_width = 0;
        let mut shared = loaded.sequence(1024)?;
        let mut separate = with_quant_reuse_disabled(|| loaded.sequence(1024))?;
        assert!(
            shared.quant_reuse_stats().1 > 0,
            "model capture must reuse input"
        );
        assert_eq!(separate.quant_reuse_stats().1, 0);
        // Include actual 51, 64, 255, 256 and 511 lengths, nonzero bases, inactive tails,
        // changed tokens across replay, and resetting from a previously long prefix.
        for length in [511, 51, 64, 255, 256, 1] {
            for (chunk, input) in tokens[..length].chunks(width).enumerate() {
                outputs(
                    &shared.prefill_batch(input, chunk * width, true)?,
                    &separate.prefill_batch(input, chunk * width, true)?,
                );
            }
            program_states(&loaded, &shared, &separate)?;
            for step in 0..8 {
                let position = shared.position();
                let token = tokens[(step + length) % tokens.len()];
                let a = shared.step(token, position, position, None, true)?;
                let b = separate.step(token, position, position, None, true)?;
                outputs(&vec![a], &vec![b]);
            }
            program_states(&loaded, &shared, &separate)?;
        }
        // Normal greedy choices from full vocabulary and rollback/replayed verification.
        let mut next = tokens[0];
        for _ in 0..16 {
            let position = shared.position();
            let a = shared.step(next, position, position, None, true)?;
            let b = separate.step(next, position, position, None, true)?;
            let token_a = greedy(&a.1)?;
            let token_b = greedy(&b.1)?;
            assert_eq!(token_a, token_b);
            outputs(&vec![a], &vec![b]);
            next = token_a;
        }
        for (count, commit) in [(3, 1), (2, 2), (1, 1), (3, 2), (3, 3)] {
            let position = shared.position();
            let input = &tokens[..count];
            outputs(
                &shared.step_batch(input, position)?,
                &separate.step_batch(input, position)?,
            );
            program_states(&loaded, &shared, &separate)?;
            shared.commit_batch(commit)?;
            separate.commit_batch(commit)?;
            program_states(&loaded, &shared, &separate)?;
        }
        shared.prefill_batch(&tokens[..51], 0, true)?;
        separate.prefill_batch(&tokens[..51], 0, true)?;
        program_states(&loaded, &shared, &separate)?;
        pooled(&loaded, &shared, &separate, &tokens)?;
        eprintln!(
            "{}",
            serde_json::json!({
                "whole_model_quant_reuse_bitwise": true, "width":width,
                "shared_capture_stats":shared.quant_reuse_stats(),
                "separate_capture_stats":separate.quant_reuse_stats(),
                "F32_state":true, "FP8_KV":true, "MTP":0,
                "greedy_steps":16, "forced_decode_steps_per_length":8,
                "pooled_inactive_reactivate_rebind":true,
            })
        );
    }
    assert_eq!(
        cache,
        read(cache_path).map_err(|e| Error::invalid(e.to_string()))?
    );
    Ok(())
}
