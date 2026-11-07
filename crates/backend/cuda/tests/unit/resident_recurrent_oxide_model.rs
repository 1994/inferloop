//! Same-checkpoint output and complete recurrent/KV-state gate for Rust SIMT attention.
use super::{LoadOptions, LoadedModel, quant_reuse_model_tests as reference};
use crate::{
    device::CudaDevice,
    resident::recurrent_prefill::oxide::{captures, without_oxide},
};
use infer_core::{Error, ModelId, Result};
use std::env::var;

#[test]
#[ignore = "requires CUDA/checkpoint/frozen511tokens/safe-run; complete model output and state equivalence"]
fn rust_gdn_preserves_checkpoint_model_bits() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens = reference::prompt()?;
    let width = var("INFER_TEST_PREFILL_WIDTH")
        .unwrap_or_else(|_| "256".into())
        .parse::<usize>()
        .map_err(|e| Error::invalid(e.to_string()))?;
    assert!([64, 256].contains(&width));
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            prefill_width: width,
            verification_width: 3,
            mtp_depth: 0,
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    let before = captures();
    let mut control = without_oxide(|| loaded.sequence(1024))?;
    assert_eq!(captures(), before, "reference must use cuTile");
    let mut candidate = loaded.sequence(1024)?;
    let actual_captures = captures() - before;
    assert!(actual_captures >= 48, "Rust GDN must actually be captured");
    let mut next = tokens[0];
    for length in [511, 51] {
        for (chunk, input) in tokens[..length].chunks(width).enumerate() {
            let position = chunk * width;
            let a = candidate.prefill_batch(input, position, true)?;
            let b = control.prefill_batch(input, position, true)?;
            reference::outputs(&a, &b);
            reference::program_states(&loaded, &candidate, &control)?;
            next = reference::greedy(
                &a.last()
                    .ok_or_else(|| Error::invariant("prefill output"))?
                    .1,
            )?;
            println!(
                "{}",
                serde_json::json!({
                    "oxide_gdn_model_prefill":true, "width":width, "length":length,
                    "base":position, "count":input.len(), "full_hidden_logits_bits":true,
                    "full_F32_state_bits":true, "full_FP8_KV_bits":true,
                    "actual_captures":actual_captures,
                })
            );
        }
        for _ in 0..16 {
            let position = candidate.position();
            let a = candidate.step(next, position, position, None, true)?;
            let b = control.step(next, position, position, None, true)?;
            reference::outputs(&vec![a.clone()], &vec![b]);
            reference::program_states(&loaded, &candidate, &control)?;
            next = reference::greedy(&a.1)?;
        }
    }
    for (count, committed) in [(3, 1), (3, 2), (3, 3)] {
        let position = candidate.position();
        reference::outputs(
            &candidate.step_batch(&tokens[..count], position)?,
            &control.step_batch(&tokens[..count], position)?,
        );
        reference::program_states(&loaded, &candidate, &control)?;
        candidate.commit_batch(committed)?;
        control.commit_batch(committed)?;
        reference::program_states(&loaded, &candidate, &control)?;
        let position = candidate.position();
        reference::outputs(
            &vec![candidate.step(next, position, position, None, true)?],
            &vec![control.step(next, position, position, None, true)?],
        );
        reference::program_states(&loaded, &candidate, &control)?;
    }
    println!(
        "{}",
        serde_json::json!({
            "oxide_gdn_complete_model_gate":true,"width":width,"actual_captures":actual_captures,
            "greedy_decode_steps":32,"verification_commit_lengths":[1,2,3],
            "full_hidden_logits_bits":true,"full_recurrent_and_KV_state_bits":true,
            "chunked_recurrent":false,
        })
    );
    Ok(())
}
