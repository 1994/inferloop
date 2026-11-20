//! Native gated norm occupancy changes in the actual checkpoint graph.
use super::{LoadOptions, LoadedModel, quant_reuse_model_tests as reference};
use crate::{device::CudaDevice, resident::gated_norm_discovery as candidate};
use infer_core::{Error, ModelId, Result};
use std::env::var;

#[test]
#[ignore = "requires CUDA/checkpoint/safe-run; full states/logits/hidden before service timing"]
fn gated_norm_occupancy_preserves_full_model_bits() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens = reference::prompt()?;
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 256,
            verification_width: 3,
            mtp_depth: 0,
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    for width in [64, 256] {
        loaded.weights.prefill_width = width;
        loaded.weights.narrow_prefill_width = 0;
        let mut control = candidate::with_model_baseline(|| loaded.sequence(1024))?;
        let mut tuned = loaded.sequence(1024)?;
        assert!(
            candidate::model_sites() > 0,
            "candidate must change real captured norms"
        );
        for length in [51, 64, 255, 256, 511] {
            for (chunk, input) in tokens[..length].chunks(width).enumerate() {
                let last = (chunk + 1) * width >= length;
                reference::outputs(
                    &control.prefill_batch_readout(input, chunk * width, last, last)?,
                    &tuned.prefill_batch_readout(input, chunk * width, last, last)?,
                );
                reference::program_states(&loaded, &tuned, &control)?;
            }
            let mut next = tokens[0];
            for _ in 0..8 {
                let position = control.position();
                let a = control.step(next, position, position, None, true)?;
                let b = tuned.step(next, position, position, None, true)?;
                next = reference::greedy(&a.1)?;
                assert_eq!(next, reference::greedy(&b.1)?);
                reference::outputs(&vec![a], &vec![b]);
                reference::program_states(&loaded, &tuned, &control)?;
            }
            println!(
                "{}",
                serde_json::json!({"gated_norm_model_bits":true,"prefill_width":width,
                "length":length,"decode_steps":8,"full_hidden_logits_F32_FP8":true,
                "candidate_capture_sites":candidate::model_sites()})
            );
        }
        for (count, commit) in [(3, 1), (2, 2), (1, 1), (3, 2), (3, 3)] {
            let position = control.position();
            reference::outputs(
                &control.step_batch(&tokens[..count], position)?,
                &tuned.step_batch(&tokens[..count], position)?,
            );
            reference::program_states(&loaded, &tuned, &control)?;
            control.commit_batch(commit)?;
            tuned.commit_batch(commit)?;
            reference::program_states(&loaded, &tuned, &control)?;
        }
    }
    Ok(())
}
