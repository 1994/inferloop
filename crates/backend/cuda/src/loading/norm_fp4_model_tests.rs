//! Actual checkpoint activations: compare full hidden/logit/state bits, not just greedy tokens.
use super::{
    LoadOptions, LoadedModel,
    quant_reuse_model_tests::{outputs, program_states},
};
use crate::{
    AttentionProfile,
    device::CudaDevice,
    resident::{norm_fp4_captures, norm_fp4_consumptions, with_separate_norm_fp4},
};
use infer_core::{Error, ModelId, Result};
use std::{env::var, fs::read};

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL and frozen 1025-token INFER_TEST_TOKENS; safe-run"]
fn norm_fp4_preserves_checkpoint_model_bits() -> Result<()> {
    checkpoint_bits(AttentionProfile::Bf16, 1024, 0)
}

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL and frozen 1025-token INFER_TEST_TOKENS; safe-run"]
fn norm_fp4_keeps_canonical_checkpoint_bits() -> Result<()> {
    checkpoint_bits(AttentionProfile::Canonical, 2048, 64)
}

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL and frozen 1025-token INFER_TEST_TOKENS; safe-run"]
fn norm_fp4_preserves_wide_bf16_checkpoint_bits() -> Result<()> {
    checkpoint_bits(AttentionProfile::Bf16, 2048, 64)
}

fn checkpoint_bits(profile: AttentionProfile, width: usize, narrow: usize) -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let token_path = var("INFER_TEST_TOKENS").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens: Vec<u32> =
        serde_json::from_slice(&read(token_path).map_err(|e| Error::invalid(e.to_string()))?)
            .map_err(|e| Error::invalid(e.to_string()))?;
    assert_eq!(tokens.len(), 1025);
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            prefill_width: width,
            verification_width: 0,
            mtp_depth: 0,
            autotune: false,
            chunked_recurrent: true,
            attention_profile: profile,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    loaded.weights.narrow_prefill_width = narrow;
    let before = norm_fp4_captures();
    let before_consumed = norm_fp4_consumptions();
    let mut fused = loaded.sequence(2048)?;
    let captured = norm_fp4_captures();
    let consumed = norm_fp4_consumptions();
    if profile.is_bf16() {
        assert!(captured > before, "must execute the fused model capture");
        assert!(
            consumed > before_consumed,
            "must skip the separate input quantizer"
        );
    } else {
        assert_eq!(captured, before, "canonical must retain separate kernels");
        assert_eq!(consumed, before_consumed, "canonical must not prime scales");
    }
    let mut separate = with_separate_norm_fp4(|| loaded.sequence(2048))?;
    assert_eq!(
        norm_fp4_captures(),
        captured,
        "reference must use separate norm and quantizer"
    );
    assert_eq!(norm_fp4_consumptions(), consumed);
    for length in [1025, 65, 1] {
        fused.reset()?;
        separate.reset()?;
        for (chunk, input) in tokens[..length].chunks(width).enumerate() {
            outputs(
                &fused.prefill_batch(input, chunk * width, true)?,
                &separate.prefill_batch(input, chunk * width, true)?,
            );
        }
        program_states(&loaded, &fused, &separate)?;
        for step in 0..4 {
            let pos = fused.position();
            let token = tokens[(length + step) % tokens.len()];
            outputs(
                &vec![fused.step(token, pos, pos, None, true)?],
                &vec![separate.step(token, pos, pos, None, true)?],
            );
        }
        program_states(&loaded, &fused, &separate)?;
        println!(
            "checkpoint bits passed: prompt={length}, fused captures={}, quantizer skips={}",
            captured - before,
            consumed - before_consumed
        );
    }
    Ok(())
}
