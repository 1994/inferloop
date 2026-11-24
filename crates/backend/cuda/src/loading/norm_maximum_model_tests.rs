//! Actual checkpoint activations: compare full hidden/logit/state bits, not just greedy tokens.
use super::{
    LoadOptions, LoadedModel,
    quant_reuse_model_tests::{outputs, program_states},
};
use crate::resident::DeviceProgram;
use crate::{
    AttentionProfile,
    attention_profile::precise_prefix_positions,
    device::CudaDevice,
    resident::{norm_maximum_captures, norm_maximum_consumptions, with_separate_norm_maximum},
};
use infer_core::{Error, ModelId, Result};
use std::{env::var, fs::read};

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL and frozen 1025-token INFER_TEST_TOKENS; safe-run"]
fn norm_maximum_preserves_checkpoint_model_bits() -> Result<()> {
    checkpoint_bits(AttentionProfile::Bf16, 1024, 0)
}

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL and frozen 1025-token INFER_TEST_TOKENS; safe-run"]
fn norm_maximum_keeps_canonical_checkpoint_bits() -> Result<()> {
    checkpoint_bits(AttentionProfile::Canonical, 2048, 64)
}

#[test]
#[ignore = "requires CUDA, INFER_TEST_MODEL and frozen 1025-token INFER_TEST_TOKENS; safe-run"]
fn norm_maximum_preserves_wide_bf16_checkpoint_bits() -> Result<()> {
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
    let before = norm_maximum_captures();
    let before_consumed = norm_maximum_consumptions();
    let mut fused = loaded.sequence(2048)?;
    let captured = norm_maximum_captures();
    let consumed = norm_maximum_consumptions();
    if profile.is_bf16() {
        assert!(captured > before, "must execute the fused model capture");
        assert!(consumed > before_consumed, "must skip the separate maximum");
    } else {
        assert_eq!(captured, before, "canonical must retain separate kernels");
        assert_eq!(consumed, before_consumed, "canonical must not prime scales");
    }
    let mut separate = with_separate_norm_maximum(|| loaded.sequence(2048))?;
    assert_eq!(
        norm_maximum_captures(),
        captured,
        "reference must use separate norm and maximum"
    );
    assert_eq!(norm_maximum_consumptions(), consumed);
    for length in [1025, 511, 1024, 1] {
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
            "checkpoint bits passed: prompt={length}, fused captures={}, maximum skips={}",
            captured - before,
            consumed - before_consumed
        );
    }
    Ok(())
}

fn precise_prefix_checkpoint(
    loaded: &LoadedModel,
    fast: &mut DeviceProgram,
    canonical: &mut DeviceProgram,
    tokens: &[u32],
    length: usize,
) -> Result<()> {
    fast.reset()?;
    canonical.reset()?;
    for (index, chunk) in tokens[..length].chunks(1024).enumerate() {
        let last = (index + 1) * 1024 >= length;
        outputs(
            &fast.prefill_batch(chunk, index * 1024, last)?,
            &canonical.prefill_batch(chunk, index * 1024, last)?,
        );
    }
    program_states(loaded, fast, canonical)?;
    for step in 0..4 {
        let pos = fast.position();
        let token = tokens[(length + step) % tokens.len()];
        outputs(
            &vec![fast.step(token, pos, pos, None, true)?],
            &vec![canonical.step(token, pos, pos, None, true)?],
        );
    }
    program_states(loaded, fast, canonical)?;
    println!("precise2K prefix canonical bits passed: prompt={length}, subsequent_steps=4");
    Ok(())
}
#[test]
#[ignore = "CUDA/checkpoint/frozen511tokens/safe-run; INFER_CUDA_ATTENTION_PRECISE_PREFIX=2048"]
fn bf16_precise_prefix_matches_canonical_checkpoint_bits() -> Result<()> {
    if precise_prefix_positions() != 2048 {
        return Err(Error::invalid(
            "set INFER_CUDA_ATTENTION_PRECISE_PREFIX=2048 for the canonical prefix check",
        ));
    }
    CudaDevice::enable_kernel_cache()?;
    let path = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let tokens: Vec<_> = super::quant_reuse_model_tests::prompt()?
        .into_iter()
        .cycle()
        .take(2049)
        .collect();
    let mut loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        path,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 1024,
            verification_width: 0,
            mtp_depth: 0,
            autotune: false,
            attention_profile: AttentionProfile::Bf16,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    loaded.weights.narrow_prefill_width = 64;
    let mut fast = loaded.sequence(4096)?;
    loaded.weights.attention_profile = AttentionProfile::Canonical;
    let mut canonical = loaded.sequence(4096)?;
    for length in [1, 51, 127, 257, 511, 1024, 1025, 1536, 2048] {
        precise_prefix_checkpoint(&loaded, &mut fast, &mut canonical, &tokens, length)?;
    }
    fast.reset()?;
    canonical.reset()?;
    for (index, chunk) in tokens[..2048].chunks(1024).enumerate() {
        fast.prefill_batch(chunk, index * 1024, false)?;
        canonical.prefill_batch(chunk, index * 1024, false)?;
    }
    let a = fast.prefill_batch(&tokens[2048..2049], 2048, true)?;
    let b = canonical.prefill_batch(&tokens[2048..2049], 2048, true)?;
    let mut changed = 0;
    for ((ah, al), (bh, bl)) in a.iter().zip(&b) {
        for (a, b) in ah.iter().chain(al).zip(bh.iter().chain(bl)) {
            assert!(a.is_finite() && b.is_finite());
            changed += usize::from(a.to_bits() != b.to_bits());
        }
    }
    assert!(changed > 0, "bulkBF16 must remain active after2048");
    println!("BF16 bulk remains active after2K: differing_values={changed}");
    Ok(())
}
