//! Complete model output/state comparison plus qualification reuse across serving states.
use super::{
    LoadOptions, LoadedModel,
    quant_reuse_model_tests::{outputs, program_states},
};
use crate::tuning::resident_gemm::ResidentTuningSession;
use crate::{
    AttentionProfile,
    device::{CudaDevice, vendor_fp8::nvfp4},
    resident::DeviceProgram,
};
use infer_core::{Error, ModelId, Result};
use std::{env::var, fs::read, sync::Arc};

#[test]
#[ignore = "requires CUDA, 27B checkpoint and frozen 1025-token INFER_TEST_TOKENS; safe-run"]
fn vendor_fp4_preserves_checkpoint_model_bits() -> Result<()> {
    checkpoint_bits(AttentionProfile::Bf16, 1024, 64)
}
#[test]
#[ignore = "requires CUDA, 27B checkpoint and frozen 1025-token INFER_TEST_TOKENS; safe-run"]
fn vendor_fp4_keeps_canonical_checkpoint_bits() -> Result<()> {
    checkpoint_bits(AttentionProfile::Canonical, 2048, 64)
}
#[test]
#[ignore = "requires CUDA, 27B checkpoint and frozen 1025-token INFER_TEST_TOKENS; safe-run"]
fn vendor_fp4_preserves_wide_bf16_checkpoint_bits() -> Result<()> {
    checkpoint_bits(AttentionProfile::Bf16, 2048, 64)
}
fn loaded_model(
    profile: AttentionProfile,
    width: usize,
    narrow: usize,
) -> Result<(LoadedModel, Vec<u32>)> {
    var("INFER_CUDA_TUNING_TABLE").map_err(|e| Error::invalid(e.to_string()))?;
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
            autotune: true,
            chunked_recurrent: true,
            attention_profile: profile,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    loaded.weights.narrow_prefill_width = narrow;
    Ok((loaded, tokens))
}
fn checkpoint_bits(profile: AttentionProfile, width: usize, narrow: usize) -> Result<()> {
    let (loaded, tokens) = loaded_model(profile, width, narrow)?;
    let before = nvfp4::captures();
    let mut candidate = loaded.sequence(2048)?;
    let captured = nvfp4::captures();
    if profile.is_bf16() {
        assert!(captured > before, "vendor path must actually be captured");
    } else {
        assert_eq!(captured, before, "canonical retains native FP4");
    }
    let mut reference = nvfp4::with_native(|| loaded.sequence(2048))?;
    assert_eq!(
        nvfp4::captures(),
        captured,
        "reference must retain native FP4"
    );
    compare_programs(
        &loaded,
        &mut candidate,
        &mut reference,
        &tokens,
        width,
        &[1025, 65, 1],
    )?;
    println!(
        "complete vendor/native checkpoint bits passed; captures={}",
        captured - before
    );
    Ok(())
}
fn compare_programs(
    loaded: &LoadedModel,
    a: &mut DeviceProgram,
    b: &mut DeviceProgram,
    tokens: &[u32],
    width: usize,
    lengths: &[usize],
) -> Result<()> {
    for &length in lengths {
        a.reset()?;
        b.reset()?;
        for (chunk, input) in tokens[..length].chunks(width).enumerate() {
            outputs(
                &a.prefill_batch(input, chunk * width, true)?,
                &b.prefill_batch(input, chunk * width, true)?,
            );
        }
        program_states(loaded, a, b)?;
        for step in 0..4 {
            let pos = a.position();
            let token = tokens[(length + step) % tokens.len()];
            outputs(
                &vec![a.step(token, pos, pos, None, true)?],
                &vec![b.step(token, pos, pos, None, true)?],
            );
        }
        program_states(loaded, a, b)?;
        println!("full hidden/logit/state bits passed: prompt={length}");
    }
    Ok(())
}
#[test]
#[ignore = "requires CUDA, 27B checkpoint and frozen 1025-token INFER_TEST_TOKENS; safe-run"]
fn vendor_fp4_reuses_model_session_qualifications() -> Result<()> {
    let (mut loaded, tokens) = loaded_model(AttentionProfile::Bf16, 1024, 64)?;
    let before = nvfp4::confirmations();
    let mut first = loaded.sequence(2048)?;
    let confirmed = nvfp4::confirmations();
    assert!(
        confirmed - before <= 1,
        "at most one fresh performance confirmation per geometry"
    );
    let captured = nvfp4::captures();
    loaded.weights.autotune = false;
    loaded.weights.resident_tuning = Arc::new(ResidentTuningSession::default());
    let mut second = loaded.sequence(2048)?;
    assert_eq!(
        nvfp4::confirmations(),
        confirmed,
        "same model must reuse its qualifications"
    );
    assert!(
        nvfp4::captures() > captured,
        "cached state must still capture and execute vendor FP4"
    );
    compare_programs(
        &loaded,
        &mut first,
        &mut second,
        &tokens,
        1024,
        &[1025, 65, 1],
    )?;
    println!(
        "model-session qualifications reused; confirmations={}",
        confirmed - before
    );
    Ok(())
}
