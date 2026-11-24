use super::LoadOptions;
use infer_core::Result;

#[test]
fn every_explicit_prompt_bucket_resolves_with_or_without_mtp() -> Result<()> {
    for prefill_width in [32, 64, 128, 256, 512, 1024, 2048] {
        for mtp_depth in 0..=2 {
            let mut options = LoadOptions {
                prefill_width,
                mtp_depth,
                ..LoadOptions::default()
            };
            options.resolve_verification()?;
            assert_eq!(options.prefill_width, prefill_width);
            assert_eq!(
                options.verification_width,
                if mtp_depth == 0 { 0 } else { mtp_depth + 1 }
            );
        }
    }
    Ok(())
}

#[test]
fn unimplemented_prompt_widths_stay_rejected() {
    for prefill_width in [2, 31, 33, 129, 511, 513, 1023, 1025, 2049] {
        let options = LoadOptions {
            prefill_width,
            ..LoadOptions::default()
        };
        assert!(options.validate().is_err(), "width {prefill_width}");
    }
}

#[test]
#[ignore = "requires CUDA, Qwen checkpoint and safe-run"]
fn explicit_wide_prompt_retains_narrow_tail_and_matching_mtp_draft() -> Result<()> {
    use super::LoadedModel;
    use crate::{AttentionProfile, device::CudaDevice};
    use infer_core::{Error, ModelId};
    use std::env::var;
    CudaDevice::enable_kernel_cache()?;
    let model = var("INFER_TEST_MODEL").map_err(|e| Error::invalid(e.to_string()))?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        model,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 2048,
            mtp_depth: 2,
            autotune: false,
            attention_profile: AttentionProfile::Bf16,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    assert!(!loaded.automatic_prefill);
    assert_eq!(loaded.weights.prefill_width, 2048);
    assert_eq!(loaded.weights.narrow_prefill_width, 64);
    let draft = loaded
        .draft
        .as_ref()
        .ok_or_else(|| Error::invariant("missing MTP draft"))?;
    assert_eq!(draft.weights.prefill_width, 2048);
    Ok(())
}
