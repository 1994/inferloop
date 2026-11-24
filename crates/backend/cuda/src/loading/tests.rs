use super::*;
use crate::constants::MAX_MTP_DEPTH;
#[test]
fn attention_profile_defaults_to_canonical() {
    let options = LoadOptions::default();
    assert_eq!(
        options.attention_profile,
        crate::AttentionProfile::Canonical
    );
    assert!(!options.attention_profile.is_bf16());
    assert!(!options.attention_profile.is_fp16());
    assert!(!options.attention_profile.is_target_mma());
}
#[test]
fn mtp_depth_controls_the_actual_verification_width() -> Result<()> {
    for depth in [1, 2, 4, 8] {
        let mut options = LoadOptions {
            mtp_depth: depth,
            prefill_width: 32,
            ..LoadOptions::default()
        };
        options.resolve_verification()?;
        assert_eq!(options.verification_width, depth + 1);
    }
    let mut mismatch = LoadOptions {
        mtp_depth: 4,
        verification_width: 3,
        ..LoadOptions::default()
    };
    assert!(mismatch.resolve_verification().is_err());
    Ok(())
}
#[test]
fn mtp_depth_bounds_are_validated() {
    assert!(
        LoadOptions {
            mtp_depth: 0,
            ..LoadOptions::default()
        }
        .validate()
        .is_ok()
    );
    assert!(
        LoadOptions {
            mtp_depth: MAX_MTP_DEPTH,
            ..LoadOptions::default()
        }
        .validate()
        .is_ok()
    );
    assert!(
        LoadOptions {
            mtp_depth: MAX_MTP_DEPTH + 1,
            ..LoadOptions::default()
        }
        .validate()
        .is_err()
    );
}
