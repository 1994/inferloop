use crate::contract::target::{Arch, ImageKernel};
use infer_core::Result;
use infer_ir::DType;

#[test]
fn maps_known_and_unknown_compute_capabilities() {
    assert_eq!(Arch::from_sm(8, 9), Arch::Sm89);
    assert_eq!(Arch::from_sm(9, 0), Arch::Sm90);
    assert_eq!(Arch::from_sm(10, 3), Arch::Sm103);
    assert_eq!(Arch::from_sm(12, 0), Arch::Sm120);
    assert_eq!(Arch::from_sm(9, 1), Arch::Other { major: 9, minor: 1 });
}

#[test]
fn names_architectures_for_manifests() {
    assert_eq!(Arch::Sm89.name(), "sm_89");
    assert_eq!(Arch::Sm100.name(), "sm_100");
    assert_eq!(Arch::Other { major: 9, minor: 1 }.name(), "sm_91");
}

#[test]
fn parses_driver_names_and_rejects_anything_else() {
    assert_eq!(Arch::from_sm_name("sm_89"), Some(Arch::Sm89));
    assert_eq!(Arch::from_sm_name("sm_120"), Some(Arch::Sm120));
    assert_eq!(
        Arch::from_sm_name("sm_121"),
        Some(Arch::Other {
            major: 12,
            minor: 1
        })
    );
    assert_eq!(Arch::from_sm_name("sm_"), None);
    assert_eq!(Arch::from_sm_name("sm_90a"), None);
    assert_eq!(Arch::from_sm_name("gpu_89"), None);
}

#[test]
fn image_program_selection_uses_compute_capability_and_driver() -> Result<()> {
    for sm in [
        "sm_70", "sm_75", "sm_80", "sm_89", "sm_90", "sm_100", "sm_130",
    ] {
        let arch = Arch::from_sm_name(sm).ok_or_else(|| infer_core::Error::invalid(sm))?;
        assert_eq!(arch.image_kernel(13_040)?, Some(ImageKernel::Portable));
    }
    assert_eq!(Arch::Sm120.image_kernel(13_040)?, Some(ImageKernel::Sm120));
    assert_eq!(
        Arch::Sm120.image_kernel(12_000)?,
        Some(ImageKernel::Portable)
    );
    assert!(Arch::Sm80.image_kernel(11_070).is_err());
    assert_eq!(Arch::from_sm(6, 9).image_kernel(13_040)?, None);
    Ok(())
}

#[test]
fn native_fp4_is_a_capability_not_a_storage_policy() {
    assert!(Arch::Sm120.supports_compute(DType::Fp4E2M1));
    assert!(!Arch::Sm90.supports_compute(DType::Fp4E2M1));
    assert!(!Arch::from_sm(99, 9).supports_compute(DType::Fp4E2M1));
    assert!(Arch::Sm90.supports_compute(DType::Fp8E4M3));
    assert!(Arch::from_sm(99, 9).supports_compute(DType::Bf16));
    assert!(Arch::Sm120.require_native_nvfp4().is_ok());
    assert!(Arch::Sm90.require_native_nvfp4().is_err());
    assert!(Arch::from_sm(99, 9).require_native_nvfp4().is_err());
}

#[test]
fn specialized_facts_stay_unknown_until_sourced() {
    assert!(Arch::Sm120.has_simt_artifacts());
    assert!(!Arch::Sm89.has_simt_artifacts());
    assert!(Arch::Sm89.tensor_rates().is_none());
}
