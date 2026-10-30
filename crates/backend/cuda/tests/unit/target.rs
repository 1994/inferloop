use super::*;

#[test]
fn image_program_selection_uses_compute_capability_and_driver() -> Result<()> {
    for sm in [
        "sm_70", "sm_75", "sm_80", "sm_89", "sm_90", "sm_100", "sm_130",
    ] {
        assert_eq!(
            CudaTarget::from_sm_name(sm).image_kernel(13_040)?,
            Some(ImageKernel::Portable)
        );
    }
    assert_eq!(
        CudaTarget::from_sm_name("sm_120").image_kernel(13_040)?,
        Some(ImageKernel::Sm120)
    );
    assert_eq!(
        CudaTarget::from_sm_name("sm_120").image_kernel(12_000)?,
        Some(ImageKernel::Portable)
    );
    assert!(
        CudaTarget::from_sm_name("sm_80")
            .image_kernel(11_070)
            .is_err()
    );
    for sm in ["sm_69", "sm_20", "sm_", "sm_x", "unknown"] {
        assert_eq!(CudaTarget::from_sm_name(sm).image_kernel(13_040)?, None);
    }
    Ok(())
}

#[test]
fn native_fp4_is_a_capability_not_a_storage_policy() {
    assert!(CudaTarget::BlackwellSm120.supports_compute(DType::Fp4E2M1));
    assert!(!CudaTarget::HopperSm90.supports_compute(DType::Fp4E2M1));
    assert!(!CudaTarget::Other("sm_999".into()).supports_compute(DType::Fp4E2M1));
    assert!(CudaTarget::HopperSm90.supports_compute(DType::Fp8E4M3));
    assert!(CudaTarget::Other("sm_999".into()).supports_compute(DType::Bf16));
}

#[test]
fn production_and_test_devices_do_not_share_fp4_capability() {
    assert!(
        CudaTarget::from_sm_name("sm_90")
            .require_native_nvfp4()
            .is_err()
    );
    assert!(
        CudaTarget::from_sm_name("sm_120")
            .require_native_nvfp4()
            .is_ok()
    );
    assert!(
        CudaTarget::from_sm_name("sm_999")
            .require_native_nvfp4()
            .is_err()
    );
}
