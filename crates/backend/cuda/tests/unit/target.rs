use super::*;

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

#[test]
fn unknown_sm_bf16_follows_driver_reported_compute_capability() {
    for name in [
        "sm_70",
        "sm_75",
        "sm_79",
        "sm_8",
        "sm_8x",
        "garbage",
        "sm_090",
        "sm_80a",
        "sm_90f",
        "sm_90af",
        "sm_999999999999999999999999",
    ] {
        assert!(
            !CudaTarget::Other(name.into()).supports_compute(DType::Bf16),
            "{name} must not claim native BF16"
        );
    }
    for name in [
        "sm_80", "sm_86", "sm_89", "sm_90a", "sm_100", "sm_100a", "sm_100f", "sm_103", "sm_120f",
        "sm_999",
    ] {
        assert!(
            CudaTarget::Other(name.into()).supports_compute(DType::Bf16),
            "{name} has BF16-capable compute capability"
        );
    }
}
