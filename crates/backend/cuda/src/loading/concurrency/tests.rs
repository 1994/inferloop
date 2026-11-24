use super::*;
use infer_cuda_kernels::contract::target::Arch;
#[test]
fn mixed_precision_roofline_scales_with_compute_bandwidth_and_verification_rows() {
    assert_eq!(bound(160, 20, 100, 10, 1).unwrap(), Some(80));
    assert_eq!(bound(160, 20, 200, 10, 1).unwrap(), Some(160));
    assert_eq!(bound(160, 20, 100, 20, 1).unwrap(), Some(40));
    assert_eq!(bound(160, 20, 100, 10, 3).unwrap(), Some(27));
}
#[test]
fn incomplete_or_overflowing_roofline_facts_are_not_published() {
    assert_eq!(bound(0, 0, 0, 0, 0).unwrap(), None);
    for data in [
        (0, 1, 1, 1, 1),
        (1, 1, 0, 1, 1),
        (1, 1, 1, 0, 1),
        (1, 1, 1, 1, 0),
    ] {
        assert!(bound(data.0, data.1, data.2, data.3, data.4).is_err());
    }
    assert!(bound(u128::MAX, 1, 2, 1, 1).is_err());
    assert_eq!(bound(1, 1000, 1, 1000, 8).unwrap(), Some(1));
}
#[test]
fn unknown_architectures_do_not_inherit_another_familys_tensor_rate() {
    assert!(Arch::from_sm(99, 9).tensor_rates().is_none());
    let hopper = Arch::Sm90.tensor_rates().unwrap();
    assert!(hopper.fp4.is_none());
    let blackwell = Arch::Sm120.tensor_rates().unwrap();
    assert!(blackwell.fp4.unwrap() > blackwell.fp8);
}
