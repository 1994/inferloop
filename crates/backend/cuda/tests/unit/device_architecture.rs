use super::{CudaArchitecture, is_sm120};

#[test]
fn reported_architecture_names_are_exact_and_unknown_names_stay_unknown() {
    for (name, expected) in [
        ("sm_90", CudaArchitecture::Sm90),
        ("sm_100", CudaArchitecture::Sm100),
        ("sm_103", CudaArchitecture::Sm103),
        ("sm_120", CudaArchitecture::Sm120),
        ("sm_90a", CudaArchitecture::Other),
        ("sm_121", CudaArchitecture::Other),
        ("sm_1000", CudaArchitecture::Other),
        ("SM_120", CudaArchitecture::Other),
        ("", CudaArchitecture::Other),
    ] {
        assert_eq!(CudaArchitecture::from_name(name), expected);
        assert_eq!(is_sm120(name), expected == CudaArchitecture::Sm120);
    }
}
