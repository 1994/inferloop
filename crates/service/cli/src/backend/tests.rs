use super::*;
#[test]
fn automatic_selection_requires_a_gpu_and_never_selects_a_test_backend() {
    assert!(resolve(BackendChoice::Auto, false, false).is_err());
    assert_eq!(
        resolve(BackendChoice::Auto, false, true).unwrap(),
        BackendChoice::Metal
    );
    assert_eq!(
        resolve(BackendChoice::Auto, true, true).unwrap(),
        BackendChoice::Cuda
    );
    assert_eq!(
        resolve(BackendChoice::Cuda, false, true).unwrap(),
        BackendChoice::Cuda
    );
}
