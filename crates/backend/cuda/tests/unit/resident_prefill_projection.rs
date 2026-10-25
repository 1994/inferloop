use super::{DENSE_REFERENCE, narrow_dense_layout, with_dense_reference};
use std::panic::catch_unwind;

#[test]
fn narrow_dense_scope_matches_the_small_projection_structure() {
    for (rows, columns) in [(48, 5120), (48, 4096), (48, 8192)] {
        assert!(narrow_dense_layout("sm_120", rows, columns));
    }
    for architecture in ["sm_100", "sm_90", "sm_121", ""] {
        assert!(!narrow_dense_layout(architecture, 48, 5120));
    }
    for (rows, columns) in [(32, 5120), (64, 5120), (48, 5121), (48, 4095)] {
        assert!(!narrow_dense_layout("sm_120", rows, columns));
    }
}

#[test]
fn dense_reference_restores_after_nested_scope_and_unwind() {
    assert!(!DENSE_REFERENCE.get());
    with_dense_reference(|| {
        assert!(DENSE_REFERENCE.get());
        with_dense_reference(|| assert!(DENSE_REFERENCE.get()));
        assert!(DENSE_REFERENCE.get());
    });
    assert!(!DENSE_REFERENCE.get());
    let failure = catch_unwind(|| with_dense_reference(|| panic!("reference unwind")));
    assert!(failure.is_err());
    assert!(!DENSE_REFERENCE.get());
}
