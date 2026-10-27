use super::{DENSE_REFERENCE, narrow_dense_layout, with_dense_reference};
use std::panic::catch_unwind;

#[test]
fn narrow_dense_control_geometry_is_not_model_specific() {
    for shape in [(64, 48, 5120), (64, 16, 96), (64, 32, 4096), (64, 64, 128)] {
        assert!(
            narrow_dense_layout("sm_120", shape),
            "graph-proven control shape {shape:?}"
        );
    }
    for architecture in ["sm_100", "sm_90", "sm_121", ""] {
        assert!(!narrow_dense_layout(architecture, (64, 48, 5120)));
    }
    for shape in [
        (1, 48, 5120),
        (3, 48, 5120),
        (32, 48, 5120),
        (128, 48, 5120),
        (64, 0, 128),
        (64, 128, 128),
        (64, 16, 0),
    ] {
        assert!(!narrow_dense_layout("sm_120", shape));
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
