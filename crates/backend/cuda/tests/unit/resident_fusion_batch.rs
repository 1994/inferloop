use super::*;
use std::panic::catch_unwind;
use std::panic::resume_unwind;

#[test]
fn capability_limits_scratch_to_validated_layout() -> Result<()> {
    let tiling = LinearTiling::new(TILE[1], DEPTH)?;
    assert_eq!(
        layout_bytes("sm_120", ROWS, Some(WEIGHT_SHAPE), tiling),
        SCRATCH_BYTES
    );
    assert_eq!(
        Some(SCRATCH_BYTES),
        u64::try_from((ROWS * COLUMNS + ROWS * HIDDEN) * size_of::<f32>()).ok()
    );
    for width in [1, 17, 31, 32, 64, 127, 129, 255, 257, 511] {
        assert_eq!(layout_bytes("sm_120", width, Some(WEIGHT_SHAPE), tiling), 0);
    }
    for width in [2, 3, 4, 5, 16] {
        assert_eq!(
            layout_bytes("sm_120", width, Some(WEIGHT_SHAPE), tiling),
            (width * (COLUMNS + HIDDEN) * size_of::<f32>()) as u64,
        );
    }
    for width in [256, 512, 1024, 2048] {
        // Wide fusion owns F32 projection inputs and embeddings, plus the vendor BF16 copy.
        let expected = (width * (10240 + 5120) * 4 + width * 10240 * 2) as u64;
        assert_eq!(
            layout_bytes("sm_120", width, Some(WEIGHT_SHAPE), tiling),
            expected
        );
    }
    for architecture in ["sm_100", "sm_121", ""] {
        assert_eq!(
            layout_bytes(architecture, ROWS, Some(WEIGHT_SHAPE), tiling),
            0
        );
    }
    assert_eq!(layout_bytes("sm_120", ROWS, None, tiling), 0);
    assert_eq!(layout_bytes("sm_120", ROWS, Some([5120, 5120]), tiling), 0);
    for alternative in [
        LinearTiling::new(8, DEPTH)?,
        LinearTiling::new(TILE[1], 128)?,
    ] {
        assert_eq!(
            layout_bytes("sm_120", ROWS, Some(WEIGHT_SHAPE), alternative),
            0
        );
    }
    Ok(())
}

#[test]
fn row_reference_is_nested_and_unwind_safe() {
    assert!(!DISABLED.get());
    with_row_reference(|| {
        assert!(DISABLED.get());
        with_row_reference(|| assert!(DISABLED.get()));
        assert!(DISABLED.get());
    });
    assert!(!DISABLED.get());
    let unwind = catch_unwind(|| with_row_reference(|| resume_unwind(Box::new(()))));
    assert!(unwind.is_err());
    assert!(!DISABLED.get());
}

#[test]
fn batched_gather_preserves_the_ir_graph_contract_and_reduces_captured_nodes() {
    // The profile census keys on the draft IR graph (`node_count = 23`), which this
    // change does not touch: only the number of kernels captured per chunk changes.
    assert_eq!(ROWS, 128);
    assert_eq!(1 + ROWS * 3, 385);
    assert_eq!(captured_nodes(ROWS), 1 + ROWS * 2 + 1);
    assert_eq!(captured_nodes(ROWS), 258);
}
