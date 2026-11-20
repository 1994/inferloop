use super::Geometry;
use infer_core::Result;

#[test]
fn validates_batch_buffers_and_pads_patch_rows() -> Result<()> {
    let geometry = Geometry::new(16, [333, 571], [320, 576])?;
    assert_eq!(geometry.batch, 16);
    assert_eq!(geometry.input_elements, 16 * 333 * 571 * 3);
    assert_eq!(geometry.horizontal_elements, 16 * 333 * 576 * 3);
    assert_eq!(geometry.padded_patches, 736);
    assert_eq!(geometry.output_elements, 16 * 736 * 1536);
    Ok(())
}

#[test]
fn rejects_invalid_and_unrepresentable_geometry() {
    assert!(Geometry::new(0, [1, 1], [32, 32]).is_err());
    assert!(Geometry::new(1, [0, 1], [32, 32]).is_err());
    assert!(Geometry::new(1, [1, 1], [33, 32]).is_err());
    assert!(Geometry::new(1, [usize::MAX, 1], [32, 32]).is_err());
    assert!(Geometry::new(usize::MAX, [1, 1], [32, 32]).is_err());
}
