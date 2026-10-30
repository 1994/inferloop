use super::{ImageInput, ImageNormalization, ImagePatchLayout, ImageTransform};
use infer_core::Result;
use std::sync::Arc;

fn transform() -> Result<ImageTransform> {
    ImageTransform::new(
        [33, 64],
        [32, 64],
        ImagePatchLayout::new(16, 2, 2)?,
        ImageNormalization::new([0.5; 3], [0.5; 3], 1.0 / 255.0)?,
    )
}

#[test]
fn rgb_input_validates_dimensions_and_retains_shared_bytes() -> Result<()> {
    let transform = transform()?;
    assert_eq!(transform.grid(), (1, 2, 4));
    assert!(ImageInput::new(transform, vec![0; 1]).is_err());
    let mut bytes = Vec::with_capacity(33 * 64 * 3 + 100);
    bytes.resize(33 * 64 * 3, 7);
    let input = ImageInput::new(transform, bytes)?;
    assert!(input.retained_bytes() >= input.pixels().len() + 100);
    let shared = input.shared_pixels();
    let copied = input.clone();
    drop(input);
    assert!(Arc::ptr_eq(&shared, &copied.shared_pixels()));
    assert!(!format!("{copied:?}").contains("7, 7"));
    Ok(())
}

#[test]
fn invalid_image_geometry_and_normalization_are_rejected() -> Result<()> {
    let norm = ImageNormalization::new([0.5; 3], [0.5; 3], 1.0 / 255.0)?;
    let layout = ImagePatchLayout::new(16, 2, 2)?;
    assert!(ImagePatchLayout::new(0, 2, 2).is_err());
    assert!(ImagePatchLayout::new(usize::MAX, 2, 2).is_err());
    assert!(ImageNormalization::new([f32::NAN; 3], [0.5; 3], 1.0).is_err());
    assert!(ImageNormalization::new([0.5; 3], [0.0; 3], 1.0).is_err());
    assert!(ImageTransform::new([0, 32], [32, 32], layout, norm).is_err());
    assert!(ImageTransform::new([32, 32], [33, 32], layout, norm).is_err());
    assert!(ImageTransform::new([usize::MAX, 2], [32, 32], layout, norm).is_err());
    Ok(())
}
