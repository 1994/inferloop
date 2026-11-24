use super::Capability;
use infer_core::Result;
use infer_ir::{ImageNormalization, ImagePatchLayout, ImageTransform};
use infer_spi::ImagePreprocessing;

#[test]
fn capability_accepts_only_qualified_layout_and_normalization() -> Result<()> {
    let layout = ImagePatchLayout::new(16, 2, 2)?;
    let norm = ImageNormalization::new([0.5; 3], [0.5; 3], 1.0 / 255.0)?;
    let canonical = ImageTransform::new([256, 256], [256, 256], layout, norm)?;
    assert!(Capability.supports(&canonical));
    let other_layout = ImageTransform::new(
        [256, 256],
        [256, 256],
        ImagePatchLayout::new(16, 1, 2)?,
        norm,
    )?;
    assert!(!Capability.supports(&other_layout));
    let other_norm = ImageTransform::new(
        [256, 256],
        [256, 256],
        layout,
        ImageNormalization::new([0.0; 3], [1.0; 3], 1.0 / 255.0)?,
    )?;
    assert!(!Capability.supports(&other_norm));
    Ok(())
}
