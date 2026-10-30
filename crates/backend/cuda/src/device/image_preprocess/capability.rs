//! Image configurations qualified for the registered CUDA kernel.
use super::geometry::{CHANNELS, Geometry};
use infer_ir::ImageTransform;
use infer_spi::ImagePreprocessing;

/// Qualified patch width and spatial/temporal merge dimensions.
const PATCH: usize = 16;
const TEMPORAL: usize = 2;
const MERGE: usize = 2;
/// Exact normalization used by the RGB8 CUDA kernel.
const NORMALIZATION: f32 = 0.5;
const RESCALE: f32 = 1.0 / 255.0;

pub struct Capability;

impl ImagePreprocessing for Capability {
    fn supports(&self, transform: &ImageTransform) -> bool {
        let layout = transform.layout();
        let normalization = transform.normalization();
        layout.patch() == PATCH
            && layout.temporal() == TEMPORAL
            && layout.merge() == MERGE
            && normalization.mean() == [NORMALIZATION; CHANNELS]
            && normalization.std() == [NORMALIZATION; CHANNELS]
            && normalization.rescale() == RESCALE
            && Geometry::new(1, transform.source(), transform.target()).is_ok()
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/vision_image_capability.rs"]
mod tests;
