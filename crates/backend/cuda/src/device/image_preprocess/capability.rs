//! Image configurations qualified for the registered CUDA kernel.
use super::geometry::{CHANNELS, Geometry};
use super::nvjpeg::LazyDecoder;
use infer_core::Result;
use infer_ir::{EncodedImageInfo, ImageTransform};
use infer_spi::{DeviceHostExecutor, ImagePreprocessing};
use std::sync::Arc;

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

pub(super) struct DeviceCapability {
    decoder: Arc<LazyDecoder>,
}

impl DeviceCapability {
    pub const fn new(decoder: Arc<LazyDecoder>) -> Self {
        Self { decoder }
    }
}

impl ImagePreprocessing for DeviceCapability {
    fn set_host_executor(&self, executor: Arc<dyn DeviceHostExecutor>) -> Result<()> {
        self.decoder.set_host_executor(executor)
    }

    fn supports(&self, transform: &ImageTransform) -> bool {
        Capability.supports(transform)
    }

    fn initialize_encoded(&self) -> Result<()> {
        let available = self.decoder.get()?.is_some();
        tracing::info!(target: "infer::cuda", available, "nvJPEG hardware image decoding");
        Ok(())
    }

    fn inspect_encoded(&self, bytes: &[u8]) -> Result<Option<EncodedImageInfo>> {
        self.decoder.inspect(bytes)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/vision_image_capability.rs"]
mod tests;
