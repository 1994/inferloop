//! CUDA image preprocessing plans and device execution.
mod axis;
mod capability;
mod geometry;
mod launch;
mod tables;

use self::geometry::{Geometry, PATCH_WIDTH};
use self::launch::Kernels;
use self::tables::Tables;
use crate::device::{CudaDevice, device_error};
use cuda_core::Function;
use cutile::cuda_async::device_operation::Zippable;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_ir::ImageTransform;
use infer_spi::ImagePreprocessing;
use std::sync::Arc;

pub use capability::Capability;

const PTX: &str =
    include_str!("../../../kernels/image-preprocess/preprocess-batch-strict-sm120.ptx");

/// An equal-geometry batch of packed RGB8 images accepted by this CUDA processor.
pub struct ImageBatch {
    pixels: Arc<Vec<u8>>,
    geometry: Geometry,
}

impl ImageBatch {
    /// Validate the image transformation and move the shared RGB8 owner into a batch.
    ///
    /// Images are consecutive in row-major RGB order and all use `transform`. The processor
    /// retains the shared owner until the device copy has completed.
    /// # Errors
    /// Rejects unsupported transforms, zero/overflowing batch dimensions or a wrong byte count.
    pub fn new(pixels: Arc<Vec<u8>>, count: usize, transform: ImageTransform) -> Result<Self> {
        if !Capability.supports(&transform) {
            return Err(Error::unsupported(
                "CUDA image processor cannot execute this transform",
            ));
        }
        let geometry = Geometry::new(count, transform.source(), transform.target())?;
        if pixels.len() != geometry.input_elements {
            return Err(Error::invalid(
                "RGB image batch does not match its dimensions",
            ));
        }
        Ok(Self { pixels, geometry })
    }
}

/// CUDA image processor for its explicitly qualified RGB8 and patch-layout contract.
///
/// A processor is bound to the supplied device stream. Callers must account for input, temporary
/// and output allocations and retain its result until all consuming device work completes.
pub struct Processor {
    horizontal: Arc<Function>,
    vertical: Arc<Function>,
    stream: usize,
}

impl Processor {
    /// Load the implementation only for the centrally qualified CUDA target.
    /// # Errors
    /// Returns an error when a qualified device cannot load its embedded CUDA kernel.
    pub fn load(device: &CudaDevice) -> Result<Option<Self>> {
        if !device.target().supports_rgb8_preprocessing() {
            return Ok(None);
        }
        let module = device
            .stream
            .device()
            .load_module_from_ptx_src(PTX)
            .map_err(device_error)?;
        Ok(Some(Self {
            horizontal: Arc::new(
                module
                    .load_function("mm_horizontal_batch")
                    .map_err(device_error)?,
            ),
            vertical: Arc::new(
                module
                    .load_function("mm_vertical_patch_batch")
                    .map_err(device_error)?,
            ),
            stream: device.stream.cu_stream() as usize,
        }))
    }

    /// Produce padded patch vectors on the device from equal-sized RGB8 images.
    ///
    /// Every source, table, intermediate, output and kernel module is retained by the device
    /// operation until its completion fence. The future resolves without a host readback.
    /// The output has shape `[count × padded_patches, 1536]`; padded rows are zero.
    /// The input owner is retained through the device fence. The caller must keep the result
    /// alive until all later device consumers finish and invoke this on the device-owner thread.
    /// # Errors
    /// Returns an error for a different stream, allocation failure, or failed CUDA operation.
    pub async fn process(
        &self,
        device: &CudaDevice,
        batch: &ImageBatch,
    ) -> Result<Arc<Tensor<f32>>> {
        let geometry = batch.geometry;
        if self.stream != device.stream.cu_stream() as usize {
            return Err(Error::invalid(
                "image processor belongs to a different CUDA stream",
            ));
        }
        let tables = Tables::new(&geometry)?;
        let horizontal_elements = geometry.horizontal_elements;
        let output_elements = geometry.output_elements;
        let rows = usize::try_from(geometry.batch)
            .ok()
            .and_then(|count| count.checked_mul(usize::try_from(geometry.padded_patches).ok()?))
            .ok_or_else(|| Error::invalid("image patch rows overflow"))?;
        let horizontal = Arc::clone(&self.horizontal);
        let vertical = Arc::clone(&self.vertical);
        let stream = self.stream;
        let operations = (
            api::copy_host_vec_to_device(&batch.pixels),
            api::copy_host_vec_to_device(&tables.values),
            api::zeros::<u8>(&[horizontal_elements]),
            api::zeros::<f32>(&[output_elements]),
        )
            .zip();
        let operation =
            operations.then(move |(input, coefficients, intermediate, output)| Kernels {
                input,
                coefficients,
                intermediate,
                output,
                geometry,
                tables,
                horizontal,
                vertical,
                stream,
            });
        device
            .run_op(operation)
            .await?
            .reshape(&[rows, PATCH_WIDTH])
            .map(Arc::new)
            .map_err(device_error)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/vision_image_preprocess.rs"]
mod tests;
