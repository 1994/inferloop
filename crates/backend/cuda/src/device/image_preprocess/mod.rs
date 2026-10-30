//! CUDA image preprocessing plans and device execution.
mod axis;
mod capability;
mod geometry;
mod launch;
mod staging;
mod tables;

use self::geometry::{Geometry, PATCH_WIDTH};
use self::launch::Kernels;
use self::staging::UploadPool;
use self::tables::Tables;
use crate::device::{CudaDevice, device_error};
use crate::target::ImageKernel;
use cuda_core::{CudaContext, Function};
use cutile::prelude::*;
use infer_core::{Error, ErrorCode, Result};
use infer_ir::ImageTransform;
use infer_spi::ImagePreprocessing;
use std::sync::Arc;

pub use capability::Capability;

const PTX: &str =
    include_str!("../../../kernels/image-preprocess/preprocess-batch-strict-sm120.ptx");
const PORTABLE_PTX: &str =
    include_str!("../../../kernels/image-preprocess-portable/preprocess-batch-strict-sm70.ptx");

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
    pixels: UploadPool<u8>,
    coefficients: UploadPool<i32>,
}

impl Processor {
    /// Load the best available implementation selected from the CUDA compute capability.
    /// # Errors
    /// Returns an error when a qualified device cannot load its embedded CUDA kernel.
    pub fn load(device: &CudaDevice) -> Result<Option<Self>> {
        let driver = CudaDevice::driver_version()?;
        let ptx = match device.target().image_kernel(driver)? {
            Some(ImageKernel::Sm120) => PTX,
            Some(ImageKernel::Portable) => PORTABLE_PTX,
            None => return Ok(None),
        };
        Self::load_kernel(device, ptx).map(Some)
    }

    fn load_kernel(device: &CudaDevice, ptx: &str) -> Result<Self> {
        let module = device
            .stream
            .device()
            .load_module_from_ptx_src(ptx)
            .map_err(|error| {
                Error::new(
                    ErrorCode::Backend,
                    format!(
                        "failed to load CUDA image preprocessing on {:?}: {error}",
                        device.target()
                    ),
                )
            })?;
        let context = CudaContext::new(device.ordinal()).map_err(device_error)?;
        Ok(Self {
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
            pixels: UploadPool::new(Arc::clone(&context)),
            coefficients: UploadPool::new(context),
        })
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
        let rows = usize::try_from(geometry.batch)
            .ok()
            .and_then(|count| count.checked_mul(usize::try_from(geometry.padded_patches).ok()?))
            .ok_or_else(|| Error::invalid("image patch rows overflow"))?;
        let horizontal = Arc::clone(&self.horizontal);
        let vertical = Arc::clone(&self.vertical);
        let pixels = self.pixels.stage(&batch.pixels)?;
        let coefficients = self.coefficients.stage(&tables.values)?;
        let operation = Kernels {
            device: device.clone(),
            pixels,
            coefficients,
            geometry,
            tables,
            horizontal,
            vertical,
            stream: self.stream,
        };
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
