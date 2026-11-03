//! Persistent pinned inputs, staged through rings that never refill a slot the device still reads.
use super::{CudaDevice, device_error};
use crate::constants::MAX_VERIFICATION_WIDTH;
use crate::resident::buffers::buffer;
use cuda_core::{DType, DeviceCopy};
use cutile::tensor::Tensor;
use infer_core::Result;
use infer_cuda_kernels::tile::runtime::{Element, Staging};

pub struct PinnedUpload<T: DeviceCopy> {
    staging: Staging<T>,
}
impl<T: DType + DeviceCopy + Element> PinnedUpload<T> {
    /// One upload in flight; a second waits for the first to be read.
    pub(crate) fn single(device: &CudaDevice, elements: usize) -> Result<Self> {
        Self::new(device, elements, 1)
    }
    /// Metadata for one draft chain, including its target-conditioned catch-up.
    pub(crate) fn chain(device: &CudaDevice, elements: usize) -> Result<Self> {
        Self::new(device, elements, MAX_VERIFICATION_WIDTH + 1)
    }
    /// At most one block-table assignment for each physical slot per flight.
    pub(crate) fn slots(device: &CudaDevice, elements: usize, slots: usize) -> Result<Self> {
        Self::new(device, elements, slots)
    }
    fn new(device: &CudaDevice, elements: usize, depth: usize) -> Result<Self> {
        Ok(Self {
            staging: device.runtime().staging(elements, depth)?,
        })
    }
    /// Enqueue a copy of `values` into the start of `destination`.
    /// # Errors
    /// Rejects more values than the ring was prepared for or `destination` holds, and a copy the
    /// driver does not enqueue.
    #[expect(
        clippy::needless_pass_by_ref_mut,
        reason = "The device writes destination; the exclusive borrow keeps a shared view from aliasing it"
    )]
    pub(crate) fn copy(
        &mut self,
        device: &CudaDevice,
        destination: &mut Tensor<T>,
        values: &[T],
    ) -> Result<()> {
        let destination = buffer(&*destination)
            .map_err(device_error)?
            .slice(0..values.len())?;
        device.enqueue(|transfers| transfers.upload(&mut self.staging, destination, values))
    }
}
