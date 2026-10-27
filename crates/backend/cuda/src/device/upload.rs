//! Persistent pinned inputs; the owning execution path bounds uploads between fences.
use super::{CudaDevice, device_error};
use crate::constants::MAX_VERIFICATION_WIDTH;
use cuda_core::{CudaContext, DType, DeviceCopy, PinnedHostBuffer};
use cutile::tensor::Tensor;
use infer_core::{Error, Result};

pub struct PinnedUpload<T: DType + DeviceCopy> {
    buffers: Vec<PinnedHostBuffer<T>>,
    next: usize,
}
impl<T: DType + DeviceCopy> PinnedUpload<T> {
    /// One upload followed by a device fence before this owner is used again.
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
        if depth == 0 {
            return Err(Error::invalid(
                "pinned upload requires a nonzero bounded depth",
            ));
        }
        let context = CudaContext::new(device.stream.device().ordinal()).map_err(device_error)?;
        let buffers = (0..depth)
            .map(|_| PinnedHostBuffer::zeroed(&context, elements).map_err(device_error))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { buffers, next: 0 })
    }
    /// Enqueue one copy. The owning graph/pool retains this ring until completion,
    /// and must fence before queuing more than its constructor's bounded depth.
    pub(crate) fn copy(
        &mut self,
        device: &CudaDevice,
        destination: &mut Tensor<T>,
        values: &[T],
    ) -> Result<()> {
        let buffer = &mut self.buffers[self.next];
        if values.len() > buffer.len() {
            return Err(Error::invalid("pinned upload exceeds prepared capacity"));
        }
        buffer.as_mut_slice()[..values.len()].copy_from_slice(values);
        device.copy_h2d_pinned(destination, buffer, values.len())?;
        self.next = (self.next + 1) % self.buffers.len();
        Ok(())
    }
}
