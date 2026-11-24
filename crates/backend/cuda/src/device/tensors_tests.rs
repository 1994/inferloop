//! cuTile tensor uploads and readbacks for tests that launch reference kernels directly.
use super::{CudaDevice, device_error};
use crate::resident::buffers::buffer;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_cuda_kernels::runtime::Element;
use infer_cuda_kernels::runtime::Stream;
use std::sync::Arc;

impl CudaDevice {
    /// The device's stream, for tests that drive cuTile operations on it directly.
    #[must_use]
    pub fn stream(&self) -> &Arc<Stream> {
        self.runtime.stream_handle()
    }

    /// A copy of `values` as a tensor of `shape`, complete when this returns.
    /// # Errors
    /// Returns a CUDA error if upload or reshape fails.
    pub fn upload<T: DType>(&self, values: Vec<T>, shape: &[usize]) -> Result<Arc<Tensor<T>>> {
        api::copy_host_vec_to_device(&Arc::new(values))
            .sync_on(self.stream())
            .map_err(device_error)?
            .reshape(shape)
            .map(Arc::new)
            .map_err(device_error)
    }

    /// # Errors
    /// Returns a CUDA transfer error.
    pub fn read<T: DType>(&self, tensor: &Arc<Tensor<T>>) -> Result<Vec<T>> {
        tensor
            .to_host_vec()
            .sync_on(self.stream())
            .map_err(device_error)
    }

    /// Write `values` over the start of `dst`, ordered before later work on the stream; the
    /// caller may drop `values` at once.
    /// # Errors
    /// Rejects a slice longer than the tensor, or a failed upload or copy.
    pub fn copy_host_slice<T: DType + Element>(
        &self,
        dst: &mut Tensor<T>,
        values: &[T],
    ) -> Result<()> {
        if values.len() > dst.size() {
            return Err(Error::invalid("host copy range outside tensor"));
        }
        let source = self.runtime.upload(values.to_vec())?;
        let target = buffer(&*dst)
            .map_err(device_error)?
            .slice(0..values.len())?;
        self.runtime
            .enqueue(|transfers| transfers.copy(target, source.operand()))
    }

    /// Copy a tensor back without taking ownership, so a borrowed snapshot can be inspected.
    /// # Errors
    /// Returns a backend error when the device is unbound or the copy fails.
    pub fn read_borrowed<T: DType + Element>(&self, tensor: &Tensor<T>) -> Result<Vec<T>> {
        self.reclaim_barrier()?;
        self.runtime
            .read_buffer(buffer(tensor).map_err(device_error)?)
    }
}
