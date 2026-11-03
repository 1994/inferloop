//! A device-to-host copy into pinned memory, as a launch that owns both ends until it completes.
#![expect(
    unsafe_code,
    reason = "Asynchronous D2H between ranges checked here; the launch retains the source allocation and the pinned destination until it completes"
)]
use super::Allocation;
use cuda_core::sys::{cuMemcpyDtoHAsync_v2, cudaError_enum_CUDA_SUCCESS};
use cuda_core::{DType, DeviceCopy, PinnedHostBuffer};
use cutile::cuda_async::device_context::with_default_device_policy;
use cutile::cuda_async::device_future::DeviceFuture;
use cutile::cuda_async::device_operation::ExecutionContext;
use cutile::prelude::*;
use cutile::tensor::KernelInputStored;
use infer_core::{Error, Result};
use std::ops::Range;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex};

/// Pinned host memory a readback writes; the host reads it only after the readback completes.
pub type Pinned<T> = Arc<Mutex<PinnedHostBuffer<T>>>;

/// Copy `range` of `source` into `destination` from element `at`, after the work before it on
/// the stream it runs on.
pub struct Readback<T: DType + DeviceCopy> {
    source: Allocation<T>,
    range: Range<usize>,
    destination: Pinned<T>,
    at: usize,
}

impl<T: DType + DeviceCopy> Readback<T> {
    /// # Errors
    /// Rejects a range outside `source`.
    pub fn new(
        source: Allocation<T>,
        range: Range<usize>,
        destination: Pinned<T>,
        at: usize,
    ) -> Result<Self> {
        if range.start > range.end || range.end > source.len() {
            return Err(Error::invalid("pinned readback range out of bounds"));
        }
        Ok(Self {
            source,
            range,
            destination,
            at,
        })
    }
}

impl<T: DType + DeviceCopy> DeviceOp for Readback<T> {
    type Output = ();

    unsafe fn execute(self, context: &ExecutionContext) -> StdResult<(), DeviceError> {
        self.source.tensor.retain(context)?;
        context.retain(Arc::clone(&self.destination))?;
        let len = self.range.len();
        let mut destination = self
            .destination
            .lock()
            .map_err(|error| DeviceError::Internal(error.to_string()))?;
        if self
            .at
            .checked_add(len)
            .is_none_or(|end| end > destination.len())
        {
            return Err(DeviceError::Internal("pinned copy size mismatch".into()));
        }
        if len == 0 {
            return Ok(());
        }
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| DeviceError::Internal("pinned copy size overflow".into()))?;
        let offset = self
            .range
            .start
            .checked_mul(size_of::<T>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| DeviceError::Internal("pinned copy offset overflow".into()))?;
        // SAFETY: `new` bounds the source range by the allocation and the check above bounds the
        // destination range by the pinned buffer. The context retains both until the submission
        // completes, and the host reads the destination only after that.
        let status = unsafe {
            cuMemcpyDtoHAsync_v2(
                destination.as_mut_ptr().wrapping_add(self.at).cast(),
                self.source.tensor.device_pointer().cu_deviceptr() + offset,
                bytes,
                context.get_cuda_stream().cu_stream(),
            )
        };
        if status == cudaError_enum_CUDA_SUCCESS {
            Ok(())
        } else {
            Err(DeviceError::Internal(format!(
                "pinned D2H CUDA status {status}"
            )))
        }
    }
}

impl<T: DType + DeviceCopy> IntoFuture for Readback<T> {
    type Output = StdResult<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), Self>;

    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => Self::IntoFuture::scheduled(self, ExecutionContext::new(stream)),
            Ok(Err(error)) | Err(error) => Self::IntoFuture::failed(error),
        }
    }
}
