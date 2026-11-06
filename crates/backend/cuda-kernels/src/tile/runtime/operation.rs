//! Stream-ordered work that makes its own memory and resolves once the device has run it.
//!
//! [`Device::operation`] runs a closure that allocates, copies and launches on the device's stream
//! without waiting. Everything the closure retains, and every allocation it makes, stays alive
//! until the device has passed its work; the future resolves only then, so polling it never
//! blocks the host on the stream.
#![expect(
    unsafe_code,
    reason = "Stream-ordered allocation and pinned uploads whose memory the submission retains until the device has passed it"
)]
use super::launch::KernelLaunch;
use super::transfer::status;
use super::{Allocation, Device};
use crate::tile::backend_error;
use cuda_core::sys::cuMemcpyHtoDAsync_v2;
use cuda_core::{DType, DeviceCopy, PinnedHostBuffer};
use cutile::cuda_async::device_context::with_default_device_policy;
use cutile::cuda_async::device_future::DeviceFuture;
use cutile::cuda_async::device_operation::ExecutionContext;
use cutile::prelude::*;
use cutile::tensor::KernelOutputStored;
use infer_core::{Error, Result};
use std::ffi::c_void;
use std::future::Future;
use std::pin::Pin;
use std::result::Result as StdResult;
use std::sync::Arc;
use std::task::{Context, Poll};

/// Work being enqueued by [`Device::operation`].
pub struct Operation<'a> {
    context: &'a ExecutionContext,
}

impl Operation<'_> {
    /// `len` uninitialized elements, alive until the device has passed this operation and for as
    /// long as the returned allocation is held after that.
    ///
    /// # Safety
    /// Work this operation enqueues must write every element before anything, on the device or the
    /// host, reads it.
    /// # Errors
    /// Rejects a size cuTile cannot describe, or reports a failed allocation.
    pub unsafe fn overwritten<T: DType>(&self, len: usize) -> Result<Allocation<T>> {
        let count =
            i32::try_from(len).map_err(|_| Error::invalid("allocation exceeds tensor extents"))?;
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::invalid("allocation size overflow"))?;
        // SAFETY: the context owns a live stream; the memory is ordered on it.
        let pointer = unsafe { self.context.alloc_async(bytes) }.map_err(backend_error)?;
        // SAFETY: the pointer is the unique allocation of `bytes` just made on this context's
        // device, and the 1-D shape and stride describe exactly those bytes. The caller writes it
        // before any read.
        let mut tensor = unsafe {
            Tensor::<T>::from_raw_parts(
                pointer,
                bytes,
                self.context.get_device_id(),
                vec![count],
                vec![1],
            )
        };
        KernelOutputStored::retain(&(&mut tensor).partition([1]), self.context)
            .map_err(backend_error)?;
        let allocation = Allocation::adopt(Arc::new(tensor))?;
        self.retain(allocation.clone())?;
        Ok(allocation)
    }

    /// Keep `owner` alive until the device has passed this operation.
    /// # Errors
    /// Reports a submission that can no longer retain.
    pub fn retain(&self, owner: impl Send + 'static) -> Result<()> {
        self.context.retain(owner).map_err(backend_error)
    }

    /// Enqueue `launch`, keeping its module loaded until the device has run it.
    /// # Errors
    /// Rejects a launch made for another stream, or returns the driver's refusal.
    pub fn launch(&self, launch: KernelLaunch) -> Result<()> {
        self.retain(launch.owner())?;
        launch.enqueue_on(self.stream())
    }

    /// Copy the first `len` values of pinned `source` into `destination`, which stays alive until
    /// the copy has run.
    ///
    /// # Safety
    /// The host must neither write nor free `source` until this operation completes; retaining its
    /// owner with [`Self::retain`] keeps it allocated.
    /// # Errors
    /// Rejects a range outside either buffer, or a copy the driver does not enqueue.
    pub unsafe fn upload<T: DType + DeviceCopy>(
        &self,
        destination: &Allocation<T>,
        source: &PinnedHostBuffer<T>,
        len: usize,
    ) -> Result<()> {
        if len > destination.len() || len > source.len() {
            return Err(Error::invalid("pinned upload range outside buffer"));
        }
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::invalid("pinned upload size overflow"))?;
        self.retain(destination.clone())?;
        // SAFETY: both ranges hold `bytes` bytes as checked above; the destination is retained
        // until the stream passes the copy, and the caller keeps the source unchanged until then.
        status(unsafe {
            cuMemcpyHtoDAsync_v2(
                destination.buffer().address(),
                source.as_ptr().cast(),
                bytes,
                self.context.get_cuda_stream().cu_stream(),
            )
        })
    }

    /// The driver stream, for libraries that enqueue on it themselves.
    #[must_use]
    pub fn stream(&self) -> *mut c_void {
        self.context.get_cuda_stream().cu_stream().cast()
    }
}

/// The closure of [`Device::operation`] as the cuTile operation it submits. Its own result is the
/// output, so a refusal keeps its error code and still waits for what was enqueued before it.
struct Work<F>(F);

impl<R, F> DeviceOp for Work<F>
where
    R: Send,
    F: FnOnce(&Operation<'_>) -> Result<R> + Send,
{
    type Output = Result<R>;

    unsafe fn execute(self, context: &ExecutionContext) -> StdResult<Result<R>, DeviceError> {
        Ok((self.0)(&Operation { context }))
    }
}

impl<R, F> IntoFuture for Work<F>
where
    R: Send,
    F: FnOnce(&Operation<'_>) -> Result<R> + Send,
{
    type Output = StdResult<Result<R>, DeviceError>;
    type IntoFuture = DeviceFuture<Result<R>, Self>;

    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => DeviceFuture::scheduled(self, ExecutionContext::new(stream)),
            Ok(Err(error)) | Err(error) => DeviceFuture::failed(error),
        }
    }
}

/// The result of an operation, once the device has passed it; see [`Device::operation`].
pub struct Pending<R, F>(DeviceFuture<Result<R>, Work<F>>)
where
    R: Send,
    F: FnOnce(&Operation<'_>) -> Result<R> + Send;

impl<R, F> Future for Pending<R, F>
where
    R: Send,
    F: FnOnce(&Operation<'_>) -> Result<R> + Send,
{
    type Output = Result<R>;

    #[inline]
    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Result<R>> {
        Pin::new(&mut self.0)
            .poll(context)
            .map(|result| match result {
                Ok(output) => output,
                Err(error) => Err(backend_error(error)),
            })
    }
}

impl Device {
    /// Enqueue `work` on this stream when the future is first polled. The future resolves to what
    /// `work` returned once the device has passed everything it enqueued, and only then releases
    /// what it retained.
    #[inline]
    #[must_use]
    pub fn operation<R, F>(&self, work: F) -> Pending<R, F>
    where
        R: Send,
        F: FnOnce(&Operation<'_>) -> Result<R> + Send,
    {
        Pending(DeviceFuture::scheduled(
            Work(work),
            ExecutionContext::new(Arc::clone(self.stream())),
        ))
    }
}
