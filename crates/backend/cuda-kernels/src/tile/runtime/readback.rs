//! A device-to-host copy into pinned memory, as a launch that owns both ends until it completes.
#![expect(
    unsafe_code,
    reason = "Asynchronous D2H between ranges checked here; the launch retains the source allocation and the pinned destination until it completes"
)]
use super::{Allocation, Device, Graph};
use crate::tile::backend_error;
use cuda_core::sys::{cuMemcpyDtoHAsync_v2, cudaError_enum_CUDA_SUCCESS};
use cuda_core::{DType, DeviceCopy, PinnedHostBuffer};
use cutile::cuda_async::device_context::with_default_device_policy;
use cutile::cuda_async::device_future::DeviceFuture;
use cutile::cuda_async::device_operation::ExecutionContext;
use cutile::cuda_async::device_operation::{BoxedDeviceOp, DeviceOpVec};
use cutile::prelude::*;
use cutile::tensor::KernelInputStored;
use infer_core::{Error, Result};
use std::future::Future;
use std::ops::Range;
use std::pin::Pin;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

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

/// Pinned readbacks submitted together, in order, as one stream operation.
pub struct Copies(DeviceOpVec<()>);

impl Copies {
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self(DeviceOpVec::with_capacity(capacity))
    }

    /// Append `copy`, which runs after every copy pushed before it.
    pub fn push<T: DType + DeviceCopy>(&mut self, copy: Readback<T>) {
        self.0.push(copy);
    }
}

/// Work submitted by [`Device::read_back`] or a replay variant; it resolves once the device has
/// finished it, and it retains every source and pinned destination until then.
///
/// The first poll enqueues the work, so a caller polls once without waiting to submit it.
pub struct Flight(DeviceFuture<Vec<()>, BoxedDeviceOp<Vec<()>>>);

impl Future for Flight {
    type Output = Result<()>;

    #[inline]
    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Result<()>> {
        Pin::new(&mut self.0)
            .poll(context)
            .map(|result| result.map(drop).map_err(backend_error))
    }
}

impl Device {
    /// Copy `copies` back once the work already on this stream is done.
    #[must_use]
    pub fn read_back(&self, copies: Copies) -> Flight {
        self.flight(copies.0.boxed())
    }

    /// Replay `graph`, then copy `copies` back, in one submission.
    #[must_use]
    pub fn replay_and_read_back(&self, graph: &Graph, copies: Copies) -> Flight {
        self.flight(graph.launch().then(move |()| copies.0).boxed())
    }

    /// Replay `graph` and then `selector`, then copy `copies` and `tokens` back, in one
    /// submission: a step's hidden rows and the tokens chosen from them.
    #[must_use]
    pub fn replay_pair_and_read_back(
        &self,
        (graph, selector): (&Graph, &Graph),
        copies: Copies,
        tokens: Copies,
    ) -> Flight {
        let selector = selector.launch();
        self.flight(
            graph
                .launch()
                .then(move |()| selector)
                .then(move |()| copies.0)
                .then(move |_| tokens.0)
                .boxed(),
        )
    }

    fn flight(&self, operation: BoxedDeviceOp<Vec<()>>) -> Flight {
        Flight(DeviceFuture::scheduled(
            operation,
            ExecutionContext::new(Arc::clone(self.stream())),
        ))
    }
}
