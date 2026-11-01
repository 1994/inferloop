//! The host's view of memory, capture and execution, implemented over cuTile.
//!
//! These are concrete types chosen at compile time, not trait objects: replaying a graph is the
//! inference hot path, and here it compiles to exactly the cuTile future it wraps.
use crate::contract::buffer::{Bf16, Buffer, Owner};
use crate::contract::recording::Recording;
use crate::tile::backend_error;
use cuda_core::{DType, Stream};
use cutile::cuda_async::cuda_graph::GraphLaunch;
use cutile::cuda_async::device_future::DeviceFuture;
use cutile::cuda_async::device_operation::{ExecutionContext, value};
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::{Error, ErrorCode, Result};
use std::cell::RefCell;
use std::sync::Arc;

/// Device memory the host owns. A graph that recorded it keeps it alive.
pub struct Allocation<T: DType> {
    tensor: Arc<Tensor<T>>,
    owner: Owner,
}

impl<T: DType> Allocation<T> {
    /// Adopt a tensor the host already holds, while it migrates away from cuTile types.
    /// # Errors
    /// Rejects storage that is not contiguous, which a buffer cannot describe.
    pub fn adopt(tensor: Arc<Tensor<T>>) -> Result<Self> {
        if !tensor.is_contiguous() {
            return Err(Error::invalid("allocation must be contiguous"));
        }
        let owner: Owner = tensor.clone();
        Ok(Self { tensor, owner })
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.tensor.size()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The whole allocation as a kernel operand.
    #[expect(
        unsafe_code,
        reason = "A contiguous allocation this value keeps alive, retained by any recording graph"
    )]
    #[must_use]
    pub fn buffer(&self) -> Buffer<'_, T> {
        let address = self.tensor.device_pointer().cu_deviceptr();
        // SAFETY: `adopt` checked contiguity, and a recording retains `owner` in its graph, so the
        // memory outlives every replay. Aliasing between operands is the caller's to avoid.
        unsafe { Buffer::owned(address, self.len(), &self.owner) }
    }

    /// The tensor behind this allocation, for code that has not migrated yet.
    #[must_use]
    pub const fn tensor(&self) -> &Arc<Tensor<T>> {
        &self.tensor
    }
}

impl Allocation<bf16> {
    /// The whole allocation as a BF16 kernel operand.
    #[expect(
        unsafe_code,
        reason = "As `buffer`: bf16 and the contract's Bf16 marker are the same two bytes"
    )]
    #[must_use]
    pub fn bf16_buffer(&self) -> Buffer<'_, Bf16> {
        let address = self.tensor.device_pointer().cu_deviceptr();
        // SAFETY: as for `buffer`.
        unsafe { Buffer::owned(address, self.len(), &self.owner) }
    }
}

/// One execution stream on one device.
#[derive(Clone)]
pub struct Device {
    stream: Arc<Stream>,
    ordinal: usize,
}

impl Device {
    #[must_use]
    pub const fn new(stream: Arc<Stream>, ordinal: usize) -> Self {
        Self { stream, ordinal }
    }

    #[must_use]
    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Record kernels into a graph. Nothing runs until a launch of it does.
    /// # Errors
    /// Returns what `record` returns, or a capture failure.
    pub fn capture(&self, record: impl FnOnce(&Recording<'_>) -> Result<()>) -> Result<Graph> {
        let owners = RefCell::new(Vec::new());
        let inner = CudaGraph::scope(&self.stream, |scope| {
            record(&Recording::retaining(scope, &owners))
                .map_err(|error| DeviceError::Internal(error.to_string()))
        })
        .map_err(backend_error)?;
        Ok(Graph {
            inner,
            _owners: owners.into_inner(),
        })
    }

    /// Run a launch on this stream; it completes when the device has.
    /// # Errors
    /// Returns device failures.
    #[inline]
    pub async fn run<O: DeviceOp<Output = ()>>(&self, launch: Launch<O>) -> Result<()> {
        DeviceFuture::scheduled(launch.0, ExecutionContext::new(Arc::clone(&self.stream)))
            .await
            .map_err(|error| Error::new(ErrorCode::Backend, error.to_string()))
    }

    /// Wait for everything queued on this stream.
    /// # Errors
    /// Returns device failures.
    pub async fn fence(&self) -> Result<()> {
        self.run(Launch(value(()))).await
    }
}

/// A captured graph, replayable any number of times.
pub struct Graph {
    inner: CudaGraph<()>,
    /// Kept for the memory the graph addresses; replay never touches them.
    _owners: Vec<Owner>,
}

impl Graph {
    #[inline]
    #[must_use]
    pub fn launch(&self) -> Launch<GraphLaunch> {
        Launch(self.inner.launch())
    }
}

/// Work ready to run. A newtype over the cuTile operation, so it costs nothing beyond it.
pub struct Launch<O>(O);

impl<O: DeviceOp<Output = ()>> Launch<O> {
    /// Run `next` after this launch, in one submission and without returning to the host.
    #[inline]
    #[must_use]
    pub fn then<N, F>(self, next: F) -> Launch<impl DeviceOp<Output = ()>>
    where
        N: DeviceOp<Output = ()>,
        F: FnOnce() -> Launch<N> + Send,
    {
        Launch(self.0.then(move |()| next().0))
    }
}
