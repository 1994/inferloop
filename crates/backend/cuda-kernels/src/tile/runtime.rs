//! The host's view of memory, capture and execution, implemented over cuTile.
//!
//! These are concrete types chosen at compile time, not trait objects: replaying a graph is the
//! inference hot path, and here it compiles to exactly the cuTile future it wraps.
use crate::contract::buffer::{Bf16, Buffer, Fp4Pair, Fp8, Owner, Width};
use crate::contract::recording::{Foreign, Recording};
use crate::tile::backend_error;
use cuda_core::sys::{CUdeviceptr, cuMemsetD8Async, cudaError_enum_CUDA_SUCCESS};
use cuda_core::{DType, DTypeId, DeviceCopy, Stream, f4e2m1fnx2, f8e4m3fn};
use cutile::cuda_async::cuda_graph::GraphLaunch;
use cutile::cuda_async::device_buffer::DeviceAllocation;
use cutile::cuda_async::device_context::with_default_device_policy;
use cutile::cuda_async::device_future::DeviceFuture;
use cutile::cuda_async::device_operation::{ExecutionContext, value};
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::KernelOutputStored;
use infer_core::{Error, ErrorCode, Result};
use std::cell::RefCell;
use std::ops::Range;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex};
use transfer::Inflight;

mod launch;
mod readback;
mod staging;
mod transfer;

pub use launch::{Argument, KernelLaunch};
pub use readback::{Pinned, Readback};
pub use staging::Staging;
pub use transfer::{HeadMajor, Transfers};

/// The contract's marker for a cuTile element type.
pub trait Element {
    type Marker: Width;
}
impl Element for f32 {
    type Marker = Self;
}
impl Element for i32 {
    type Marker = Self;
}
impl Element for u32 {
    type Marker = Self;
}
impl Element for bf16 {
    type Marker = Bf16;
}
impl Element for f4e2m1fnx2 {
    type Marker = Fp4Pair;
}
impl Element for f8e4m3fn {
    type Marker = Fp8;
}

/// Device memory the host owns. A graph that recorded it keeps it alive.
///
/// A clone shares the memory, as an `Arc` does; nothing is copied.
pub struct Allocation<T: DType> {
    tensor: Arc<Tensor<T>>,
    owner: Owner,
}

impl<T: DType> Clone for Allocation<T> {
    fn clone(&self) -> Self {
        Self {
            tensor: Arc::clone(&self.tensor),
            owner: Arc::clone(&self.owner),
        }
    }
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

    /// Ordinal of the device holding the memory.
    #[must_use]
    pub fn device(&self) -> usize {
        self.tensor.device_id()
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

    /// Elements `range` as an allocation of their own, holding a share of this memory, so it stays
    /// valid whatever happens to the allocation it came from: one row of a batched owner that a
    /// launch takes alone.
    /// # Errors
    /// Rejects a range outside the allocation.
    pub fn slice(&self, range: Range<usize>) -> Result<Self> {
        let len = range.end.saturating_sub(range.start);
        Self::adopt(Arc::new(self.view(range, &[len])?))
    }

    /// The allocation as `count` equal, disjoint rows, each an allocation of its own as
    /// [`Self::slice`] makes, so each can be a launch's output.
    /// # Errors
    /// Rejects zero rows and memory that does not divide into `count` rows.
    pub fn rows(&self, count: usize) -> Result<Vec<Self>> {
        if count == 0 || !self.len().is_multiple_of(count) {
            return Err(Error::invalid(format!(
                "{} elements do not divide into {count} rows",
                self.len()
            )));
        }
        let row = self.len() / count;
        (0..count)
            .map(|index| self.slice(index * row..(index + 1) * row))
            .collect()
    }

    /// All of this memory read as `U`s, as [`Self::view_as`] reads it, as an allocation of its own
    /// holding a share of the memory: an unused F32 product doubles as FP8 byte scratch.
    /// # Errors
    /// Rejects memory whose bytes do not divide into `U`s, and what [`Self::view_as`] rejects.
    pub fn reinterpret<U: DType>(&self) -> Result<Allocation<U>> {
        let bytes = self.len() * size_of::<T>();
        if !bytes.is_multiple_of(size_of::<U>()) {
            return Err(Error::invalid(format!(
                "{bytes} bytes do not divide into elements of {} bytes",
                size_of::<U>()
            )));
        }
        let view = self.view_as::<U>(0..self.len(), &[bytes / size_of::<U>()])?;
        Allocation::adopt(Arc::new(view))
    }

    /// Elements `range` as a contiguous tensor of `shape`, holding a share of this memory, so the
    /// view stays valid whatever happens to the allocation it came from. Transitional, for
    /// recordings that still take cuTile tensors.
    /// # Errors
    /// Rejects a range outside the allocation, and a shape that does not cover the range.
    pub fn view(&self, range: Range<usize>, shape: &[usize]) -> Result<Tensor<T>> {
        self.view_as(range, shape)
    }

    /// Elements `range` read as `U`s of `shape`, as [`Self::view`]: an unused F32 product doubles
    /// as the byte scratch of an FP8 operand. Every bit pattern is a valid number, so only `bool`,
    /// whose device stores are 0 or 1, is refused.
    /// # Errors
    /// Rejects a range outside the allocation, a shape whose bytes differ from the range's, a
    /// start misaligned for `U`, and `bool`.
    #[expect(
        unsafe_code,
        reason = "The view owns the memory it addresses, and its range, size and alignment are checked against it"
    )]
    pub fn view_as<U: DType>(&self, range: Range<usize>, shape: &[usize]) -> Result<Tensor<U>> {
        if U::DTYPE == DTypeId::Bool {
            return Err(Error::invalid("a bool view of untyped memory"));
        }
        let len = range
            .end
            .checked_sub(range.start)
            .filter(|_| range.end <= self.len())
            .ok_or_else(|| {
                Error::invalid(format!(
                    "view {range:?} of an allocation of {} elements",
                    self.len()
                ))
            })?;
        let width = size_of::<T>();
        let bytes = len * width;
        if shape
            .iter()
            .try_fold(size_of::<U>(), |all, &dim| all.checked_mul(dim))
            != Some(bytes)
        {
            return Err(Error::invalid(format!(
                "view shape {shape:?} does not cover {bytes} bytes"
            )));
        }
        let offset = range
            .start
            .checked_mul(width)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| Error::invalid("view offset overflows"))?;
        let pointer = self.tensor.device_pointer().cu_deviceptr() + offset;
        if !pointer.is_multiple_of(align_of::<U>() as u64) {
            return Err(Error::invalid("view start is misaligned"));
        }
        let dims = shape
            .iter()
            .map(|&dim| i32::try_from(dim).map_err(|_| Error::invalid("view dimension")))
            .collect::<Result<Vec<_>>>()?;
        let mut strides = vec![1i32; dims.len()];
        for axis in (0..dims.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1]
                .checked_mul(dims[axis + 1])
                .ok_or_else(|| Error::invalid("view strides"))?;
        }
        let memory = View {
            pointer,
            bytes,
            memory: self.clone(),
        };
        // SAFETY: `memory` owns the allocation, so it outlives the tensor, and its bytes lie
        // inside the allocation, as checked above. The strides are contiguous over `shape`, the
        // start is aligned for `U`, and any bytes are a valid `U` (`DType`), `bool` excepted.
        Ok(unsafe { Tensor::from_foreign(Arc::new(memory), dims, strides) })
    }
}

/// A range of an allocation as cuTile sees it; holding the allocation keeps the range valid.
struct View<T: DType> {
    memory: Allocation<T>,
    pointer: CUdeviceptr,
    bytes: usize,
}

#[expect(
    unsafe_code,
    reason = "`memory` keeps the allocation alive and `Allocation::view` checks the range"
)]
// SAFETY: the getters are constant, and `pointer`/`bytes` lie inside `memory`, which this value
// keeps alive, as `Allocation::view` checked when it made them.
unsafe impl<T: DType> DeviceAllocation for View<T> {
    fn device_ptr(&self) -> CUdeviceptr {
        self.pointer
    }

    fn len_bytes(&self) -> usize {
        self.bytes
    }

    fn device_id(&self) -> usize {
        self.memory.device()
    }
}

impl<T: DType + Element> Allocation<T> {
    /// The whole allocation as the contract types it, retained by any recording graph.
    #[expect(
        unsafe_code,
        reason = "As `buffer`: each `Element` marker names exactly its element's size and layout"
    )]
    #[must_use]
    pub fn operand(&self) -> Buffer<'_, T::Marker> {
        let address = self.tensor.device_pointer().cu_deviceptr();
        // SAFETY: as for `buffer`; the marker is declared for this element type above.
        unsafe { Buffer::owned(address, self.len(), &self.owner) }
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

/// One execution stream on one device. Clones share the stream and its pending transfers.
#[derive(Clone)]
pub struct Device {
    stream: Arc<Stream>,
    ordinal: usize,
    inflight: Arc<Mutex<Inflight>>,
}

impl Device {
    #[must_use]
    pub fn new(stream: Arc<Stream>, ordinal: usize) -> Self {
        Self {
            stream,
            ordinal,
            inflight: Arc::default(),
        }
    }

    #[must_use]
    pub const fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// The stream every launch and transfer of this device is ordered on.
    pub(crate) const fn stream(&self) -> &Arc<Stream> {
        &self.stream
    }

    /// `len` zeroed elements, complete when this returns.
    /// # Errors
    /// Reports a failed allocation.
    pub fn zeros<T: DType>(&self, len: usize) -> Result<Allocation<T>> {
        let tensor = api::zeros::<T>(&[len])
            .sync_on(&self.stream)
            .map_err(backend_error)?;
        Allocation::adopt(Arc::new(tensor))
    }

    /// A copy of `values`, complete when this returns.
    /// # Errors
    /// Reports a failed allocation or transfer.
    pub fn upload<T: DType>(&self, values: Vec<T>) -> Result<Allocation<T>> {
        let tensor = api::copy_host_vec_to_device(&Arc::new(values))
            .sync_on(&self.stream)
            .map_err(backend_error)?;
        Allocation::adopt(Arc::new(tensor))
    }

    /// The contents of `allocation`, after the work already on this stream.
    /// # Errors
    /// Reports a failed transfer.
    pub fn read<T: DType>(&self, allocation: &Allocation<T>) -> Result<Vec<T>> {
        Arc::clone(&allocation.tensor)
            .to_host_vec()
            .sync_on(&self.stream)
            .map_err(backend_error)
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

    /// Enqueue `work` on this stream without waiting. Every owned buffer it transfers stays alive
    /// until the stream has passed the batch, as a recorded buffer does for its graph.
    /// # Errors
    /// Returns what `work` returns, or a failure to track the batch.
    pub fn enqueue<R>(&self, work: impl FnOnce(&Transfers<'_>) -> Result<R>) -> Result<R> {
        transfer::enqueue(&self.inflight, &self.stream, work)
    }

    /// Pinned staging for uploads of up to `elements` values, `depth` slots deep.
    /// # Errors
    /// Reports a failed pinned allocation or event creation.
    pub fn staging<T: DeviceCopy>(&self, elements: usize, depth: usize) -> Result<Staging<T>> {
        Staging::new(&self.stream, elements, depth)
    }

    /// Release what finished transfers retained, without waiting; a fence or synchronization of
    /// this stream beforehand makes that everything enqueued until then.
    /// # Errors
    /// Returns a device failure the completion query reports.
    pub fn reap(&self) -> Result<()> {
        transfer::reap(&self.inflight)
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
        self.run(Launch(value(()))).await?;
        self.reap()
    }
}

/// A captured graph, replayable any number of times.
pub struct Graph {
    inner: CudaGraph<()>,
    /// Kept for the memory the graph addresses; replay never touches them.
    _owners: Vec<Owner>,
}

impl Graph {
    /// Capture a graph whose recording still uses cuTile directly. Transitional, until recordings
    /// move behind the kernel families and [`Device::capture`].
    /// # Errors
    /// Returns what `record` returns, or a capture failure.
    pub fn scope(
        stream: &Arc<Stream>,
        record: impl FnOnce(&Scope) -> StdResult<(), DeviceError>,
    ) -> StdResult<Self, DeviceError> {
        Ok(Self {
            inner: CudaGraph::scope(stream, record)?,
            _owners: Vec::new(),
        })
    }

    #[inline]
    #[must_use]
    pub fn launch(&self) -> Launch<GraphLaunch> {
        Launch(self.inner.launch())
    }

    /// Apply an update ordered before this graph's next launch on its stream. Transitional, until
    /// updates move behind the kernel families.
    /// # Errors
    /// Returns what the capture's stream reports.
    pub fn update<N>(&self, op: N) -> StdResult<(), DeviceError>
    where
        N: GraphNode + DeviceOp<Output = ()>,
    {
        self.inner.update(op)
    }

    /// The stream this graph was captured on. Transitional, like [`Self::update`].
    #[must_use]
    pub fn stream(&self) -> &Arc<Stream> {
        self.inner.stream()
    }
}

/// Work ready to run. A newtype over the cuTile operation, so it costs nothing beyond it.
pub struct Launch<O>(O);

/// Lets code that has not migrated keep combining launches; every method forwards.
#[expect(
    unsafe_code,
    reason = "Forwards to the wrapped operation, with its obligations"
)]
impl<O: DeviceOp> DeviceOp for Launch<O> {
    type Output = <O as DeviceOp>::Output;

    #[inline]
    unsafe fn execute(
        self,
        context: &ExecutionContext,
    ) -> StdResult<<O as DeviceOp>::Output, DeviceError> {
        // SAFETY: the caller upholds `DeviceOp::execute` for the wrapped operation.
        unsafe { self.0.execute(context) }
    }
}

impl<O: DeviceOp> IntoFuture for Launch<O> {
    type Output = StdResult<<O as DeviceOp>::Output, DeviceError>;
    type IntoFuture = DeviceFuture<<O as DeviceOp>::Output, Self>;

    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => DeviceFuture::scheduled(self, ExecutionContext::new(stream)),
            Ok(Err(error)) | Err(error) => DeviceFuture::failed(error),
        }
    }
}

impl<O: DeviceOp<Output = ()>> Launch<O> {
    /// Run `next` after this launch, in one submission and without returning to the host.
    #[inline]
    #[must_use]
    pub fn followed_by<N, F>(self, next: F) -> Launch<impl DeviceOp<Output = ()>>
    where
        N: DeviceOp<Output = ()>,
        F: FnOnce() -> Launch<N> + Send,
    {
        Launch(self.0.then(move |()| next().0))
    }
}

/// Discard only the host return value of a graph-safe operation. Resource leases
/// remain owned by its execution context until the enclosing replay completes.
pub(in crate::tile) struct GraphUpdate<N>(pub N);

#[expect(
    unsafe_code,
    reason = "Audited forwarding of the caller's live execution context to a graph-safe operation"
)]
impl<N: GraphNode> DeviceOp for GraphUpdate<N> {
    type Output = ();

    unsafe fn execute(self, context: &ExecutionContext) -> StdResult<(), DeviceError> {
        // SAFETY: preserve the caller's context and submission lifetime unchanged.
        unsafe { self.0.execute(context) }.map(|_| ())
    }
}

impl<N: GraphNode> GraphNode for GraphUpdate<N> {}

impl<N: GraphNode> IntoFuture for GraphUpdate<N> {
    type Output = StdResult<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), Self>;

    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => Self::IntoFuture::scheduled(self, ExecutionContext::new(stream)),
            Ok(Err(error)) | Err(error) => Self::IntoFuture::failed(error),
        }
    }
}

/// Elements a clear retains its write lease over at a time.
const CLEAR_LEASE_TILE: usize = 256;

/// Zero every byte of a tensor with a driver memset, as one graph node.
pub(in crate::tile) struct Clear<'a>(pub &'a mut Tensor<f32>);

#[expect(
    unsafe_code,
    reason = "The live tensor bounds the memset extent; its write lease is retained by the capture context and replay submission"
)]
impl DeviceOp for Clear<'_> {
    type Output = ();

    unsafe fn execute(self, context: &ExecutionContext) -> StdResult<(), DeviceError> {
        let pointer = self.0.device_pointer().cu_deviceptr();
        let bytes = self.0.num_bytes();
        KernelOutputStored::retain(&self.0.partition([CLEAR_LEASE_TILE]), context)?;
        // SAFETY: the tensor addresses exactly this byte range on the captured stream's device.
        let status =
            unsafe { cuMemsetD8Async(pointer, 0, bytes, context.get_cuda_stream().cu_stream()) };
        if status == cudaError_enum_CUDA_SUCCESS {
            Ok(())
        } else {
            Err(DeviceError::Launch(format!("state clear: {status:?}")))
        }
    }
}

impl GraphNode for Clear<'_> {}

impl IntoFuture for Clear<'_> {
    type Output = StdResult<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), Self>;

    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => Self::IntoFuture::scheduled(self, ExecutionContext::new(stream)),
            Ok(Err(error)) | Err(error) => Self::IntoFuture::failed(error),
        }
    }
}

/// A [`Foreign`] call as one graph node, holding what the graph must keep alive.
struct ForeignNode<F> {
    call: F,
    keep: Vec<Owner>,
}

#[expect(
    unsafe_code,
    reason = "The node hands the call the stream its execution context records on; the call owns its FFI"
)]
impl<F: Foreign> DeviceOp for ForeignNode<F> {
    type Output = ();

    unsafe fn execute(self, context: &ExecutionContext) -> StdResult<(), DeviceError> {
        if !self.keep.is_empty() {
            context.retain(self.keep)?;
        }
        self.call
            .enqueue(context.get_cuda_stream().cu_stream().cast())
            .map_err(|error| DeviceError::Internal(error.to_string()))
    }
}

impl<F: Foreign> GraphNode for ForeignNode<F> {}

impl<F: Foreign> IntoFuture for ForeignNode<F> {
    type Output = StdResult<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), Self>;

    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => Self::IntoFuture::scheduled(self, ExecutionContext::new(stream)),
            Ok(Err(error)) | Err(error) => Self::IntoFuture::failed(error),
        }
    }
}

/// Record `call` into the graph `scope` captures.
pub(crate) fn record_foreign(scope: &Scope, call: impl Foreign, keep: Vec<Owner>) -> Result<()> {
    scope
        .record(ForeignNode { call, keep })
        .map_err(backend_error)
}
