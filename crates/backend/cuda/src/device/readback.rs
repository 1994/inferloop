//! Owned pinned destinations; host slices are exposed only after submission completion.
use super::{CudaDevice, device_error};
use crate::constants::MIB;
use cuda_core::sys::cuMemcpyDtoHAsync_v2;
use cuda_core::sys::cudaError_enum_CUDA_SUCCESS;
use cuda_core::{CudaContext, DType, DeviceCopy, PinnedHostBuffer};
use cutile::cuda_async::device_context::with_default_device_policy;
use cutile::cuda_async::device_future::DeviceFuture;
use cutile::{prelude::*, tensor::KernelInputStored};
use infer_core::{Error, ErrorCode, Result};
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex};

/// Total pinned bytes admitted across one batch of readback copies.
const MAX_READBACK_STAGING_BYTES: usize = 64 * MIB;

#[cfg(test)]
#[path = "../../tests/unit/device_readback_check.rs"]
mod check;

mod flight;

type Buffer<T> = Arc<Mutex<PinnedHostBuffer<T>>>;
type Rows<T> = Vec<Vec<T>>;

/// A half-open element range `[skip, skip + len)` of a device tensor to copy out.
#[derive(Clone)]
pub struct ReadbackSource<T: DType + DeviceCopy = f32> {
    pub tensor: Arc<Tensor<T>>,
    pub skip: usize,
    pub len: usize,
}

impl<T: DType + DeviceCopy> ReadbackSource<T> {
    #[must_use]
    pub fn whole(tensor: Arc<Tensor<T>>) -> Self {
        let len = tensor.size();
        Self {
            tensor,
            skip: 0,
            len,
        }
    }
}

#[derive(Default)]
pub struct Readbacks<T: DType + DeviceCopy = f32> {
    context: Option<Arc<CudaContext>>,
    buffers: Vec<Option<Buffer<T>>>,
    poisoned: bool,
    flight: Option<flight::Flight<T>>,
    owner: Arc<()>,
}

impl<T: DType + DeviceCopy> Readbacks<T> {
    pub(crate) fn run(
        &mut self,
        device: &CudaDevice,
        graph: &CudaGraph<()>,
        sources: &[Option<ReadbackSource<T>>],
    ) -> Result<Rows<T>> {
        if self.poisoned || !Arc::ptr_eq(graph.stream(), &device.stream) {
            return Err(Error::invalid(
                "poisoned readback or mismatched CUDA graph stream",
            ));
        }
        self.prepare(device, sources)?;
        let copies = self.copies(sources)?;
        self.poisoned = true;
        device
            .stream
            .device()
            .bind_to_thread()
            .map_err(device_error)?;
        graph
            .launch()
            .then(move |()| copies)
            .sync_on(&device.stream)
            .map_err(device_error)?;
        // sync_on succeeded: every queued D2H finished. No GPU work can touch
        // these private buffers until another exclusive &mut self call.
        let results = self.results(sources)?;
        self.poisoned = false;
        Ok(results)
    }

    fn copies(&self, sources: &[Option<ReadbackSource<T>>]) -> Result<DeviceOpVec<()>> {
        let mut copies = DeviceOpVec::with_capacity(sources.len());
        for (source, buffer) in sources.iter().zip(&self.buffers) {
            if let Some(source) = source {
                copies.push(PinnedCopy {
                    source: Arc::clone(&source.tensor),
                    skip: source.skip,
                    len: source.len,
                    destination: Arc::clone(
                        buffer
                            .as_ref()
                            .ok_or_else(|| Error::invariant("pinned slot"))?,
                    ),
                });
            }
        }
        Ok(copies)
    }

    fn results(&self, sources: &[Option<ReadbackSource<T>>]) -> Result<Rows<T>> {
        sources
            .iter()
            .zip(&self.buffers)
            .map(|(source, buffer)| {
                if source.is_none() {
                    return Ok(Vec::new());
                }
                let buffer = buffer
                    .as_ref()
                    .ok_or_else(|| Error::invariant("pinned result"))?;
                Ok(buffer.lock().map_err(device_error)?.as_slice().to_vec())
            })
            .collect()
    }

    fn staging_bytes(
        &self,
        device: &CudaDevice,
        sources: &[Option<ReadbackSource<T>>],
    ) -> Result<usize> {
        let mut bytes = 0usize;
        for (index, source) in sources.iter().enumerate() {
            if let Some(source) = source {
                if source.tensor.device_id() != device.stream.device().ordinal() {
                    return Err(Error::invalid("pinned readback device mismatch"));
                }
                if source
                    .skip
                    .checked_add(source.len)
                    .is_none_or(|end| end > source.tensor.size())
                {
                    return Err(Error::invalid("pinned readback range out of bounds"));
                }
                bytes = bytes
                    .checked_add(
                        source
                            .len
                            .checked_mul(size_of::<T>())
                            .ok_or_else(|| Error::invalid("pinned size overflow"))?,
                    )
                    .ok_or_else(|| Error::invalid("pinned size overflow"))?;
            } else if let Some(Some(buffer)) = self.buffers.get(index) {
                bytes = bytes
                    .checked_add(buffer.lock().map_err(device_error)?.num_bytes())
                    .ok_or_else(|| Error::invalid("pinned size overflow"))?;
            }
        }
        if bytes > MAX_READBACK_STAGING_BYTES {
            return Err(Error::new(
                ErrorCode::Capacity,
                "pinned readbacks exceed 64 MiB",
            ));
        }
        Ok(bytes)
    }

    fn prepare(
        &mut self,
        device: &CudaDevice,
        sources: &[Option<ReadbackSource<T>>],
    ) -> Result<usize> {
        let bytes = self.staging_bytes(device, sources)?;
        self.buffers.resize_with(sources.len(), || None);
        for (source, slot) in sources.iter().zip(&mut self.buffers) {
            let Some(source) = source else {
                continue;
            };
            if let Some(buffer) = slot
                && buffer.lock().map_err(device_error)?.len() == source.len
            {
                continue;
            }
            if self.context.is_none() {
                self.context =
                    Some(CudaContext::new(device.stream.device().ordinal()).map_err(device_error)?);
            }
            let context = self
                .context
                .as_ref()
                .ok_or_else(|| Error::invariant("pinned context"))?;
            *slot = Some(Arc::new(Mutex::new(
                PinnedHostBuffer::zeroed(context, source.len).map_err(device_error)?,
            )));
        }
        Ok(bytes)
    }
}

impl Readbacks<f32> {
    /// Execute the model and token selector on one stream, then expose both readouts
    /// after one completion fence. Either owner stays poisoned on submission failure.
    pub(crate) fn run_with_tokens(
        &mut self,
        device: &CudaDevice,
        graph: &CudaGraph<()>,
        sources: &[Option<ReadbackSource>],
        tail: (&CudaGraph<()>, &mut Readbacks<i32>, Arc<Tensor<i32>>),
    ) -> Result<(Rows<f32>, Rows<i32>)> {
        let (selector, tokens, tensor) = tail;
        if self.poisoned
            || tokens.poisoned
            || !Arc::ptr_eq(graph.stream(), &device.stream)
            || !Arc::ptr_eq(selector.stream(), &device.stream)
        {
            return Err(Error::invalid(
                "poisoned readback or mismatched CUDA graph stream",
            ));
        }
        let token_sources = [Some(ReadbackSource::whole(tensor))];
        let bytes = self
            .staging_bytes(device, sources)?
            .checked_add(tokens.staging_bytes(device, &token_sources)?)
            .ok_or_else(|| Error::invalid("pinned size overflow"))?;
        if bytes > MAX_READBACK_STAGING_BYTES {
            return Err(Error::new(
                ErrorCode::Capacity,
                "pinned readbacks exceed 64 MiB",
            ));
        }
        self.prepare(device, sources)?;
        tokens.prepare(device, &token_sources)?;
        let copies = self.copies(sources)?;
        let token_copies = tokens.copies(&token_sources)?;
        self.poisoned = true;
        tokens.poisoned = true;
        device
            .stream
            .device()
            .bind_to_thread()
            .map_err(device_error)?;
        graph
            .launch()
            .then(move |()| selector.launch())
            .then(move |()| copies)
            .then(move |_| token_copies)
            .sync_on(&device.stream)
            .map_err(device_error)?;
        let output = (self.results(sources)?, tokens.results(&token_sources)?);
        self.poisoned = false;
        tokens.poisoned = false;
        Ok(output)
    }
}

struct PinnedCopy<T: DType + DeviceCopy> {
    source: Arc<Tensor<T>>,
    skip: usize,
    len: usize,
    destination: Buffer<T>,
}

#[expect(
    unsafe_code,
    reason = "Audited asynchronous D2H: the [skip, skip + len) element range is bounds-checked \
              against the source before enqueue; source access lease and pinned destination are \
              retained by the live submission"
)]
impl<T: DType + DeviceCopy> DeviceOp for PinnedCopy<T> {
    type Output = ();

    unsafe fn execute(self, context: &ExecutionContext) -> StdResult<(), DeviceError> {
        self.source.retain(context)?;
        context.retain(Arc::clone(&self.destination))?;
        if self
            .skip
            .checked_add(self.len)
            .is_none_or(|end| end > self.source.size())
            || self.len.checked_mul(size_of::<T>()).is_none()
        {
            return Err(DeviceError::Internal(
                "pinned copy range out of bounds".into(),
            ));
        }
        let mut destination = self
            .destination
            .lock()
            .map_err(|e| DeviceError::Internal(e.to_string()))?;
        if destination.len() != self.len {
            return Err(DeviceError::Internal("pinned copy size mismatch".into()));
        }
        if self.len == 0 {
            return Ok(());
        }
        let bytes = self.len * size_of::<T>();
        let offset = self
            .skip
            .checked_mul(size_of::<T>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| DeviceError::Internal("pinned copy offset overflow".into()))?;
        // SAFETY: owned, initialized destination has exactly self.len elements; the
        // checked source range keeps the offset read inside the source tensor.
        // Both allocations outlive the submission. Host reads happen only after
        // sync_on; failures poison the owner and submission cleanup retains work.
        let status = unsafe {
            cuMemcpyDtoHAsync_v2(
                destination.as_mut_ptr().cast(),
                self.source.device_pointer().cu_deviceptr() + offset,
                bytes,
                context.get_cuda_stream().cu_stream(),
            )
        };
        if status != cudaError_enum_CUDA_SUCCESS {
            return Err(DeviceError::Internal(format!(
                "pinned D2H CUDA status {status}"
            )));
        }
        Ok(())
    }
}

impl<T: DType + DeviceCopy> IntoFuture for PinnedCopy<T> {
    type Output = StdResult<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), Self>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| policy.next_stream()) {
            Ok(Ok(stream)) => Self::IntoFuture::scheduled(self, ExecutionContext::new(stream)),
            Ok(Err(error)) | Err(error) => Self::IntoFuture::failed(error),
        }
    }
}

/// Queue a bounded host gate for device-completion tests.
#[cfg(test)]
#[expect(
    unsafe_code,
    reason = "Test-only stream callback: closure owns its data, is bounded and calls no CUDA APIs"
)]
fn queue_test_gate(device: &CudaDevice, gate: impl FnOnce() + Send + 'static) -> Result<()> {
    // SAFETY: callers supply a bounded, CUDA-free closure with owned state; the stream
    // and every captured resource remain alive through the test's completion drain.
    unsafe { device.stream.launch_host_function(gate) }.map_err(device_error)
}
