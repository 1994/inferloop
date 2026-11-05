//! Owned pinned destinations; host slices are exposed only after submission completion.
use super::{CudaDevice, device_error};
use crate::constants::MIB;
use cuda_core::{CudaContext, DType, DeviceCopy, PinnedHostBuffer};
use cutile::prelude::*;
use infer_core::{Error, ErrorCode, Result};
#[cfg(test)]
use infer_cuda_kernels::tile::runtime::Graph;
use infer_cuda_kernels::tile::runtime::{Allocation, Pinned, Readback};
use std::sync::{Arc, Mutex};

/// Total pinned bytes admitted across one batch of readback copies.
const MAX_READBACK_STAGING_BYTES: usize = 64 * MIB;

#[cfg(test)]
#[path = "../../tests/unit/device_readback_check.rs"]
mod check;

mod flight;

type Buffer<T> = Pinned<T>;
type Rows<T> = Vec<Vec<T>>;

/// A half-open element range `[skip, skip + len)` of device memory to copy out.
#[derive(Clone)]
pub struct ReadbackSource<T: DType + DeviceCopy = f32> {
    pub source: Allocation<T>,
    pub skip: usize,
    pub len: usize,
}

impl<T: DType + DeviceCopy> ReadbackSource<T> {
    #[must_use]
    pub fn whole(source: Allocation<T>) -> Self {
        let len = source.len();
        Self {
            source,
            skip: 0,
            len,
        }
    }

    /// `[skip, skip + len)` of `source`.
    #[must_use]
    pub fn slice(source: &Allocation<T>, skip: usize, len: usize) -> Self {
        Self {
            source: source.clone(),
            skip,
            len,
        }
    }

    /// All of `source`.
    #[must_use]
    pub fn all(source: &Allocation<T>) -> Self {
        Self::whole(source.clone())
    }
}

pub struct Readbacks<T: DType + DeviceCopy = f32> {
    context: Option<Arc<CudaContext>>,
    buffers: Vec<Option<Buffer<T>>>,
    arena: Option<Buffer<T>>,
    poisoned: bool,
    flight: Option<flight::Flight<T>>,
    owner: Arc<()>,
}

impl<T: DType + DeviceCopy> Default for Readbacks<T> {
    fn default() -> Self {
        Self {
            context: None,
            buffers: Vec::new(),
            arena: None,
            poisoned: false,
            flight: None,
            owner: Arc::new(()),
        }
    }
}

impl<T: DType + DeviceCopy> Readbacks<T> {
    pub(crate) fn allocated(device: &CudaDevice, len: usize) -> Result<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::invalid("pinned size overflow"))?;
        if bytes > MAX_READBACK_STAGING_BYTES {
            return Err(Error::new(
                ErrorCode::Capacity,
                "pinned readbacks exceed 64 MiB",
            ));
        }
        let context = CudaContext::new(device.stream.device().ordinal()).map_err(device_error)?;
        let buffer = PinnedHostBuffer::zeroed(&context, len).map_err(device_error)?;
        Ok(Self {
            context: Some(context),
            buffers: vec![Some(Arc::new(Mutex::new(buffer)))],
            ..Self::default()
        })
    }

    #[cfg(test)]
    pub(crate) fn run(
        &mut self,
        device: &CudaDevice,
        graph: &Graph,
        sources: &[Option<ReadbackSource<T>>],
    ) -> Result<Rows<T>> {
        super::asynchronous::block_on(self.run_async(device, graph, sources))
    }

    fn copies(&self, sources: &[Option<ReadbackSource<T>>]) -> Result<DeviceOpVec<()>> {
        let mut copies = DeviceOpVec::with_capacity(sources.len());
        let mut offset = 0;
        for (index, source) in sources.iter().enumerate() {
            if let Some(source) = source {
                let (buffer, destination_skip) = self.destination(index, offset)?;
                let end = source
                    .skip
                    .checked_add(source.len)
                    .ok_or_else(|| Error::invalid("pinned readback range out of bounds"))?;
                copies.push(Readback::new(
                    source.source.clone(),
                    source.skip..end,
                    Arc::clone(buffer),
                    destination_skip,
                )?);
                offset += source.len;
            }
        }
        Ok(copies)
    }

    fn results(&self, sources: &[Option<ReadbackSource<T>>]) -> Result<Rows<T>> {
        let mut offset = 0;
        sources
            .iter()
            .enumerate()
            .map(|(index, source)| {
                let Some(source) = source else {
                    return Ok(Vec::new());
                };
                let (buffer, start) = self.destination(index, offset)?;
                let result = buffer.lock().map_err(device_error)?.as_slice()
                    [start..start + source.len]
                    .to_vec();
                offset += source.len;
                Ok(result)
            })
            .collect()
    }

    fn destination(&self, index: usize, offset: usize) -> Result<(&Buffer<T>, usize)> {
        if let Some(arena) = &self.arena {
            return Ok((arena, offset));
        }
        self.buffers
            .get(index)
            .and_then(Option::as_ref)
            .map(|buffer| (buffer, 0))
            .ok_or_else(|| Error::invariant("pinned slot"))
    }

    fn staging_bytes(
        &self,
        device: &CudaDevice,
        sources: &[Option<ReadbackSource<T>>],
    ) -> Result<usize> {
        if let Some(arena) = &self.arena {
            let capacity = arena.lock().map_err(device_error)?.num_bytes();
            let requested = sources.iter().flatten().try_fold(0usize, |sum, source| {
                sum.checked_add(source_bytes(device, source)?)
                    .ok_or_else(|| Error::invalid("pinned size overflow"))
            })?;
            if requested > capacity {
                return Err(Error::new(
                    ErrorCode::Capacity,
                    "pinned readbacks exceed prepared arena",
                ));
            }
            return Ok(capacity);
        }
        let mut bytes = 0usize;
        for index in 0..sources.len().max(self.buffers.len()) {
            let retained = self
                .buffers
                .get(index)
                .and_then(Option::as_ref)
                .map(|buffer| {
                    buffer
                        .lock()
                        .map(|buffer| buffer.num_bytes())
                        .map_err(device_error)
                })
                .transpose()?
                .unwrap_or(0);
            let requested = if let Some(Some(source)) = sources.get(index) {
                source_bytes(device, source)?
            } else {
                0
            };
            bytes = bytes
                .checked_add(retained.max(requested))
                .ok_or_else(|| Error::invalid("pinned size overflow"))?;
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
        if self.arena.is_some() {
            return Ok(bytes);
        }
        if self.buffers.len() < sources.len() {
            self.buffers.resize_with(sources.len(), || None);
        }
        for (source, slot) in sources.iter().zip(&mut self.buffers) {
            let Some(source) = source else {
                continue;
            };
            if let Some(buffer) = slot
                && buffer.lock().map_err(device_error)?.len() >= source.len
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
    #[cfg(test)]
    pub(crate) fn run_with_tokens(
        &mut self,
        device: &CudaDevice,
        graph: &Graph,
        sources: &[Option<ReadbackSource>],
        tail: (&Graph, &mut Readbacks<i32>, Allocation<i32>),
    ) -> Result<(Rows<f32>, Rows<i32>)> {
        super::asynchronous::block_on(self.run_with_tokens_async(device, graph, sources, tail))
    }
}

/// Queue a bounded host gate for device-completion tests.
#[cfg(test)]
#[expect(
    unsafe_code,
    reason = "Test-only stream callback: closure owns its data, is bounded and calls no CUDA APIs"
)]
pub fn queue_test_gate(device: &CudaDevice, gate: impl FnOnce() + Send + 'static) -> Result<()> {
    // SAFETY: callers supply a bounded, CUDA-free closure with owned state; the stream
    // and every captured resource remain alive through the test's completion drain.
    unsafe { device.stream.launch_host_function(gate) }.map_err(device_error)
}

fn source_bytes<T: DType + DeviceCopy>(
    device: &CudaDevice,
    source: &ReadbackSource<T>,
) -> Result<usize> {
    if source.source.device() != device.stream.device().ordinal() {
        return Err(Error::invalid("pinned readback device mismatch"));
    }
    if source
        .skip
        .checked_add(source.len)
        .is_none_or(|end| end > source.source.len())
    {
        return Err(Error::invalid("pinned readback range out of bounds"));
    }
    source
        .len
        .checked_mul(size_of::<T>())
        .ok_or_else(|| Error::invalid("pinned size overflow"))
}

impl Readbacks {
    fn prepared_rows(device: &CudaDevice, sources: &[Option<ReadbackSource>]) -> Result<Self> {
        let mut owner = Self::default();
        let token_bytes = sources
            .len()
            .checked_mul(size_of::<i32>())
            .ok_or_else(|| Error::invalid("token staging overflow"))?;
        match owner.staging_bytes(device, sources) {
            Ok(bytes) if bytes <= MAX_READBACK_STAGING_BYTES.saturating_sub(token_bytes) => {}
            Ok(_) => return Self::bounded_arena(device, token_bytes),
            Err(error) if error.code == ErrorCode::Capacity => {
                return Self::bounded_arena(device, token_bytes);
            }
            Err(error) => return Err(error),
        }
        owner.prepare(device, sources)?;
        Ok(owner)
    }

    fn bounded_arena(device: &CudaDevice, token_bytes: usize) -> Result<Self> {
        // Large verify graphs cannot retain every possible logit row within the bound.
        // A compact arena admits any active subset without allocating on lane changes.
        let bytes = MAX_READBACK_STAGING_BYTES
            .checked_sub(token_bytes)
            .ok_or_else(|| Error::invalid("token staging exceeds readback budget"))?;
        let mut owner = Self::allocated(device, bytes / size_of::<f32>())?;
        owner.arena = Some(
            owner
                .buffers
                .pop()
                .flatten()
                .ok_or_else(|| Error::invariant("prepared pinned arena"))?,
        );
        Ok(owner)
    }

    pub(crate) fn pair(
        device: &CudaDevice,
        hidden: &Allocation<f32>,
        logits: &Allocation<f32>,
    ) -> Result<Self> {
        Self::prepared_rows(
            device,
            &[
                Some(ReadbackSource::all(hidden)),
                Some(ReadbackSource::all(logits)),
            ],
        )
    }
    pub(crate) fn pairs(
        device: &CudaDevice,
        hidden: &[Allocation<f32>],
        logits: &[Allocation<f32>],
    ) -> Result<Self> {
        let mut sources = Vec::with_capacity(2 * hidden.len());
        for (hidden, logits) in hidden.iter().zip(logits) {
            sources.push(Some(ReadbackSource::all(hidden)));
            sources.push(Some(ReadbackSource::all(logits)));
        }
        Self::prepared_rows(device, &sources)
    }
    pub(crate) fn contiguous(
        device: &CudaDevice,
        hidden: &Allocation<f32>,
        logits: &Allocation<f32>,
        rows: usize,
    ) -> Result<Self> {
        let mut sources = Vec::with_capacity(2 * rows);
        for row in 0..rows {
            let (hidden_len, logits_len) = (hidden.len() / rows, logits.len() / rows);
            sources.push(Some(ReadbackSource::slice(
                hidden,
                row * hidden_len,
                hidden_len,
            )));
            sources.push(Some(ReadbackSource::slice(
                logits,
                row * logits_len,
                logits_len,
            )));
        }
        Self::prepared_rows(device, &sources)
    }
}
