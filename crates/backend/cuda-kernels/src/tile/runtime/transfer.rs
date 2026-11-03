//! Stream-ordered work outside a graph, which keeps its operands alive as a graph does.
//!
//! cuda-async frees memory on its own deallocator stream, so dropping the last owner of a buffer
//! right after enqueueing a copy over it would free the memory under the copy. A batch of
//! transfers therefore retains every owner it was given and records one event; the device drops
//! the owners once that event has passed, which a later batch or a fence checks without waiting.
#![expect(
    unsafe_code,
    reason = "Driver copies between contract buffers whose extents are checked here and whose owners the batch retains until its event passes"
)]
use crate::contract::buffer::{Buffer, Owner, Width};
use crate::tile::backend_error;
use cuda_core::sys::{
    CUDA_MEMCPY3D, CUmemorytype_enum_CU_MEMORYTYPE_DEVICE, CUresult, cuMemcpy3DAsync_v2,
    cuMemcpyDtoDAsync_v2, cudaError_enum_CUDA_SUCCESS,
};
use cuda_core::{Event, Stream};
use infer_core::{Error, Result};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::fmt::Display;
use std::mem::ManuallyDrop;
use std::ptr::{null, null_mut};
use std::sync::{Arc, Mutex};

/// Extents of a head-major layout: heads, positions, width.
const HEAD_MAJOR_RANK: usize = 3;
/// Completed events kept for reuse, enough for the batches one decode step enqueues.
const SPARE_EVENTS: usize = 8;

/// A head-major `[heads, capacity, dim]` buffer, as a KV cache lays out its positions.
#[derive(Debug)]
pub struct HeadMajor<'a, T> {
    buffer: Buffer<'a, T>,
    heads: usize,
    capacity: usize,
    dim: usize,
}

impl<T> Clone for HeadMajor<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for HeadMajor<'_, T> {}

impl<'a, T> HeadMajor<'a, T> {
    /// # Errors
    /// Rejects a buffer that is not exactly `heads * capacity * dim` elements.
    pub fn new(
        buffer: Buffer<'a, T>,
        [heads, capacity, dim]: [usize; HEAD_MAJOR_RANK],
    ) -> Result<Self> {
        let len = heads
            .checked_mul(capacity)
            .and_then(|positions| positions.checked_mul(dim));
        if len != Some(buffer.len()) {
            return Err(Error::invalid("head-major buffer extent"));
        }
        Ok(Self {
            buffer,
            heads,
            capacity,
            dim,
        })
    }
}

/// Transfers enqueued on one stream as one batch; see the module documentation.
pub struct Transfers<'a> {
    stream: &'a Stream,
    owners: RefCell<Vec<Owner>>,
}

impl<'a> Transfers<'a> {
    pub(super) const fn new(stream: &'a Stream) -> Self {
        Self {
            stream,
            owners: RefCell::new(Vec::new()),
        }
    }

    pub(super) fn into_owners(self) -> Vec<Owner> {
        self.owners.into_inner()
    }

    /// Copy `src` into `dst` after the work already on the stream. Copying a buffer onto itself
    /// does nothing.
    /// # Errors
    /// Rejects buffers of different lengths or partly overlapping memory, and a copy the driver
    /// does not enqueue.
    pub fn copy<T: Width>(&self, dst: Buffer<'_, T>, src: Buffer<'_, T>) -> Result<()> {
        if dst.len() != src.len() {
            return Err(Error::invalid("device copy lengths differ"));
        }
        let bytes = bytes::<T>(src.len())?;
        if dst.address() == src.address() || bytes == 0 {
            return Ok(());
        }
        if overlaps((dst.address(), bytes), (src.address(), bytes)) {
            return Err(Error::invalid("device copy ranges overlap"));
        }
        self.retain([dst.owner(), src.owner()]);
        // SAFETY: both buffers address `bytes` bytes of device memory, disjoint as checked above;
        // their owners stay retained until the batch event passes, and unowned buffers carry the
        // same obligation from `Buffer::new`.
        status(unsafe {
            cuMemcpyDtoDAsync_v2(dst.address(), src.address(), bytes, self.stream.cu_stream())
        })
    }

    /// Copy the first `rows` positions of every head; the capacities may differ, and the
    /// positions past `rows` are not touched.
    /// # Errors
    /// Rejects layouts with different heads or widths, `rows` beyond either capacity, a buffer
    /// partly overlapping the other, and a copy the driver does not enqueue.
    pub fn copy_heads<T: Width>(
        &self,
        dst: HeadMajor<'_, T>,
        src: HeadMajor<'_, T>,
        rows: usize,
    ) -> Result<()> {
        if dst.heads != src.heads
            || dst.dim != src.dim
            || rows > dst.capacity
            || rows > src.capacity
        {
            return Err(Error::invalid("head prefix copy layout or range"));
        }
        if rows == 0 || src.heads == 0 || src.dim == 0 {
            return Ok(());
        }
        if dst.buffer.address() == src.buffer.address() {
            return if dst.capacity == src.capacity {
                Ok(())
            } else {
                Err(Error::invalid("aliased head copy layout"))
            };
        }
        if rows == src.capacity && dst.capacity == src.capacity {
            return self.copy(dst.buffer, src.buffer);
        }
        let dst_bytes = bytes::<T>(dst.buffer.len())?;
        let src_bytes = bytes::<T>(src.buffer.len())?;
        if overlaps(
            (dst.buffer.address(), dst_bytes),
            (src.buffer.address(), src_bytes),
        ) {
            return Err(Error::invalid("head copy ranges overlap"));
        }
        let pitch = bytes::<T>(src.dim)?;
        let copy = CUDA_MEMCPY3D {
            srcXInBytes: 0,
            srcY: 0,
            srcZ: 0,
            srcLOD: 0,
            srcMemoryType: CUmemorytype_enum_CU_MEMORYTYPE_DEVICE,
            srcHost: null(),
            srcDevice: src.buffer.address(),
            srcArray: null_mut(),
            reserved0: null_mut(),
            srcPitch: pitch,
            srcHeight: src.capacity,
            dstXInBytes: 0,
            dstY: 0,
            dstZ: 0,
            dstLOD: 0,
            dstMemoryType: CUmemorytype_enum_CU_MEMORYTYPE_DEVICE,
            dstHost: null_mut(),
            dstDevice: dst.buffer.address(),
            dstArray: null_mut(),
            reserved1: null_mut(),
            dstPitch: pitch,
            dstHeight: dst.capacity,
            WidthInBytes: pitch,
            Height: rows,
            Depth: src.heads,
        };
        self.retain([dst.buffer.owner(), src.buffer.owner()]);
        // SAFETY: `HeadMajor::new` bounds every [head, row, column] the descriptor reaches to its
        // buffer, the buffers are disjoint as checked above, and unused host, array and reserved
        // fields are NULL. Owners stay retained until the batch event passes.
        status(unsafe { cuMemcpy3DAsync_v2(&raw const copy, self.stream.cu_stream()) })
    }

    fn retain(&self, owners: [Option<&Owner>; 2]) {
        let mut retained = self.owners.borrow_mut();
        for owner in owners.into_iter().flatten() {
            // Loops copy many ranges of one buffer; one reference keeps it alive.
            if !retained.last().is_some_and(|last| Arc::ptr_eq(last, owner)) {
                retained.push(Arc::clone(owner));
            }
        }
    }
}

/// Batches the stream has not yet been seen to pass, oldest first.
#[derive(Default)]
pub(super) struct Inflight {
    batches: VecDeque<Batch>,
    spare: Vec<Event>,
}

struct Batch {
    event: Event,
    owners: Vec<Owner>,
}

impl Inflight {
    /// Retain `owners` until the stream passes the work enqueued so far.
    pub(super) fn retain(&mut self, stream: &Arc<Stream>, owners: Vec<Owner>) -> Result<()> {
        let event = match self.spare.pop() {
            Some(event) => event,
            None => match stream.device().new_event() {
                Ok(event) => event,
                Err(error) => return Err(settle(stream, owners, error)),
            },
        };
        if let Err(error) = event.record(stream) {
            return Err(settle(stream, owners, error));
        }
        self.batches.push_back(Batch { event, owners });
        Ok(())
    }

    /// Drop the owners of every batch the stream has passed. Events on one stream complete in
    /// order, so this stops at the first one still pending.
    pub(super) fn reap(&mut self) -> Result<()> {
        while let Some(batch) = self.batches.front() {
            if !batch.event.query().map_err(backend_error)? {
                break;
            }
            let Some(batch) = self.batches.pop_front() else {
                break;
            };
            if self.spare.len() < SPARE_EVENTS {
                self.spare.push(batch.event);
            }
        }
        Ok(())
    }
}

impl Drop for Inflight {
    fn drop(&mut self) {
        for batch in self.batches.drain(..) {
            // Memory the device may still be using leaks rather than being freed under it.
            if batch.event.synchronize().is_err() {
                leak(batch.owners);
            }
        }
    }
}

/// Wait for the stream before dropping `owners` after a failure left their batch untracked, or
/// leak them when even that fails.
fn settle(stream: &Stream, owners: Vec<Owner>, error: impl Display) -> Error {
    // SAFETY: the stream is live and owned by its device; this only waits for its work.
    if unsafe { stream.synchronize() }.is_err() {
        leak(owners);
    }
    backend_error(error)
}

/// Keep memory the device may still be using allocated for good rather than free it under the
/// device; only a failed device reaches this.
fn leak(owners: Vec<Owner>) {
    let _leaked = ManuallyDrop::new(owners);
}

fn bytes<T: Width>(elements: usize) -> Result<usize> {
    elements
        .checked_mul(T::BYTES)
        .ok_or_else(|| Error::invalid("device copy size overflow"))
}

/// Whether two `(address, bytes)` ranges share a byte.
fn overlaps((left, left_bytes): (u64, usize), (right, right_bytes): (u64, usize)) -> bool {
    let end = |address: u64, bytes: usize| {
        u64::try_from(bytes)
            .ok()
            .and_then(|bytes| address.checked_add(bytes))
            .unwrap_or(u64::MAX)
    };
    left < end(right, right_bytes) && right < end(left, left_bytes)
}

fn status(status: CUresult) -> Result<()> {
    if status == cudaError_enum_CUDA_SUCCESS {
        Ok(())
    } else {
        Err(backend_error(format!(
            "device copy: CUDA driver status {status}"
        )))
    }
}

/// Run `work` as one batch on `stream` and retain the owners it gave until the stream passes it.
pub(super) fn enqueue<R>(
    inflight: &Mutex<Inflight>,
    stream: &Arc<Stream>,
    work: impl FnOnce(&Transfers<'_>) -> Result<R>,
) -> Result<R> {
    let transfers = Transfers::new(stream);
    let result = work(&transfers);
    let owners = transfers.into_owners();
    if owners.is_empty() {
        return result;
    }
    let tracked = match inflight.lock() {
        Ok(mut inflight) => inflight
            .reap()
            .and_then(|()| inflight.retain(stream, owners)),
        Err(_) => Err(settle(stream, owners, "transfer queue poisoned")),
    };
    tracked.and(result)
}

/// Drop the owners of every batch `stream` has passed.
pub(super) fn reap(inflight: &Mutex<Inflight>) -> Result<()> {
    inflight
        .lock()
        .map_err(|_| Error::invariant("transfer queue poisoned"))?
        .reap()
}
