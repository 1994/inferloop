//! Pinned host memory that stages uploads, refilled only after the stream has read it.
//!
//! An upload from pinned memory is asynchronous: the device reads the host bytes some time after
//! the copy is enqueued. Each slot records an event after its copy and waits on it before being
//! refilled, so a caller may enqueue uploads without fencing at all; a caller that fences anyway
//! pays one completion query per upload.
#![expect(
    unsafe_code,
    reason = "Host-to-device copies out of pinned slots that are refilled only after the event recorded behind their last copy has passed"
)]
use super::Element;
use super::transfer::{Transfers, status};
use crate::contract::buffer::Buffer;
use crate::tile::backend_error;
use cuda_core::sys::cuMemcpyHtoDAsync_v2;
use cuda_core::{CudaContext, DeviceCopy, Event, PinnedHostBuffer, Stream};
use infer_core::{Error, Result};
use std::mem::{ManuallyDrop, size_of};
use std::sync::Arc;

/// A ring of pinned host slots, each holding up to `elements` values.
pub struct Staging<T: DeviceCopy> {
    slots: Vec<Slot<T>>,
    next: usize,
}

struct Slot<T: DeviceCopy> {
    pinned: PinnedHostBuffer<T>,
    /// Recorded after the slot's last copy; meaningless until `pending` is first set.
    read: Event,
    pending: bool,
}

impl<T: DeviceCopy> Staging<T> {
    pub(super) fn new(stream: &Arc<Stream>, elements: usize, depth: usize) -> Result<Self> {
        if depth == 0 {
            return Err(Error::invalid("staging needs at least one slot"));
        }
        let context = CudaContext::new(stream.device().ordinal()).map_err(backend_error)?;
        let slots = (0..depth)
            .map(|_| {
                Ok(Slot {
                    pinned: PinnedHostBuffer::zeroed(&context, elements).map_err(backend_error)?,
                    read: stream.device().new_event().map_err(backend_error)?,
                    pending: false,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self { slots, next: 0 })
    }
}

impl<T: DeviceCopy> Drop for Staging<T> {
    fn drop(&mut self) {
        for slot in self.slots.drain(..) {
            // Memory the device may still be reading leaks rather than being freed under it.
            if slot.pending && slot.read.synchronize().is_err() {
                let _leaked = ManuallyDrop::new(slot.pinned);
            }
        }
    }
}

impl Transfers<'_> {
    /// Copy `values` into `dst` through the next slot of `staging`, waiting first only while the
    /// stream has not yet read what that slot last staged.
    /// # Errors
    /// Rejects `values` longer than a slot or of another length than `dst`, and a copy the
    /// driver does not enqueue or track.
    pub fn upload<T: DeviceCopy + Element>(
        &self,
        staging: &mut Staging<T>,
        dst: Buffer<'_, T::Marker>,
        values: &[T],
    ) -> Result<()> {
        let depth = staging.slots.len();
        let slot = staging
            .slots
            .get_mut(staging.next)
            .ok_or_else(|| Error::invariant("staging cursor"))?;
        if values.len() > slot.pinned.len() || values.len() != dst.len() {
            return Err(Error::invalid("staged upload extent"));
        }
        if slot.pending && !slot.read.query().map_err(backend_error)? {
            slot.read.synchronize().map_err(backend_error)?;
        }
        slot.pending = false;
        let Some(staged) = slot.pinned.as_mut_slice().get_mut(..values.len()) else {
            return Err(Error::invariant("staging slot extent"));
        };
        staged.copy_from_slice(values);
        let bytes = values
            .len()
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::invalid("staged upload size overflow"))?;
        self.retain([dst.owner(), None]);
        // SAFETY: `dst` addresses `values.len()` elements of `T`, whose marker it is typed by; the
        // pinned slot holds them and is not refilled until the event recorded below has passed,
        // and the batch retains the owner of `dst`.
        status(unsafe {
            cuMemcpyHtoDAsync_v2(
                dst.address(),
                slot.pinned.as_ptr().cast(),
                bytes,
                self.stream().cu_stream(),
            )
        })?;
        if let Err(error) = slot.read.record(self.stream()) {
            // Untracked, the slot could be refilled while the copy still reads it.
            // SAFETY: the stream is live and owned by its device; this only waits for its work.
            unsafe { self.stream().synchronize() }.map_err(backend_error)?;
            return Err(backend_error(error));
        }
        slot.pending = true;
        staging.next = (staging.next + 1) % depth;
        Ok(())
    }
}
