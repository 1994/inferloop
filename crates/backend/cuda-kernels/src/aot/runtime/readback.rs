//! A device-to-host copy into pinned memory, run after the work already on the stream.
#![expect(
    unsafe_code,
    reason = "D2H copies between ranges checked here, into pinned memory locked until the stream has passed them"
)]
use super::{Allocation, DType, Device, DeviceCopy, Graph, PinnedHostBuffer};
use infer_core::{Error, Result};
use std::future::Future;
use std::mem::size_of;
use std::ops::Range;
use std::pin::Pin;
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

/// One readback with its element type erased, so copies of several types run in one batch.
trait Copyback: Send {
    /// Copy on `device` and wait, while the destination is locked.
    fn run(&self, device: &Device) -> Result<()>;
}

impl<T: DType + DeviceCopy> Copyback for Readback<T> {
    fn run(&self, device: &Device) -> Result<()> {
        let len = self.range.len();
        let mut destination = self
            .destination
            .lock()
            .map_err(|_| Error::invariant("pinned readback destination poisoned"))?;
        if self
            .at
            .checked_add(len)
            .is_none_or(|end| end > destination.len())
        {
            return Err(Error::invalid("pinned copy size mismatch"));
        }
        if len == 0 {
            return Ok(());
        }
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::invalid("pinned copy size overflow"))?;
        let source = self.source.slice(self.range.clone())?;
        // SAFETY: `new` bounds the source range by the allocation and the check above bounds the
        // destination range by the pinned buffer. Both stay alive and the destination stays locked
        // until the stream has passed the copy below.
        unsafe {
            device.driver.enqueue_read(
                destination.as_mut_ptr().wrapping_add(self.at).cast(),
                source.address(),
                bytes,
            )
        }?;
        device.synchronize()
    }
}

/// Pinned readbacks submitted together, in order.
pub struct Copies(Vec<Box<dyn Copyback>>);

impl Copies {
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    /// Append `copy`, which runs after every copy pushed before it.
    pub fn push<T: DType + DeviceCopy>(&mut self, copy: Readback<T>) {
        self.0.push(Box::new(copy));
    }

    fn run(&self, device: &Device) -> Result<()> {
        self.0.iter().try_for_each(|copy| copy.run(device))
    }
}

/// Work submitted by [`Device::read_back`] or a replay variant. The first poll runs it and waits
/// for the device, so the future is ready from then on.
pub struct Flight(Option<Box<dyn FnOnce() -> Result<()> + Send>>);

impl Future for Flight {
    type Output = Result<()>;

    fn poll(mut self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Result<()>> {
        Poll::Ready(self.0.take().map_or_else(
            || Err(Error::invariant("readback polled after completion")),
            |work| work(),
        ))
    }
}

impl Device {
    /// Copy `copies` back once the work already on this stream is done.
    #[must_use]
    pub fn read_back(&self, copies: Copies) -> Flight {
        let device = self.clone();
        Flight(Some(Box::new(move || copies.run(&device))))
    }

    /// Replay `graph`, then copy `copies` back, in one submission.
    #[must_use]
    pub fn replay_and_read_back(&self, graph: &Graph, copies: Copies) -> Flight {
        let (device, graph) = (self.clone(), Arc::clone(&graph.0));
        Flight(Some(Box::new(move || {
            graph.launch()?;
            copies.run(&device)
        })))
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
        let device = self.clone();
        let (graph, selector) = (Arc::clone(&graph.0), Arc::clone(&selector.0));
        Flight(Some(Box::new(move || {
            graph.launch()?;
            selector.launch()?;
            copies.run(&device)?;
            tokens.run(&device)
        })))
    }
}
