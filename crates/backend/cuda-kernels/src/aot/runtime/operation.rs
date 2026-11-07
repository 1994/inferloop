//! Stream-ordered work that makes its own memory and resolves once the device has run it.
//!
//! [`Device::operation`] runs a closure that allocates, copies and launches on the device's
//! stream. This runtime runs it on the first poll and waits for the stream before resolving, so
//! everything the closure retained is released only after the device has passed its work.
#![expect(
    unsafe_code,
    reason = "Pinned uploads whose source the operation retains until the stream has passed them"
)]
use super::launch::KernelLaunch;
use super::{Allocation, DType, Device, DeviceCopy, PinnedHostBuffer};
use infer_core::{Error, Result};
use std::any::Any;
use std::cell::RefCell;
use std::ffi::c_void;
use std::future::Future;
use std::marker::PhantomData;
use std::mem::size_of;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Work being enqueued by [`Device::operation`].
pub struct Operation<'a> {
    device: &'a Device,
    retained: RefCell<Vec<Box<dyn Any + Send>>>,
}

impl Operation<'_> {
    /// `len` uninitialized elements, alive until the device has passed this operation and for as
    /// long as the returned allocation is held after that.
    ///
    /// # Safety
    /// Work this operation enqueues must write every element before anything, on the device or the
    /// host, reads it.
    /// # Errors
    /// Reports a failed allocation.
    pub unsafe fn overwritten<T: DType>(&self, len: usize) -> Result<Allocation<T>> {
        let allocation = self.device.allocate(len)?;
        self.retain(allocation.clone())?;
        Ok(allocation)
    }

    /// Keep `owner` alive until the device has passed this operation.
    /// # Errors
    /// Never fails; the signature matches the cuTile build's.
    pub fn retain(&self, owner: impl Send + 'static) -> Result<()> {
        self.retained.borrow_mut().push(Box::new(owner));
        Ok(())
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
        unsafe {
            self.device
                .driver
                .enqueue_write(destination.address(), source.as_ptr().cast(), bytes)
        }
    }

    /// The driver stream, for libraries that enqueue on it themselves.
    #[must_use]
    pub fn stream(&self) -> *mut c_void {
        self.device.raw_stream()
    }
}

/// The result of an operation, once the device has passed it; see [`Device::operation`].
pub struct Pending<R, F> {
    device: Device,
    work: Option<F>,
    output: PhantomData<fn() -> R>,
}

// The closure is moved out, never pinned in place.
impl<R, F> Unpin for Pending<R, F> {}

impl<R, F> Future for Pending<R, F>
where
    R: Send,
    F: FnOnce(&Operation<'_>) -> Result<R> + Send,
{
    type Output = Result<R>;

    fn poll(mut self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Result<R>> {
        let Some(work) = self.work.take() else {
            return Poll::Ready(Err(Error::invariant("operation polled after completion")));
        };
        let operation = Operation {
            device: &self.device,
            retained: RefCell::new(Vec::new()),
        };
        let output = work(&operation);
        // What was enqueued before a refusal still runs, so wait either way before releasing.
        let drained = self.device.synchronize();
        drop(operation);
        Poll::Ready(drained.and(output))
    }
}

impl Device {
    /// Run `work` on this stream when the future is first polled. The future resolves to what
    /// `work` returned once the device has passed everything it enqueued, and only then releases
    /// what it retained.
    #[must_use]
    pub fn operation<R, F>(&self, work: F) -> Pending<R, F>
    where
        R: Send,
        F: FnOnce(&Operation<'_>) -> Result<R> + Send,
    {
        Pending {
            device: self.clone(),
            work: Some(work),
            output: PhantomData,
        }
    }
}
