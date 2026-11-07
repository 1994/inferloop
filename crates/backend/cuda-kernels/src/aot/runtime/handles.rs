//! The device objects the host still makes directly: PTX modules, page-locked host memory and
//! timing events.
#![expect(
    unsafe_code,
    reason = "Typed views of page-locked memory this buffer owns, for element types that promise every bit pattern is valid"
)]
use super::super::driver::{self, Function};
use super::{Device, DeviceCopy};
use infer_core::{Error, Result};
use std::ffi::CString;
use std::marker::PhantomData;
use std::mem::size_of;
use std::slice;

/// A loaded PTX module; its functions keep it loaded.
pub struct Module(driver::Module);

impl Module {
    /// The kernel `name` in this module.
    /// # Errors
    /// Rejects a name holding a NUL, or returns the driver's failure to find it.
    pub fn function(&self, name: &str) -> Result<Function> {
        let name = CString::new(name).map_err(|_| Error::invalid("kernel name holds a NUL"))?;
        self.0.function(&name)
    }
}

/// A stream event for device timing; recording it only enqueues.
pub struct Event(driver::Event);

/// The event [`timing_event`](super::timing_event) makes; this runtime has one event type.
pub type CudaEvent = Event;

impl Event {
    /// Record this event at the current end of `device`'s stream.
    /// # Errors
    /// Rejects a device of another context, or returns the driver's refusal.
    pub fn record(&self, device: &Device) -> Result<()> {
        self.0.record(&device.driver)
    }

    /// Milliseconds of device time from this event to `end`, both recorded and complete.
    /// # Errors
    /// Returns the driver's failure, for example while either event is pending.
    pub fn elapsed_ms(&self, end: &Self) -> Result<f32> {
        self.0.elapsed_ms(&end.0)
    }

    pub(super) const fn driver(&self) -> &driver::Event {
        &self.0
    }
}

/// `len` values of page-locked host memory, zeroed when made.
pub struct PinnedHostBuffer<T: DeviceCopy> {
    memory: driver::Pinned,
    len: usize,
    element: PhantomData<T>,
}

impl<T: DeviceCopy> PinnedHostBuffer<T> {
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.memory.pointer().cast_const().cast()
    }

    #[must_use]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.memory.pointer().cast()
    }

    #[must_use]
    pub const fn as_slice(&self) -> &[T] {
        // SAFETY: the memory holds `len` values, zeroed when made and written only as `T`, and
        // `DType` makes every bit pattern a valid `T`; the borrow of `self` bounds the view.
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as for `as_slice`, and the exclusive borrow of `self` makes the view unique.
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }
}

impl Device {
    /// Load `ptx`, which the driver compiles for this device.
    /// # Errors
    /// Rejects text holding a NUL, or returns the driver's failure to load it.
    pub fn load_ptx(&self, ptx: &str) -> Result<Module> {
        let ptx = CString::new(ptx).map_err(|_| Error::invalid("PTX holds a NUL"))?;
        self.driver.load_module(ptx.as_bytes_with_nul()).map(Module)
    }

    /// `len` zeroed values of page-locked host memory on this device's context.
    /// # Errors
    /// Rejects a size that overflows, or returns the driver's failure to allocate.
    pub fn pinned<T: DeviceCopy>(&self, len: usize) -> Result<PinnedHostBuffer<T>> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::invalid("pinned allocation size overflow"))?;
        Ok(PinnedHostBuffer {
            memory: self.driver.pinned(bytes)?,
            len,
            element: PhantomData,
        })
    }

    /// A timing event on this device.
    /// # Errors
    /// Returns the driver's failure to create it.
    pub fn event(&self) -> Result<Event> {
        self.driver.event().map(Event)
    }
}
