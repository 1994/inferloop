//! The device objects the host still makes directly: PTX modules, pinned host memory, timing
//! events and the raw stream vendor libraries take.
use super::Device;
use crate::tile::backend_error;
use cuda_core::{CudaContext, DeviceCopy, Function, PinnedHostBuffer};
use infer_core::Result;
use std::ffi::c_void;
use std::sync::Arc;

/// A loaded PTX module; its functions keep it loaded.
pub struct Module(Arc<cuda_core::Module>);

impl Module {
    /// The kernel `name` in this module.
    /// # Errors
    /// Returns the driver's failure to find it.
    pub fn function(&self, name: &str) -> Result<Function> {
        self.0.load_function(name).map_err(backend_error)
    }
}

/// A stream event for device timing; recording it only enqueues.
pub struct Event(cuda_core::Event);

impl Event {
    /// Record this event at the current end of `device`'s stream.
    /// # Errors
    /// Returns the driver's refusal.
    pub fn record(&self, device: &Device) -> Result<()> {
        self.0.record(device.stream()).map_err(backend_error)
    }

    /// Milliseconds of device time from this event to `end`, both recorded and complete.
    /// # Errors
    /// Returns the driver's failure, for example while either event is pending.
    pub fn elapsed_ms(&self, end: &Self) -> Result<f32> {
        self.0.elapsed_time(&end.0).map_err(backend_error)
    }
}

impl Device {
    /// Load `ptx`, which the driver compiles for this device.
    /// # Errors
    /// Returns the driver's failure to load it.
    pub fn load_ptx(&self, ptx: &str) -> Result<Module> {
        self.stream()
            .device()
            .load_module_from_ptx_src(ptx)
            .map(Module)
            .map_err(backend_error)
    }

    /// `len` zeroed values of page-locked host memory on this device's context.
    /// # Errors
    /// Returns the driver's failure to create the context or allocate.
    pub fn pinned<T: DeviceCopy>(&self, len: usize) -> Result<PinnedHostBuffer<T>> {
        let context = CudaContext::new(self.ordinal()).map_err(backend_error)?;
        PinnedHostBuffer::zeroed(&context, len).map_err(backend_error)
    }

    /// A timing event on this device.
    /// # Errors
    /// Returns the driver's failure to create it.
    pub fn event(&self) -> Result<Event> {
        self.stream()
            .device()
            .new_event()
            .map(Event)
            .map_err(backend_error)
    }

    /// Device bytes the allocation pool holds but no allocation uses, which a later allocation
    /// can take without asking the driver. The driver counts them as used until it trims the pool.
    /// # Errors
    /// Returns the driver's failure to bind the context or read the pool.
    pub fn pool_reclaimable_bytes(&self) -> Result<u64> {
        self.bind_to_thread()?;
        let pool = self
            .stream()
            .device()
            .default_mem_pool()
            .map_err(backend_error)?;
        let stats = pool.mem_stats().map_err(backend_error)?;
        Ok(stats.reserved_current.saturating_sub(stats.used_current))
    }

    /// Wait until this device's stream has finished everything queued on it.
    /// # Errors
    /// Returns the driver's failure to bind the context or synchronize.
    #[expect(unsafe_code, reason = "Waiting on this device's own live stream")]
    pub fn drain_stream(&self) -> Result<()> {
        self.bind_to_thread()?;
        // SAFETY: the stream is live and its context is current; this only waits for its work.
        unsafe { self.stream().synchronize() }.map_err(backend_error)
    }

    /// The stream's identity, which a [`KernelLaunch`](super::KernelLaunch) is pinned to.
    #[must_use]
    pub fn stream_id(&self) -> usize {
        self.stream().cu_stream() as usize
    }

    /// The stream handle a vendor library enqueues on.
    #[must_use]
    pub fn raw_stream(&self) -> *mut c_void {
        self.stream().cu_stream().cast()
    }
}
