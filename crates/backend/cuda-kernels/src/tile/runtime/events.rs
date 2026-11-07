//! Timing events recorded inside captures, and synchronous reads of device memory.
#![expect(
    unsafe_code,
    reason = "Driver event records and D2H copies over memory whose extent a buffer's contract guarantees"
)]
use super::{Device, Element};
use crate::contract::buffer::{Buffer, Owner};
use crate::contract::recording::{Foreign, Recording};
use crate::contract::target::Arch;
use crate::tile::backend_error;
use cuda_core::sys::{
    CUevent_flags_enum_CU_EVENT_DEFAULT, CUevent_record_flags_enum_CU_EVENT_RECORD_EXTERNAL,
    cuEventRecordWithFlags, cuMemcpyDtoH_v2, cudaError_enum_CUDA_SUCCESS,
};
use cuda_core::{CudaContext, CudaEvent, DType};
use infer_core::{Error, Result};
use std::ffi::c_void;
use std::sync::Arc;

/// A timing event on `context`'s device, for [`record_event`].
/// # Errors
/// Returns the driver's failure to create it.
pub fn timing_event(context: &Arc<CudaContext>) -> Result<Arc<CudaEvent>> {
    context
        .new_event(Some(CUevent_flags_enum_CU_EVENT_DEFAULT))
        .map(Arc::new)
        .map_err(backend_error)
}

/// Record `event` where `recording` puts its launches; a captured graph keeps the event alive and
/// every replay records it again, queryable from the host afterwards.
/// # Errors
/// Returns the driver's refusal to record it.
pub fn record_event(recording: &Recording<'_>, event: &Arc<CudaEvent>) -> Result<()> {
    recording.foreign(
        RecordEvent(Arc::clone(event)),
        vec![Arc::clone(event) as Owner],
    )
}

/// Records an external timing event on the stream it is handed, as one node of a capture.
struct RecordEvent(Arc<CudaEvent>);

impl Foreign for RecordEvent {
    fn enqueue(self, stream: *mut c_void) -> Result<()> {
        // SAFETY: the event is a live RAII wrapper on the device that owns `stream`, and the
        // recording keeps it alive with the graph; the external flag is the documented way to
        // keep capture-time event records observable from the host.
        let status = unsafe {
            cuEventRecordWithFlags(
                self.0.cu_event(),
                stream.cast(),
                CUevent_record_flags_enum_CU_EVENT_RECORD_EXTERNAL,
            )
        };
        if status == cudaError_enum_CUDA_SUCCESS {
            Ok(())
        } else {
            Err(backend_error(format!("event record CUDA status {status}")))
        }
    }
}

impl Device {
    /// Open device `ordinal` with a new stream, and the architecture it reports. An unreadable
    /// architecture name is an unknown architecture, so only general paths apply to it.
    /// # Errors
    /// Fails when the device or its stream is unavailable.
    pub fn open(ordinal: usize) -> Result<(Self, Arch)> {
        let device = cuda_core::Device::new(ordinal).map_err(backend_error)?;
        // SAFETY: `Device::new` initialized CUDA and owns this live device handle.
        let name =
            unsafe { cuda_core::get_device_sm_name(device.cu_device()) }.map_err(backend_error)?;
        let arch = Arch::from_sm_name(&name).unwrap_or(Arch::Other { major: 0, minor: 0 });
        let stream = device.new_stream().map_err(backend_error)?;
        Ok((Self::new(stream, ordinal), arch))
    }

    /// The stream itself, for vendor-library calls the host still makes on it. Transitional,
    /// until those calls move behind the kernel families.
    #[must_use]
    pub const fn stream_handle(&self) -> &Arc<cuda_core::Stream> {
        self.stream()
    }

    /// The `buffer.len()` elements at `buffer`, copied back synchronously.
    /// # Errors
    /// Returns the driver's failure to bind the context or copy.
    pub fn read_buffer<T: DType + Element>(&self, buffer: Buffer<'_, T::Marker>) -> Result<Vec<T>> {
        self.bind_to_thread()?;
        let len = buffer.len();
        let mut host = vec![T::zero(); len];
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| Error::invalid("readback size overflow"))?;
        // SAFETY: the buffer's contract guarantees `len` live elements at its address, `Element`
        // ties the marker's width to `T`, the device's context is current, and the destination is
        // a live host vector of the same length.
        let status = unsafe { cuMemcpyDtoH_v2(host.as_mut_ptr().cast(), buffer.address(), bytes) };
        if status == cudaError_enum_CUDA_SUCCESS {
            Ok(host)
        } else {
            Err(backend_error(format!("readback CUDA status {status}")))
        }
    }
}
