//! Timing events recorded inside captures.
#![expect(
    unsafe_code,
    reason = "Driver event records on the stream a recording hands its foreign calls"
)]
use super::{CudaEvent, Device};
use crate::contract::buffer::Owner;
use crate::contract::recording::{Foreign, Recording};
use infer_core::Result;
use std::ffi::c_void;
use std::sync::Arc;

/// A timing event on `device`, for [`record_event`].
/// # Errors
/// Returns the driver's failure to create it.
pub fn timing_event(device: &Device) -> Result<Arc<CudaEvent>> {
    device.event().map(Arc::new)
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
        // SAFETY: a recording hands foreign calls a live stream for the call; an event of another
        // context is a driver error, not undefined behavior. The external flag keeps capture-time
        // records observable from the host.
        unsafe { self.0.driver().record_on(stream, true) }
    }
}
