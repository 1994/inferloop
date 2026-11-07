//! Device time of one execution step, reported to the engine's cost model.
use super::CudaDevice;
use infer_core::Error;
use infer_cuda_kernels::runtime::Event;
use std::time::Duration;

/// Milliseconds per second, the unit conversion for `cuEventElapsedTime`.
const MILLIS_PER_SECOND: f32 = 1000.0;
/// Nanoseconds per microsecond, the unit of `ExecutionTiming`.
const NANOS_PER_MICRO: u128 = 1000;

/// Two reusable events bracketing a step on the device's single execution stream.
///
/// Recording only enqueues on the stream, so neither edge waits for the device.
pub struct StepClock {
    start: Event,
    end: Event,
}

impl StepClock {
    /// `None` leaves the step unmeasured: the cost model keeps its estimate and execution
    /// does not depend on timing.
    pub fn new(device: &CudaDevice) -> Option<Self> {
        let created = (|| {
            Ok::<_, Error>(Self {
                start: device.tile().event()?,
                end: device.tile().event()?,
            })
        })();
        created
            .inspect_err(|error| {
                tracing::warn!(target: "infer::cuda", %error, "step device timing unavailable");
            })
            .ok()
    }
    pub fn start(&self, device: &CudaDevice) -> bool {
        self.start.record(device.tile()).is_ok()
    }
    pub fn stop(&self, device: &CudaDevice) -> bool {
        self.end.record(device.tile()).is_ok()
    }
    /// Device microseconds between both edges. The caller has fenced the stream.
    pub fn elapsed_us(&self) -> Option<u64> {
        millis_to_micros(self.start.elapsed_ms(&self.end).ok()?)
    }
}

/// A sample the engine can use: positive, finite and rounded up so it never reads as zero.
fn millis_to_micros(milliseconds: f32) -> Option<u64> {
    let elapsed = Duration::try_from_secs_f32(milliseconds / MILLIS_PER_SECOND).ok()?;
    let micros = u64::try_from(elapsed.as_nanos().div_ceil(NANOS_PER_MICRO)).ok()?;
    (micros > 0).then_some(micros)
}

#[cfg(test)]
#[path = "../../tests/unit/device_step_clock.rs"]
mod tests;
