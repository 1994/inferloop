//! Device time of each execution phase of a step, reported raw as the step's timeline.
//!
//! Every phase transition records one event on the device's execution stream and names the
//! phase that owns the time from there on, so nested phases attribute each interval to exactly
//! one phase, and the intervals of a step add up to its device time: work outside any phase is
//! [`TimedPhase::Other`]. Only sampled steps record; the rest touch nothing.
use super::{CudaDevice, device_error};
use infer_core::{Error, Result};
use infer_cuda_kernels::runtime::Event;
use infer_ir::{DeviceTimingConfig, DeviceTimingSample, TimedInterval, TimedPhase};
use std::time::Duration;

/// Milliseconds per second, the unit conversion for `cuEventElapsedTime`.
const MILLIS_PER_SECOND: f32 = 1000.0;

/// Phase transitions of the step being measured and the events that mark them.
pub struct DeviceTimer {
    config: DeviceTimingConfig,
    executions: u64,
    sampling: bool,
    rows: u32,
    stack: Vec<TimedPhase>,
    /// Event index and the phase owning the time after it.
    marks: Vec<(usize, TimedPhase)>,
    /// Events reused across steps; grows to the most transitions one step has needed.
    events: Vec<Event>,
}

impl DeviceTimer {
    #[must_use]
    pub const fn new(config: DeviceTimingConfig) -> Self {
        Self {
            config,
            executions: 0,
            sampling: false,
            rows: 0,
            stack: Vec::new(),
            marks: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Start a step over `rows` tasks; decides whether this step is sampled.
    /// # Errors
    /// Fails when an event cannot be created or recorded on the stream.
    pub fn begin_step(&mut self, device: &CudaDevice, rows: usize) -> Result<()> {
        self.sampling = self.config.samples(self.executions);
        self.executions = self.executions.wrapping_add(1);
        self.rows = u32::try_from(rows).unwrap_or(u32::MAX);
        self.stack.clear();
        self.marks.clear();
        if self.sampling {
            self.mark(device, TimedPhase::Other)?;
        }
        Ok(())
    }

    /// `phase` owns the device time from now until it exits or a nested phase enters.
    /// # Errors
    /// Fails when the transition event cannot be recorded.
    pub fn enter(&mut self, device: &CudaDevice, phase: TimedPhase) -> Result<()> {
        if !self.sampling {
            return Ok(());
        }
        self.stack.push(phase);
        self.mark(device, phase)
    }

    /// The innermost phase exits; the time returns to the phase around it.
    /// # Errors
    /// Fails when the transition event cannot be recorded.
    pub fn exit(&mut self, device: &CudaDevice) -> Result<()> {
        if !self.sampling {
            return Ok(());
        }
        self.stack.pop();
        let owner = self.stack.last().copied().unwrap_or(TimedPhase::Other);
        self.mark(device, owner)
    }

    /// Close the step's last interval. Read [`Self::sample`] after the stream is fenced.
    /// # Errors
    /// Fails when the closing event cannot be recorded.
    pub fn end_step(&mut self, device: &CudaDevice) -> Result<()> {
        if !self.sampling {
            return Ok(());
        }
        self.mark(device, TimedPhase::Other)
    }

    /// The step's timeline once the stream is fenced; `None` when the step was not sampled.
    ///
    /// Every mark is timed from the step's first one, so offsets do not accumulate rounding;
    /// consecutive stretches owned by the same phase merge into one interval.
    /// # Errors
    /// Fails when an event has not completed or its time cannot be read.
    pub fn sample(&mut self) -> Result<Option<DeviceTimingSample>> {
        if !self.sampling {
            return Ok(None);
        }
        self.sampling = false;
        let first = self
            .marks
            .first()
            .and_then(|(index, _)| self.events.get(*index))
            .ok_or_else(|| Error::invariant("device timer start"))?;
        let mut marks = Vec::with_capacity(self.marks.len());
        for (index, phase) in &self.marks {
            let event = self
                .events
                .get(*index)
                .ok_or_else(|| Error::invariant("device timer event"))?;
            marks.push((
                nanos(first.elapsed_ms(event).map_err(device_error)?)?,
                *phase,
            ));
        }
        let intervals = intervals(&marks);
        Ok(Some(DeviceTimingSample {
            rows: self.rows,
            intervals,
            segments: Vec::new(),
            host: Vec::new(),
        }))
    }

    fn mark(&mut self, device: &CudaDevice, owner: TimedPhase) -> Result<()> {
        let index = self.marks.len();
        if index == self.events.len() {
            self.events.push(device.tile().event()?);
        }
        let event = self
            .events
            .get(index)
            .ok_or_else(|| Error::invariant("device timer event"))?;
        event.record(device.tile())?;
        self.marks.push((index, owner));
        Ok(())
    }
}

/// Intervals between consecutive `(offset, owner)` marks, each owned by the earlier mark's
/// phase; empty stretches drop and adjacent stretches of one phase merge.
fn intervals(marks: &[(u64, TimedPhase)]) -> Vec<TimedInterval> {
    let mut intervals: Vec<TimedInterval> = Vec::with_capacity(marks.len());
    for window in marks.windows(2) {
        let [(start, phase), (end, _)] = window else {
            continue;
        };
        let duration = end.saturating_sub(*start);
        if duration == 0 {
            continue;
        }
        match intervals.last_mut() {
            Some(last) if last.phase == *phase && last.end_ns() == *start => {
                last.duration_ns = last.duration_ns.saturating_add(duration);
            }
            _ => intervals.push(TimedInterval {
                phase: *phase,
                start_ns: *start,
                duration_ns: duration,
            }),
        }
    }
    intervals
}

/// Non-negative nanoseconds of a `cuEventElapsedTime` reading.
fn nanos(milliseconds: f32) -> Result<u64> {
    let elapsed = Duration::try_from_secs_f32(milliseconds.max(0.0) / MILLIS_PER_SECOND)
        .map_err(device_error)?;
    Ok(u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX))
}

#[cfg(test)]
#[path = "../../tests/unit/device_timer.rs"]
mod tests;
