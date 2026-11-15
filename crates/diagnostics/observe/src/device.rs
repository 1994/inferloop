//! Retention of raw device timelines: the one source every device-time view reads.
//!
//! The runtime keeps each sampled step's timeline exactly as the backend reported it, with
//! the step, program and runtime clock that place it among the other observations. Summaries,
//! dashboards, trace viewers and exporters are built on top of [`DeviceTimelines::query`] and
//! stay outside this crate's collection path, so adding one never changes what is collected.
use infer_core::{Error, Result};
use infer_ir::DeviceTimingSample;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Sampled steps retained; older ones are evicted and counted.
pub const DEVICE_TIMELINE_CAPACITY: usize = 64;
/// Records one query returns at most.
pub const MAX_DEVICE_TIMELINE_QUERY: usize = DEVICE_TIMELINE_CAPACITY;

/// One sampled step's device timeline and where it sits in the runtime.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepTimeline {
    /// Position in retention order; a query cursor.
    pub sequence: u64,
    pub step: u64,
    pub program: u64,
    /// Runtime logical clock when the step completed, the clock of every other observation.
    pub completed_us: u64,
    pub sample: DeviceTimingSample,
}

/// Records after a cursor, with what a reader needs to interpret them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceTimelineQuery {
    pub records: Vec<StepTimeline>,
    /// Pass as `after` to continue from the last record returned.
    pub next: u64,
    /// Records evicted before any reader saw them since the runtime started.
    pub evicted: u64,
    /// The backend's implementation names, which segment variants index.
    pub variants: Vec<String>,
}

/// The most recent sampled steps, in fixed capacity.
#[derive(Debug, Clone)]
pub struct DeviceTimelines {
    records: VecDeque<StepTimeline>,
    next: u64,
    evicted: u64,
    variants: Vec<String>,
}

impl DeviceTimelines {
    /// An empty retention whose segment variants index `variants`, the backend's declared names.
    #[must_use]
    pub fn new(variants: Vec<String>) -> Self {
        Self {
            records: VecDeque::with_capacity(DEVICE_TIMELINE_CAPACITY),
            next: 0,
            evicted: 0,
            variants,
        }
    }

    /// Retain one sampled step, evicting the oldest when full.
    pub fn push(&mut self, step: u64, program: u64, completed_us: u64, sample: DeviceTimingSample) {
        if self.records.len() == DEVICE_TIMELINE_CAPACITY {
            self.records.pop_front();
            self.evicted = self.evicted.saturating_add(1);
        }
        self.next = self.next.saturating_add(1);
        self.records.push_back(StepTimeline {
            sequence: self.next,
            step,
            program,
            completed_us,
            sample,
        });
    }

    /// Up to `limit` records with a sequence after `after`, oldest first.
    /// # Errors
    /// Rejects a limit of zero or above [`MAX_DEVICE_TIMELINE_QUERY`].
    pub fn query(&self, after: u64, limit: usize) -> Result<DeviceTimelineQuery> {
        if limit == 0 || limit > MAX_DEVICE_TIMELINE_QUERY {
            return Err(Error::invalid(
                "device timeline query limit must be 1 to the retained capacity",
            ));
        }
        let records: Vec<StepTimeline> = self
            .records
            .iter()
            .filter(|record| record.sequence > after)
            .take(limit)
            .cloned()
            .collect();
        Ok(DeviceTimelineQuery {
            next: records.last().map_or(after, |record| record.sequence),
            records,
            evicted: self.evicted,
            variants: self.variants.clone(),
        })
    }
}

#[cfg(test)]
#[path = "../tests/unit/device.rs"]
mod tests;
