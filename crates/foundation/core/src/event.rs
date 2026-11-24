use rtrb::{Consumer, Producer, RingBuffer};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u16)]
pub enum EventKind {
    Accepted,
    Deferred,
    Scheduled,
    Submitted,
    Completed,
    Failed,
    Progress,
    Finished,
    Cancelled,
    StateReserved,
    StateReleased,
    InvariantViolation,
    PrefixHit,
    Preempted,
    Rejected,
    TokenProduced,
    FirstToken,
    RequestLatency,
    ExecutionTiming,
    CheckpointCaptured,
    Quarantined,
    Backpressure,
    CpuOutputCompleted,
    ResourceAcknowledged,
    LaunchAcknowledged,
    CpuStageTiming,
}
impl EventKind {
    pub const ALL: [Self; 26] = [
        Self::Accepted,
        Self::Deferred,
        Self::Scheduled,
        Self::Submitted,
        Self::Completed,
        Self::Failed,
        Self::Progress,
        Self::Finished,
        Self::Cancelled,
        Self::StateReserved,
        Self::StateReleased,
        Self::InvariantViolation,
        Self::PrefixHit,
        Self::Preempted,
        Self::Rejected,
        Self::TokenProduced,
        Self::FirstToken,
        Self::RequestLatency,
        Self::ExecutionTiming,
        Self::CheckpointCaptured,
        Self::Quarantined,
        Self::Backpressure,
        Self::CpuOutputCompleted,
        Self::ResourceAcknowledged,
        Self::LaunchAcknowledged,
        Self::CpuStageTiming,
    ];
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::Deferred => "deferred",
            Self::Scheduled => "scheduled",
            Self::Submitted => "submitted",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Progress => "progress",
            Self::Finished => "finished",
            Self::Cancelled => "cancelled",
            Self::StateReserved => "state_reserved",
            Self::StateReleased => "state_released",
            Self::InvariantViolation => "invariant_violation",
            Self::PrefixHit => "prefix_hit",
            Self::Preempted => "preempted",
            Self::Rejected => "rejected",
            Self::TokenProduced => "token_produced",
            Self::FirstToken => "first_token",
            Self::RequestLatency => "request_latency",
            Self::ExecutionTiming => "execution_timing",
            Self::CheckpointCaptured => "checkpoint_captured",
            Self::Quarantined => "quarantined",
            Self::Backpressure => "backpressure",
            Self::CpuOutputCompleted => "cpu_output_completed",
            Self::ResourceAcknowledged => "resource_acknowledged",
            Self::LaunchAcknowledged => "launch_acknowledged",
            Self::CpuStageTiming => "cpu_stage_timing",
        }
    }
}
/// Low-cardinality CPU owner service categories. Durations exclude device waiting.
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum CpuStage {
    Ready,
    Planning,
    Dispatch,
    Completion,
    Owner,
}
impl CpuStage {
    pub const LABELS: [&'static str; 5] = ["ready", "planning", "dispatch", "completion", "owner"];
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u16)]
pub enum ObjectKind {
    Request,
    Decision,
    Step,
    State,
    Program,
    Op,
    Kernel,
}

/// Fixed-size hot-path record. Strings and JSON are added by cold-path exporters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct SemanticEvent {
    pub timestamp_us: u64,
    pub kind: EventKind,
    pub object_kind: ObjectKind,
    pub reserved: u32,
    pub object_id: u64,
    pub correlation_id: u64,
    pub arg0: u64,
    pub arg1: u64,
}
pub struct EventWriter {
    producer: Producer<SemanticEvent>,
    dropped: u64,
    published: u64,
}
pub type EventReader = Consumer<SemanticEvent>;
#[must_use]
pub fn event_ring(capacity: usize) -> (EventWriter, EventReader) {
    let (producer, consumer) = RingBuffer::new(capacity);
    (
        EventWriter {
            producer,
            dropped: 0,
            published: 0,
        },
        consumer,
    )
}
impl EventWriter {
    pub fn emit(&mut self, event: SemanticEvent) {
        if self.producer.push(event).is_err() {
            self.dropped = self.dropped.saturating_add(1);
        } else {
            self.published = self.published.saturating_add(1);
        }
    }
    pub const fn published(&self) -> u64 {
        self.published
    }
    pub const fn dropped(&self) -> u64 {
        self.dropped
    }
}

#[cfg(test)]
mod tests;
