use crate::store::{FINISH_CODE_CANCELLED, FINISH_CODE_FAILED};
use infer_core::event::CpuStage;
use infer_core::{CancelSource, ErrorCode, event::EventKind, event::SemanticEvent};
use serde::{Deserialize, Serialize};

pub const LATENCY_BOUNDS_US: [u64; 16] = [
    10, 50, 100, 250, 500, 1000, 2500, 5000, 10_000, 25_000, 50_000, 100_000, 250_000, 500_000,
    1_000_000, 5_000_000,
];
/// Finite latency buckets of [`LATENCY_BOUNDS_US`] plus the overflow bucket.
pub const LATENCY_BUCKET_COUNT: usize = 17;
/// Event-kind slots of [`EventKind::ALL`].
pub const EVENT_KIND_COUNT: usize = EventKind::ALL.len();
/// CPU pipeline stages of [`infer_core::event::CpuStage::LABELS`].
pub const CPU_STAGE_COUNT: usize = CpuStage::LABELS.len();
/// Whole-percent scale of the requested percentile argument.
const PERCENT_SCALE: u8 = 100;

/// Non-cumulative bucket counts plus overflow; no allocations when recording.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatencyHistogram {
    pub buckets: [u64; LATENCY_BUCKET_COUNT],
    pub count: u64,
    pub sum_us: u64,
    pub max_us: u64,
}
impl LatencyHistogram {
    pub fn record(&mut self, value: u64) {
        let index = LATENCY_BOUNDS_US.partition_point(|bound| *bound < value);
        self.buckets[index] = self.buckets[index].saturating_add(1);
        self.count = self.count.saturating_add(1);
        self.sum_us = self.sum_us.saturating_add(value);
        self.max_us = self.max_us.max(value);
    }
    /// The finite bucket upper bound, or actual maximum for the overflow bucket.
    #[must_use]
    pub fn percentile_upper_bound(&self, percentile: u8) -> Option<u64> {
        if self.count == 0 || percentile == 0 || percentile > PERCENT_SCALE {
            return None;
        }
        let rank = u64::try_from(
            (u128::from(self.count) * u128::from(percentile)).div_ceil(u128::from(PERCENT_SCALE)),
        )
        .ok()?;
        let mut count = 0_u64;
        for (index, bucket) in self.buckets.iter().enumerate() {
            count = count.saturating_add(*bucket);
            if count >= rank {
                return Some(LATENCY_BOUNDS_US.get(index).copied().unwrap_or(self.max_us));
            }
        }
        None
    }
}

/// Always-on counters are updated before enqueueing; full trace rings cannot lose metrics.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeMetrics {
    pub events: [u64; EVENT_KIND_COUNT],
    pub deferred_reasons: [u64; infer_ir::DeferReason::LABELS.len()],
    pub successful_requests: u64,
    pub failed_requests: u64,
    /// Cancelled requests by [`CancelSource::ALL`] position.
    #[serde(default)]
    pub cancelled_by: [u64; CancelSource::ALL.len()],
    /// Failed requests by [`ErrorCode::ALL`] position.
    #[serde(default)]
    pub failed_by: [u64; ErrorCode::ALL.len()],
    pub prefix_tokens_reused: u64,
    pub e2e: LatencyHistogram,
    pub ttft: LatencyHistogram,
    pub tpot: LatencyHistogram,
    pub cpu_execution: LatencyHistogram,
    pub cpu_stages: [LatencyHistogram; CPU_STAGE_COUNT],
    pub cpu_stage_items: [u64; CPU_STAGE_COUNT],
    pub gpu_execution: LatencyHistogram,
    pub cpu_output: LatencyHistogram,
    pub resource_preparation: LatencyHistogram,
    /// Device time the backend measured, by phase, operator class, layer and implementation.
    #[serde(default)]
    pub device: crate::DeviceTimingMetrics,
}
impl RuntimeMetrics {
    pub fn record(&mut self, event: &SemanticEvent) {
        let index = event.kind as usize;
        self.events[index] = self.events[index].saturating_add(1);
        match event.kind {
            EventKind::CpuStageTiming => {
                if let Some(histogram) = self.cpu_stages.get_mut(event.reserved as usize) {
                    histogram.record(event.arg0);
                    self.cpu_stage_items[event.reserved as usize] =
                        self.cpu_stage_items[event.reserved as usize].saturating_add(event.arg1);
                }
            }
            EventKind::Deferred => {
                if let Some(count) = self.deferred_reasons.get_mut(event.reserved as usize) {
                    *count = count.saturating_add(1);
                }
            }
            EventKind::Finished => {
                self.e2e.record(event.arg1);
                if event.arg0 <= 2 {
                    self.successful_requests = self.successful_requests.saturating_add(1);
                } else {
                    self.failed_requests = self.failed_requests.saturating_add(1);
                }
                let cause = (event.reserved as usize).checked_sub(1);
                let counter = match event.arg0 {
                    FINISH_CODE_CANCELLED => {
                        cause.and_then(|index| self.cancelled_by.get_mut(index))
                    }
                    FINISH_CODE_FAILED => cause.and_then(|index| self.failed_by.get_mut(index)),
                    _ => None,
                };
                if let Some(count) = counter {
                    *count = count.saturating_add(1);
                }
            }
            EventKind::FirstToken => self.ttft.record(event.arg0),
            EventKind::RequestLatency => self.tpot.record(event.arg0),
            EventKind::ExecutionTiming if event.arg1 == 0 => self.cpu_execution.record(event.arg0),
            EventKind::ExecutionTiming => self.gpu_execution.record(event.arg0),
            EventKind::CpuOutputCompleted => self.cpu_output.record(event.arg0),
            EventKind::ResourceAcknowledged => self.resource_preparation.record(event.arg1),
            EventKind::PrefixHit => {
                self.prefix_tokens_reused = self.prefix_tokens_reused.saturating_add(event.arg0);
            }
            _ => {}
        }
    }
    #[must_use]
    pub const fn count(&self, kind: EventKind) -> u64 {
        self.events[kind as usize]
    }
}
