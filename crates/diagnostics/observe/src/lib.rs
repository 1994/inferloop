//! Fixed-storage recording and bounded cold-path semantic observation/export.
mod constants;
mod device;
mod diagnostic;
pub mod export;
mod metrics;
mod store;
#[cfg(test)]
mod tests;
pub mod trace;
mod window;

pub use device::{
    DEVICE_TIMELINE_CAPACITY, DeviceTimelineQuery, DeviceTimelines, MAX_DEVICE_TIMELINE_QUERY,
    StepTimeline,
};
pub use diagnostic::{Diagnostic, DiagnosticCode, Severity};
pub use metrics::{LATENCY_BOUNDS_US, LatencyHistogram, RuntimeMetrics};
pub use store::{
    EventQuery, ObservationStore, ObservedEvent, diagnostic_for_event, error_code, finish_cause,
    finish_code,
};
pub use window::ObservationWindow;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ObservationQuery {
    Summary,
    Metrics,
    Events {
        after: u64,
        limit: usize,
        request: Option<infer_core::RequestId>,
    },
    Diagnostics,
    Timeline,
    Otlp,
    /// Raw sampled device timelines after a cursor; see [`DeviceTimelines::query`].
    DeviceTimelines {
        after: u64,
        limit: usize,
    },
}
