//! Immutable observation ownership handed to exporters outside the engine owner.
use super::collector;
use crate::EngineFault;
use infer_core::Result;
use infer_observe::export::prometheus_retention;
use infer_observe::trace::coverage;
use infer_observe::trace::otlp;
use infer_observe::trace::timeline;
use infer_observe::{
    ObservationQuery, ObservationWindow, RuntimeMetrics, export::ResourceGauges,
    trace::TraceContext,
};
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// A consistent logical-clock snapshot. Rendering does not borrow or lock the live engine.
pub(super) enum WindowSource {
    Ready(ObservationWindow),
    Pending(collector::CollectionTicket),
}
impl WindowSource {
    fn resolve(self) -> Result<collector::CollectedSnapshot> {
        match self {
            Self::Ready(window) => Ok(collector::CollectedSnapshot {
                window,
                parents: BTreeMap::new(),
                dropped_metadata: 0,
            }),
            Self::Pending(ticket) => ticket.finish(),
        }
    }
}
pub struct ObservationSnapshot {
    pub(super) query: ObservationQuery,
    /// Taken with the snapshot for a device timeline query; rendered as recorded.
    pub(super) device: Option<infer_observe::DeviceTimelineQuery>,
    pub(super) metrics: RuntimeMetrics,
    pub(super) resources: ResourceGauges,
    pub(super) fault: Option<EngineFault>,
    pub(super) window: WindowSource,
    pub(super) parents: BTreeMap<u64, TraceContext>,
    pub(super) namespace: u64,
    pub(super) unix_origin_us: u64,
}
impl ObservationSnapshot {
    /// # Errors
    /// Rejects event query limits outside the bounded response budget.
    pub fn render(self) -> Result<Value> {
        let collected = self.window.resolve()?;
        let window = collected.window;
        let parents = if collected.parents.is_empty() {
            self.parents
        } else {
            collected.parents
        };
        Ok(match self.query {
            ObservationQuery::Summary => json!({
                "metrics":self.metrics,"resources":self.resources,"fault":self.fault,
                "session":format!("{:016x}",self.namespace),
                "retained_events":window.retained(),"history_evicted":window.history_evicted,
                "diagnostics_evicted":window.diagnostics_evicted,"clock":"runtime_logical",
                "trace_coverage":coverage(&window.timeline()),
                "unix_origin_us":self.unix_origin_us, "collector_metadata_dropped": collected.dropped_metadata,
            }),
            ObservationQuery::Metrics => json!(prometheus_retention(
                &self.metrics,
                &self.resources,
                window.history_evicted,
                window.diagnostics_evicted
            )),
            ObservationQuery::Events {
                after,
                limit,
                request,
            } => {
                let mut events =
                    window.query(after, limit, request, self.resources.event_ring_dropped)?;
                events.session = Some(format!("{:016x}", self.namespace));
                json!(events)
            }
            ObservationQuery::Diagnostics => json!(window.diagnostics()),
            ObservationQuery::Timeline => timeline(
                &window.timeline(),
                self.resources.event_ring_dropped,
                window.history_evicted,
            ),
            ObservationQuery::Otlp => otlp(
                &window.timeline(),
                &parents,
                self.namespace,
                self.unix_origin_us,
            ),
            ObservationQuery::DeviceTimelines { .. } => json!(self.device),
        })
    }
}
