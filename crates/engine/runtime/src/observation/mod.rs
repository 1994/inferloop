pub mod collector;
mod snapshot;
use crate::{Engine, EngineFault};
use infer_core::event::SemanticEvent;
use infer_core::{Error, RequestId, Result, StepId, event::EventKind, event::ObjectKind};
use infer_observe::{
    Diagnostic, DiagnosticCode, EventQuery, ObservationStore, RuntimeMetrics, Severity,
    export::ResourceGauges, trace::TraceContext,
};
use infer_spi::{BackendProvider, SchedulingPolicy};
use serde_json::{Value, json};
use std::process::id as process_id;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
use std::{collections::BTreeMap, sync::atomic::AtomicU64, sync::atomic::Ordering};

pub use snapshot::ObservationSnapshot;
/// High half of the namespace carries the process id so separate processes cannot collide.
const NAMESPACE_PROCESS_ID_SHIFT: u32 = 32;
/// Largest event window an exporter may request from one observation query.
const MAX_EVENT_QUERY_LIMIT: usize = 4096;
static NEXT_NAMESPACE: AtomicU64 = AtomicU64::new(0);
fn next_namespace(clock: u64) -> Result<u64> {
    let mut previous = NEXT_NAMESPACE.load(Ordering::Relaxed);
    loop {
        let next = previous
            .max(clock)
            .checked_add(1)
            .ok_or_else(|| Error::invariant("observation session identities exhausted"))?;
        match NEXT_NAMESPACE.compare_exchange_weak(
            previous,
            next,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => {
                return Ok(next ^ (u64::from(process_id()) << NAMESPACE_PROCESS_ID_SHIFT));
            }
            Err(observed) => previous = observed,
        }
    }
}

pub struct Observations {
    pub metrics: RuntimeMetrics,
    pub store: ObservationStore,
    /// Raw sampled device timelines, retained apart from metrics so snapshots never copy them
    /// unless asked.
    pub device: infer_observe::DeviceTimelines,
    pub collector: Option<collector::Collector>,
    pub parents: BTreeMap<u64, TraceContext>,
    pub parents_dirty: bool,
    pub namespace: u64,
    pub unix_origin_us: u64,
    pub legacy_cursor: u64,
}
impl Observations {
    pub fn new(capacity: usize) -> Result<Self> {
        let unix_origin_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| {
                u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
            });
        Ok(Self {
            metrics: RuntimeMetrics::default(),
            store: ObservationStore::new(capacity)?,
            device: infer_observe::DeviceTimelines::new(Vec::new()),
            collector: None,
            parents: BTreeMap::new(),
            parents_dirty: false,
            namespace: next_namespace(unix_origin_us)?,
            unix_origin_us,
            legacy_cursor: 0,
        })
    }
}
impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    /// # Errors
    /// Returns invalid input for an event query outside the bounded response limit.
    pub fn observe(&mut self, query: &infer_observe::ObservationQuery) -> Result<Value> {
        self.observation_snapshot(query.clone())?.render()
    }
    /// Capture shared event pages and immutable counters; exporters can render on another thread.
    /// # Errors
    /// Rejects event limits outside the bounded response budget before capturing ownership.
    pub fn observation_snapshot(
        &mut self,
        query: infer_observe::ObservationQuery,
    ) -> Result<ObservationSnapshot> {
        use infer_observe::ObservationQuery;
        if matches!(query, ObservationQuery::Events { limit, .. } if limit == 0 || limit > MAX_EVENT_QUERY_LIMIT)
        {
            return Err(Error::invalid("event query limit must be 1 to 4096"));
        }
        let device = match query {
            ObservationQuery::DeviceTimelines { after, limit } => {
                Some(self.observations.device.query(after, limit)?)
            }
            _ => None,
        };
        self.collect_observations();
        let include_events = !matches!(
            query,
            ObservationQuery::Metrics | ObservationQuery::Diagnostics
        );
        let parents =
            if self.observations.collector.is_none() && matches!(query, ObservationQuery::Otlp) {
                self.observations.parents.clone()
            } else {
                BTreeMap::new()
            };
        let include_parents = matches!(query, ObservationQuery::Otlp);
        Ok(ObservationSnapshot {
            query,
            device,
            metrics: self.observations.metrics.clone(),
            resources: self.resource_gauges(),
            fault: self.fault.clone(),
            window: if let Some(collector) = &self.observations.collector {
                snapshot::WindowSource::Pending(collector.snapshot(
                    self.events.published(),
                    include_events,
                    include_parents,
                )?)
            } else {
                snapshot::WindowSource::Ready(self.observations.store.snapshot(include_events))
            },
            parents,
            namespace: self.observations.namespace,
            unix_origin_us: self.observations.unix_origin_us,
        })
    }
    /// Drain the fixed POD ring into bounded cold-path retention.
    pub fn collect_observations(&mut self) {
        if let Some(collector) = &self.observations.collector {
            collector.wake();
            return;
        }
        let evicted = self.observations.store.history_evicted;
        if let Some(reader) = &mut self.event_reader {
            while let Ok(event) = reader.pop() {
                self.observations.store.record(event);
            }
        }
        // Only eviction or request-identity changes can invalidate retained trace parents.
        if !self.observations.parents_dirty && evicted == self.observations.store.history_evicted {
            return;
        }
        self.observations.parents_dirty = false;
        self.observations.parents.retain(|id, _| {
            RequestId::new(*id).is_ok_and(|request| self.host.requests.contains_key(request))
                || self.observations.store.contains_request(*id)
        });
    }
    pub const fn metrics(&self) -> &RuntimeMetrics {
        &self.observations.metrics
    }
    /// Raw sampled device timelines after `after`, oldest first.
    /// # Errors
    /// Rejects a limit of zero or above the retained capacity.
    pub fn device_timelines(
        &self,
        after: u64,
        limit: usize,
    ) -> Result<infer_observe::DeviceTimelineQuery> {
        self.observations.device.query(after, limit)
    }
    /// # Errors
    /// Returns invalid input for event limits outside 1 to 4096.
    pub fn observed_events(
        &mut self,
        after: u64,
        limit: usize,
        request: Option<RequestId>,
    ) -> Result<EventQuery> {
        let collected = self.retained_snapshot(true, false)?;
        let mut query = collected
            .window
            .query(after, limit, request, self.events.dropped())?;
        query.session = Some(format!("{:016x}", self.observations.namespace));
        Ok(query)
    }
    fn retained_snapshot(
        &mut self,
        events: bool,
        parents: bool,
    ) -> Result<collector::CollectedSnapshot> {
        self.collect_observations();
        if let Some(collector) = &self.observations.collector {
            collector
                .snapshot(self.events.published(), events, parents)?
                .finish()
        } else {
            Ok(collector::CollectedSnapshot {
                window: self.observations.store.snapshot(events),
                parents: if parents {
                    self.observations.parents.clone()
                } else {
                    BTreeMap::new()
                },
                dropped_metadata: 0,
            })
        }
    }
    pub fn diagnostics(&mut self) -> Vec<Diagnostic> {
        match self.retained_snapshot(false, false) {
            Ok(snapshot) => snapshot.window.diagnostics(),
            Err(error) => vec![Diagnostic {
                sequence: 0,
                timestamp_us: self.now_us,
                code: DiagnosticCode::InvariantViolation,
                severity: Severity::Error,
                request: None,
                step: None,
                error,
                progress_epoch: self.global_progress_epoch,
                source: "crates/engine/runtime/src/observation/collector.rs".into(),
                checkpoint_available: false,
                resource_release_pending: !self.is_idle(),
            }],
        }
    }
    fn render_observation(&mut self, query: &infer_observe::ObservationQuery) -> Value {
        self.observe(query)
            .unwrap_or_else(|error| json!({"error":error}))
    }
    pub fn observability(&mut self) -> Value {
        self.render_observation(&infer_observe::ObservationQuery::Summary)
    }
    pub fn prometheus(&mut self) -> String {
        match self.observe(&infer_observe::ObservationQuery::Metrics) {
            Ok(Value::String(metrics)) => metrics,
            Ok(_) => "# Invalid metrics response\n".into(),
            Err(error) => format!(
                "# Collector error: {}\n",
                error.to_string().replace(['\n', '\r'], " ")
            ),
        }
    }
    pub fn timeline(&mut self) -> Value {
        self.render_observation(&infer_observe::ObservationQuery::Timeline)
    }
    pub fn otlp(&mut self) -> Value {
        self.render_observation(&infer_observe::ObservationQuery::Otlp)
    }
    /// # Errors
    /// Returns a collector failure while preserving the legacy cursor for retry.
    pub fn drain_legacy_events(&mut self) -> Result<Vec<SemanticEvent>> {
        let snapshot = self.retained_snapshot(true, false)?;
        let events = snapshot
            .window
            .timeline()
            .into_iter()
            .filter(|event| event.sequence > self.observations.legacy_cursor)
            .collect::<Vec<_>>();
        if let Some(last) = events.last() {
            self.observations.legacy_cursor = last.sequence;
        }
        Ok(events.into_iter().map(|event| event.event).collect())
    }
    fn resource_gauges(&self) -> ResourceGauges {
        let kv = self.backend.kv_cache();
        ResourceGauges {
            backend: self.backend.capabilities().backend_kind().label().into(),
            ready: self.fault.is_none(),
            active_requests: self
                .host
                .requests
                .values()
                .filter(|r| !r.status.terminal())
                .count(),
            inflight_steps: usize::from(self.inflight.is_some()),
            resource_waiters: self.resources_pending.len(),
            output_waiters: self.output_owners.len(),
            output_batches: self.output_pending.len(),
            logical_pages: self.state.allocated_pages(),
            free_logical_pages: self.state.free_pages(),
            active_kv_blocks: kv.as_ref().map_or(0, |kv| kv.active_blocks),
            cached_kv_blocks: kv.as_ref().map_or(0, |kv| kv.cached_blocks),
            available_kv_blocks: kv.as_ref().map_or(0, |kv| kv.available_blocks),
            kv_pool_bytes: kv.as_ref().map_or(0, |kv| kv.pool_bytes),
            event_ring_dropped: self.events.dropped(),
            collector_metadata_dropped: self
                .observations
                .collector
                .as_ref()
                .map_or(0, collector::Collector::dropped_metadata),
        }
    }
    pub(crate) fn retain_diagnostic(
        &mut self,
        code: DiagnosticCode,
        error: &Error,
        request: Option<RequestId>,
        step: Option<StepId>,
    ) {
        let diagnostic = Diagnostic {
            sequence: 0,
            timestamp_us: self.now_us,
            code,
            severity: if matches!(
                code,
                DiagnosticCode::AdmissionRejected
                    | DiagnosticCode::OutputBackpressure
                    | DiagnosticCode::DeadlineExceeded
            ) {
                Severity::Warning
            } else {
                Severity::Error
            },
            request,
            step,
            error: error.clone(),
            progress_epoch: self.global_progress_epoch,
            source: if code == DiagnosticCode::SubmissionFailed {
                "crates/engine/runtime/src/pipeline/dispatch.rs"
            } else {
                "crates/engine/runtime/src/engine/mod.rs"
            }
            .into(),
            checkpoint_available: self.diagnostic_snapshot.is_some(),
            resource_release_pending: self.inflight.is_some()
                || !self.resources_pending.is_empty()
                || !self.output_pending.is_empty()
                || self.backend.pending_resource_releases()
                || (self.fault.is_some() && !self.is_idle()),
        };
        if let Some(collector) = &self.observations.collector {
            collector.diagnostic(diagnostic);
        } else {
            self.observations.store.diagnostic(diagnostic);
        }
    }
    pub(crate) fn isolate(&mut self, code: DiagnosticCode, error: &Error) {
        if self.fault.is_some() {
            return;
        }
        self.fault = Some(EngineFault {
            error: error.clone(),
            since_us: self.now_us,
        });
        self.diagnostic_snapshot = Some(self.snapshot_for_diagnostic());
        let step = self.inflight.as_ref().map(|flight| flight.step.id);
        self.retain_diagnostic(code, error, None, step);
        self.event(
            EventKind::Quarantined,
            ObjectKind::Step,
            step.map_or(0, StepId::get),
            0,
            self.global_progress_epoch,
            0,
        );
    }
    pub fn record_backpressure(&mut self, request: RequestId) {
        self.event(
            EventKind::Backpressure,
            ObjectKind::Request,
            request.get(),
            0,
            0,
            0,
        );
        self.retain_diagnostic(
            DiagnosticCode::OutputBackpressure,
            &Error::new(
                infer_core::ErrorCode::Capacity,
                "output stream is full or disconnected",
            ),
            Some(request),
            None,
        );
    }
}

#[cfg(test)]
#[path = "../../tests/unit/observation.rs"]
mod tests;
