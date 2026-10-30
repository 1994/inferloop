//! Inspection responsibilities.
use super::{Engine, RequestRecord, RuntimeConfig, RuntimeInspection};
use infer_core::{Error, ErrorCode, RequestId, Result};
use infer_ir::{ExecutionProgram, ModelIr, SchedulingDecision};
use infer_spi::{BackendProvider, SchedulingPolicy, WorkloadProvider};
use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::thread::available_parallelism;

impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    /// Largest accepted or retired identity, including restored request history.
    #[must_use]
    pub fn request_id_high_watermark(&self) -> u64 {
        self.seen_requests
            .last()
            .map_or(self.retired_request_floor, |id| {
                id.get().max(self.retired_request_floor)
            })
    }
    pub const fn program(&self) -> &ExecutionProgram {
        &self.program
    }
    pub const fn model(&self) -> &ModelIr {
        &self.model
    }
    pub fn set_waker(&mut self, wake: Arc<dyn Fn() + Send + Sync>) {
        self.backend.set_waker(wake.clone());
        self.host.credits.set_waker(wake.clone());
        self.host.bytes.set_waker(wake.clone());
        if let Some(worker) = &self.output_worker {
            worker.set_waker(wake);
        }
    }
    pub const fn backend(&self) -> &B {
        &self.backend
    }
    pub const fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
    pub fn workload_identity(&self) -> &str {
        self.workloads.identity()
    }
    /// # Errors
    /// Returns an unsupported error if the installed workload provider cannot be forked.
    pub fn fork_workloads(&self) -> Result<Box<dyn WorkloadProvider + Send + Sync>> {
        self.workloads.fork().ok_or_else(|| {
            Error::unsupported("workload provider cannot fork for isolated quality/replay")
        })
    }
    pub const fn config(&self) -> &RuntimeConfig {
        &self.config
    }
    /// Effective single-sequence length limits, resolved once at construction.
    pub const fn length_limits(&self) -> crate::ResolvedLengthLimits {
        self.limits
    }
    pub const fn now_us(&self) -> u64 {
        self.now_us
    }
    pub fn request_records(&self) -> impl Iterator<Item = &RequestRecord> {
        self.host.requests.values()
    }
    pub fn is_idle(&self) -> bool {
        self.inflight.is_none()
            && self.resources_pending.is_empty()
            && self.output_pending.is_empty()
            && !self.backend.pending_resource_releases()
            && self.host.requests.values().all(|r| r.status.terminal())
    }
    pub fn inspect(&self) -> RuntimeInspection {
        RuntimeInspection {
            state_recipe: self.backend.state_recipe().cloned(),
            ready: self.fault.is_none(),
            fault: self.fault.clone(),
            resource_release_pending: !self.resources_pending.is_empty()
                || !self.output_pending.is_empty()
                || self.backend.pending_resource_releases()
                || self
                    .host
                    .requests
                    .values()
                    .any(|r| r.pending_finish.is_some())
                || self.fault.is_some() && !self.is_idle(),
            resource_waiters: self.resources_pending.len(),
            output_waiters: self.output_owners.len(),
            output_batches: self.output_pending.len(),
            backend_kind: self.backend.capabilities().backend_kind(),
            weight_backed_dataflow: self.backend.weight_backed_dataflow(),
            backend: self.backend.identity().into(),
            model: self.model.id,
            model_name: self
                .config
                .served_model_name
                .clone()
                .unwrap_or_else(|| self.model.id.to_string()),
            program: self.program.id,
            active_requests: self
                .host
                .requests
                .values()
                .filter(|r| !r.status.terminal())
                .count(),
            completed_requests: self
                .host
                .requests
                .values()
                .filter(|r| r.status.terminal())
                .count(),
            inflight_step: self.inflight.as_ref().map(|s| s.step.id),
            global_progress_epoch: self.global_progress_epoch,
            state: self.state.snapshot(),
            dropped_events: self.events.dropped(),
            dropped_actions: self.dropped_actions,
            cost_model: self.costs.inspect(),
            pending_cost_observations: self.pending_cost_observations.len(),
            preemptions: self.host.requests.values().map(|r| r.preemptions).sum(),
            kv_cache: self.backend.kv_cache(),
            execution_profile: self.backend.execution_profile(),
            scheduler: self.config.scheduler.clone(),
            lengths: self.limits,
            queues: self.host.queues.inspect(),
            cpu: crate::CpuRuntimeInspection {
                config: self.config.cpu.clone(),
                configured_storage_bytes: self
                    .config
                    .cpu
                    .fixed_bytes_for(&self.config, self.model.vocab_size)
                    .unwrap_or(usize::MAX),
                retained_host_tokens: self.host.credits.used(),
                retained_host_bytes: self.host.bytes.used(),
                candidate_capacity: self.config.candidate_limit.min(self.config.max_requests),
                logical_cpus: available_parallelism().map_or(1, NonZeroUsize::get),
                placement: self.host.placement.clone(),
                prepared_next: self.host.prepared_next.is_some(),
            },
        }
    }
    /// # Errors
    /// Returns a not-found error for an unknown or retired request identity.
    pub fn request(&self, id: RequestId) -> Result<&RequestRecord> {
        self.host
            .requests
            .get(id)
            .ok_or_else(|| Error::new(ErrorCode::NotFound, "unknown request"))
    }
    /// # Errors
    /// Returns a not-found error for an unknown request or a serialization error for invalid inspection data.
    pub fn explain(&self, id: RequestId) -> Result<serde_json::Value> {
        let request = self.request(id)?;
        let last = self
            .decisions
            .iter()
            .rev()
            .find_map(|d| d.deferred.iter().find(|w| w.request == id));
        let selected = self
            .decisions
            .iter()
            .rev()
            .find_map(|d| d.selected.iter().find(|w| w.request == id));
        let latest = self.decisions.back();
        let coverage = latest.map(|decision| {
            if decision.selected.iter().any(|work| work.request == id) {
                "selected"
            } else if decision.deferred.iter().any(|work| work.request == id) {
                "deferred"
            } else {
                "outside_candidate_window"
            }
        });
        Ok(
            serde_json::json!({"decision_window":latest.and_then(|d| d.window.as_ref()), "latest_window_status":coverage, "request":id,"status":request.status,"progress_epoch":request.progress_epoch,"state":request.state,"program":request.plan.program,"qos":request.request.qos,"admission":request.admission,"last_selection":selected,"last_deferral":last,"cost_model":self.costs.inspect(),"source":"crates/engine/runtime/src/engine/inspection.rs"}),
        )
    }
    pub const fn decisions(&self) -> &VecDeque<SchedulingDecision> {
        &self.decisions
    }
}
