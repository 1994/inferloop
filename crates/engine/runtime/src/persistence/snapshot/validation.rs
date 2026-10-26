//! Validation responsibilities.
use super::{Engine, RequestRecord, RuntimeSnapshot, validate_active_record};
use crate::preparation::validate_plan;
use infer_core::{Error, RequestId, RequestStatus, Result};
use infer_ir::RequestInput;
use infer_scheduler::CostAwarePolicy;
use infer_spi::{
    AdmissionPolicy, BackendProvider, CostModelProvider, SchedulingPolicy, WorkloadProvider,
};

impl RuntimeSnapshot {
    pub(super) fn validate_compatibility(
        &self,
        backend: &impl BackendProvider,
        workloads: &impl WorkloadProvider,
        costs: &impl CostModelProvider,
        admission: &impl AdmissionPolicy,
    ) -> Result<()> {
        if self
            .requests
            .values()
            .any(|record| record.constraint.is_some())
        {
            return Err(Error::unsupported(
                "live generation constraints cannot be restored",
            ));
        }
        if self.fault.is_some() {
            return Err(Error::invalid(
                "cannot restore a quarantined diagnostic snapshot",
            ));
        }
        if self.policy_identity != CostAwarePolicy.identity()
            || !self.checkpoint_supported
            || !backend.supports_control_checkpoint()
        {
            return Err(Error::unsupported(
                "snapshot policy or backend checkpoint capability differs",
            ));
        }
        if self.inflight.is_some() {
            return Err(Error::unsupported(
                "in-flight GPU tickets cannot be restored",
            ));
        }
        if self.backend_identity != backend.identity() {
            return Err(Error::invalid(
                "snapshot backend/weights fingerprint mismatch",
            ));
        }
        if self.workload_identity != workloads.identity() {
            return Err(Error::invalid("snapshot workload provider differs"));
        }
        if self.cost_provider_identity != costs.identity() {
            return Err(Error::invalid("snapshot cost provider differs"));
        }
        if self.admission_provider_identity != admission.identity() {
            return Err(Error::invalid("snapshot admission provider differs"));
        }
        Ok(())
    }
    pub(super) fn validate_capacities(&self) -> Result<()> {
        self.state.check_invariants()?;
        self.queues.check_invariants()?;
        if self.priority_epoch > self.priority_sweep_epoch
            || self.priority_sweep_epoch > self.cost_epoch
            || (self.priority_cursor.is_some() && self.priority_sweep_epoch == self.priority_epoch)
        {
            return Err(Error::invalid("snapshot priority refresh epochs mismatch"));
        }
        self.ids.validate_after(
            self.decisions
                .iter()
                .flat_map(|d| [d.id.get(), d.step.as_ref().map_or(0, |s| s.id.get())])
                .max()
                .unwrap_or(0),
        )?;
        if !self.guard.valid_for(self.config.progress_limit) {
            return Err(Error::invalid(
                "snapshot progress guard differs from configuration",
            ));
        }
        if self.state.snapshot().total_pages != self.config.num_gpu_blocks
            || self.state.block_size() != self.config.block_size
            || self.requests.len() > self.config.max_requests
            || self.decisions.len() > self.config.history_capacity
            || self.actions.len() > self.config.history_capacity
            || self.actions.iter().fold(0usize, |count, action| {
                count.saturating_add(action.retained_tokens())
            }) > self.config.max_history_tokens
            || self.actions.iter().fold(0usize, |bytes, action| {
                bytes.saturating_add(action.retained_bytes())
            }) > self.config.max_history_bytes
            || self.pending_cost_observations.len() > self.config.history_capacity
        {
            return Err(Error::invalid("snapshot capacities mismatch"));
        }
        Ok(())
    }
}

impl<B: BackendProvider> Engine<B, CostAwarePolicy> {
    pub(super) fn validate_checkpoint_record(
        &self,
        id: RequestId,
        record: &RequestRecord,
        snapshot: &RuntimeSnapshot,
    ) -> Result<()> {
        if !matches!(&record.request.input, RequestInput::Sequence { media, .. } if media.is_empty())
            || record.media.is_some()
        {
            return Err(Error::unsupported(
                "multimodal requests cannot be checkpointed while media state is live",
            ));
        }
        let expected =
            self.workloads
                .plan(&record.request, &snapshot.model, snapshot.program.id)?;
        // A restored record must still satisfy the length limits resolved for this engine, so a
        // checkpoint cannot reintroduce a request the current deployment would reject.
        validate_plan(
            &record.request,
            &record.plan,
            &snapshot.model,
            snapshot.program.id,
            &snapshot.config,
            &self.limits,
        )?;
        if id != record.request.id
            || record.plan != expected
            || record.unit > record.plan.units.len()
            || record.context.len() > record.plan.reserved_tokens
            || record.accepted_us > snapshot.now_us
            || record.progress_epoch > snapshot.global_progress_epoch
            || record.pending_finish.is_some()
            || record.last_service_us < record.accepted_us
            || record.last_service_us > snapshot.now_us
            || record.admission.rejection.is_some()
            || matches!(record.status, RequestStatus::Running { .. })
        {
            return Err(Error::invalid("invalid snapshot request lifecycle/cursor"));
        }
        if !record.status.terminal() {
            validate_active_record(record)?;
        }
        if record
            .generated
            .iter()
            .any(|t| *t as usize >= snapshot.model.vocab_size)
            || record
                .first_token_us
                .is_some_and(|t| t < record.accepted_us || t > snapshot.now_us)
            || record
                .last_token_us
                .is_some_and(|t| t < record.accepted_us || t > snapshot.now_us)
        {
            return Err(Error::invalid("invalid snapshot tokens/timestamps"));
        }
        if snapshot
            .tenants
            .get(record.request.qos.tenant.as_str())
            .is_none_or(|t| t.weight != record.request.qos.weight)
        {
            return Err(Error::invalid("snapshot tenant metadata mismatch"));
        }
        Ok(())
    }
}
