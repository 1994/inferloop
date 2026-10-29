//! Cancellation and terminal ownership converge through one finish operation.
use crate::{CompletedRequest, Engine, EngineOutput};
use infer_core::{
    Error, ErrorCode, FinishReason, RequestId, RequestStatus, Result, event::EventKind,
    event::ObjectKind,
};
use infer_ir::WorkloadOutput;
use infer_observe::DiagnosticCode;
use infer_quality::RequestMeasurement;
use infer_spi::{BackendProvider, SchedulingPolicy};
use std::mem::take;

impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    /// # Errors
    /// Returns a not-found error for an unknown request or a backend/state error if cancellation cleanup fails.
    pub fn cancel(&mut self, id: RequestId) -> Result<Vec<EngineOutput>> {
        let mut emitted = Vec::new();
        self.cancel_into(id, &mut emitted)?;
        Ok(emitted)
    }
    /// Reuse delivery storage during ordinary cancellation; state remains leased until all fences settle.
    /// # Errors
    /// Returns missing request or ownership/backend errors.
    pub fn cancel_into(&mut self, id: RequestId, emitted: &mut Vec<EngineOutput>) -> Result<()> {
        let record = self.request(id)?;
        if record.status.terminal() || record.pending_finish.is_some() {
            return Ok(());
        }
        self.host.queues.cancel(id)?;
        self.record_action(crate::ReplayAction::Cancel(id));
        self.event(EventKind::Cancelled, ObjectKind::Request, id.get(), 0, 0, 0);
        emitted.extend(self.request_termination(id, FinishReason::Cancelled)?);
        Ok(())
    }

    pub(crate) fn request_termination(
        &mut self,
        id: RequestId,
        reason: FinishReason,
    ) -> Result<Option<EngineOutput>> {
        if self
            .resources_pending
            .get(id)
            .is_some_and(|pending| pending.ticket.is_none())
        {
            // No command was published: release can safely supersede the deferred reset.
            self.resources_pending.remove(id);
        }
        let record = self
            .host
            .requests
            .get_mut(id)
            .ok_or_else(|| Error::invariant("known request"))?;
        if !matches!(record.status, RequestStatus::Running { .. })
            && !self.resources_pending.contains_key(id)
            && !self.output_owners.contains_key(&id)
        {
            return self.finish(id, reason, None).map(Some);
        }
        let progress = record.pending_finish.is_none() || reason == FinishReason::Cancelled;
        record.pending_finish.get_or_insert(reason);
        if progress {
            self.progress(id)?;
        }
        Ok(None)
    }

    pub(crate) fn expire_requests(&mut self, emitted: &mut Vec<EngineOutput>) -> Result<()> {
        let mut ids = take(&mut self.host.expiry_buffer);
        self.host
            .queues
            .expired_into(self.now_us, self.config.cpu.maintenance_items, &mut ids);
        let result = ids.iter().copied().try_for_each(|id| {
            let reason = if self
                .host
                .requests
                .known(id)?
                .request
                .qos
                .deadline_us
                .is_some_and(|deadline| deadline <= self.now_us)
            {
                FinishReason::Deadline
            } else {
                FinishReason::Failed("scheduler stage queue wait timed out".into())
            };
            self.host.queues.cancel(id)?;
            emitted.extend(self.request_termination(id, reason)?);
            Ok(())
        });
        self.host.expiry_buffer = ids;
        result
    }

    pub(crate) fn drain_quarantined_requests(
        &mut self,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        let Some(failure) = self.fault.as_ref().map(|fault| fault.error.to_string()) else {
            return Ok(());
        };
        let active: Vec<_> = self
            .host
            .requests
            .iter()
            .filter(|(_, record)| !record.status.terminal())
            .map(|(id, _)| *id)
            .collect();
        for id in active {
            let step = self
                .inflight
                .as_ref()
                .filter(|flight| flight.step.work.iter().any(|work| work.request == id))
                .map(|flight| flight.step.id);
            if let Some(step) = step {
                self.host.queues.cancel(id)?;
                let record = self
                    .host
                    .requests
                    .get_mut(id)
                    .ok_or_else(|| Error::invariant("known request"))?;
                record.status = RequestStatus::Running { step };
                record.pending_finish = Some(FinishReason::Failed(failure.clone()));
            } else {
                let reason = FinishReason::Failed(failure.clone());
                if self.resources_pending.contains_key(id) || self.output_owners.contains_key(&id) {
                    if self.host.queues.state(id).is_some() {
                        self.host.queues.cancel(id)?;
                    }
                    emitted.extend(self.request_termination(id, reason)?);
                } else {
                    emitted.push(self.finish(id, reason, None)?);
                }
            }
        }
        Ok(())
    }
    pub(crate) fn finish(
        &mut self,
        id: RequestId,
        reason: FinishReason,
        mut output: Option<WorkloadOutput>,
    ) -> Result<EngineOutput> {
        let r = self
            .host
            .requests
            .get_mut(id)
            .ok_or_else(|| Error::invariant("known request"))?;
        if r.status.terminal() {
            return Err(Error::invariant("request completed twice"));
        }
        let state = r
            .state
            .ok_or_else(|| Error::invariant("active request missing state"))?;
        self.backend.release_state(state)?;
        if let Some(
            infer_scheduler::QueueState::Running { step }
            | infer_scheduler::QueueState::CancelPending { step },
        ) = self.host.queues.state(id)
        {
            self.host.queues.complete(id, step)?;
        }
        self.host.queues.remove(id)?;
        self.state.release(state)?;
        let tenant = self
            .tenants
            .get_mut(r.request.qos.tenant.as_str())
            .ok_or_else(|| Error::invariant("finishing tenant disappeared"))?;
        tenant.active = tenant
            .active
            .checked_sub(1)
            .ok_or_else(|| Error::invariant("tenant active count underflow"))?;
        tenant.tokens = tenant
            .tokens
            .checked_sub(r.plan.reserved_tokens)
            .ok_or_else(|| Error::invariant("tenant token count underflow"))?;
        tenant.pages = tenant
            .pages
            .checked_sub(r.plan.reserved_tokens.div_ceil(self.config.block_size))
            .ok_or_else(|| Error::invariant("tenant page count underflow"))?;
        r.state = None;
        r.status
            .transition(RequestStatus::Finished(reason.clone()))?;
        r.pending_finish = None;
        r.constraint = None;
        r.outputs.clear();
        if let Some(output) = &mut output {
            output.attach_credit(
                r.byte_lease
                    .clone()
                    .ok_or_else(|| Error::invariant("result byte lease missing"))?,
            );
        }
        let successful = !matches!(
            reason,
            FinishReason::Cancelled | FinishReason::Deadline | FinishReason::Failed(_)
        );
        let completed = CompletedRequest {
            request: id,
            reason,
            output,
            measurement: RequestMeasurement {
                ttft_us: r.first_token_us.map(|t| t - r.accepted_us),
                max_tpot_us: r.max_tpot_us,
                e2e_us: self.now_us - r.accepted_us,
                output_tokens: r.generated.len(),
                successful,
                cached_tokens: r.cached_tokens,
            },
        };
        log_finish(&completed);
        r.completed = Some(completed.clone());
        self.resources_changed()?;
        if self.preemption_focus == Some(id) {
            self.preemption_focus = None;
        }
        self.event(
            EventKind::StateReleased,
            ObjectKind::State,
            state.get(),
            id.get(),
            0,
            0,
        );
        self.event(
            EventKind::Finished,
            ObjectKind::Request,
            id.get(),
            0,
            infer_observe::finish_code(&completed.reason),
            completed.measurement.e2e_us,
        );
        if completed.reason == FinishReason::Deadline {
            self.retain_diagnostic(
                DiagnosticCode::DeadlineExceeded,
                &Error::new(ErrorCode::Capacity, "request deadline expired"),
                Some(id),
                None,
            );
        }
        self.progress(id)?;
        Ok(EngineOutput::Finished(completed))
    }
    /// Client termination can be delivered before a GPU fence; the record and KV stay owned.
    #[must_use]
    pub fn pending_terminal(&self, id: RequestId) -> Option<CompletedRequest> {
        let record = self.host.requests.get(id)?;
        let reason = record.pending_finish.clone()?;
        Some(CompletedRequest {
            request: id,
            reason,
            output: None,
            measurement: RequestMeasurement {
                ttft_us: record.first_token_us.map(|t| t - record.accepted_us),
                max_tpot_us: record.max_tpot_us,
                e2e_us: self.now_us.saturating_sub(record.accepted_us),
                output_tokens: record.generated.len(),
                successful: false,
                cached_tokens: record.cached_tokens,
            },
        })
    }
    /// Results count against admission until consumed, providing bounded backpressure.
    ///
    /// # Errors
    /// Returns a not-found error for an unknown request or an invariant error if completion ownership is inconsistent.
    pub fn take_completed(&mut self, id: RequestId) -> Result<Option<CompletedRequest>> {
        if !self.request(id)?.status.terminal() {
            return Ok(None);
        }
        let record = self
            .host
            .requests
            .remove(id)
            .ok_or_else(|| Error::invariant("known request"))?;
        if let Some(collector) = &mut self.observations.collector {
            if self.observations.parents.remove(&id.get()).is_some() {
                collector.retire(id.get(), self.events.published());
            }
        } else {
            self.observations.parents_dirty = true;
        }
        self.record_action(crate::ReplayAction::Drain(id));
        let tenant = self
            .tenants
            .get_mut(record.request.qos.tenant.as_str())
            .ok_or_else(|| Error::invariant("draining tenant disappeared"))?;
        tenant.owners = tenant
            .owners
            .checked_sub(1)
            .ok_or_else(|| Error::invariant("tenant owner count underflow"))?;
        if tenant.owners == 0 {
            self.tenants.remove(record.request.qos.tenant.as_str());
        }
        Ok(record.completed)
    }
}

/// Report a finished request with the measurement an operator reads: how many tokens were
/// produced and how long the request took end to end.
fn log_finish(completed: &CompletedRequest) {
    let measurement = &completed.measurement;
    if measurement.successful {
        tracing::info!(
            target: "infer::request",
            request = completed.request.get(),
            reason = ?completed.reason,
            output_tokens = measurement.output_tokens,
            ttft_us = measurement.ttft_us,
            e2e_us = measurement.e2e_us,
            "request finished"
        );
    } else {
        tracing::warn!(
            target: "infer::request",
            request = completed.request.get(),
            reason = ?completed.reason,
            output_tokens = measurement.output_tokens,
            e2e_us = measurement.e2e_us,
            "request finished unsuccessfully"
        );
    }
}
