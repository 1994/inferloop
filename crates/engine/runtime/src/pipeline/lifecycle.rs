//! Cancellation and terminal ownership converge through one finish operation.
use crate::{CompletedRequest, Engine, EngineOutput, Termination};
use infer_core::{
    CancelSource, Error, ErrorCode, FinishReason, RequestId, RequestStage, RequestStatus, Result,
    event::EventKind, event::ObjectKind, event::SemanticEvent,
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
        self.cancel_from(id, CancelSource::Requested)
    }
    /// Cancel one request and record who stopped wanting it.
    ///
    /// The source changes nothing about execution; it reaches the request's terminal log line and
    /// the cancellation metrics, so a disconnected client is distinguishable from a shutdown.
    /// # Errors
    /// Returns a not-found error for an unknown request or a backend/state error if cancellation cleanup fails.
    pub fn cancel_from(
        &mut self,
        id: RequestId,
        source: CancelSource,
    ) -> Result<Vec<EngineOutput>> {
        let mut emitted = Vec::new();
        self.cancel_from_into(id, source, &mut emitted)?;
        Ok(emitted)
    }
    /// Reuse delivery storage during ordinary cancellation; state remains leased until all fences settle.
    /// # Errors
    /// Returns missing request or ownership/backend errors.
    pub fn cancel_into(&mut self, id: RequestId, emitted: &mut Vec<EngineOutput>) -> Result<()> {
        self.cancel_from_into(id, CancelSource::Requested, emitted)
    }
    /// [`Self::cancel_from`] reusing delivery storage.
    /// # Errors
    /// Returns missing request or ownership/backend errors.
    pub fn cancel_from_into(
        &mut self,
        id: RequestId,
        source: CancelSource,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        let record = self.request(id)?;
        if record.status.terminal() || record.pending_finish.is_some() {
            return Ok(());
        }
        if let Some(record) = self.host.requests.get_mut(id) {
            record.termination.cancel = Some(source);
        }
        self.host.queues.cancel(id)?;
        self.record_action(crate::ReplayAction::Cancel(id));
        self.event(EventKind::Cancelled, ObjectKind::Request, id.get(), 0, 0, 0);
        emitted.extend(self.request_termination(id, FinishReason::Cancelled)?);
        Ok(())
    }

    /// A failure ending that keeps its error code for the terminal line; the message stays the
    /// one `FinishReason::Failed` has always carried. An ending already decided, such as a cancel
    /// waiting on its fence, keeps its own cause.
    pub(crate) fn failure(
        &mut self,
        id: RequestId,
        code: ErrorCode,
        message: impl Into<String>,
    ) -> FinishReason {
        if let Some(record) = self.host.requests.get_mut(id)
            && record.pending_finish.is_none()
        {
            record.termination.error.get_or_insert(code);
        }
        FinishReason::Failed(message.into())
    }
    /// Remember where the request was when its ending was first decided. A deferred ending
    /// finishes later, after the stage that was running has already handed the request back.
    fn mark_stage(&mut self, id: RequestId) {
        let output = self.output_owners.contains_key(&id);
        if let Some(record) = self.host.requests.get_mut(id)
            && record.termination.stage.is_none()
        {
            record.termination.stage = Some(if output {
                RequestStage::Output
            } else if record.termination.first_dispatch_us.is_none() {
                RequestStage::Queued
            } else if record.first_token_us.is_none() {
                RequestStage::Prefill
            } else {
                RequestStage::Decode
            });
        }
    }

    /// Remember the stage of a request whose step failed while being handed to the backend.
    pub(crate) fn mark_dispatching(&mut self, id: RequestId) {
        if let Some(record) = self.host.requests.get_mut(id)
            && record.termination.stage.is_none()
        {
            record.termination.stage = Some(if record.first_token_us.is_none() {
                RequestStage::Prefill
            } else {
                RequestStage::Decode
            });
        }
    }

    pub(crate) fn request_termination(
        &mut self,
        id: RequestId,
        reason: FinishReason,
    ) -> Result<Option<EngineOutput>> {
        self.mark_stage(id);
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
                self.failure(
                    id,
                    ErrorCode::Capacity,
                    "scheduler stage queue wait timed out",
                )
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
        let Some((code, failure)) = self
            .fault
            .as_ref()
            .map(|fault| (fault.error.code, fault.error.to_string()))
        else {
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
                self.mark_stage(id);
                let reason = self.failure(id, code, failure.clone());
                let record = self
                    .host
                    .requests
                    .get_mut(id)
                    .ok_or_else(|| Error::invariant("known request"))?;
                record.status = RequestStatus::Running { step };
                record.pending_finish = Some(reason);
            } else {
                let reason = self.failure(id, code, failure.clone());
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
        self.mark_stage(id);
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
        let termination = r.termination;
        log_finish(&TerminalLine {
            completed: &completed,
            termination,
            queue_us: termination
                .first_dispatch_us
                .map(|t| t.saturating_sub(r.accepted_us)),
            last_token_age_us: r.last_token_us.map(|t| self.now_us.saturating_sub(t)),
        });
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
        self.report_finished(&completed, termination);
        self.progress(id)?;
        Ok(EngineOutput::Finished(completed))
    }
    /// Publish a finished request: its event carries the cancel source or error code so the
    /// per-cause counters need no second record, and an expired deadline is kept as a diagnostic.
    fn report_finished(&mut self, completed: &CompletedRequest, termination: Termination) {
        self.emit_semantic(SemanticEvent {
            timestamp_us: self.now_us,
            kind: EventKind::Finished,
            object_kind: ObjectKind::Request,
            reserved: infer_observe::finish_cause(
                &completed.reason,
                termination.cancel,
                termination.error,
            ),
            object_id: completed.request.get(),
            correlation_id: 0,
            arg0: infer_observe::finish_code(&completed.reason),
            arg1: completed.measurement.e2e_us,
        });
        if completed.reason == FinishReason::Deadline {
            self.retain_diagnostic(
                DiagnosticCode::DeadlineExceeded,
                &Error::new(ErrorCode::Capacity, "request deadline expired"),
                Some(completed.request),
                None,
            );
        }
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

/// What the one terminal line of a finished request reports.
struct TerminalLine<'a> {
    completed: &'a CompletedRequest,
    termination: Termination,
    /// Acceptance to first dispatch; absent for a request that never left its queue.
    queue_us: Option<u64>,
    /// Last token to the ending: a stalled request and a requester that just left differ here.
    last_token_age_us: Option<u64>,
}

/// Report a finished request exactly once, from the one place a request becomes terminal.
///
/// Field names are a contract (`docs/guides/serving-stability.md`): `phase`, `cancel`, `stage` and
/// `error_code` locate how and where it ended; the rest measure how far it got. Every field is a
/// static string or an integer, so the line adds no allocation to the successful path.
fn log_finish(line: &TerminalLine<'_>) {
    let completed = line.completed;
    let measurement = &completed.measurement;
    let cancel = line.termination.cancel.map_or("none", CancelSource::as_str);
    let stage = line.termination.stage.map_or("none", RequestStage::as_str);
    let error_code = line.termination.error.map_or("none", ErrorCode::as_str);
    if measurement.successful {
        tracing::info!(
            target: "infer::request",
            request = completed.request.get(),
            phase = "finished",
            reason = ?completed.reason,
            cancel,
            stage,
            error_code,
            output_tokens = measurement.output_tokens,
            cached_tokens = measurement.cached_tokens,
            ttft_us = measurement.ttft_us,
            max_tpot_us = measurement.max_tpot_us,
            e2e_us = measurement.e2e_us,
            queue_us = line.queue_us,
            last_token_age_us = line.last_token_age_us,
            "request finished"
        );
    } else {
        tracing::warn!(
            target: "infer::request",
            request = completed.request.get(),
            phase = "finished",
            reason = ?completed.reason,
            cancel,
            stage,
            error_code,
            output_tokens = measurement.output_tokens,
            cached_tokens = measurement.cached_tokens,
            ttft_us = measurement.ttft_us,
            max_tpot_us = measurement.max_tpot_us,
            e2e_us = measurement.e2e_us,
            queue_us = line.queue_us,
            last_token_age_us = line.last_token_age_us,
            "request finished unsuccessfully"
        );
    }
}
