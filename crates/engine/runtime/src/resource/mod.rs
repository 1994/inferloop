//! Preparation waiters retain state ownership until the actual resource acknowledgement.
mod waiters;
use crate::stages::prefill::retained_readout;
use crate::{Engine, EngineOutput};
use infer_core::event::EventKind;
use infer_core::event::ObjectKind;
use infer_core::event::SemanticEvent;
use infer_core::{Error, ErrorCode, FinishReason, RequestId, RequestStatus, Result};
use infer_scheduler::BlockedOn;
use infer_spi::{
    BackendProvider, ResourceCommand, ResourceReply, ResourceTicket, SchedulingPolicy,
};
use std::mem::take;
pub use waiters::ResourceWaiters;

#[derive(Clone, Copy)]
pub enum ResourcePhase {
    Reserve,
    Reset,
    Prefix,
}
pub struct PendingResource {
    pub ticket: Option<ResourceTicket>,
    pub phase: ResourcePhase,
    pub started: u64,
    /// Epoch at which the latest reservation attempt was issued. A rejected intent is
    /// retried only after resources or execution advance past this epoch, so progress
    /// that races with the acknowledgement is not discarded.
    pub retry_epoch: Option<u64>,
}
impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    pub(crate) fn check_resource_ownership(&self) -> Result<()> {
        for (id, pending) in self.resources_pending.iter() {
            if pending.ticket.is_none()
                && !matches!(pending.phase, ResourcePhase::Reset | ResourcePhase::Reserve)
            {
                return Err(Error::invariant(
                    "unpublished resource intent cannot be retried",
                ));
            }
            let record = self
                .host
                .requests
                .get(*id)
                .ok_or_else(|| Error::invariant("resource waiter has no request"))?;
            if !matches!(record.status, RequestStatus::Waiting(_))
                || record.state.is_none()
                || (record.pending_finish.is_none()
                    && !matches!(
                        self.host.queues.state(*id),
                        Some(infer_scheduler::QueueState::Blocked {
                            reason: BlockedOn::Preparation,
                            ..
                        })
                    ))
                || (record.pending_finish.is_some() && self.host.queues.state(*id).is_some())
            {
                return Err(Error::invariant("resource waiter ownership mismatch"));
            }
        }
        Ok(())
    }
    pub(crate) fn park_resource(
        &mut self,
        id: RequestId,
        ticket: ResourceTicket,
        phase: ResourcePhase,
    ) -> Result<()> {
        self.park_pending(
            id,
            PendingResource {
                ticket: Some(ticket),
                phase,
                // A reservation attempt is armed at the epoch it starts in. If its
                // rejection is observed after further progress, one retry must still
                // probe that newer epoch instead of waiting for the next one.
                retry_epoch: matches!(phase, ResourcePhase::Reserve).then_some(self.resource_epoch),
                started: self.now_us,
            },
        )
    }
    /// Park an intent that carries no live ticket. A reservation rejected because the
    /// resident budget is busy keeps host state ownership and must not re-issue until
    /// `retry_epoch` proves that resources or execution progressed.
    pub(crate) fn park_pending(&mut self, id: RequestId, pending: PendingResource) -> Result<()> {
        if self.resources_pending.contains_key(id) {
            return Err(Error::invariant("resource command already pending"));
        }
        self.host
            .queues
            .block(id, BlockedOn::Preparation, self.resource_epoch)?;
        self.host
            .requests
            .get_mut(id)
            .ok_or_else(|| Error::invariant("resource request disappeared"))?
            .status
            .transition(RequestStatus::Waiting(infer_core::WaitReason::BackendBusy))?;
        self.emit_semantic(SemanticEvent {
            timestamp_us: self.now_us,
            kind: EventKind::Deferred,
            object_kind: ObjectKind::Request,
            reserved: infer_ir::DeferReason::Preparation.code(),
            object_id: id.get(),
            correlation_id: 0,
            arg0: 1,
            arg1: 0,
        });
        self.resources_pending.insert(id, pending)?;
        Ok(())
    }
    pub(crate) fn reset_request_state(&mut self, id: RequestId) -> Result<()> {
        let state = self
            .host
            .requests
            .known(id)?
            .state
            .ok_or_else(|| Error::invariant("reset has no state"))?;
        let mut ticket = match self
            .backend
            .begin_resource(ResourceCommand::Reset { state })
        {
            Ok(ticket) => ticket,
            Err(error) if error.code == ErrorCode::Capacity => {
                self.park_deferred_reset(id)?;
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        match ticket.poll()? {
            Some(ResourceReply::Reset) => {
                self.state.reset(state)?;
                self.resources_changed()?;
            }
            Some(_) => return Err(Error::invariant("reset acknowledgement type mismatch")),
            None => self.park_resource(id, ticket, ResourcePhase::Reset)?,
        }
        Ok(())
    }
    fn park_deferred_reset(&mut self, id: RequestId) -> Result<()> {
        self.host
            .queues
            .block(id, BlockedOn::Preparation, self.resource_epoch)?;
        self.host
            .requests
            .get_mut(id)
            .ok_or_else(|| Error::invariant("reset owner lost"))?
            .status
            .transition(RequestStatus::Waiting(infer_core::WaitReason::BackendBusy))?;
        self.resources_pending.insert(
            id,
            PendingResource {
                ticket: None,
                phase: ResourcePhase::Reset,
                retry_epoch: None,
                started: self.now_us,
            },
        )?;
        Ok(())
    }
    fn resource_reply(&mut self, id: RequestId) -> Result<Option<ResourceReply>> {
        let pending = self
            .resources_pending
            .get_mut(id)
            .ok_or_else(|| Error::invariant("resource waiter lost"))?;
        if pending.ticket.is_none() {
            if pending.retry_epoch == Some(self.resource_epoch) {
                return Ok(None);
            }
            let record = self.host.requests.known(id)?;
            let state = record
                .state
                .ok_or_else(|| Error::invariant("resource state lost"))?;
            let command = match pending.phase {
                ResourcePhase::Reserve => {
                    let capacity = record.plan.reserved_tokens;
                    let readout = retained_readout(&record.request.workload);
                    if record.media.is_some() {
                        ResourceCommand::ReserveMedia {
                            state,
                            capacity,
                            readout,
                        }
                    } else {
                        ResourceCommand::Reserve {
                            state,
                            capacity,
                            readout,
                        }
                    }
                }
                ResourcePhase::Reset => ResourceCommand::Reset { state },
                ResourcePhase::Prefix => {
                    return Err(Error::invariant("prefix intent has no ticket"));
                }
            };
            match self.backend.begin_resource(command) {
                Ok(ticket) => {
                    // A retry after resource progress starts a fresh acknowledgement wait.
                    if pending.retry_epoch.take().is_some() {
                        pending.started = self.now_us;
                    }
                    // Re-arm at the epoch this attempt starts in; its rejection may be
                    // observed only after a later progress epoch.
                    if matches!(pending.phase, ResourcePhase::Reserve) {
                        pending.retry_epoch = Some(self.resource_epoch);
                    }
                    pending.ticket = Some(ticket);
                }
                Err(error) if error.code == ErrorCode::Capacity => {
                    // A synchronous rejection is also an attempt at this epoch. Re-arm it
                    // so a full control lane is not polled again until progress advances.
                    if pending.retry_epoch.take().is_some() {
                        pending.started = self.now_us;
                    }
                    if matches!(pending.phase, ResourcePhase::Reserve) {
                        pending.retry_epoch = Some(self.resource_epoch);
                    }
                    return Ok(None);
                }
                Err(error) => return Err(error),
            }
        }
        pending
            .ticket
            .as_mut()
            .ok_or_else(|| Error::invariant("resource ticket lost"))?
            .poll()
    }
    pub(crate) fn poll_resources(&mut self, emitted: &mut Vec<EngineOutput>) -> Result<()> {
        let mut ids = take(&mut self.host.resource_poll_buffer);
        self.resources_pending
            .poll_window(self.config.cpu.maintenance_items, &mut ids)?;
        let result = ids
            .iter()
            .copied()
            .try_for_each(|id| self.poll_resource(id, emitted));
        self.host.resource_poll_buffer = ids;
        result
    }
    fn poll_resource(&mut self, id: RequestId, emitted: &mut Vec<EngineOutput>) -> Result<()> {
        let mut result = self.resource_reply(id);
        // A quote can race with other reservations on the backend owner. Capacity
        // is backpressure, not a failed generation. Keep the request and its host
        // state, but publish no retry until execution or resource ownership changes.
        if matches!(&result, Err(error) if error.code == ErrorCode::Capacity)
            && matches!(
                self.resources_pending.get(id).map(|p| p.phase),
                Some(ResourcePhase::Reserve)
            )
            && self.host.requests.known(id)?.pending_finish.is_none()
        {
            let pending = self
                .resources_pending
                .get_mut(id)
                .ok_or_else(|| Error::invariant("resource waiter lost"))?;
            pending.ticket = None;
            // Keep the epoch of the attempt that produced this rejection. If progress
            // happened while the reply was in flight, the next poll probes that newer
            // epoch instead of discarding the progress.
            pending.retry_epoch.get_or_insert(self.resource_epoch);
            result = Ok(None);
        }
        if matches!(result, Ok(None)) {
            let pending = self
                .resources_pending
                .get(id)
                .ok_or_else(|| Error::invariant("resource waiter lost"))?;
            if self.now_us.saturating_sub(pending.started) >= self.config.resource_timeout_us
                && self.host.requests.known(id)?.pending_finish.is_none()
            {
                self.retain_diagnostic(
                    infer_observe::DiagnosticCode::SubmissionTimeout,
                    &Error::new(
                        ErrorCode::Backend,
                        "resource preparation acknowledgement timed out",
                    ),
                    Some(id),
                    None,
                );
                self.host.queues.cancel(id)?;
                emitted.extend(self.request_termination(
                    id,
                    FinishReason::Failed("resource preparation acknowledgement timed out".into()),
                )?);
            }
            return Ok(());
        }
        let pending = self
            .resources_pending
            .remove(id)
            .ok_or_else(|| Error::invariant("resource acknowledgement lost owner"))?;
        self.event(
            EventKind::ResourceAcknowledged,
            ObjectKind::Request,
            id.get(),
            0,
            pending.phase as u64,
            self.now_us.saturating_sub(pending.started),
        );
        if let Some(reason) = self
            .host
            .requests
            .get_mut(id)
            .and_then(|record| record.pending_finish.take())
        {
            emitted.push(self.finish(id, reason, None)?);
            return Ok(());
        }
        match result {
            Ok(Some(reply)) => self.complete_resource(id, pending.phase, reply)?,
            Err(error) => {
                emitted.push(self.finish(id, FinishReason::Failed(error.to_string()), None)?);
                if matches!(error.code, ErrorCode::Backend | ErrorCode::Invariant) {
                    self.isolate(infer_observe::DiagnosticCode::SubmissionFailed, &error);
                }
            }
            Ok(None) => return Err(Error::invariant("pending acknowledgement changed")),
        }
        Ok(())
    }
    fn complete_resource(
        &mut self,
        id: RequestId,
        phase: ResourcePhase,
        reply: ResourceReply,
    ) -> Result<()> {
        let state = self
            .host
            .requests
            .known(id)?
            .state
            .ok_or_else(|| Error::invariant("resource state lost"))?;
        match (phase, reply) {
            (ResourcePhase::Reserve, ResourceReply::Reserved) => {}
            (ResourcePhase::Reset, ResourceReply::Reset) => {
                self.state.reset(state)?;
            }
            (ResourcePhase::Prefix, ResourceReply::Prefix(tokens)) => {
                if self.state.ensure_tokens(state, tokens).is_err() && tokens > 0 {
                    let ticket = match self
                        .backend
                        .begin_resource(ResourceCommand::Reset { state })
                    {
                        Ok(ticket) => Some(ticket),
                        Err(error) if error.code == ErrorCode::Capacity => None,
                        Err(error) => return Err(error),
                    };
                    self.resources_pending.insert(
                        id,
                        PendingResource {
                            ticket,
                            phase: ResourcePhase::Reset,
                            retry_epoch: None,
                            started: self.now_us,
                        },
                    )?;
                    return Ok(());
                }
                if tokens > 0 {
                    self.state.commit(state, tokens)?;
                    let record = self
                        .host
                        .requests
                        .get_mut(id)
                        .ok_or_else(|| Error::invariant("prefix request lost"))?;
                    record.prefill_done = tokens;
                    record.cached_tokens += tokens;
                    self.event(
                        EventKind::PrefixHit,
                        ObjectKind::Request,
                        id.get(),
                        state.get(),
                        tokens as u64,
                        0,
                    );
                }
            }
            _ => return Err(Error::invariant("resource acknowledgement type mismatch")),
        }
        self.host.queues.unblock(id)?;
        self.host
            .requests
            .get_mut(id)
            .ok_or_else(|| Error::invariant("resource request lost"))?
            .status
            .transition(RequestStatus::Runnable)?;
        self.refresh_queue(id)?;
        self.progress(id)?;
        self.resources_changed()
    }
}
