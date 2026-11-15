//! GPU completion releases compute ownership; CPU output leases retire independently.
use super::CompletedBatch;
use crate::stages::output::ProcessedOutput;
use crate::stages::prefill::readout as prefill_readout;
use crate::{
    Engine, EngineOutput, stages::output::OutputJob, stages::output::OutputShape,
    stages::worker::OutputCredit, stages::worker::OutputTicket,
};
use infer_core::event::EventKind;
use infer_core::event::ObjectKind;
use infer_core::{Error, ErrorCode, RequestStatus, Result};
use infer_ir::{ExecutionRole, ModelOutput, OutputReadout, PlannedWork, StepPlan, Workload};
use infer_spi::{BackendProvider, SchedulingPolicy};
use std::mem::take;
use std::sync::Arc;

pub struct PendingOutput {
    pub step: Arc<StepPlan>,
    pub ticket: OutputTicket,
    pub started: u64,
}
/// Completion credit fixes the capacity; polling does not allocate map nodes.
#[derive(Default)]
pub struct OutputFlights {
    slots: [Option<PendingOutput>; 2],
}
impl OutputFlights {
    pub fn is_empty(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }
    /// A short, nonblocking grace period lets rows from the preceding output batch
    /// rejoin one device dispatch. A slow CPU job cannot stall independent work
    /// beyond this bound; deadlines and resource polling continue on every tick.
    pub fn coalescing(&self, now_us: u64) -> bool {
        const OUTPUT_BATCH_GRACE_US: u64 = 500;
        self.slots
            .iter()
            .flatten()
            .map(|flight| flight.started)
            .min()
            .is_some_and(|started| now_us.saturating_sub(started) < OUTPUT_BATCH_GRACE_US)
    }
    pub fn len(&self) -> usize {
        self.slots.iter().flatten().count()
    }
    fn insert(&mut self, pending: PendingOutput) -> Result<()> {
        if self
            .slots
            .iter()
            .flatten()
            .any(|flight| flight.step.id == pending.step.id)
        {
            return Err(Error::invariant("duplicate CPU output flight"));
        }
        let slot = self
            .slots
            .iter_mut()
            .find(|slot| slot.is_none())
            .ok_or_else(|| Error::invariant("credited CPU output slot unavailable"))?;
        *slot = Some(pending);
        Ok(())
    }
}
impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    pub(super) fn output_job(
        &mut self,
        work: &PlannedWork,
        output: ModelOutput,
    ) -> Result<OutputJob> {
        let r = self
            .host
            .requests
            .get_mut(work.request)
            .ok_or_else(|| Error::invariant("output owner lost"))?;
        let rows = if work.role == ExecutionRole::Decode {
            r.context.len()
        } else {
            r.prefill_done + work.token_count
        };
        let readout = if work.role == ExecutionRole::Decode {
            OutputReadout::Logits
        } else {
            prefill_readout(r, rows)
        };
        let generate = matches!(r.request.workload, Workload::Generate { .. });
        Ok(OutputJob {
            constraint: r.constraint.take(),
            request: r.request.clone(),
            generated: r.generated.clone(),
            output,
            shape: OutputShape {
                logits: if readout == OutputReadout::None {
                    0
                } else {
                    self.model.vocab_size
                },
                rows: if readout == OutputReadout::Full {
                    rows
                } else {
                    0
                },
                width: self.model.hidden_size,
            },
            sample: (generate && readout == OutputReadout::Logits).then_some(r.generated.len()),
            project: (!generate && rows == r.context.len() && r.unit + 1 == r.plan.units.len())
                .then(|| take(&mut r.outputs)),
        })
    }
    pub(super) fn start_output_stage(
        &mut self,
        step: Arc<StepPlan>,
        mut outputs: CompletedBatch,
        credit: OutputCredit,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        let mut jobs = take(&mut self.host.output_jobs);
        jobs.clear();
        let result = self
            .prepare_output_jobs(&step, &mut outputs, &mut jobs, emitted)
            .and_then(|()| {
                if jobs.is_empty() {
                    return Ok(None);
                }
                self.output_worker
                    .as_ref()
                    .ok_or_else(|| Error::invariant("output worker lost"))?
                    .submit_indexed(&mut jobs, credit)
                    .map(Some)
            });
        self.backend.recycle_batch(outputs.outputs)?;
        jobs.clear();
        self.host.output_jobs = jobs;
        let Some(ticket) = result? else {
            return Ok(());
        };
        for (index, work) in step.work.iter().enumerate() {
            if !ticket.pending(index) {
                continue;
            }
            self.host.queues.complete_blocked(
                work.request,
                step.id,
                infer_scheduler::BlockedOn::Preparation,
                self.resource_epoch,
            )?;
            self.host
                .requests
                .get_mut(work.request)
                .ok_or_else(|| Error::invariant("output owner lost"))?
                .status
                .transition(RequestStatus::Waiting(infer_core::WaitReason::BackendBusy))?;
            self.output_owners.insert(work.request, step.id)?;
        }
        self.output_pending.insert(PendingOutput {
            step,
            ticket,
            started: self.now_us,
        })?;
        Ok(())
    }
    fn prepare_output_jobs(
        &mut self,
        step: &StepPlan,
        outputs: &mut CompletedBatch,
        jobs: &mut Vec<(usize, OutputJob)>,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        if step.work.len() > jobs.capacity() {
            return Err(Error::invariant("output scratch capacity exceeded"));
        }
        for (index, work) in step.work.iter().enumerate() {
            let output = outputs.take(work)?;
            if let Some(reason) = self
                .host
                .requests
                .get_mut(work.request)
                .and_then(|record| record.pending_finish.take())
            {
                self.backend.recycle_output(work.state, output)?;
                emitted.push(self.finish(work.request, reason, None)?);
            } else {
                jobs.push((index, self.output_job(work, output)?));
            }
        }
        Ok(())
    }
    pub(crate) fn poll_output_stages(&mut self, emitted: &mut Vec<EngineOutput>) -> Result<()> {
        for index in 0..self.output_pending.slots.len() {
            let Some(mut pending) = self.output_pending.slots[index].take() else {
                continue;
            };
            let result = self.poll_output_batch(&mut pending, emitted);
            if !pending.ticket.done() {
                self.output_pending.slots[index] = Some(pending);
            }
            result?;
        }
        Ok(())
    }
    fn poll_output_batch(
        &mut self,
        pending: &mut PendingOutput,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        for (index, work) in pending.step.work.iter().enumerate() {
            if !pending.ticket.pending(index) {
                continue;
            }
            if let Some(ack) = pending.ticket.poll(index) {
                self.event(
                    EventKind::CpuOutputCompleted,
                    ObjectKind::Request,
                    work.request.get(),
                    pending.step.id.get(),
                    ack.elapsed_us,
                    0,
                );
                self.commit_output_reply(&pending.step, work, ack.result, emitted)?;
            } else if self.now_us.saturating_sub(pending.started) >= self.config.output_timeout_us
                && self
                    .host
                    .requests
                    .known(work.request)?
                    .pending_finish
                    .is_none()
            {
                self.retain_diagnostic(
                    infer_observe::DiagnosticCode::SubmissionTimeout,
                    &Error::new(ErrorCode::Backend, "CPU output stage timed out"),
                    Some(work.request),
                    Some(pending.step.id),
                );
                self.host.queues.cancel(work.request)?;
                let reason = self.failure(
                    work.request,
                    ErrorCode::Backend,
                    "CPU output stage timed out",
                );
                self.request_termination(work.request, reason)?;
            }
        }
        Ok(())
    }
    fn commit_output_reply(
        &mut self,
        step: &StepPlan,
        work: &PlannedWork,
        output: Result<ProcessedOutput>,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        self.output_owners.remove(&work.request);
        if let Some(reason) = self
            .host
            .requests
            .get_mut(work.request)
            .and_then(|record| record.pending_finish.take())
        {
            if let Ok(mut output) = output
                && let Some(buffer) = output.output.take()
            {
                self.backend.recycle_output(work.state, buffer)?;
            }
            emitted.push(self.finish(work.request, reason, None)?);
            return Ok(());
        }
        self.host
            .requests
            .get_mut(work.request)
            .ok_or_else(|| Error::invariant("CPU output request lost"))?
            .status
            .transition(RequestStatus::Runnable)?;
        // A plan prepared while this request was awaiting CPU sampling cannot
        // include its next decode. Rebuild so newly ready rows can join the batch.
        self.discard_prepared();
        match output.and_then(|output| self.complete_prepared_work(step, work, output, emitted)) {
            Ok(()) => {}
            Err(error) => {
                self.retain_diagnostic(
                    infer_observe::DiagnosticCode::InvalidCompletion,
                    &error,
                    Some(work.request),
                    Some(step.id),
                );
                let reason = self.failure(work.request, error.code, error.to_string());
                emitted.push(self.finish(work.request, reason, None)?);
                if error.code == ErrorCode::Invariant {
                    self.isolate(infer_observe::DiagnosticCode::InvalidCompletion, &error);
                }
            }
        }
        Ok(())
    }
    pub(crate) fn check_output_ownership(&self) -> Result<()> {
        let mut count = 0;
        for pending in self.output_pending.slots.iter().flatten() {
            let step = pending.step.id;
            for (index, work) in pending.step.work.iter().enumerate() {
                if !pending.ticket.pending(index) {
                    continue;
                }
                count += 1;
                let record = self
                    .host
                    .requests
                    .get(work.request)
                    .ok_or_else(|| Error::invariant("CPU output request lost"))?;
                if self.output_owners.get(&work.request) != Some(&step)
                    || record.state != Some(work.state)
                    || !matches!(record.status, RequestStatus::Waiting(_))
                    || (record.pending_finish.is_none()
                        && !matches!(
                            self.host.queues.state(work.request),
                            Some(infer_scheduler::QueueState::Blocked {
                                reason: infer_scheduler::BlockedOn::Preparation,
                                ..
                            })
                        ))
                    || (record.pending_finish.is_some()
                        && self.host.queues.state(work.request).is_some())
                {
                    return Err(Error::invariant("CPU output ownership mismatch"));
                }
            }
        }
        if count != self.output_owners.len() {
            return Err(Error::invariant("orphaned CPU output owner"));
        }
        Ok(())
    }
}
