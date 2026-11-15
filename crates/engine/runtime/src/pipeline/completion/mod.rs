//! Completed executor output is validated before committing cursors or producing client output.
use crate::stages::decode::commit;
use crate::stages::prefill::commit as prefill_commit;
use crate::{Engine, EngineOutput, InFlight, stages::output::ProcessedOutput};
use infer_core::event::CpuStage;
use infer_core::{Error, FinishReason, RequestStatus, Result, event::EventKind, event::ObjectKind};
use infer_ir::{
    CostObservation, ExecutionRole, ModelOutput, PlannedWork, StepPlan, TaskOutput, Workload,
    WorkloadOutput,
};
use infer_observe::DiagnosticCode;
use infer_spi::{BackendProvider, SchedulingPolicy};
use std::mem::replace;
use std::mem::take as mem_take;
use std::ops::ControlFlow;
use std::time::Instant;

pub mod cpu;

struct CompletedBatch {
    outputs: Vec<TaskOutput>,
    consumed: u64,
}
impl CompletedBatch {
    fn validate(step: &StepPlan, mut outputs: Vec<TaskOutput>) -> Result<Self> {
        outputs.sort_unstable_by_key(|output| output.request);
        if outputs.len() != step.work.len()
            || outputs.len() > infer_gpu_api::MAX_SUBMISSION_BATCH
            || outputs
                .windows(2)
                .any(|pair| pair[0].request == pair[1].request)
            || step.work.iter().any(|work| {
                outputs
                    .binary_search_by_key(&work.request, |output| output.request)
                    .is_err()
            })
        {
            return Err(Error::invariant(
                "backend completion does not match submitted requests",
            ));
        }
        Ok(Self {
            outputs,
            consumed: 0,
        })
    }
    fn take(&mut self, work: &PlannedWork) -> Result<ModelOutput> {
        let slot = self
            .outputs
            .binary_search_by_key(&work.request, |output| output.request)
            .map_err(|_| Error::invariant("validated output missing"))?;
        let bit = 1_u64 << slot;
        if self.consumed & bit != 0 {
            return Err(Error::invariant("output consumed twice"));
        }
        self.consumed |= bit;
        Ok(replace(
            &mut self.outputs[slot].output,
            ModelOutput {
                next_token: None,
                logits: Vec::new(),
                hidden: Vec::new(),

                tokens: Vec::new(),
            },
        ))
    }
}

impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    pub(crate) fn poll_execution(
        &mut self,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<ControlFlow<()>> {
        let Some(mut running) = self.inflight.take() else {
            return Ok(ControlFlow::Continue(()));
        };
        if !running.launch_acked
            && let Some(accepted) = self.backend.launch_accepted(&running.ticket)
        {
            running.launch_acked = true;
            running.launched_us.get_or_insert(self.now_us);
            self.event(
                EventKind::LaunchAcknowledged,
                ObjectKind::Step,
                running.step.id.get(),
                running.step.program.get(),
                u64::from(accepted),
                self.now_us.saturating_sub(running.submitted_us),
            );
        }
        match self.backend.poll(&mut running.ticket) {
            Ok(Some(outputs)) => self.commit_execution(running, outputs, emitted)?,
            Ok(None) => {
                // Only device execution is timed; encoding has not finished before acceptance.
                let timed_out = running.launched_us.is_some_and(|launched| {
                    self.now_us.saturating_sub(launched) >= self.config.submission_timeout_us
                });
                // Timeout is not a completion fence: put the ticket back before reporting it.
                self.inflight = Some(running);
                if timed_out && self.fault.is_none() {
                    let error = Error::invariant(
                        "SubmittedStepTimeout: backend has not completed or failed",
                    );
                    self.isolate(DiagnosticCode::SubmissionTimeout, &error);
                    return Err(error);
                }
                return Ok(ControlFlow::Break(()));
            }
            Err(error) => self.reject_execution(
                &running.step,
                DiagnosticCode::CompletionFailed,
                &error,
                emitted,
            )?,
        }
        Ok(ControlFlow::Continue(()))
    }

    fn record_execution_completion(&mut self, running: &InFlight<B::Ticket>) -> Result<()> {
        if let Some(timing) = self.backend.completion_timing(&running.ticket) {
            self.event(
                EventKind::ExecutionTiming,
                ObjectKind::Step,
                running.step.id.get(),
                running.step.program.get(),
                timing.elapsed_us,
                match timing.source {
                    infer_ir::TimingSource::CpuWall => 0,
                    infer_ir::TimingSource::MetalGpu => 1,
                    infer_ir::TimingSource::CudaGpu => 2,
                },
            );
            if !self.replaying && self.fault.is_none() {
                self.enqueue_cost_observation(
                    CostObservation {
                        step: running.step.id,
                        work: running.cost_queries.clone(),
                        timing,
                    },
                    true,
                )?;
            }
        }
        self.event(
            EventKind::Completed,
            ObjectKind::Step,
            running.step.id.get(),
            running.step.program.get(),
            0,
            0,
        );
        Ok(())
    }

    fn commit_execution(
        &mut self,
        running: InFlight<B::Ticket>,
        outputs: Vec<TaskOutput>,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        let start = Instant::now();
        let id = running.step.id.get();
        let items = running.step.work.len();
        let result = self.commit_execution_inner(running, outputs, emitted);
        self.cpu_stage(CpuStage::Completion, id, start, items);
        result
    }
    fn commit_execution_inner(
        &mut self,
        mut running: InFlight<B::Ticket>,
        outputs: Vec<TaskOutput>,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        self.resources_changed()?;
        self.record_execution_completion(&running)?;
        let mut completed = match CompletedBatch::validate(&running.step, outputs) {
            Ok(completed) => completed,
            Err(error) => {
                return self.reject_execution(
                    &running.step,
                    DiagnosticCode::InvalidCompletion,
                    &error,
                    emitted,
                );
            }
        };
        if let Some(credit) = running.output_credit.take() {
            return self.start_output_stage(running.step, completed, credit, emitted);
        }
        let mut first_error = None;
        for work in &running.step.work {
            let output = completed.take(work)?;
            match self.complete_work(&running.step, work, output, emitted) {
                Ok(()) => {}
                Err(error) => {
                    self.retain_diagnostic(
                        DiagnosticCode::InvalidCompletion,
                        &error,
                        Some(work.request),
                        Some(running.step.id),
                    );
                    let reason = self.failure(work.request, error.code, error.to_string());
                    emitted.push(self.finish(work.request, reason, None)?);
                    first_error.get_or_insert(error);
                }
            }
        }
        self.backend.recycle_batch(completed.outputs)?;
        if let Some(error) = first_error {
            self.isolate(DiagnosticCode::InvalidCompletion, &error);
        }
        Ok(())
    }

    fn reject_execution(
        &mut self,
        step: &StepPlan,
        code: DiagnosticCode,
        error: &Error,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        self.retain_diagnostic(code, error, None, Some(step.id));
        emitted.extend(self.fail_step(step, error)?);
        self.isolate(code, error);
        Ok(())
    }
    pub(crate) fn fail_step(
        &mut self,
        step: &StepPlan,
        error: &Error,
    ) -> Result<Vec<EngineOutput>> {
        self.event(
            EventKind::Failed,
            ObjectKind::Step,
            step.id.get(),
            step.program.get(),
            step.work.len() as u64,
            0,
        );
        let message = error.to_string();
        step.work
            .iter()
            .map(|w| {
                // The step was being dispatched, even if the backend never took it.
                self.mark_dispatching(w.request);
                let reason = self.failure(w.request, error.code, message.clone());
                self.finish(w.request, reason, None)
            })
            .collect()
    }
    pub(crate) fn complete_work(
        &mut self,
        step: &StepPlan,
        work: &PlannedWork,
        output: ModelOutput,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        let r = self
            .host
            .requests
            .get_mut(work.request)
            .ok_or_else(|| Error::invariant("submitted request"))?;
        if r.status != (RequestStatus::Running { step: step.id }) {
            return Err(Error::invariant("completion arrived for wrong step"));
        }
        if let Some(reason) = r.pending_finish.take() {
            self.backend.recycle_output(work.state, output)?;
            emitted.push(self.finish(work.request, reason, None)?);
            return Ok(());
        }
        self.host.queues.complete(work.request, step.id)?;
        r.status.transition(RequestStatus::Runnable)?;
        let output = self
            .output_job(work, output)?
            .process(self.workloads.as_ref(), &mut self.host.sampling)?;
        self.complete_prepared_work(step, work, output, emitted)
    }
    pub(super) fn complete_prepared_work(
        &mut self,
        step: &StepPlan,
        work: &PlannedWork,
        mut output: ProcessedOutput,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        if matches!(
            self.host.requests.known(work.request)?.request.workload,
            Workload::Generate { .. }
        ) && let Some(buffer) = output.output.take()
        {
            self.backend.recycle_output(work.state, buffer)?;
        }
        let decided = mem_take(&mut output.tokens);
        let r = self
            .host
            .requests
            .get_mut(work.request)
            .ok_or_else(|| Error::invariant("completion request lost"))?;
        r.constraint = output.constraint.take();
        if r.status != RequestStatus::Runnable {
            return Err(Error::invariant("completion owner is not runnable"));
        }
        let committed = self.state.get(work.state)?.committed_tokens;
        let next = if work.role == ExecutionRole::Decode {
            commit(r, work.token_count)?
        } else {
            prefill_commit(r, work.token_count)?
        };
        self.state.commit(work.state, committed.max(next))?;
        r.status.transition(RequestStatus::Runnable)?;
        if work.role != ExecutionRole::Decode && r.prefill_done < r.context.len() {
            self.refresh_queue(work.request)?;
            self.progress(work.request)?;
            return Ok(());
        }
        if let Workload::Generate { max_new_tokens } = r.request.workload {
            if decided.is_empty() {
                return Err(Error::invariant("generation step produced no token"));
            }
            self.complete_generation(step, work, decided.as_slice(), max_new_tokens, emitted)?;
        } else {
            if let Some(output) = output.output {
                r.outputs.push(output);
            }
            r.unit += 1;
            if r.unit < r.plan.units.len() {
                r.context = crate::TokenContext::new(r.plan.units[r.unit].clone());
                r.prefill_done = 0;
                r.prefill_target = r.context.len();
                r.prefix_attempted = false;
                let _ = r;
                self.refresh_queue(work.request)?;
                self.reset_request_state(work.request)?;
                // Pair units reuse one reservation; its commit cursor represents capacity,
                // not the token position of a different pair.
                if self.host.requests.known(work.request)?.status == RequestStatus::Runnable {
                    self.refresh_queue(work.request)?;
                }
                self.progress(work.request)?;
            } else {
                let output = output
                    .projection
                    .ok_or_else(|| Error::invariant("CPU projection produced no output"))?;
                emitted.push(self.finish(work.request, FinishReason::Completed, Some(output))?);
            }
        }
        Ok(())
    }
    /// Record every token this step decided. A speculative backend may return several, so the
    /// terminal reason can be reached before the list ends; the tail is then discarded.
    fn complete_generation(
        &mut self,
        step: &StepPlan,
        work: &PlannedWork,
        tokens: &[u32],
        max_new_tokens: usize,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        for token in tokens {
            if self.append_generated(step, work, *token, max_new_tokens, emitted)? {
                break;
            }
        }
        Ok(())
    }
    /// Append one decided token and report whether it finished the request.
    fn append_generated(
        &mut self,
        step: &StepPlan,
        work: &PlannedWork,
        token: u32,
        max_new_tokens: usize,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<bool> {
        let (index, first_token, ttft, tpot, terminal) = {
            let r = self
                .host
                .requests
                .get_mut(work.request)
                .ok_or_else(|| Error::invariant("known generation request"))?;
            let index = r.generated.len();
            let first_token = r.first_token_us.is_none();
            let ttft = self.now_us - r.accepted_us;
            let tpot = r.last_token_us.map(|last| self.now_us - last);
            r.generated.push(token);
            r.context.append()?;
            if r.first_token_us.is_none() {
                r.first_token_us = Some(self.now_us);
            }
            if let Some(last) = r.last_token_us {
                r.max_tpot_us = Some(r.max_tpot_us.unwrap_or(0).max(self.now_us - last));
            }
            r.last_token_us = Some(self.now_us);
            let terminal = if r.request.sampling.is_eos(token) {
                Some(FinishReason::Eos)
            } else if r.generated.len() >= max_new_tokens {
                Some(FinishReason::Length)
            } else {
                None
            };
            (index, first_token, ttft, tpot, terminal)
        };
        emitted.push(EngineOutput::Token {
            request: work.request,
            token,
            index,
        });
        let terminal_output = if terminal.is_some() {
            Some(WorkloadOutput::Tokens(
                self.host
                    .requests
                    .get(work.request)
                    .ok_or_else(|| Error::invariant("known generation request"))?
                    .generated
                    .clone(),
            ))
        } else {
            None
        };
        if first_token {
            self.event(
                EventKind::FirstToken,
                ObjectKind::Request,
                work.request.get(),
                step.id.get(),
                ttft,
                0,
            );
        }
        if let Some(tpot) = tpot {
            self.event(
                EventKind::RequestLatency,
                ObjectKind::Request,
                work.request.get(),
                step.id.get(),
                tpot,
                0,
            );
        }
        self.event(
            EventKind::TokenProduced,
            ObjectKind::Request,
            work.request.get(),
            step.id.get(),
            u64::from(token),
            index as u64,
        );
        self.refresh_queue(work.request)?;
        self.progress(work.request)?;
        if let Some(reason) = terminal {
            emitted.push(self.finish(work.request, reason, terminal_output)?);
            return Ok(true);
        }
        Ok(false)
    }
}
