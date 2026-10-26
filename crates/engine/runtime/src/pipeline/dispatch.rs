//! The dispatch transaction separates planner output, reservation, and executor ownership.
use crate::stages::decode::prepare;
use crate::stages::prefill::prepare as prefill_prepare;
use crate::stages::worker::OutputWorker;
use crate::{Engine, EngineOutput, InFlight};
use infer_core::event::CpuStage;
use infer_core::{Error, RequestStatus, Result, event::EventKind, event::ObjectKind};
use infer_ir::{CostQuery, ExecutionRole, ExecutionTask, ReadyWork, StepPlan};
use infer_observe::DiagnosticCode;
use infer_spi::{BackendProvider, SchedulingPolicy};
use std::sync::Arc;
use std::time::Instant;

pub struct PreparedDispatch {
    tasks: Vec<ExecutionTask>,
    growth: Vec<(infer_core::StateId, usize)>,
    charges: Vec<u64>,
    cost_queries: Vec<CostQuery>,
    ids: Vec<infer_core::RequestId>,
}
impl PreparedDispatch {
    pub fn new(batch: usize) -> Result<Self> {
        fn buffer<T>(batch: usize) -> Result<Vec<T>> {
            let mut buffer = Vec::new();
            buffer
                .try_reserve_exact(batch)
                .map_err(|e| Error::new(infer_core::ErrorCode::Capacity, e.to_string()))?;
            Ok(buffer)
        }
        Ok(Self {
            tasks: buffer(batch)?,
            growth: buffer(batch)?,
            charges: buffer(batch)?,
            cost_queries: buffer(batch)?,
            ids: buffer(batch)?,
        })
    }
    fn clear(&mut self) {
        self.tasks.clear();
        self.growth.clear();
        self.charges.clear();
        self.cost_queries.clear();
        self.ids.clear();
    }
}
impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    fn prepare_dispatch(
        &self,
        step: &StepPlan,
        ready: &[ReadyWork],
        prepared: &mut PreparedDispatch,
    ) -> Result<()> {
        prepared.clear();
        infer_scheduler::step_cost_queries_into(step, ready, &mut prepared.cost_queries)?;
        for (index, work) in step.work.iter().enumerate() {
            let record = self
                .host
                .requests
                .get(work.request)
                .ok_or_else(|| Error::invariant("validated request disappeared"))?;
            if record.status != RequestStatus::Runnable || record.state != Some(work.state) {
                return Err(Error::invariant(
                    "dispatch request is not runnable or changed state",
                ));
            }
            let tokens = if work.role == ExecutionRole::Decode {
                prepare(record, work.token_count)?
            } else {
                prefill_prepare(record, work.token_count)?
            };
            let tenant = self
                .tenants
                .get(record.request.qos.tenant.as_str())
                .ok_or_else(|| Error::invariant("dispatch tenant is absent"))?;
            let cost = self.costs.estimate(&[prepared.cost_queries[index]])?;
            prepared
                .charges
                .push(infer_scheduler::service_charge(cost.gpu_us, tenant.weight));
            prepared.growth.push((work.state, tokens.len()));
            prepared.ids.push(work.request);
            prepared.tasks.push(ExecutionTask {
                plain_greedy: matches!(
                    record.request.workload,
                    infer_ir::Workload::Generate { .. }
                ) && record.constraint.is_none()
                    && record.request.sampling.uses_plain_greedy(),
                request: work.request,
                state: work.state,
                tokens,
                // Full sampling parameters are needed only for speculative acceptance.
                // Plain output decisions use the allocation-free request flag above.
                // A grammar owns token selection; None also prevents draft proposals.
                sampling: (work.role == ExecutionRole::Decode
                    && record.constraint.is_none()
                    && self.backend.speculation_capability().draft_depth > 0)
                    .then(|| record.request.sampling.clone()),
                media: (work.role == ExecutionRole::Prefill)
                    .then(|| record.media.clone())
                    .flatten(),
            });
        }
        Ok(())
    }
    pub(crate) fn dispatch(
        &mut self,
        step: &mut StepPlan,
        ready: &[ReadyWork],
    ) -> Result<Vec<EngineOutput>> {
        let start = Instant::now();
        let step = match self.host.steps.seal(step) {
            Ok(step) => step,
            Err(error) if error.code == infer_core::ErrorCode::Capacity => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        let mut prepared = self
            .host
            .dispatch
            .take()
            .ok_or_else(|| Error::invariant("dispatch scratch already borrowed"))?;
        let result = self
            .prepare_dispatch(&step, ready, &mut prepared)
            .and_then(|()| self.dispatch_prepared_step(&step, &prepared));
        prepared.clear();
        self.host.dispatch = Some(prepared);
        self.cpu_stage(CpuStage::Dispatch, step.id.get(), start, step.work.len());
        result
    }
    fn dispatch_prepared_step(
        &mut self,
        step: &Arc<StepPlan>,
        prepared: &PreparedDispatch,
    ) -> Result<Vec<EngineOutput>> {
        let output_credit = self
            .output_worker
            .as_ref()
            .map(OutputWorker::reserve)
            .transpose()?;
        let cost_queries = match self.host.queries.seal(&prepared.cost_queries) {
            Ok(queries) => queries,
            Err(error) if error.code == infer_core::ErrorCode::Capacity => return Ok(Vec::new()),
            Err(error) => return Err(error),
        };
        // Allocation is all-or-nothing. No request has transitioned to Running yet.
        self.state.ensure_batch(&prepared.growth)?;
        let ticket =
            match self
                .backend
                .submit_shared_borrowed(&self.program, step.clone(), &prepared.tasks)
            {
                Ok(ticket) => ticket,
                Err(error) if error.code == infer_core::ErrorCode::Capacity => {
                    // A full owner lane has no ticket and owns no new GPU work. Keep requests
                    // runnable, preserve their reservations, and retry within their queue deadline.
                    return Ok(Vec::new());
                }
                Err(error) => {
                    self.retain_diagnostic(
                        DiagnosticCode::SubmissionFailed,
                        &error,
                        None,
                        Some(step.id),
                    );
                    let outputs = self.fail_step(step, &error.to_string())?;
                    if matches!(
                        error.code,
                        infer_core::ErrorCode::Backend | infer_core::ErrorCode::Invariant
                    ) {
                        self.isolate(DiagnosticCode::SubmissionFailed, &error);
                    }
                    return Ok(outputs);
                }
            };
        // Install the ticket before committing lifecycle changes, preserving ownership
        // even if a later invariant detects corrupted control state.
        self.inflight = Some(InFlight {
            step: step.clone(),
            ticket,
            submitted_us: self.now_us,
            launch_acked: false,
            cost_queries,
            output_credit,
        });
        self.host.queues.dispatch(step.id, &prepared.ids)?;
        for work in &step.work {
            let record = self
                .host
                .requests
                .get_mut(work.request)
                .ok_or_else(|| Error::invariant("dispatched request disappeared"))?;
            record.status = RequestStatus::Running { step: step.id };
            record.last_service_us = self.now_us;
        }
        for (index, work) in step.work.iter().enumerate() {
            let record = self
                .host
                .requests
                .get(work.request)
                .ok_or_else(|| Error::invariant("validated request disappeared"))?;
            let service = self
                .tenants
                .get_mut(record.request.qos.tenant.as_str())
                .ok_or_else(|| Error::invariant("validated tenant disappeared"))?;
            service.virtual_finish = service
                .virtual_finish
                .saturating_add(prepared.charges[index]);
            self.host
                .queues
                .update_tenant(&record.request.qos.tenant, service.virtual_finish);
            self.event(
                EventKind::Scheduled,
                ObjectKind::Request,
                work.request.get(),
                step.id.get(),
                work.token_count as u64,
                step.decision.get(),
            );
            self.progress(work.request)?;
        }
        self.event(
            EventKind::Submitted,
            ObjectKind::Step,
            step.id.get(),
            step.program.get(),
            step.work.len() as u64,
            0,
        );
        Ok(Vec::new())
    }
}
