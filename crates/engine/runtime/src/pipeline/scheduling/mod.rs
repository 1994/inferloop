//! Schedule one batch: resource preparation → queue candidates → pure policy → validated commit.
use crate::{Engine, EngineOutput};
use infer_core::event::CpuStage;
use infer_core::{
    Error, RequestStatus, Result, event::EventKind, event::ObjectKind, event::SemanticEvent,
};
use infer_ir::{DeferReason, ReadyWork, ResourceSnapshot, SchedulingDecision};
use infer_observe::DiagnosticCode;
use infer_spi::{BackendProvider, SchedulingPolicy};
use std::mem::take;
use std::time::Instant;

mod admission;
mod evidence;
mod feedback;
pub mod index;
pub mod overlap;
mod preemption;
mod ready;
mod resources;
impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    pub(crate) fn schedule_next(&mut self, emitted: &mut Vec<EngineOutput>) -> Result<()> {
        self.reuse_ready_prefixes()?;
        self.poll_resources(emitted)?;
        if self
            .output_worker
            .as_ref()
            .is_some_and(|worker| !worker.available())
        {
            return Ok(());
        }
        if self.output_pending.coalescing(self.now_us) {
            return Ok(());
        }
        if self.dispatch_prepared(emitted)? {
            return Ok(());
        }
        let mut ready = self
            .host
            .ready_buffer
            .take()
            .ok_or_else(|| Error::invariant("ready window already borrowed"))?;
        let result = self
            .preempt_for_progress(&mut ready)
            .and_then(|()| self.plan_ready(&ready, emitted));
        self.host.ready_buffer = Some(ready);
        result
    }
    fn plan_ready(&mut self, ready: &[ReadyWork], emitted: &mut Vec<EngineOutput>) -> Result<()> {
        if ready.is_empty() {
            return Ok(());
        }
        let resources = self.resources()?;
        let decision = self.build_decision(ready, &resources)?;
        self.commit_decision(decision, ready, &resources, emitted)
    }
    pub(super) fn build_decision(
        &mut self,
        ready: &[ReadyWork],
        resources: &ResourceSnapshot,
    ) -> Result<SchedulingDecision> {
        let decision_id = self.ids.allocate()?;
        let step_id = self.ids.allocate()?;
        let start = Instant::now();
        let decision = self.policy.plan_into(
            infer_spi::PlanningContext {
                ready,
                resources,
                now_us: self.now_us,
                decision: decision_id,
                step: step_id,
                costs: self.costs.as_ref(),
            },
            &mut self.planning_workspace,
            &mut self.host.decisions,
        )?;
        self.cpu_stage(CpuStage::Planning, decision_id.get(), start, ready.len());
        if decision.id != decision_id
            || decision
                .step
                .as_ref()
                .is_some_and(|s| s.id != step_id || s.decision != decision_id)
        {
            self.host.decisions.reclaim(decision);
            return self.diagnostic(Error::invariant(
                "scheduler changed allocated step/decision identities",
            ));
        }
        Ok(decision)
    }
    pub(super) fn commit_decision(
        &mut self,
        mut decision: SchedulingDecision,
        ready: &[ReadyWork],
        resources: &ResourceSnapshot,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        let result = self.commit_decision_inner(&mut decision, ready, resources, emitted);
        self.host.decisions.reclaim(decision);
        result
    }
    fn commit_decision_inner(
        &mut self,
        decision: &mut SchedulingDecision,
        ready: &[ReadyWork],
        resources: &ResourceSnapshot,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        let feasible = self.any_feasible(ready, resources)?;
        if let Err(error) = self.validate_decision(decision, ready, resources) {
            return self.diagnostic(error);
        }
        if let Err(error) = self
            .guard
            .observe(ready.len(), feasible, decision.step.is_some())
        {
            self.isolate(DiagnosticCode::ProgressStall, &error);
            return self.diagnostic(error);
        }
        if decision.window.is_none() {
            self.append_focus_deferrals(decision)?;
        }
        self.publish_decision(decision)?;
        self.apply_deferrals(decision, emitted)?;
        if decision.step.is_none()
            && let Some(last) = ready.last()
        {
            self.host.queues.advance_candidate_window(last.request);
        }
        if let Some(step) = decision.step.as_mut() {
            emitted.extend(self.dispatch(step, ready)?);
        }
        Ok(())
    }

    fn publish_decision(&mut self, decision: &SchedulingDecision) -> Result<()> {
        for deferred in &decision.deferred {
            self.emit_semantic(SemanticEvent {
                timestamp_us: self.now_us,
                kind: EventKind::Deferred,
                object_kind: ObjectKind::Request,
                reserved: deferred.reason.code(),
                object_id: deferred.request.get(),
                correlation_id: decision.id.get(),
                arg0: deferred.required,
                arg1: deferred.available,
            });
        }
        let old = if self.decisions.len() == self.config.history_capacity {
            self.decisions.pop_front()
        } else {
            None
        };
        self.decisions
            .push_back(self.host.history.record(old, decision)?);
        Ok(())
    }

    fn apply_deferrals(
        &mut self,
        decision: &SchedulingDecision,
        emitted: &mut Vec<EngineOutput>,
    ) -> Result<()> {
        for rejected in &decision.deferred {
            if rejected.reason == DeferReason::AtomicCostLimit {
                let reason = self.failure(
                    rejected.request,
                    infer_core::ErrorCode::Capacity,
                    "calibrated minimum execution exceeds atomic dispatch limit",
                );
                emitted.push(self.finish(rejected.request, reason, None)?);
            }
        }
        if decision.step.is_some() {
            for deferred in &decision.deferred {
                if deferred.reason == DeferReason::StateCapacity {
                    self.host.queues.block(
                        deferred.request,
                        infer_scheduler::BlockedOn::Memory,
                        self.resource_epoch,
                    )?;
                    self.host
                        .requests
                        .get_mut(deferred.request)
                        .ok_or_else(|| Error::invariant("deferred request disappeared"))?
                        .status
                        .transition(RequestStatus::Waiting(
                            infer_core::WaitReason::StateCapacity {
                                required: usize::try_from(deferred.required).unwrap_or(usize::MAX),
                                available: usize::try_from(deferred.available)
                                    .unwrap_or(usize::MAX),
                            },
                        ))?;
                }
            }
        }
        Ok(())
    }
    pub(crate) fn any_feasible(
        &self,
        ready: &[ReadyWork],
        resources: &ResourceSnapshot,
    ) -> Result<bool> {
        infer_scheduler::any_feasible(ready, resources, self.costs.as_ref())
    }
    pub(crate) fn validate_decision(
        &mut self,
        decision: &SchedulingDecision,
        ready: &[ReadyWork],
        resources: &ResourceSnapshot,
    ) -> Result<()> {
        infer_scheduler::validate_decision_reusing(
            decision,
            ready,
            resources,
            self.program.id,
            self.costs.as_ref(),
            &mut self.host.validation,
        )
    }
    pub(crate) fn resources_changed(&mut self) -> Result<()> {
        self.resource_epoch = self.resource_epoch.saturating_add(1);
        self.wake_memory_waiters()
    }
    pub(crate) fn wake_memory_waiters(&mut self) -> Result<()> {
        let mut ids = take(&mut self.host.wake_buffer);
        let result = self
            .host
            .queues
            .wake_into(
                infer_scheduler::BlockedOn::Memory,
                self.resource_epoch,
                self.config.cpu.maintenance_items,
                &mut ids,
            )
            .and_then(|()| {
                ids.iter().copied().try_for_each(|id| {
                    self.host
                        .requests
                        .get_mut(id)
                        .ok_or_else(|| Error::invariant("woken request disappeared"))?
                        .status
                        .transition(RequestStatus::Runnable)?;
                    self.refresh_queue(id)?;
                    Ok(())
                })
            });
        self.host.wake_buffer = ids;
        result
    }
}
