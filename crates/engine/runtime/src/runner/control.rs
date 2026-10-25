//! Nonblocking resource commands and shared submissions preserve backend ownership.
use super::{RunnerTicket, ThreadedBackend, worker::Job};
use infer_core::{Error, ErrorCode, Result, StateId};
use infer_ir::{
    DeviceCapabilities, ExecutionProgram, ExecutionTask, ExecutionTiming, KvCacheInspection,
    ModelIr, PageGrowth, StepPlan, TaskOutput,
};
use infer_spi::BackendProvider;
use std::{sync::Arc, sync::atomic::Ordering};

impl<B: BackendProvider + Send + 'static> BackendProvider for ThreadedBackend<B>
where
    B::Ticket: Send,
{
    type Ticket = RunnerTicket;
    fn state_recipe(&self) -> Option<&infer_ir::StateRecipe> {
        self.recipe.as_ref()
    }
    fn execution_profile(&self) -> Option<infer_ir::ExecutionProfileInspection> {
        self.execution_profile.clone()
    }
    fn recycle_output(&mut self, state: StateId, output: infer_ir::ModelOutput) -> Result<()> {
        self.recycle(super::Recycled::Output(state, output))
    }
    fn recycle_batch(&mut self, mut outputs: Vec<TaskOutput>) -> Result<()> {
        outputs.clear();
        self.recycle(super::Recycled::Batch(outputs))
    }

    fn begin_resource(
        &mut self,
        command: infer_spi::ResourceCommand,
    ) -> Result<infer_spi::ResourceTicket> {
        let (ticket, reply) = self.resources.channel()?;
        let reserve = if let infer_spi::ResourceCommand::Reserve { state, .. }
        | infer_spi::ResourceCommand::ReserveMedia { state, .. } = &command
        {
            Some(*state)
        } else {
            None
        };
        self.shared.controls.fetch_add(1, Ordering::AcqRel);
        let credit = super::ControlCredit(self.shared.clone());
        self.enqueue(Job::Resource {
            command,
            reply,
            reserve,
            credit,
        })?;
        if let Some(state) = reserve {
            self.state_ids.insert(state)?;
            Ok(ticket.on_abandon_state(self.shared.clone(), state))
        } else {
            Ok(ticket)
        }
    }
    fn identity(&self) -> &str {
        &self.identity
    }
    fn weight_backed_dataflow(&self) -> bool {
        self.weight_backed
    }
    fn supports_multimodal(&self) -> bool {
        self.multimodal.supported()
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.capabilities.clone()
    }
    fn speculation_capability(&self) -> infer_ir::SpeculationCapability {
        self.speculation
    }
    fn maintenance(&mut self) -> Result<()> {
        if self.shared.flight_abandoned.load(Ordering::Acquire)
            && let Ok(handle) = self.completions.pop()
        {
            let _ = self
                .shared
                .batches
                .take_completion(handle)?
                .ok_or_else(|| Error::invariant("abandoned flight has no fence"))?;
            self.active = None;
            self.shared.flight_abandoned.store(false, Ordering::Release);
        }
        while let Some(state) = self.pending_releases.front().copied() {
            let job = Job::Release(state);
            match self.enqueue(job) {
                Ok(()) => {
                    self.pending_releases.pop_front();
                }
                Err(error) if error.code == ErrorCode::Capacity => break,
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
    fn pending_resource_releases(&self) -> bool {
        self.shared.releases.load(Ordering::Acquire) > 0
            || self.shared.controls.load(Ordering::Acquire) > 0
    }
    fn tracks_reservation_intent(&self) -> bool {
        true
    }
    fn resource_epoch(&self) -> u64 {
        self.shared.epoch.load(Ordering::Acquire)
    }
    fn control_ready(&self) -> bool {
        !self.shared.encoding.load(Ordering::Acquire)
    }
    fn set_waker(&mut self, wake: Arc<dyn Fn() + Send + Sync>) {
        if let Ok(mut w) = self.shared.wake.lock() {
            *w = Some(wake);
        }
    }
    fn requires_async_checkpoint(&self) -> bool {
        true
    }
    fn supports_control_checkpoint(&self) -> bool {
        self.checkpoint
    }
    fn supports_recompute_preemption(&self) -> bool {
        self.recompute
    }
    fn state_reservation_bytes(&self, _capacity: usize) -> Result<Option<u64>> {
        Err(Error::unsupported(
            "threaded sizing requires an admission quote",
        ))
    }
    fn reuse_prefix_shared(
        &mut self,
        _state: StateId,
        _tokens: infer_ir::TokenBuffer,
        _maximum: usize,
    ) -> Result<usize> {
        Err(Error::unsupported(
            "threaded prefix reuse requires a resource command",
        ))
    }
    fn state_capacity_bytes(&self) -> Result<Option<u64>> {
        let snapshot = self.shared.snapshot()?;
        if let Some(error) = &snapshot.error {
            return Err(error.clone());
        }
        Ok(snapshot.state_capacity_bytes)
    }
    fn free_state_bytes(&self) -> Result<Option<u64>> {
        let snapshot = self.shared.snapshot()?;
        if let Some(error) = &snapshot.error {
            return Err(error.clone());
        }
        Ok(snapshot.free_bytes)
    }
    fn kv_cache(&self) -> Option<KvCacheInspection> {
        self.shared.snapshot().ok().and_then(|s| s.kv.clone())
    }
    fn state_prefill_geometry(&self, state: StateId) -> Result<Option<infer_ir::PrefillGeometry>> {
        let snapshot = self.shared.snapshot()?;
        if let Some(error) = &snapshot.error {
            return Err(error.clone());
        }
        Ok(snapshot.prefill.get(&state).copied())
    }
    fn state_page_growth(&self, state: StateId) -> Result<Option<PageGrowth>> {
        Ok(self.shared.snapshot()?.growth.get(&state).copied())
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        if model != &self.model {
            return Err(Error::invalid("execution graph model mismatch"));
        }
        Ok(self.program.dataflow.clone())
    }
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        if model != &self.model || program != &*self.program {
            return Err(Error::invalid("device runner program mismatch"));
        }
        Ok(())
    }
    fn reserve_state(&mut self, _state: StateId, _capacity: usize) -> Result<()> {
        Err(Error::unsupported(
            "threaded reservation requires a resource command",
        ))
    }
    fn reset_state(&mut self, _state: StateId) -> Result<()> {
        Err(Error::unsupported(
            "threaded reset requires a resource command",
        ))
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        if !self.state_ids.remove(&state) {
            return Err(Error::invalid("release unknown runner state"));
        }
        self.shared.releases.fetch_add(1, Ordering::AcqRel);
        self.pending_releases.push_back(state);
        self.maintenance()
    }
    fn capture_execution_state(&self) -> Result<Option<Vec<u8>>> {
        Err(Error::unsupported(
            "threaded owner checkpoint requires a coordinated resource/flight snapshot",
        ))
    }
    fn restore_execution_state(&mut self, _state: Option<&[u8]>) -> Result<()> {
        Err(Error::unsupported(
            "threaded owner restore requires resource identity reconstruction",
        ))
    }
    fn validate_state_ownership(&self, _states: &[(StateId, usize, usize)]) -> Result<()> {
        Err(Error::unsupported(
            "threaded ownership validation requires an owner checkpoint",
        ))
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<RunnerTicket> {
        self.submit_shared(program, Arc::new(step.clone()), tasks)
    }
    fn submit_shared(
        &mut self,
        program: &ExecutionProgram,
        step: Arc<StepPlan>,
        tasks: Vec<ExecutionTask>,
    ) -> Result<RunnerTicket> {
        self.submit_shared_borrowed(program, step, &tasks)
    }
    fn submit_shared_borrowed(
        &mut self,
        program: &ExecutionProgram,
        step: Arc<StepPlan>,
        tasks: &[ExecutionTask],
    ) -> Result<RunnerTicket> {
        self.maintenance()?;
        if self.active.is_some() {
            return Err(Error::new(ErrorCode::Conflict, "runner ticket still owned"));
        }
        self.validate_program(&self.model, program)?;
        let id = step.id;
        let batch = self.shared.batches.seal(step, tasks)?;
        self.shared.encoding.store(true, Ordering::Release);
        if self.submissions.push(batch).is_err() {
            self.shared.batches.revoke(batch)?;
            self.shared.encoding.store(false, Ordering::Release);
            return Err(Error::new(
                ErrorCode::Capacity,
                "device submission ring full",
            ));
        }
        if let Some(thread) = self.shared.device_thread.get() {
            thread.unpark();
        }
        self.active = Some(id);
        Ok(RunnerTicket {
            owner: self.shared.clone(),
            step: id,
            batch,
            timing: None,
            done: false,
        })
    }
    fn launch_accepted(&self, ticket: &RunnerTicket) -> Option<bool> {
        self.shared
            .batches
            .launch_accepted(ticket.batch)
            .ok()
            .flatten()
    }
    fn poll(&mut self, ticket: &mut RunnerTicket) -> Result<Option<Vec<TaskOutput>>> {
        if !Arc::ptr_eq(&self.shared, &ticket.owner)
            || ticket.done
            || self.active != Some(ticket.step)
        {
            return Err(Error::invariant("device runner ticket ownership mismatch"));
        }
        let Ok(handle) = self.completions.pop() else {
            return Ok(None);
        };
        if handle != ticket.batch {
            return Err(Error::invariant("completion ring generation mismatch"));
        }
        let result = self
            .shared
            .batches
            .take_completion(handle)?
            .ok_or_else(|| Error::invariant("completion handle has no fence"))?;
        ticket.done = true;
        self.active = None;
        result.map(|done| {
            ticket.timing = done.timing;
            Some(done.outputs)
        })
    }
    fn completion_timing(&self, ticket: &RunnerTicket) -> Option<ExecutionTiming> {
        ticket.timing
    }
}

impl<B: BackendProvider> ThreadedBackend<B> {
    fn recycle(&mut self, buffer: super::Recycled) -> Result<()> {
        self.recycled
            .push(buffer)
            .map_err(|_| Error::invariant("credited readback recycler full"))?;
        if let Some(thread) = self.shared.device_thread.get() {
            thread.unpark();
        }
        Ok(())
    }
}
