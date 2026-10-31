use super::CudaBackend;
use crate::executor::{CudaTicket, submission::TicketState, validation::check_program};
use infer_core::{Error, ErrorCode, Result, StateId};
use infer_ir::{
    DeviceCapabilities, ExecutionProgram, ExecutionTask, ExecutionTiming, ModelIr, OutputReadout,
    PageGrowth, PrefillGeometry, StepPlan, TaskOutput, TimingSource,
};
use infer_spi::BackendProvider;
use std::collections::BTreeSet;
use std::{mem::take, sync::Arc};
impl BackendProvider for CudaBackend {
    type Ticket = CudaTicket;
    fn identity(&self) -> &str {
        &self.view.identity
    }
    fn weight_backed_dataflow(&self) -> bool {
        true
    }
    fn supports_multimodal(&self) -> bool {
        self.view.multimodal
    }
    fn image_preprocessing(&self) -> Option<Arc<dyn infer_spi::ImagePreprocessing>> {
        self.view.image_preprocessing.clone()
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.view.capabilities.clone()
    }
    fn speculation_capability(&self) -> infer_ir::SpeculationCapability {
        self.view.capabilities.speculation
    }
    fn execution_profile(&self) -> Option<infer_ir::ExecutionProfileInspection> {
        Some(self.view.profile.clone())
    }
    fn control_ready(&self) -> bool {
        !self.busy && self.fatal.is_none()
    }
    fn maintenance(&mut self) -> Result<()> {
        self.fatal
            .as_ref()
            .map_or(Ok(()), |error| Err(error.clone()))
    }
    fn set_waker(&mut self, wake: Arc<dyn Fn() + Send + Sync>) {
        self.notification.set_waker(wake);
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<infer_ir::DataflowGraph> {
        if model != self.model() {
            return Err(Error::invalid("execution graph model mismatch"));
        }
        Ok(self.view.program.dataflow.clone())
    }
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        let expected = &self.view.program;
        check_program(
            (
                &self.view.model,
                &expected.dataflow,
                &expected.operations,
                expected.workspace_bytes,
            ),
            model,
            program,
        )
    }
    fn prepare_state_reservation(&mut self, capacity: usize, readout: OutputReadout) -> Result<()> {
        self.control(|executor| executor.prepare_state_reservation(capacity, readout))
    }
    fn state_reservation_bytes(&self, capacity: usize) -> Result<Option<u64>> {
        self.state_reservation_bytes_for(capacity, OutputReadout::Full)
    }
    fn state_reservation_bytes_for(
        &self,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<Option<u64>> {
        self.idle()?.state_reservation_bytes_for(capacity, readout)
    }
    fn prepare_state_reservation_for_media(
        &mut self,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        self.control(|executor| executor.prepare_state_reservation_for_media(capacity, readout))
    }
    fn state_reservation_bytes_for_media(
        &self,
        capacity: usize,
        readout: OutputReadout,
        media: &infer_ir::MediaPlan,
    ) -> Result<Option<u64>> {
        self.idle()?
            .state_reservation_bytes_for_media(capacity, readout, media)
    }
    fn state_capacity_bytes(&self) -> Result<Option<u64>> {
        Ok(self.view.capacity)
    }
    fn free_state_bytes(&self) -> Result<Option<u64>> {
        Ok(self.view.free)
    }
    fn kv_cache(&self) -> Option<infer_ir::KvCacheInspection> {
        self.view.kv.clone()
    }
    fn state_page_growth(&self, state: StateId) -> Result<Option<PageGrowth>> {
        Ok(self.view.state(state)?.growth)
    }
    fn state_prefill_geometry(&self, state: StateId) -> Result<Option<PrefillGeometry>> {
        self.prefill_geometry_for(state).map(Some)
    }
    fn reserve_state(&mut self, state: StateId, capacity: usize) -> Result<()> {
        self.reserve_state_for(state, capacity, OutputReadout::Full)
    }
    fn reserve_state_for(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        self.control(|executor| {
            executor.reserve(state, capacity, readout, true)?;
            if let Err(error) = executor.prepare_execution() {
                executor.release(state)?;
                return Err(error);
            }
            Ok(())
        })
    }
    fn reserve_state_for_media(
        &mut self,
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
        media: &infer_ir::MediaPlan,
    ) -> Result<()> {
        self.control(|executor| {
            executor.reserve_media(state, capacity, readout, media)?;
            if let Err(error) = executor.prepare_execution() {
                executor.release(state)?;
                return Err(error);
            }
            Ok(())
        })
    }
    fn reset_state(&mut self, state: StateId) -> Result<()> {
        self.control(|executor| executor.reset(state))
    }
    fn release_state(&mut self, state: StateId) -> Result<()> {
        self.control(|executor| executor.release(state))
    }
    fn reuse_prefix_shared(
        &mut self,
        state: StateId,
        tokens: infer_ir::TokenBuffer,
        maximum: usize,
    ) -> Result<usize> {
        self.control(|executor| executor.reuse_prefix_shared(state, &tokens, maximum))
    }
    fn validate_state_ownership(&self, states: &[(StateId, usize, usize)]) -> Result<()> {
        if !self.busy
            && let Some(executor) = &self.ready
        {
            return executor.ownership(states);
        }

        if states.len() != self.view.states.len()
            || states
                .iter()
                .map(|entry| entry.0)
                .collect::<BTreeSet<_>>()
                .len()
                != states.len()
            || states.iter().any(|(id, capacity, cursor)| {
                self.view.states.get(id).is_none_or(|state| {
                    state.poisoned || state.capacity != *capacity || state.cursor != *cursor
                })
            })
        {
            return Err(Error::invariant("CUDA state ownership mismatch"));
        }
        Ok(())
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<CudaTicket> {
        self.start(program, step, tasks)
    }
    fn submit_borrowed(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: &[ExecutionTask],
    ) -> Result<CudaTicket> {
        self.idle()?;
        let mut owned = take(&mut self.recycled_tasks);
        owned.extend_from_slice(tasks);
        self.start(program, step, owned)
    }
    fn completion_timing(&self, ticket: &CudaTicket) -> Option<ExecutionTiming> {
        Some(ExecutionTiming {
            elapsed_us: ticket.device_us?,
            source: TimingSource::CudaGpu,
        })
    }
    fn poll(&mut self, ticket: &mut CudaTicket) -> Result<Option<Vec<TaskOutput>>> {
        if !Arc::ptr_eq(&self.owner, &ticket.owner)
            || ticket.generation != self.generation
            || matches!(ticket.state, TicketState::Consumed)
            || !self.busy
        {
            return Err(Error::invalid("foreign or consumed CUDA completion"));
        }
        self.progress()?;
        if let Some(error) = &self.fatal {
            ticket.state = TicketState::Consumed;
            return Err(error.clone());
        }
        let Some(result) = self.view.result.take() else {
            return Ok(None);
        };
        ticket.state = TicketState::Consumed;
        ticket.device_us = self.view.device_us.take();
        self.busy = false;
        let result = if let Some(executor) = self.ready.as_deref() {
            let refresh = self.view.refresh(executor);
            result.and_then(|outputs| refresh.map(|()| outputs))
        } else {
            Err(Error::invariant("completed CUDA executor missing"))
        };
        if let Err(error) = &result
            && error.code == ErrorCode::Backend
        {
            self.fatal = Some(error.clone());
        }
        result.map(Some)
    }
}
