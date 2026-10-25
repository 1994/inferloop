use super::CudaExecutor;
use infer_core::{Error, Result};
use infer_ir::{
    BackendKind, ExecutionProgram, ExecutionRole, ExecutionTask, ModelIr, OutputReadout,
    PrecisionPlan, StepPlan,
};
use std::collections::BTreeSet;
impl CudaExecutor {
    pub(super) fn check_program(&self, model: &ModelIr, p: &ExecutionProgram) -> Result<()> {
        if model != self.model()
            || p.backend != BackendKind::Cuda
            || p.model != model.id
            || p.precision != PrecisionPlan::f32()
            || p.dataflow != *self.loaded.graph()
            || p.operations != self.operations
            || p.workspace_bytes != self.workspace_bytes
        {
            return Err(Error::invalid("CUDA program/model/kernel mismatch"));
        }
        Ok(())
    }
    pub(super) fn check_tasks(
        &self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: &[ExecutionTask],
    ) -> Result<()> {
        self.idle()?;
        self.check_program(self.model(), program)?;
        if step.program != program.id
            || step.graph.is_some()
            || tasks.is_empty()
            || tasks.len() != step.work.len()
            || tasks.len() > self.maximum_states
            || tasks.iter().map(|t| t.state).collect::<BTreeSet<_>>().len() != tasks.len()
            || tasks
                .iter()
                .map(|t| t.request)
                .collect::<BTreeSet<_>>()
                .len()
                != tasks.len()
        {
            return Err(Error::invalid(
                "CUDA batch/program/graph descriptor mismatch",
            ));
        }
        for (task, work) in tasks.iter().zip(&step.work) {
            let state = self
                .states
                .get(&task.state)
                .ok_or_else(|| Error::invalid("unknown CUDA state"))?;
            let delta =
                task.tokens
                    .validate(&state.history, state.capacity, self.model().vocab_size)?;
            // A slot-leased sequence decodes through the shared pool graph, so its own
            // program position is frozen at bind time and the lease cursor is authoritative.
            let position = state
                .slot
                .as_ref()
                .map_or_else(|| state.program.position(), |lease| lease.position);
            if state.poisoned
                || position != state.history.len()
                || task.request != work.request
                || task.state != work.state
                || work.token_count != delta.len()
                || work.role == ExecutionRole::Decode && work.token_count != 1
                || step.role != ExecutionRole::Mixed && step.role != work.role
                || task.tokens.readout() == OutputReadout::Full
                    && state.readout != OutputReadout::Full
            {
                return Err(Error::invalid("CUDA task/state/cursor/readout mismatch"));
            }
        }
        Ok(())
    }
}
