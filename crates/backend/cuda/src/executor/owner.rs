//! Provider ownership stays outside the suspended device execution state.
mod notification;
mod provider;
mod view;
use super::{
    CudaExecutor, CudaTicket, PoolInspection,
    submission::{Submission, TicketState},
};
use crate::loading::LoadedModel;
use infer_core::{Error, ErrorCode, ProgramId, Result, StateId};
use infer_ir::{ExecutionProfileInspection, ExecutionProgram, ExecutionTask, ModelIr};
use notification::Notification;
use std::sync::Arc;
use view::View;

pub struct CudaBackend {
    ready: Option<Box<CudaExecutor>>,
    running: Option<Submission>,
    view: View,
    owner: Arc<()>,
    generation: u64,
    busy: bool,
    fatal: Option<Error>,
    notification: Notification,
    recycled_tasks: Vec<ExecutionTask>,
}
impl CudaBackend {
    /// # Errors
    /// Rejects invalid state limits, unavailable devices, or failed initialization.
    pub fn new(loaded: LoadedModel, budget: u64, maximum_states: usize) -> Result<Self> {
        Self::from_executor(CudaExecutor::new_bounded(loaded, budget, maximum_states)?)
    }
    /// Derive the default batch from the loaded graph's roofline and full memory quote.
    /// # Errors
    /// Rejects invalid device/model facts, capacity or initialization failures.
    pub fn new_automatic(loaded: LoadedModel, budget: u64, maximum_states: usize) -> Result<Self> {
        Self::from_executor(CudaExecutor::new_automatic(loaded, budget, maximum_states)?)
    }
    /// Capture the funded serving pool now instead of at the first reservation that needs it.
    ///
    /// The zero-demand quote already funds every remaining KV page while keeping one full
    /// context, which is the pool the first short request would capture, so this moves that
    /// one-time cost (tens of seconds for a large model) from a waiting request to start-up.
    /// Longer demand still rebuilds the pool at an idle boundary, and a failed capture leaves
    /// the quote for the first reservation to retry.
    /// # Errors
    /// Rejects a backend that is already executing.
    pub fn with_pool_captured(mut self) -> Result<Self> {
        let mut executor = self
            .ready
            .take()
            .ok_or_else(|| Error::invariant("CUDA executor is not idle"))?;
        executor.capture_funded_pool();
        Self::from_executor(*executor)
    }
    fn from_executor(executor: CudaExecutor) -> Result<Self> {
        let executor = Box::new(executor);
        let view = View::new(&executor)?;
        Ok(Self {
            ready: Some(executor),
            running: None,
            view,
            owner: Arc::new(()),
            generation: 0,
            busy: false,
            fatal: None,
            notification: Notification::new(),
            recycled_tasks: Vec::new(),
        })
    }
    #[must_use]
    pub const fn model(&self) -> &ModelIr {
        &self.view.model
    }
    /// # Errors
    /// This cached lookup is infallible; the result type preserves API compatibility.
    pub fn compile(&self, id: ProgramId) -> Result<ExecutionProgram> {
        let mut program = self.view.program.clone();
        program.id = id;
        Ok(program)
    }
    #[must_use]
    pub fn execution_profile(&self) -> ExecutionProfileInspection {
        self.view.profile.clone()
    }
    #[must_use]
    pub const fn prefix_inspection(&self) -> (u64, u64, u64, usize) {
        self.view.prefix
    }
    #[must_use]
    pub fn pool_inspection(&self) -> PoolInspection {
        self.view.pool.clone()
    }
    /// # Errors
    /// Rejects unknown states.
    pub fn prefill_width_for(&self, state: StateId) -> Result<usize> {
        Ok(self.view.state(state)?.program_width)
    }
    /// # Errors
    /// Rejects unknown states.
    pub fn prefill_geometry_for(&self, state: StateId) -> Result<infer_ir::PrefillGeometry> {
        Ok(self.view.state(state)?.geometry)
    }
    /// # Errors
    /// Rejects trimming while a device flight owns the pool.
    pub fn trim_state_pool(&mut self) -> Result<()> {
        self.control(CudaExecutor::trim_state_pool)
    }
    fn idle(&self) -> Result<&CudaExecutor> {
        if let Some(error) = &self.fatal {
            return Err(error.clone());
        }
        if self.busy {
            return Err(Error::new(
                ErrorCode::Conflict,
                "CUDA completion must be consumed first",
            ));
        }
        self.ready
            .as_deref()
            .ok_or_else(|| Error::invariant("CUDA executor missing"))
    }
    fn control<T>(&mut self, action: impl FnOnce(&mut CudaExecutor) -> Result<T>) -> Result<T> {
        self.idle()?;
        let executor = self
            .ready
            .as_deref_mut()
            .ok_or_else(|| Error::invariant("CUDA executor missing"))?;
        let result = action(executor);
        let refreshed = self.view.refresh(executor);
        result.and_then(|value| refreshed.map(|()| value))
    }
    fn start(
        &mut self,
        program: &ExecutionProgram,
        step: &infer_ir::StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<CudaTicket> {
        self.idle()?.check_tasks(program, step, &tasks)?;
        let generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorCode::Capacity, "CUDA ticket generations exhausted"))?;
        let executor = self
            .ready
            .take()
            .ok_or_else(|| Error::invariant("CUDA executor missing"))?;
        self.running = Some(Submission::new(executor, tasks));
        self.generation = generation;
        self.busy = true;
        self.notification.arm();
        // Poll once to enqueue the first graph before returning ownership to the caller.
        self.progress()?;
        Ok(CudaTicket {
            owner: Arc::clone(&self.owner),
            generation,
            state: TicketState::Pending,
            device_us: None,
        })
    }
    fn progress(&mut self) -> Result<()> {
        let Some(running) = self.running.as_mut() else {
            return Ok(());
        };
        if !self.notification.should_poll() {
            return Ok(());
        }
        let completed = match running.poll(self.notification.waker()) {
            Ok(Some(completed)) => completed,
            Ok(None) => return Ok(()),
            Err(error) => {
                self.fatal = Some(error);
                return Ok(());
            }
        };
        self.running = None;
        self.ready = Some(completed.executor);
        self.recycled_tasks = completed.tasks;
        self.recycled_tasks.clear();
        self.busy = true;
        let executor = self
            .ready
            .as_deref_mut()
            .ok_or_else(|| Error::invariant("CUDA executor missing"))?;
        if !completed.fenced {
            self.fatal = completed.result.err();
            return Ok(());
        }
        executor.busy = false;
        self.view.result = Some(completed.result);
        self.view.device_us = completed.device_us;
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../tests/unit/executor_owner.rs"]
mod tests;
