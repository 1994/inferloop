use infer_backend_cuda::executor::CudaBackend;
use infer_core::{DecisionId, Error, KernelId, Result, StateId, StepId};
use infer_ir::{ExecutionProgram, ExecutionRole, ExecutionTask, PlannedWork, StepPlan, TaskOutput};
use infer_spi::BackendProvider;
use std::{
    thread::yield_now,
    time::{Duration, Instant},
};

/// Wall-clock budget for an actual-model provider completion.
const COMPLETION_TIMEOUT: Duration = Duration::from_secs(30);
fn step(program: &ExecutionProgram, tasks: &[ExecutionTask], count: usize) -> StepPlan {
    StepPlan {
        id: StepId::ONE,
        decision: DecisionId::ONE,
        program: program.id,
        role: ExecutionRole::Forward,
        work: tasks
            .iter()
            .map(|t| PlannedWork {
                request: t.request,
                state: t.state,
                token_count: count,
                role: ExecutionRole::Forward,
            })
            .collect(),
        cost: infer_ir::CostEstimate::default(),
        graph: None,
        quantum_overrun: false,
    }
}
pub fn submit(
    backend: &mut CudaBackend,
    program: &ExecutionProgram,
    tasks: &[ExecutionTask],
    count: usize,
) -> Result<Vec<TaskOutput>> {
    let step = step(program, tasks, count);
    let mut ticket = backend.submit_borrowed(program, &step, tasks)?;
    if backend.trim_state_pool().is_ok()
        || backend.release_state(tasks[0].state).is_ok()
        || backend.reset_state(tasks[0].state).is_ok()
        || backend.submit_borrowed(program, &step, tasks).is_ok()
    {
        return Err(Error::invariant(
            "CUDA pending completion did not retain state ownership",
        ));
    }
    let deadline = Instant::now() + COMPLETION_TIMEOUT;
    let outputs = loop {
        if let Some(outputs) = backend.poll(&mut ticket)? {
            break outputs;
        }
        if Instant::now() >= deadline {
            return Err(Error::invariant("CUDA completion timed out"));
        }
        yield_now();
    };
    if backend.poll(&mut ticket).is_ok() {
        return Err(Error::invariant("CUDA completion consumed twice"));
    }
    Ok(outputs)
}
pub fn invalid_reservations(
    backend: &mut CudaBackend,
    state: StateId,
    capacity: usize,
) -> Result<()> {
    if backend.reserve_state(state, capacity).is_ok()
        || backend.reserve_state(StateId::new(3)?, capacity).is_ok()
        || backend.state_reservation_bytes(32769).is_ok()
        || backend.state_reservation_bytes(0).is_ok()
    {
        return Err(Error::invariant("CUDA reservation limits not enforced"));
    }
    Ok(())
}
pub fn invalid_submissions(
    backend: &mut CudaBackend,
    program: &ExecutionProgram,
    tasks: &[ExecutionTask],
) -> Result<()> {
    let mut plan = step(program, tasks, tasks[0].tokens.len());
    plan.work[0].token_count += 1;
    if backend.submit_borrowed(program, &plan, tasks).is_ok() {
        return Err(Error::invariant("CUDA invalid cursor accepted"));
    }
    let mut wrong = program.clone();
    wrong.operations[0].kernel = KernelId::ONE;
    if backend.validate_program(backend.model(), &wrong).is_ok() {
        return Err(Error::invariant("CUDA unregistered kernel accepted"));
    }
    Ok(())
}
pub fn argmax(values: &[f32]) -> Result<u32> {
    values
        .iter()
        .enumerate()
        .max_by(|(a, x), (b, y)| x.total_cmp(y).then_with(|| b.cmp(a)))
        .map(|(i, _)| u32::try_from(i))
        .transpose()
        .map_err(|e| Error::invalid(e.to_string()))?
        .ok_or_else(|| Error::invalid("empty logits"))
}
