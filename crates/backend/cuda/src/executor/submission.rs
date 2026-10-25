//! One owned coroutine spans prefill, draft, verification and final device fencing.
use super::{
    CudaExecutor,
    profiling::{PhaseScope, StepProfile},
};
use crate::device::{CudaDevice, device_error};
use infer_core::Result;
use infer_ir::{ExecutionTask, TaskOutput};
use std::{
    future::Future,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll, Waker},
};

pub struct CudaTicket {
    pub(super) owner: Arc<()>,
    pub(super) generation: u64,
    pub(super) state: TicketState,
}
pub(super) enum TicketState {
    Pending,
    Consumed,
}
pub(super) struct Completed {
    pub executor: Box<CudaExecutor>,
    pub tasks: Vec<ExecutionTask>,
    pub result: Result<Vec<TaskOutput>>,
    pub fenced: bool,
}
pub(super) struct Submission {
    future: Pin<Box<dyn Future<Output = Completed> + Send>>,
    device: CudaDevice,
    phases: Option<super::profiling::PhaseRecords>,
}
impl Submission {
    pub(super) fn new(mut executor: Box<CudaExecutor>, tasks: Vec<ExecutionTask>) -> Self {
        let device = executor.loaded.device().clone();
        let profile = StepProfile::start(&executor, &tasks);
        let phases = profile.as_ref().map(|_| PhaseScope::records());
        let report_phases = phases.clone();
        let future = Box::pin(async move {
            let result = executor.execute(&tasks).await;
            // Detached restores and draft catch-up are part of the same completion.
            let fence = executor.loaded.device().fence().await;
            let fenced = fence.is_ok();
            let result = result.and_then(|outputs| fence.map(|()| outputs));
            if let Some(profile) = profile {
                profile.finish(
                    &executor,
                    result.as_deref().ok(),
                    PhaseScope::take_records(report_phases),
                );
            }
            Completed {
                executor,
                tasks,
                result,
                fenced,
            }
        });
        Self {
            future,
            device,
            phases,
        }
    }
    pub(super) fn poll(&mut self, waker: &Waker) -> Result<Option<Completed>> {
        self.device
            .stream
            .device()
            .bind_to_thread()
            .map_err(device_error)?;
        let _phases = PhaseScope::bind(self.phases.clone());
        match self.future.as_mut().poll(&mut Context::from_waker(waker)) {
            Poll::Ready(completed) => Ok(Some(completed)),
            Poll::Pending => Ok(None),
        }
    }
}
