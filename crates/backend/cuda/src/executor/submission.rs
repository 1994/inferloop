//! One owned coroutine spans prefill, draft, verification and final device fencing.
use super::profiling;
use super::{
    CudaExecutor,
    profiling::{PhaseScope, StepProfile, TimerBinding},
};
use crate::device::CudaDevice;
use crate::device::device_timer::DeviceTimer;
use infer_core::Result;
use infer_ir::{DeviceTimingSample, ExecutionTask, TaskOutput};
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
    pub(super) device_us: Option<u64>,
    pub(super) device_timing: Option<DeviceTimingSample>,
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
    pub device_us: Option<u64>,
    pub device_timing: Option<DeviceTimingSample>,
}
pub(super) struct Submission {
    future: Pin<Box<dyn Future<Output = Completed> + Send>>,
    device: CudaDevice,
    phases: Option<profiling::PhaseRecords>,
    timer: TimerBinding,
}

/// Run one change on the step timer; timing is diagnostic, so a failure only skips it.
fn timed<T>(
    timer: &TimerBinding,
    change: impl FnOnce(&mut DeviceTimer, &CudaDevice) -> Result<T>,
) -> Option<T> {
    match timer.timer.lock() {
        Ok(mut guard) => change(&mut guard, &timer.device)
            .inspect_err(|error| {
                tracing::debug!(target: "infer::profile", %error, "device step timing skipped");
            })
            .ok(),
        Err(error) => {
            tracing::error!(target: "infer::profile", %error, "device timer poisoned");
            None
        }
    }
}
impl Submission {
    pub(super) fn new(mut executor: Box<CudaExecutor>, tasks: Vec<ExecutionTask>) -> Self {
        let device = executor.loaded.device().clone();
        let profile = StepProfile::start(&executor, &tasks);
        let phases = profile.as_ref().map(|_| PhaseScope::records());
        let report_phases = phases.clone();
        let timer = TimerBinding {
            device: device.clone(),
            timer: Arc::clone(&executor.timer),
        };
        let step_timer = timer.clone();
        let future = Box::pin(async move {
            let clocked = executor
                .clock
                .as_ref()
                .is_some_and(|clock| clock.start(executor.loaded.device()));
            let rows = tasks.len();
            timed(&step_timer, |timer, device| timer.begin_step(device, rows));
            let result = executor.execute(&tasks).await;
            timed(&step_timer, DeviceTimer::end_step);
            let clocked = clocked
                && executor
                    .clock
                    .as_ref()
                    .is_some_and(|clock| clock.stop(executor.loaded.device()));
            // Detached restores and draft catch-up are part of the same completion.
            let fence = executor.loaded.device().fence().await;
            let fenced = fence.is_ok();
            let device_us = (clocked && fenced && result.is_ok())
                .then(|| executor.clock.as_ref()?.elapsed_us())
                .flatten();
            // Events are only readable once the fence has passed; an unfenced or failed step
            // reports no timeline, and the timer still resets for the next one.
            let device_timing = timed(&step_timer, |timer, _| timer.sample())
                .flatten()
                .filter(|_| fenced && result.is_ok());
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
                device_us,
                device_timing,
            }
        });
        Self {
            future,
            device,
            phases,
            timer,
        }
    }
    pub(super) fn poll(&mut self, waker: &Waker) -> Result<Option<Completed>> {
        self.device.tile().bind_to_thread()?;
        let _phases = PhaseScope::bind(self.phases.clone(), Some(self.timer.clone()));
        match self.future.as_mut().poll(&mut Context::from_waker(waker)) {
            Poll::Ready(completed) => Ok(Some(completed)),
            Poll::Pending => Ok(None),
        }
    }
}
