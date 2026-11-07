//! Submit device-library host work without blocking the CUDA owner thread.
use futures_channel::oneshot;
use infer_core::{Error, ErrorCode, Result};
use infer_cuda_kernels::runtime::Allocation;
use infer_spi::DeviceHostExecutor;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

type Pending = Pin<Box<dyn Future<Output = Result<Allocation<f32>>> + Send>>;

pub(super) enum Submission {
    Ready(Allocation<f32>),
    Pending(Pending),
}

impl Submission {
    pub async fn finish(self) -> Result<Allocation<f32>> {
        match self {
            Self::Ready(output) => Ok(output),
            Self::Pending(future) => future.await,
        }
    }
}

/// Poll once on a worker: any library-side initialization or reallocation stays off the owner.
pub(super) fn kickoff(
    operation: impl Future<Output = Result<Allocation<f32>>> + Send + 'static,
) -> Result<Submission> {
    let mut future = Box::pin(operation);
    match future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(output) => output.map(Submission::Ready),
        Poll::Pending => Ok(Submission::Pending(future)),
    }
}

/// Hand a pending CUDA future back to its owner; cancellation leaves it owned by the worker.
pub(super) async fn submit<T: Send + 'static>(
    executor: &dyn DeviceHostExecutor,
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    let (sender, receiver) = oneshot::channel();
    executor.submit(Box::new(move || {
        if sender.is_canceled() {
            return;
        }
        let result = work();
        if let Err(abandoned) = sender.send(result) {
            // Dropping an abandoned device future on this worker waits for its stream fence.
            drop(abandoned);
        }
    }))?;
    receiver
        .await
        .map_err(|_| Error::new(ErrorCode::Backend, "device host worker stopped"))?
}
