//! Submit device-library host work without blocking the CUDA owner thread.
use crate::device::{CudaDevice, device_error};
use cutile::cuda_async::device_future::DeviceFuture;
use cutile::prelude::{DeviceError, DeviceOp, ExecutionContext, Tensor};
use futures_channel::oneshot;
use infer_core::{Error, ErrorCode, Result};
use infer_spi::DeviceHostExecutor;
use std::future::Future;
use std::pin::Pin;
use std::result::Result as StdResult;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

type Pending = Pin<Box<dyn Future<Output = StdResult<Tensor<f32>, DeviceError>> + Send>>;

pub(super) enum Submission {
    Ready(Tensor<f32>),
    Pending(Pending),
}

impl Submission {
    pub async fn finish(self) -> Result<Tensor<f32>> {
        match self {
            Self::Ready(output) => Ok(output),
            Self::Pending(future) => future.await.map_err(device_error),
        }
    }
}

/// Poll once on a worker: any library-side initialization or reallocation stays off the owner.
pub(super) fn kickoff<O: DeviceOp<Output = Tensor<f32>> + 'static>(
    device: &CudaDevice,
    operation: O,
) -> Result<Submission> {
    let context = ExecutionContext::new(Arc::clone(&device.stream));
    let mut future = Box::pin(DeviceFuture::scheduled(operation, context));
    match future
        .as_mut()
        .poll(&mut Context::from_waker(Waker::noop()))
    {
        Poll::Ready(Ok(output)) => Ok(Submission::Ready(output)),
        Poll::Ready(Err(error)) => Err(device_error(error)),
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
