//! Explicit stream futures for execution; synchronous adapters are diagnostic compatibility.
use super::{CudaDevice, device_error};
use cutile::cuda_async::device_future::DeviceFuture;
use cutile::cuda_async::device_operation::value;
use cutile::prelude::*;
use infer_core::Result;
use std::{
    future::Future,
    pin::pin,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
    thread::{self, Thread},
};

impl CudaDevice {
    pub(crate) async fn run_op<O: DeviceOp>(
        &self,
        operation: O,
    ) -> Result<<O as DeviceOp>::Output> {
        DeviceFuture::scheduled(operation, ExecutionContext::new(Arc::clone(&self.stream)))
            .await
            .map_err(device_error)
    }
    pub(crate) async fn fence(&self) -> Result<()> {
        self.run_op(value(())).await
    }
}

struct ThreadWake(Thread);
impl Wake for ThreadWake {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}
/// Drive synchronous diagnostic APIs using the same async implementation. The provider
/// never calls this adapter: its owner polls the stored execution future instead.
pub fn block_on<T>(future: impl Future<Output = T>) -> T {
    let waker = Waker::from(Arc::new(ThreadWake(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => thread::park(),
        }
    }
}
