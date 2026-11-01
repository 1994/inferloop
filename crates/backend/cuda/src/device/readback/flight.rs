//! Split submission and completion. The future retains all DMA and graph owners.
use super::{ReadbackSource, Readbacks, Rows, device_error};
use crate::device::CudaDevice;
use cuda_core::{DType, DeviceCopy};
use cutile::cuda_async::device_future::DeviceFuture;
use cutile::cuda_async::device_operation::BoxedDeviceOp;
use cutile::prelude::*;
use infer_core::{Error, ErrorCode, Result};
use infer_cuda_kernels::tile::runtime::Graph;
use std::future::{Future, poll_fn};
use std::pin::Pin;
use std::task::{Context, Poll, Waker};

type Completion = DeviceFuture<Vec<()>, BoxedDeviceOp<Vec<()>>>;

pub(super) struct Flight<T: DType + DeviceCopy> {
    completion: Completion,
    ready: bool,
    error: Option<Error>,
    token_owner: Option<Arc<()>>,
    sources: Vec<Option<ReadbackSource<T>>>,
}

impl<T: DType + DeviceCopy> Readbacks<T> {
    pub(crate) async fn read_async(
        &mut self,
        device: &CudaDevice,
        source: Arc<Tensor<T>>,
    ) -> Result<Vec<T>> {
        if self.poisoned || self.flight.is_some() {
            return Err(Error::invalid("busy or poisoned tensor readback"));
        }
        let sources = [Some(ReadbackSource::whole(source))];
        self.prepare(device, &sources)?;
        let copies = self.copies(&sources)?;
        self.start(device, copies.boxed(), &sources)?;
        let mut rows = poll_fn(|context| match self.poll_with(context) {
            Ok(Some(rows)) => Poll::Ready(Ok(rows)),
            Ok(None) => Poll::Pending,
            Err(error) => Poll::Ready(Err(error)),
        })
        .await?;
        rows.pop()
            .ok_or_else(|| Error::invariant("tensor readback row"))
    }

    /// Enqueue replay and D2H without waiting. This owner cannot be reused until poll succeeds.
    pub(crate) fn submit(
        &mut self,
        device: &CudaDevice,
        graph: &Graph,
        sources: &[Option<ReadbackSource<T>>],
    ) -> Result<()> {
        self.validate_submit(device, graph)?;
        self.prepare(device, sources)?;
        let copies = self.copies(sources)?;
        self.start(
            device,
            graph.launch().then(move |()| copies).boxed(),
            sources,
        )
    }

    fn validate_submit(&self, device: &CudaDevice, graph: &Graph) -> Result<()> {
        if self.poisoned || self.flight.is_some() || !Arc::ptr_eq(graph.stream(), &device.stream) {
            return Err(Error::invalid(
                "busy/poisoned readback or mismatched CUDA graph stream",
            ));
        }
        Ok(())
    }

    fn start(
        &mut self,
        device: &CudaDevice,
        op: BoxedDeviceOp<Vec<()>>,
        sources: &[Option<ReadbackSource<T>>],
    ) -> Result<()> {
        self.poisoned = true;
        let completion =
            DeviceFuture::scheduled(op, ExecutionContext::new(Arc::clone(&device.stream)));
        self.flight = Some(Flight {
            completion,
            ready: false,
            error: None,
            token_owner: None,
            sources: sources.to_vec(),
        });
        // First poll actually enqueues the operation. Keep the future on failure too:
        // its submission owns leases even when enqueue only partially succeeded.
        self.progress()?;
        Ok(())
    }

    fn progress(&mut self) -> Result<bool> {
        self.progress_with(&mut Context::from_waker(Waker::noop()))
    }
    fn progress_with(&mut self, context: &mut Context<'_>) -> Result<bool> {
        let flight = self
            .flight
            .as_mut()
            .ok_or_else(|| Error::invalid("no CUDA readback flight"))?;
        if let Some(error) = &flight.error {
            return Err(error.clone());
        }
        if !flight.ready {
            match Pin::new(&mut flight.completion).poll(context) {
                Poll::Pending => return Ok(false),
                Poll::Ready(result) => {
                    if let Err(error) = result {
                        let error = device_error(error);
                        flight.error = Some(error.clone());
                        return Err(error);
                    }
                    flight.ready = true;
                }
            }
        }
        Ok(true)
    }

    pub(crate) fn poll(&mut self) -> Result<Option<Rows<T>>> {
        self.poll_with(&mut Context::from_waker(Waker::noop()))
    }
    fn poll_with(&mut self, context: &mut Context<'_>) -> Result<Option<Rows<T>>> {
        if self
            .flight
            .as_ref()
            .is_some_and(|flight| flight.token_owner.is_some())
        {
            return Err(Error::invalid("paired readback requires its token owner"));
        }
        if !self.progress_with(context)? {
            return Ok(None);
        }
        let flight = self
            .flight
            .as_ref()
            .ok_or_else(|| Error::invariant("readback flight"))?;
        let rows = self.results(&flight.sources)?;
        self.flight = None;
        self.poisoned = false;
        Ok(Some(rows))
    }
}

impl Readbacks<f32> {
    pub(crate) fn submit_with_tokens(
        &mut self,
        device: &CudaDevice,
        graph: &Graph,
        sources: &[Option<ReadbackSource>],
        tail: (&Graph, &mut Readbacks<i32>, Arc<Tensor<i32>>),
    ) -> Result<()> {
        let (selector, tokens, tensor) = tail;
        self.validate_submit(device, graph)?;
        tokens.validate_submit(device, selector)?;
        let token_sources = [Some(ReadbackSource::whole(tensor))];
        let bytes = self
            .staging_bytes(device, sources)?
            .checked_add(tokens.staging_bytes(device, &token_sources)?)
            .ok_or_else(|| Error::invalid("pinned size overflow"))?;
        if bytes > super::MAX_READBACK_STAGING_BYTES {
            return Err(Error::new(
                ErrorCode::Capacity,
                "pinned readbacks exceed 64 MiB",
            ));
        }
        self.prepare(device, sources)?;
        tokens.prepare(device, &token_sources)?;
        let copies = self.copies(sources)?;
        let token_copies = tokens.copies(&token_sources)?;
        let selector = selector.launch();
        tokens.poisoned = true;
        self.start(
            device,
            graph
                .launch()
                .then(move |()| selector)
                .then(move |()| copies)
                .then(move |_| token_copies)
                .boxed(),
            sources,
        )?;
        self.flight
            .as_mut()
            .ok_or_else(|| Error::invariant("paired flight"))?
            .token_owner = Some(Arc::clone(&tokens.owner));
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn poll_with_tokens(
        &mut self,
        tokens: &mut Readbacks<i32>,
    ) -> Result<Option<(Rows<f32>, Rows<i32>)>> {
        self.poll_tokens_with(tokens, &mut Context::from_waker(Waker::noop()))
    }
    fn poll_tokens_with(
        &mut self,
        tokens: &mut Readbacks<i32>,
        context: &mut Context<'_>,
    ) -> Result<Option<(Rows<f32>, Rows<i32>)>> {
        if !self
            .flight
            .as_ref()
            .and_then(|flight| flight.token_owner.as_ref())
            .is_some_and(|owner| Arc::ptr_eq(owner, &tokens.owner))
        {
            return Err(Error::invalid("foreign token readback owner"));
        }
        if !self.progress_with(context)? {
            return Ok(None);
        }
        let integers = tokens
            .buffers
            .iter()
            .map(|buffer| {
                let buffer = buffer
                    .as_ref()
                    .ok_or_else(|| Error::invariant("token pinned buffer"))?;
                Ok(buffer.lock().map_err(device_error)?.as_slice().to_vec())
            })
            .collect::<Result<_>>()?;
        self.flight
            .as_mut()
            .ok_or_else(|| Error::invariant("paired flight"))?
            .token_owner = None;
        let rows = self
            .poll()?
            .ok_or_else(|| Error::invariant("completed readback"))?;
        tokens.poisoned = false;
        Ok(Some((rows, integers)))
    }
}

impl<T: DType + DeviceCopy> Readbacks<T> {
    pub(crate) async fn run_async(
        &mut self,
        device: &CudaDevice,
        graph: &Graph,
        sources: &[Option<ReadbackSource<T>>],
    ) -> Result<Rows<T>> {
        self.submit(device, graph, sources)?;
        poll_fn(|context| match self.poll_with(context) {
            Ok(Some(rows)) => Poll::Ready(Ok(rows)),
            Ok(None) => Poll::Pending,
            Err(error) => Poll::Ready(Err(error)),
        })
        .await
    }
}
impl Readbacks<f32> {
    pub(crate) async fn run_with_tokens_async(
        &mut self,
        device: &CudaDevice,
        graph: &Graph,
        sources: &[Option<ReadbackSource>],
        tail: (&Graph, &mut Readbacks<i32>, Arc<Tensor<i32>>),
    ) -> Result<(Rows<f32>, Rows<i32>)> {
        let (selector, tokens, tensor) = tail;
        self.submit_with_tokens(device, graph, sources, (selector, tokens, tensor))?;
        poll_fn(|context| match self.poll_tokens_with(tokens, context) {
            Ok(Some(rows)) => Poll::Ready(Ok(rows)),
            Ok(None) => Poll::Pending,
            Err(error) => Poll::Ready(Err(error)),
        })
        .await
    }
}
