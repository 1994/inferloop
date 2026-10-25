//! Resource commands carry typed acknowledgements and cancellation ownership.
use crate::BackendProvider;
use infer_core::{Error, ErrorCode, Result, StateId};
use infer_ir::{OutputReadout, TokenBuffer};
use std::result::Result as StdResult;
use std::{sync::Arc, sync::atomic::AtomicBool, sync::atomic::Ordering, sync::mpsc};

mod pool;
pub use pool::ResourcePool;
pub trait ResourceAbandon: Send + Sync {
    fn abandon_state(&self, state: StateId);
}
enum Abandon {
    Callback(Box<dyn FnOnce() + Send>),
    State(Arc<dyn ResourceAbandon>, StateId),
}
enum Receiver {
    Channel(mpsc::Receiver<Result<ResourceReply>>, Arc<AtomicBool>),
    Pool(pool::Reader),
}
enum Responder {
    Channel(mpsc::SyncSender<Result<ResourceReply>>, Arc<AtomicBool>),
    Pool(pool::Reply),
}
#[derive(Clone)]
pub enum ResourceCommand {
    ReservationBytes {
        capacity: usize,
        readout: OutputReadout,
    },
    Reserve {
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
    },
    /// Reserve state for a media request, which must not create speculative state.
    ReserveMedia {
        state: StateId,
        capacity: usize,
        readout: OutputReadout,
    },
    Reset {
        state: StateId,
    },
    Checkpoint {
        states: Vec<(StateId, usize, usize)>,
    },
    Prefix {
        state: StateId,
        tokens: TokenBuffer,
        maximum: usize,
    },
}
#[derive(Debug)]
pub enum ResourceReply {
    ReservationBytes(Option<u64>),
    Reserved,
    Reset,
    Checkpoint(Option<Vec<u8>>),
    Prefix(usize),
}
/// Dropping a ticket abandons the reply, not backend ownership. Asynchronous owners
/// must install compensation with `on_abandon` before publishing a reservation ticket.
pub struct ResourceTicket {
    receiver: Receiver,
    done: bool,
    abandon: Option<Abandon>,
}
pub struct ResourceResponder {
    reply: Responder,
}
impl ResourceTicket {
    #[must_use]
    pub fn channel() -> (Self, ResourceResponder) {
        let (reply, receiver) = mpsc::sync_channel(1);
        let cancelled = Arc::new(AtomicBool::new(false));
        (
            Self {
                receiver: Receiver::Channel(receiver, cancelled.clone()),
                done: false,
                abandon: None,
            },
            ResourceResponder {
                reply: Responder::Channel(reply, cancelled),
            },
        )
    }
    const fn pooled(reader: pool::Reader) -> Self {
        Self {
            receiver: Receiver::Pool(reader),
            done: false,
            abandon: None,
        }
    }
    #[must_use]
    pub fn on_abandon_state(mut self, owner: Arc<dyn ResourceAbandon>, state: StateId) -> Self {
        self.abandon = Some(Abandon::State(owner, state));
        self
    }
    /// Install owner-side compensation that also covers a published but unconsumed reply.
    #[must_use]
    pub fn on_abandon(mut self, action: impl FnOnce() + Send + 'static) -> Self {
        self.abandon = Some(Abandon::Callback(Box::new(action)));
        self
    }
    /// # Errors
    /// Returns a command failure or a disconnected owner error; never waits.
    pub fn poll(&mut self) -> Result<Option<ResourceReply>> {
        if self.done {
            return Err(Error::invariant(
                "resource acknowledgement already consumed",
            ));
        }
        let Receiver::Channel(receiver, _) = &self.receiver else {
            let Receiver::Pool(reader) = &self.receiver else {
                return Err(Error::invariant("invalid resource receiver"));
            };
            let result = reader.poll();
            self.done = !matches!(result, Ok(None));
            return result;
        };
        match receiver.try_recv() {
            Ok(result) => {
                self.done = true;
                result.map(Some)
            }
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => {
                self.done = true;
                Err(Error::new(
                    ErrorCode::Backend,
                    "resource owner disconnected",
                ))
            }
        }
    }
}
impl Drop for ResourceTicket {
    fn drop(&mut self) {
        if !self.done {
            match &self.receiver {
                Receiver::Channel(_, cancelled) => cancelled.store(true, Ordering::Release),
                Receiver::Pool(reader) => reader.cancel(),
            }
            match self.abandon.take() {
                Some(Abandon::Callback(action)) => action(),
                Some(Abandon::State(owner, state)) => owner.abandon_state(state),
                None => {}
            }
        }
    }
}
impl ResourceResponder {
    const fn pooled(reply: pool::Reply) -> Self {
        Self {
            reply: Responder::Pool(reply),
        }
    }
    #[must_use]
    pub fn abandoned(&self) -> bool {
        match &self.reply {
            Responder::Channel(_, cancelled) => cancelled.load(Ordering::Acquire),
            Responder::Pool(reply) => reply.reader.cell.cancelled.load(Ordering::Acquire),
        }
    }
    /// Publish once. The returned reply retains ownership if the receiver disappeared.
    /// # Errors
    /// Returns the unpublished result when the receiving ticket has been dropped.
    pub fn send(self, result: Result<ResourceReply>) -> StdResult<(), Result<ResourceReply>> {
        match self.reply {
            Responder::Channel(reply, _) => reply.send(result).map_err(|error| error.0),
            Responder::Pool(reply) => reply.send(result),
        }
    }
}
/// Execute on the backend's owning thread. The generic engine never calls this on a proxy.
/// # Errors
/// Returns the underlying allocator/cache/layout error without hiding resource failure.
pub fn execute_resource<B: BackendProvider + ?Sized>(
    backend: &mut B,
    command: ResourceCommand,
) -> Result<ResourceReply> {
    match command {
        ResourceCommand::ReservationBytes { capacity, readout } => {
            backend.prepare_state_reservation(capacity, readout)?;
            backend
                .state_reservation_bytes_for(capacity, readout)
                .map(ResourceReply::ReservationBytes)
        }
        ResourceCommand::Reserve {
            state,
            capacity,
            readout,
        } => backend
            .reserve_state_for(state, capacity, readout)
            .map(|()| ResourceReply::Reserved),
        ResourceCommand::ReserveMedia {
            state,
            capacity,
            readout,
        } => backend
            .reserve_state_for_media(state, capacity, readout)
            .map(|()| ResourceReply::Reserved),
        ResourceCommand::Reset { state } => {
            backend.reset_state(state).map(|()| ResourceReply::Reset)
        }
        ResourceCommand::Checkpoint { states } => {
            backend.validate_state_ownership(&states)?;
            backend
                .capture_execution_state()
                .map(ResourceReply::Checkpoint)
        }
        ResourceCommand::Prefix {
            state,
            tokens,
            maximum,
        } => backend
            .reuse_prefix_shared(state, tokens, maximum)
            .map(ResourceReply::Prefix),
    }
}

#[cfg(test)]
// The case bodies live under `tests/`; the module stays a child of this one so the tests keep
// access to its private items without opening a product API.
#[path = "../tests/unit/resource.rs"]
mod tests;
