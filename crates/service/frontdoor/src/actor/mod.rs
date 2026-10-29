//! Command admission, engine driving, and bounded output delivery.
mod admission;
mod commands;
mod delivery;
mod driver;
mod handle;
mod startup;
use crate::constants::GENERATED_TOKEN_STAGING_BYTES;
use crate::constants::TOKEN_STAGING_BYTES;
use crate::cpu::CpuConfig;
use crate::cpu::CpuPool;
use crate::cpu::Credit;
use crate::ingress::Pending;
use crate::ingress::Registry;
use infer_core::{Error, RequestId, Result};
use infer_ir::CanonicalRequest;
use infer_observe::{ObservationQuery, trace::TraceContext};
use infer_runtime::{Engine, EngineOutput, RuntimeInspection};
use infer_spi::BackendProvider;
use std::sync::atomic::AtomicU64;
use std::thread::Thread;
use std::{
    collections::BTreeMap, collections::VecDeque, sync::Arc, sync::atomic::AtomicBool,
    sync::atomic::Ordering, sync::mpsc, sync::mpsc::SyncSender, time::Instant,
};
use tokio::{sync::mpsc as async_mpsc, sync::oneshot};

type OutputChannel = async_mpsc::Sender<Result<EngineOutput>>;
/// Maximum deferred bulk commands buffered before control commands are served again.
const DEFERRED_QUEUE_CAPACITY: usize = 256;
/// Pre-allocated capacity of the pending admission-quote queue.
const QUOTING_QUEUE_CAPACITY: usize = 256;
/// Maximum pending shutdown waiters the actor keeps before rejecting further stops.
const STOP_WAITER_CAPACITY: usize = 256;
/// Extra observed events charged beyond the configured history capacity.
const OBSERVATION_RESERVED_EVENTS: usize = 256;
/// Conservative exporter expansion multiplier applied to each retained event's footprint.
const OBSERVATION_EXPORT_EXPANSION: usize = 32;
/// Staging bytes charged per admitted parent request.
const OBSERVATION_PARENT_BYTES: usize = 1024;
/// Fixed observation staging headroom kept for exporter overhead.
const OBSERVATION_HEADROOM_BYTES: usize = 64 << 10;
enum Command {
    Submit {
        request: Box<infer_runtime::PreparedRequest>,
        pending: Arc<Pending>,
        trace: Option<TraceContext>,
        output: OutputChannel,
        reply: oneshot::Sender<Result<()>>,
    },
    Cancel {
        request: RequestId,
        preparing: bool,
        reply: oneshot::Sender<Result<()>>,
    },
    Inspect {
        reply: oneshot::Sender<RuntimeInspection>,
    },
    Observe {
        query: ObservationQuery,
        credit: Credit,
        reply: oneshot::Sender<Result<(infer_runtime::ObservationSnapshot, Credit)>>,
    },
    Stop {
        reply: oneshot::Sender<Result<()>>,
    },
}
struct Subscriber {
    channel: OutputChannel,
    terminal: Option<Result<EngineOutput>>,
}
#[derive(Clone)]
pub struct RuntimeHandle {
    commands: Arc<SyncSender<Command>>,
    control: Arc<SyncSender<Command>>,
    wake: Thread,
    stop: Arc<AtomicBool>,
    preparer: infer_runtime::RequestPreparer,
    cpu: CpuPool,
    delivery: CpuPool,
    observation: CpuPool,
    observation_bytes: usize,
    config: CpuConfig,
    ingress: Arc<Registry>,
    clock_started: Instant,
    clock_origin_us: u64,
    request_ids: Arc<AtomicU64>,
}

struct RuntimeActor<B: BackendProvider> {
    engine: Engine<B>,
    receiver: mpsc::Receiver<Command>,
    control: mpsc::Receiver<Command>,
    stop: Arc<AtomicBool>,
    deferred: VecDeque<Command>,
    /// Submissions awaiting their sizing quote. `None` waits for room in the device owner's
    /// command lane; the quote is issued again on a later pass instead of refusing the request.
    quoting: VecDeque<(Command, Option<infer_spi::ResourceTicket>)>,
    subscribers: BTreeMap<RequestId, Subscriber>,
    start: Instant,
    clock_origin_us: u64,
    stopping: bool,
    stop_replies: Vec<oneshot::Sender<Result<()>>>,
    /// Reused identity buffer for the drain passes. Reaping and cancellation both need a
    /// `RequestId` list before mutating the engine, and both run on every actor iteration, so the
    /// buffer is retained across iterations instead of reallocated per pass.
    scratch: Vec<RequestId>,
}
impl Subscriber {
    fn flush(&mut self) -> bool {
        if self.channel.is_closed() && self.terminal.is_some() {
            return false;
        }
        let Some(done) = self.terminal.take() else {
            return true;
        };
        match self.channel.try_send(done) {
            Ok(()) | Err(async_mpsc::error::TrySendError::Closed(_)) => false,
            Err(async_mpsc::error::TrySendError::Full(done)) => {
                self.terminal = Some(done);
                true
            }
        }
    }
}

fn preparation_bytes(request: &CanonicalRequest) -> Result<usize> {
    let tokens = match &request.input {
        infer_ir::RequestInput::Sequence { tokens, .. } => tokens.len(),
        infer_ir::RequestInput::Pairs { query, documents } => documents
            .iter()
            .try_fold(0usize, |n, doc| {
                n.checked_add(query.len())
                    .and_then(|n| n.checked_add(doc.len()))
            })
            .ok_or_else(|| Error::invalid("preparation token size overflow"))?,
    };
    let generated = if let infer_ir::Workload::Generate { max_new_tokens } = request.workload {
        max_new_tokens
    } else {
        0
    };
    tokens
        .checked_mul(TOKEN_STAGING_BYTES)
        .and_then(|n| {
            generated
                .checked_mul(GENERATED_TOKEN_STAGING_BYTES)
                .and_then(|tail| n.checked_add(tail))
        })
        .ok_or_else(|| Error::invalid("preparation byte size overflow"))
}

impl Drop for RuntimeHandle {
    fn drop(&mut self) {
        if Arc::strong_count(&self.commands) == 1 {
            if let Err(error) = self.ingress.stop() {
                tracing::error!(target: "infer::frontdoor", %error, "failed to stop ingress during handle drop");
            }
            self.stop.store(true, Ordering::Release);
            self.wake.unpark();
        }
    }
}

// Charge retained pages and conservative exporter expansion before the scheduler captures them.
fn observation_bytes(config: &infer_runtime::RuntimeConfig) -> Result<usize> {
    config
        .history_capacity
        .checked_add(OBSERVATION_RESERVED_EVENTS)
        .and_then(|events| {
            events.checked_mul(
                size_of::<infer_observe::ObservedEvent>() * OBSERVATION_EXPORT_EXPANSION,
            )
        })
        .and_then(|bytes| {
            config
                .max_requests
                .checked_mul(OBSERVATION_PARENT_BYTES)
                .and_then(|parents| bytes.checked_add(parents))
        })
        .and_then(|bytes| bytes.checked_add(OBSERVATION_HEADROOM_BYTES))
        .ok_or_else(|| Error::invalid("observation staging budget overflow"))
}

#[cfg(test)]
#[path = "../../tests/unit/actor.rs"]
mod tests;
