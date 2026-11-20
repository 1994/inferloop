//! Retention, parent lifetimes and export barriers belong to one bounded collector owner.
use infer_core::{Error, ErrorCode, Result, event::EventReader};
use infer_observe::{Diagnostic, ObservationStore, ObservationWindow, trace::TraceContext};
use std::thread::Builder;
use std::thread::Thread;
use std::thread::park;
use std::{
    collections::BTreeMap, sync::Arc, sync::atomic::AtomicBool, sync::atomic::AtomicU64,
    sync::atomic::AtomicUsize, sync::atomic::Ordering, sync::mpsc,
};

/// Bounded command channel depth for snapshot and diagnostic commands.
const COMMAND_CHANNEL_CAPACITY: usize = 258;
/// Events drained per collector iteration before other work is serviced.
const EVENT_DRAIN_QUANTUM: u64 = 128;
/// Metadata commands drained per collector iteration before events are drained.
const METADATA_DRAIN_QUANTUM: u64 = 64;
/// Commands serviced per collector iteration before parking.
const COMMAND_DRAIN_QUANTUM: usize = 64;

pub struct CollectedSnapshot {
    pub window: ObservationWindow,
    pub parents: BTreeMap<u64, TraceContext>,
    pub dropped_metadata: u64,
}
enum Command {
    Snapshot {
        events_target: u64,
        metadata_target: u64,
        events: bool,
        parents: bool,
        reply: mpsc::SyncSender<Result<CollectedSnapshot>>,
    },
    Diagnostic(Diagnostic),
}
enum Metadata {
    Parent {
        id: u64,
        context: TraceContext,
        credit: ParentCredit,
    },
    Retire {
        id: u64,
        target: u64,
    },
}
struct Shared {
    stop: AtomicBool,
    dropped_metadata: AtomicU64,
    dropped_diagnostics: AtomicU64,
    parents: AtomicUsize,
    parent_capacity: usize,
}
struct ParentCredit(Arc<Shared>);
impl Drop for ParentCredit {
    fn drop(&mut self) {
        self.0.parents.fetch_sub(1, Ordering::AcqRel);
    }
}
struct Parent {
    context: TraceContext,
    active: bool,
    _credit: ParentCredit,
}
pub struct CollectionTicket {
    receiver: mpsc::Receiver<Result<CollectedSnapshot>>,
}
impl CollectionTicket {
    /// Resolve only on an exporter worker; the scheduler never waits for retention.
    pub fn finish(self) -> Result<CollectedSnapshot> {
        self.receiver.recv().map_err(|_| {
            Error::new(
                ErrorCode::Backend,
                "event collector stopped before snapshot",
            )
        })?
    }
}
pub struct Collector {
    commands: mpsc::SyncSender<Command>,
    metadata: mpsc::SyncSender<Metadata>,
    issued: u64,
    shared: Arc<Shared>,
    thread: Thread,
}
impl Collector {
    pub fn new(
        reader: EventReader,
        store: ObservationStore,
        drained: u64,
        parent_capacity: usize,
    ) -> Result<Self> {
        let shared = Arc::new(Shared {
            stop: AtomicBool::new(false),
            dropped_metadata: AtomicU64::new(0),
            dropped_diagnostics: AtomicU64::new(0),
            parents: AtomicUsize::new(0),
            parent_capacity,
        });
        let (commands, receiver) = mpsc::sync_channel(COMMAND_CHANNEL_CAPACITY);
        let (metadata, metadata_receiver) = mpsc::sync_channel(parent_capacity * 2);
        let worker = shared.clone();
        let thread = Builder::new()
            .name("infer-event-collector".into())
            .spawn(move || {
                CollectionState {
                    reader,
                    store,
                    drained,
                    metadata_done: 0,
                    parents: BTreeMap::new(),
                }
                .run(&receiver, &metadata_receiver, &worker);
            })
            .map_err(|error| Error::new(ErrorCode::Backend, error.to_string()))?;
        Ok(Self {
            commands,
            metadata,
            issued: 0,
            shared,
            thread: thread.thread().clone(),
        })
    }
    pub fn wake(&self) {
        self.thread.unpark();
    }
    pub fn snapshot(
        &self,
        events_target: u64,
        events: bool,
        parents: bool,
    ) -> Result<CollectionTicket> {
        let (reply, receiver) = mpsc::sync_channel(1);
        self.commands
            .try_send(Command::Snapshot {
                events_target,
                metadata_target: self.issued,
                events,
                parents,
                reply,
            })
            .map_err(|error| match error {
                mpsc::TrySendError::Full(_) => Error::new(
                    ErrorCode::Capacity,
                    "event collector command budget exhausted",
                ),
                mpsc::TrySendError::Disconnected(_) => {
                    Error::new(ErrorCode::Backend, "event collector stopped")
                }
            })?;
        self.wake();
        Ok(CollectionTicket { receiver })
    }
    pub fn parent(&mut self, id: u64, context: TraceContext) -> bool {
        let mut used = self.shared.parents.load(Ordering::Acquire);
        loop {
            if used >= self.shared.parent_capacity {
                self.drop_metadata();
                return false;
            }
            match self.shared.parents.compare_exchange_weak(
                used,
                used + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(actual) => used = actual,
            }
        }
        let credit = ParentCredit(self.shared.clone());
        if self
            .metadata
            .try_send(Metadata::Parent {
                id,
                context,
                credit,
            })
            .is_err()
        {
            self.drop_metadata();
            return false;
        }
        self.issued = self.issued.saturating_add(1);
        self.wake();
        true
    }
    pub fn retire(&mut self, id: u64, target: u64) {
        // Each registered parent holds a credit for both metadata messages until retirement.
        if self
            .metadata
            .try_send(Metadata::Retire { id, target })
            .is_err()
        {
            self.drop_metadata();
        } else {
            self.issued = self.issued.saturating_add(1);
        }
        self.wake();
    }
    pub fn diagnostic(&self, diagnostic: Diagnostic) {
        if self
            .commands
            .try_send(Command::Diagnostic(diagnostic))
            .is_err()
        {
            self.shared
                .dropped_diagnostics
                .fetch_add(1, Ordering::Relaxed);
        }
        self.wake();
    }
    pub fn dropped_metadata(&self) -> u64 {
        self.shared.dropped_metadata.load(Ordering::Relaxed)
    }
    fn drop_metadata(&self) {
        self.shared.dropped_metadata.fetch_add(1, Ordering::Relaxed);
    }
}
impl Drop for Collector {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        self.wake();
    }
}
struct CollectionState {
    reader: EventReader,
    store: ObservationStore,
    drained: u64,
    metadata_done: u64,
    parents: BTreeMap<u64, Parent>,
}
impl CollectionState {
    fn drain(&mut self, target: u64) {
        while self.drained < target {
            let Ok(event) = self.reader.pop() else {
                break;
            };
            if let Some(id) = self.store.record_retirement(event)
                && self
                    .parents
                    .get(&id.get())
                    .is_some_and(|parent| !parent.active)
            {
                self.parents.remove(&id.get());
            }
            self.drained = self.drained.saturating_add(1);
        }
    }
    fn metadata(&mut self, receiver: &mpsc::Receiver<Metadata>, target: u64) {
        while self.metadata_done < target {
            let Ok(command) = receiver.try_recv() else {
                break;
            };
            match command {
                Metadata::Parent {
                    id,
                    context,
                    credit,
                } => {
                    self.parents.insert(
                        id,
                        Parent {
                            context,
                            active: true,
                            _credit: credit,
                        },
                    );
                }
                Metadata::Retire { id, target } => {
                    self.drain(target);
                    if !self.store.contains_request(id) {
                        self.parents.remove(&id);
                    } else if let Some(parent) = self.parents.get_mut(&id) {
                        parent.active = false;
                    }
                }
            }
            self.metadata_done = self.metadata_done.saturating_add(1);
        }
    }
    fn snapshot(
        &mut self,
        metadata: &mpsc::Receiver<Metadata>,
        events_target: u64,
        metadata_target: u64,
        events: bool,
        parents: bool,
        shared: &Shared,
    ) -> Result<CollectedSnapshot> {
        self.metadata(metadata, metadata_target);
        self.drain(events_target);
        if self.drained < events_target || self.metadata_done < metadata_target {
            return Err(Error::invariant(
                "collector snapshot publication barrier not reached",
            ));
        }
        let mut window = self.store.snapshot(true).as_of(events_target, events);
        let dropped_metadata = shared.dropped_metadata.load(Ordering::Relaxed);
        window.diagnostics_evicted = window
            .diagnostics_evicted
            .saturating_add(shared.dropped_diagnostics.load(Ordering::Relaxed));
        let parents = if parents {
            self.parents
                .iter()
                .map(|(id, parent)| (*id, parent.context.clone()))
                .collect()
        } else {
            BTreeMap::new()
        };
        Ok(CollectedSnapshot {
            window,
            parents,
            dropped_metadata,
        })
    }
    fn run(
        mut self,
        receiver: &mpsc::Receiver<Command>,
        metadata: &mpsc::Receiver<Metadata>,
        shared: &Shared,
    ) {
        loop {
            self.drain(self.drained.saturating_add(EVENT_DRAIN_QUANTUM));
            let metadata_before = self.metadata_done;
            self.metadata(
                metadata,
                self.metadata_done.saturating_add(METADATA_DRAIN_QUANTUM),
            );
            let metadata_progress = self.metadata_done != metadata_before;
            let mut commands = 0;
            while commands < COMMAND_DRAIN_QUANTUM {
                let Ok(command) = receiver.try_recv() else {
                    break;
                };
                match command {
                    Command::Snapshot {
                        events_target,
                        metadata_target,
                        events,
                        parents,
                        reply,
                    } => {
                        let _ = reply.send(self.snapshot(
                            metadata,
                            events_target,
                            metadata_target,
                            events,
                            parents,
                            shared,
                        ));
                    }
                    Command::Diagnostic(diagnostic) => self.store.diagnostic(diagnostic),
                }
                commands += 1;
            }
            if shared.stop.load(Ordering::Acquire)
                && self.reader.is_empty()
                && commands == 0
                && !metadata_progress
            {
                break;
            }
            if self.reader.is_empty() && commands == 0 && !metadata_progress {
                park();
            }
        }
    }
}

#[cfg(test)]
mod tests;
