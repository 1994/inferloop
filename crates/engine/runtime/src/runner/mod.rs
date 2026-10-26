//! Bounded CPU submission owner. Backend encoding/readback cannot block the scheduler owner.
mod control;
mod worker;
use crate::constants::MAX_BOUNDED_CAPACITY;
use crate::constants::MAX_POLL_INTERVAL_US;
use infer_core::map::BoundedMap;
use infer_core::placement::ThreadPlacement;
use infer_core::set::BoundedSet;
use infer_core::{Error, ErrorCode, Result, StateId, StepId};
use infer_ir::{
    DeviceCapabilities, ExecutionProgram, ExecutionTiming, KvCacheInspection, ModelIr, PageGrowth,
    TaskOutput,
};
use infer_spi::BackendProvider;
use std::collections::VecDeque;
use std::marker::PhantomData;
use std::sync::OnceLock;
use std::thread::Thread;
use std::{
    sync::Arc, sync::Mutex, sync::atomic::AtomicBool, sync::atomic::AtomicU64,
    sync::atomic::AtomicUsize, sync::atomic::Ordering, sync::mpsc,
};
use worker::{Job, Worker};

/// Recycled readback entries reserved for each batch cell.
const RECYCLED_ENTRIES_PER_BATCH: usize = 4;
/// Extra recycled readback entries reserved for traffic already in flight.
const RECYCLED_ENTRY_SLACK: usize = 8;
/// Immutable device snapshots published to readers between owner refreshes.
const SNAPSHOT_SLOTS: usize = 3;
/// Resource cells reserved beyond queued control commands and tracked states.
const RESOURCE_POOL_SLACK: usize = 4;
/// Smallest state ledger accepted by the default runner configuration.
const MIN_TRACKED_STATES: usize = 256;

/// CPU submission limits are frozen independently of the backend's device architecture.
#[derive(Debug, Clone, Copy)]
pub struct RunnerConfig {
    pub control_capacity: usize,
    pub max_num_seqs: usize,
    pub max_states: usize,
    pub batch_slots: usize,
    pub poll_min_us: u64,
    pub poll_max_us: u64,
}
impl RunnerConfig {
    /// # Errors
    /// Rejects invalid ring dimensions and unbounded poll latency.
    pub fn validate(self) -> Result<()> {
        if self.control_capacity == 0
            || self.max_states == 0
            || self.max_states > MAX_BOUNDED_CAPACITY
            || self.max_num_seqs == 0
            || self.max_num_seqs > infer_gpu_api::MAX_SUBMISSION_BATCH
            || self.batch_slots < 2
            || self.batch_slots > infer_gpu_api::MAX_SUBMISSION_BATCH
            || self.poll_min_us == 0
            || self.poll_max_us < self.poll_min_us
            || self.poll_max_us > MAX_POLL_INTERVAL_US
        {
            return Err(Error::invalid("invalid CPU device owner limits"));
        }
        Ok(())
    }
}
type Wake = Arc<dyn Fn() + Send + Sync>;
#[derive(Clone)]
struct Snapshot {
    free_bytes: Option<u64>,
    state_capacity_bytes: Option<u64>,
    kv: Option<KvCacheInspection>,
    growth: BoundedMap<StateId, PageGrowth>,
    prefill: BoundedMap<StateId, infer_ir::PrefillGeometry>,
    error: Option<Error>,
}
impl Snapshot {
    fn new(states: usize) -> Result<Self> {
        Ok(Self {
            free_bytes: None,
            state_capacity_bytes: None,
            kv: None,
            growth: BoundedMap::new(states)?,
            prefill: BoundedMap::new(states)?,
            error: None,
        })
    }
}
struct Shared {
    config: RunnerConfig,
    encoding: AtomicBool,
    flight_abandoned: AtomicBool,
    batches: Arc<infer_gpu_api::BatchArena<Result<Completion>>>,
    device_thread: OnceLock<Thread>,
    snapshot: Mutex<Arc<Snapshot>>,
    wake: Mutex<Option<Wake>>,
    releases: AtomicUsize,
    controls: AtomicUsize,
    abandoned: Mutex<BoundedSet<StateId>>,
    epoch: AtomicU64,
}
struct ControlCredit(Arc<Shared>);
impl Drop for ControlCredit {
    fn drop(&mut self) {
        self.0.controls.fetch_sub(1, Ordering::AcqRel);
        self.0.notify();
    }
}
impl Shared {
    fn notify(&self) {
        // Clone the waker out and release the lock before invoking it. The callback is opaque, so
        // waking under the lock would serialize every notify behind it and deadlock any waker that
        // re-entered `notify`/`set_waker`.
        let wake = self.wake.lock().ok().and_then(|wake| wake.clone());
        if let Some(wake) = wake {
            wake();
        }
    }
    fn snapshot(&self) -> Result<Arc<Snapshot>> {
        self.snapshot
            .lock()
            .map(|s| s.clone())
            .map_err(|_| Error::invariant("device snapshot poisoned"))
    }
    fn remember_abandoned(&self, state: StateId) -> Result<bool> {
        self.abandoned
            .lock()
            .map_err(|_| Error::invariant("abandoned-state ledger poisoned"))?
            .insert(state)
    }
}
impl infer_spi::ResourceAbandon for Shared {
    fn abandon_state(&self, state: StateId) {
        match self.remember_abandoned(state) {
            Ok(true) => {
                self.releases.fetch_add(1, Ordering::AcqRel);
            }
            Ok(false) => {}
            Err(error) => {
                if let Ok(mut snapshot) = self.snapshot.lock() {
                    Arc::make_mut(&mut snapshot).error = Some(error);
                }
            }
        }
        self.notify();
        if let Some(thread) = self.device_thread.get() {
            thread.unpark();
        }
    }
}
enum Recycled {
    Output(StateId, infer_ir::ModelOutput),
    Batch(Vec<TaskOutput>),
}
struct Completion {
    outputs: Vec<TaskOutput>,
    timing: Option<ExecutionTiming>,
}
/// A pending launch and GPU completion share one ticket; timeout never destroys driver ownership.
pub struct RunnerTicket {
    owner: Arc<Shared>,
    step: StepId,
    batch: infer_gpu_api::BatchLease,
    timing: Option<ExecutionTiming>,
    done: bool,
}
/// The service uses this adapter; direct CLI/quality execution can retain a synchronous backend.
#[derive(Clone, Copy)]
enum Multimodal {
    Enabled,
    Disabled,
}
impl Multimodal {
    const fn supported(self) -> bool {
        matches!(self, Self::Enabled)
    }
}
pub struct ThreadedBackend<B: BackendProvider> {
    recycled: infer_gpu_api::Producer<Recycled>,
    recipe: Option<infer_ir::StateRecipe>,
    execution_profile: Option<infer_ir::ExecutionProfileInspection>,
    jobs: mpsc::SyncSender<Job>,
    submissions: infer_gpu_api::Producer<infer_gpu_api::BatchLease>,
    completions: infer_gpu_api::Consumer<infer_gpu_api::BatchLease>,
    shared: Arc<Shared>,
    resources: infer_spi::ResourcePool,
    marker: PhantomData<B>,
    identity: String,
    capabilities: DeviceCapabilities,
    speculation: infer_ir::SpeculationCapability,
    model: ModelIr,
    program: Arc<ExecutionProgram>,
    weight_backed: bool,
    multimodal: Multimodal,
    recompute: bool,
    checkpoint: bool,
    active: Option<StepId>,
    state_ids: BoundedSet<StateId>,
    pending_releases: VecDeque<StateId>,
}
impl<B: BackendProvider + Send + 'static> ThreadedBackend<B>
where
    B::Ticket: Send,
{
    /// # Errors
    /// Rejects invalid limits or failure to initialize/start the driver owner.
    pub fn new(
        backend: B,
        model: ModelIr,
        program: ExecutionProgram,
        queue: usize,
    ) -> Result<Self> {
        let cpu = crate::CpuRuntimeConfig::default();
        Self::with_config(
            backend,
            model,
            program,
            RunnerConfig {
                control_capacity: queue,
                max_num_seqs: infer_gpu_api::MAX_SUBMISSION_BATCH,
                max_states: queue.max(MIN_TRACKED_STATES),
                batch_slots: 2,
                poll_min_us: cpu.poll_min_us,
                poll_max_us: cpu.poll_max_us,
            },
        )
    }
    /// # Errors
    /// Rejects invalid CPU limits or inability to start the dedicated device owner.
    pub fn with_config(
        backend: B,
        model: ModelIr,
        program: ExecutionProgram,
        config: RunnerConfig,
    ) -> Result<Self> {
        Self::with_states(backend, model, program, config, &[])
    }
    /// Attach already restored states before starting the asynchronous owner.
    /// # Errors
    /// Rejects invalid CPU budgets, duplicate states or owner startup failure.
    pub fn with_states(
        backend: B,
        model: ModelIr,
        program: ExecutionProgram,
        config: RunnerConfig,
        states: &[StateId],
    ) -> Result<Self> {
        Self::with_states_placed(
            backend,
            model,
            program,
            config,
            states,
            &ThreadPlacement::default(),
        )
    }
    /// Initialize owner storage under its policy before first-touch and apply it to the worker.
    /// # Errors
    /// Rejects unavailable placement, invalid dimensions or owner startup failure.
    pub fn with_states_placed(
        backend: B,
        model: ModelIr,
        program: ExecutionProgram,
        config: RunnerConfig,
        states: &[StateId],
        placement: &ThreadPlacement,
    ) -> Result<Self> {
        placement.scope(|| Self::initialize(backend, model, program, config, states, placement))
    }
    fn initialize(
        backend: B,
        model: ModelIr,
        program: ExecutionProgram,
        config: RunnerConfig,
        states: &[StateId],
        placement: &ThreadPlacement,
    ) -> Result<Self> {
        config.validate()?;
        let mut state_ids = BoundedSet::new(config.max_states)?;
        for state in states {
            if !state_ids.insert(*state)? {
                return Err(Error::invalid("duplicate restored device states"));
            }
        }
        let mut snapshot = Snapshot::new(config.max_states)?;
        snapshot.free_bytes = backend.free_state_bytes()?;
        snapshot.state_capacity_bytes = backend.state_capacity_bytes()?;
        snapshot.kv = backend.kv_cache();
        let snapshot = Arc::new(snapshot);
        let snapshots = [
            snapshot.clone(),
            Arc::new(Snapshot::new(config.max_states)?),
            Arc::new(Snapshot::new(config.max_states)?),
        ];
        let shared = Arc::new(Shared {
            config,
            encoding: AtomicBool::new(false),
            flight_abandoned: AtomicBool::new(false),
            batches: Arc::new(infer_gpu_api::BatchArena::new(
                config.batch_slots,
                config.max_num_seqs,
            )?),
            device_thread: OnceLock::new(),
            snapshot: Mutex::new(snapshot),
            wake: Mutex::new(None),
            releases: AtomicUsize::new(0),
            controls: AtomicUsize::new(0),
            abandoned: Mutex::new(BoundedSet::new(config.max_states)?),
            epoch: AtomicU64::new(0),
        });
        let (jobs, receiver) = mpsc::sync_channel(config.control_capacity);
        let (submissions, submit_rx) = infer_gpu_api::RingBuffer::new(config.batch_slots);
        let (complete_tx, completions) = infer_gpu_api::RingBuffer::new(config.batch_slots);
        let (recycled, recycled_rx) = infer_gpu_api::RingBuffer::new(
            config.max_num_seqs * RECYCLED_ENTRIES_PER_BATCH + RECYCLED_ENTRY_SLACK,
        );
        let program = Arc::new(program);
        let runner = Self {
            recipe: backend.state_recipe().cloned(),
            execution_profile: backend.execution_profile(),
            recycled,
            resources: infer_spi::ResourcePool::new(
                config
                    .control_capacity
                    .checked_add(config.max_states)
                    .and_then(|n| n.checked_add(RESOURCE_POOL_SLACK))
                    .ok_or_else(|| Error::invalid("resource pool dimensions overflow"))?,
            )?,
            marker: PhantomData,
            identity: backend.identity().into(),
            capabilities: backend.capabilities(),
            speculation: backend.speculation_capability(),
            weight_backed: backend.weight_backed_dataflow(),
            multimodal: if backend.supports_multimodal() {
                Multimodal::Enabled
            } else {
                Multimodal::Disabled
            },
            recompute: backend.supports_recompute_preemption(),
            checkpoint: backend.supports_control_checkpoint(),
            jobs,
            submissions,
            completions,
            shared: shared.clone(),
            model,
            program: program.clone(),
            active: None,
            state_ids: state_ids.clone(),
            pending_releases: VecDeque::with_capacity(config.max_states),
        };
        let worker_shared = shared.clone();
        let thread = placement.spawn("infer-device-submit".into(), move |_| {
            Worker::new(
                backend,
                program,
                worker_shared,
                receiver,
                submit_rx,
                complete_tx,
                state_ids,
            )
            .with_recycler(recycled_rx)
            .with_snapshots(snapshots)
            .prepare()
            .run();
        })?;
        shared
            .device_thread
            .set(thread.thread().clone())
            .map_err(|_| Error::invariant("device owner registered twice"))?;
        Ok(runner)
    }
    fn enqueue(&self, job: Job) -> Result<()> {
        self.jobs.try_send(job).map_err(|e| match e {
            mpsc::TrySendError::Full(_) => {
                Error::new(ErrorCode::Capacity, "device command queue full")
            }
            mpsc::TrySendError::Disconnected(_) => {
                Error::new(ErrorCode::Backend, "device owner stopped")
            }
        })?;
        if let Some(thread) = self.shared.device_thread.get() {
            thread.unpark();
        }
        Ok(())
    }
}
impl Drop for RunnerTicket {
    fn drop(&mut self) {
        if !self.done {
            let _ = self.owner.batches.abandon(self.batch);
            self.owner.flight_abandoned.store(true, Ordering::Release);
            if let Some(thread) = self.owner.device_thread.get() {
                thread.unpark();
            }
        }
    }
}
