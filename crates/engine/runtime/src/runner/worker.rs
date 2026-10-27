use super::{Completion, Shared};
use infer_core::set::BoundedSet;
use infer_core::{Error, Result, StateId};
use infer_gpu_api::BatchLease;
use infer_ir::ExecutionProgram;
use infer_spi::BackendProvider;
use std::thread::{current, park_timeout};
use std::{sync::Arc, sync::atomic::Ordering, sync::mpsc, time::Duration};

/// Control commands serviced before a pending compute submission is allowed to launch.
const CONTROL_LAUNCH_QUANTUM: usize = 8;
/// Abandoned states retired per maintenance pass before the drain loop yields.
const ABANDONED_DRAIN_BATCH: usize = 8;

pub(super) enum Job {
    Resource {
        command: infer_spi::ResourceCommand,
        reply: infer_spi::ResourceResponder,
        reserve: Option<StateId>,
        credit: super::ControlCredit,
    },
    Release(StateId),
}
struct Flight<T> {
    ticket: T,
    handle: BatchLease,
}
pub(super) struct Worker<B: BackendProvider> {
    pub backend: B,
    execution_profile: Option<Arc<infer_ir::ExecutionProfileInspection>>,
    pub states: BoundedSet<StateId>,
    program: Arc<ExecutionProgram>,
    pub shared: Arc<Shared>,
    receiver: mpsc::Receiver<Job>,
    flight: Option<Flight<B::Ticket>>,
    config: super::RunnerConfig,
    recycler: Option<infer_gpu_api::Consumer<super::Recycled>>,
    snapshots: Option<[Arc<super::Snapshot>; super::SNAPSHOT_SLOTS]>,
    submissions: infer_gpu_api::Consumer<BatchLease>,
    completions: infer_gpu_api::Producer<BatchLease>,
}
impl<B: BackendProvider> Worker<B> {
    pub fn new(
        backend: B,
        program: Arc<ExecutionProgram>,
        shared: Arc<Shared>,
        receiver: mpsc::Receiver<Job>,
        submissions: infer_gpu_api::Consumer<BatchLease>,
        completions: infer_gpu_api::Producer<BatchLease>,
        states: BoundedSet<StateId>,
    ) -> Self {
        let config = shared.config;
        let execution_profile = backend.execution_profile().map(Arc::new);
        Self {
            backend,
            execution_profile,
            states,
            program,
            shared,
            receiver,
            flight: None,
            config,
            recycler: None,
            snapshots: None,
            submissions,
            completions,
        }
    }
    pub fn with_recycler(mut self, recycler: infer_gpu_api::Consumer<super::Recycled>) -> Self {
        self.recycler = Some(recycler);
        self
    }
    fn recycle(&mut self) {
        let Some(recycler) = &mut self.recycler else {
            return;
        };
        let mut error = None;
        for _ in 0..self.config.max_num_seqs * super::RECYCLED_ENTRIES_PER_BATCH
            + super::RECYCLED_ENTRY_SLACK
        {
            let Ok(buffer) = recycler.pop() else {
                break;
            };
            let result = match buffer {
                super::Recycled::Output(state, output) => {
                    self.backend.recycle_output(state, output)
                }
                super::Recycled::Batch(batch) => self.backend.recycle_batch(batch),
            };
            if let Err(failure) = result {
                error.get_or_insert(failure);
            }
        }
        if let Some(error) = error {
            self.fail(error);
        }
    }
    pub fn with_snapshots(
        mut self,
        snapshots: [Arc<super::Snapshot>; super::SNAPSHOT_SLOTS],
    ) -> Self {
        self.snapshots = Some(snapshots);
        self
    }
    pub fn prepare(mut self) -> Self {
        self.refresh();
        self
    }
    /// Fetch owned diagnostic strings only after control-plane changes. Decode snapshots
    /// share this immutable Arc and never allocate another profile per token.
    fn refresh_profile(&mut self) {
        let profile = self.backend.execution_profile();
        if self.execution_profile.as_deref() != profile.as_ref() {
            self.execution_profile = profile.map(Arc::new);
        }
    }

    pub fn refresh(&mut self) {
        let Some(pool) = &mut self.snapshots else {
            return;
        };
        let Some(index) = pool
            .iter()
            .position(|snapshot| Arc::strong_count(snapshot) == 1)
        else {
            return;
        };
        let result = (|| {
            let snapshot = Arc::get_mut(&mut pool[index])
                .ok_or_else(|| Error::invariant("snapshot reader acquired unpublished slot"))?;
            snapshot.free_bytes = self.backend.free_state_bytes()?;
            snapshot.state_capacity_bytes = self.backend.state_capacity_bytes()?;
            snapshot.kv = self.backend.kv_cache();
            snapshot
                .execution_profile
                .clone_from(&self.execution_profile);
            snapshot.growth.clear();
            snapshot.prefill.clear();
            for state in self.states.iter() {
                if let Some(growth) = self.backend.state_page_growth(*state)? {
                    snapshot.growth.insert(*state, growth)?;
                }
                if let Some(geometry) = self.backend.state_prefill_geometry(*state)? {
                    snapshot.prefill.insert(*state, geometry)?;
                }
            }
            Ok(())
        })();
        if let Ok(mut published) = self.shared.snapshot.lock() {
            match result {
                Ok(()) => {
                    // Slot filling runs outside the publication lock. Readers hold immutable Arcs.
                    if let Some(snapshot) = Arc::get_mut(&mut pool[index]) {
                        snapshot.error.clone_from(&published.error);
                    }
                    let changed = published.execution_profile != pool[index].execution_profile
                        || published.free_bytes != pool[index].free_bytes
                        || published.state_capacity_bytes != pool[index].state_capacity_bytes
                        || published.growth != pool[index].growth
                        || published.prefill != pool[index].prefill
                        || published
                            .kv
                            .as_ref()
                            .map(|kv| (kv.available_blocks, kv.free_blocks))
                            != pool[index]
                                .kv
                                .as_ref()
                                .map(|kv| (kv.available_blocks, kv.free_blocks));
                    *published = pool[index].clone();
                    if changed {
                        let epoch = self.shared.epoch.load(Ordering::Relaxed);
                        self.shared
                            .epoch
                            .store(epoch.saturating_add(1), Ordering::Release);
                    }
                }
                Err(error) => Arc::make_mut(&mut published).error = Some(error),
            }
        }
    }
    fn collect_abandoned(&mut self) {
        // The snapshot is republished once for the whole drain. Rebuilding it per retired state
        // is O(retired x tracked states) of backend growth queries for a view no reader can
        // observe between iterations, because this owner is the only writer of `states`.
        let mut drained = false;
        for _ in 0..ABANDONED_DRAIN_BATCH {
            let Some(state) = self
                .shared
                .abandoned
                .lock()
                .ok()
                .and_then(|mut states| states.pop_first())
            else {
                break;
            };
            drained = true;
            let result = if self.states.contains(&state) {
                self.backend.release_state(state)
            } else {
                Ok(())
            };
            match result {
                Ok(()) => {
                    self.states.remove(&state);
                    self.shared.epoch.fetch_add(1, Ordering::AcqRel);
                    self.shared.releases.fetch_sub(1, Ordering::AcqRel);
                }
                Err(error) => {
                    if let Ok(mut snapshot) = self.shared.snapshot.lock() {
                        Arc::make_mut(&mut snapshot).error = Some(error);
                    }
                    if let Err(error) = self.shared.remember_abandoned(state) {
                        self.fail(error);
                    }
                }
            }
        }
        if drained {
            self.refresh_profile();
            self.refresh();
        }
    }
    fn execute(&mut self, job: Job) {
        match job {
            Job::Resource {
                command,
                reply,
                reserve,
                credit,
            } => {
                let _credit = credit;
                if reply.abandoned() {
                    return;
                }
                let changes_state = matches!(
                    &command,
                    infer_spi::ResourceCommand::ReservationBytes { .. }
                        | infer_spi::ResourceCommand::Reserve { .. }
                        | infer_spi::ResourceCommand::ReserveMedia { .. }
                        | infer_spi::ResourceCommand::Reset { .. }
                        | infer_spi::ResourceCommand::Prefix { .. }
                );
                let result = infer_spi::execute_resource(&mut self.backend, command);
                if result.is_ok()
                    && let Some(state) = reserve
                    && let Err(error) = self.states.insert(state)
                {
                    self.fail(error);
                }
                if changes_state {
                    self.refresh_profile();
                    self.refresh();
                }
                let _ = reply.send(result);
            }
            Job::Release(state) => {
                let result = if self.states.contains(&state) {
                    self.backend.release_state(state)
                } else {
                    Ok(())
                };
                match result {
                    Ok(()) => {
                        self.states.remove(&state);
                        self.shared.epoch.fetch_add(1, Ordering::AcqRel);
                        self.shared.releases.fetch_sub(1, Ordering::AcqRel);
                    }
                    Err(error) => {
                        self.fail(error);
                        if let Err(error) = self.shared.remember_abandoned(state) {
                            self.fail(error);
                        }
                    }
                }
                self.refresh_profile();
                self.refresh();
            }
        }
    }
    pub fn run(mut self) {
        let owner = current();
        self.backend.set_waker(Arc::new(move || owner.unpark()));
        let mut disconnected = false;
        let mut poll_us = self.config.poll_min_us;
        let mut controls = 0usize;
        loop {
            self.recycle();
            if let Err(error) = self.backend.maintenance() {
                self.fail(error);
            }
            if self.poll() {
                poll_us = self.config.poll_min_us;
            } else {
                poll_us = poll_us.saturating_mul(2).min(self.config.poll_max_us);
            }
            if self.flight.is_none() {
                self.collect_abandoned();
                // Resource traffic cannot starve an already published compute batch.
                if controls >= CONTROL_LAUNCH_QUANTUM {
                    controls = 0;
                    if let Ok(handle) = self.submissions.pop() {
                        self.launch(handle);
                        continue;
                    }
                }
                match self.receiver.try_recv() {
                    Ok(job) => {
                        controls += 1;
                        self.execute(job);
                        self.shared.notify();
                        continue;
                    }
                    Err(mpsc::TryRecvError::Disconnected) => disconnected = true,
                    Err(mpsc::TryRecvError::Empty) => {}
                }
                if let Ok(handle) = self.submissions.pop() {
                    controls = 0;
                    self.launch(handle);
                    continue;
                }
            }
            if disconnected && self.flight.is_none() && self.submissions.is_abandoned() {
                if let Err(error) = self.shared.batches.reclaim_abandoned() {
                    self.fail(error);
                }
                break;
            }
            park_timeout(if self.flight.is_some() {
                Duration::from_micros(poll_us)
            } else {
                Duration::from_secs(1)
            });
        }
    }
    fn fail(&self, error: Error) {
        if let Ok(mut snapshot) = self.shared.snapshot.lock() {
            Arc::make_mut(&mut snapshot).error = Some(error);
        }
        self.shared.notify();
    }
    fn publish(&mut self, handle: BatchLease, completion: Result<Completion>) {
        let result = self
            .shared
            .batches
            .complete(handle, completion)
            .and_then(|()| {
                self.completions
                    .push(handle)
                    .map_err(|_| Error::invariant("reserved completion ring credit missing"))
            });
        if let Err(error) = result {
            self.fail(error);
        }
        self.shared.notify();
    }
    fn launch(&mut self, handle: BatchLease) {
        let result = self.shared.batches.apply(handle, |step, tasks| {
            self.backend.submit_borrowed(&self.program, step, tasks)
        });
        if let Err(error) = self.shared.batches.acknowledge(handle, result.is_ok()) {
            self.fail(error);
        }
        match result {
            Ok(ticket) => self.flight = Some(Flight { ticket, handle }),
            Err(error) => self.publish(handle, Err(error)),
        }
        self.refresh();
        self.shared.encoding.store(false, Ordering::Release);
        self.shared.notify();
    }
    fn poll(&mut self) -> bool {
        let Some(mut flight) = self.flight.take() else {
            return false;
        };
        self.shared.encoding.store(true, Ordering::Release);
        match self.backend.poll(&mut flight.ticket) {
            Ok(None) => self.flight = Some(flight),
            result => {
                self.refresh();
                let timing = self.backend.completion_timing(&flight.ticket);
                let result = result.and_then(|outputs| {
                    outputs
                        .map(|outputs| Completion { outputs, timing })
                        .ok_or_else(|| Error::invariant("completion disappeared"))
                });
                self.shared.encoding.store(false, Ordering::Release);
                self.publish(flight.handle, result);
                self.shared.notify();
            }
        }
        self.shared.encoding.store(false, Ordering::Release);
        self.flight.is_none()
    }
}
