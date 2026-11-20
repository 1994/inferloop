//! Generation-checked sealed batches. Rings carry handles; work storage stays in its slot.
use crate::constants::HANDLE_GENERATION_SHIFT;
use crate::{MAX_SUBMISSION_BATCH, SubmissionDescriptor};
use infer_core::{BatchHandle, Error, ErrorCode, OwnerId, Result, new_owner_id};
use infer_ir::{ExecutionTask, StepPlan};
use std::{
    sync::Arc, sync::Mutex, sync::atomic::AtomicBool, sync::atomic::AtomicU8,
    sync::atomic::AtomicU32, sync::atomic::Ordering,
};

/// A generation is local to a pool. The owner identity rejects handles from another pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatchLease {
    owner: OwnerId,
    handle: BatchHandle,
}
impl BatchLease {
    #[must_use]
    pub const fn handle(self) -> BatchHandle {
        self.handle
    }
}
const FREE: u8 = 0;
const BUILDING: u8 = 1;
const PUBLISHED: u8 = 2;
const APPLYING: u8 = 3;
const LAUNCHED: u8 = 4;
const COMPLETE: u8 = 5;
const RETIRED: u8 = 6;
struct Storage<C> {
    step: Option<Arc<StepPlan>>,
    tasks: Vec<ExecutionTask>,
    descriptor: Option<SubmissionDescriptor>,
    completion: Option<C>,
}
struct Slot<C> {
    generation: AtomicU32,
    state: AtomicU8,
    launch: AtomicU8,
    abandoned: AtomicBool,
    storage: Mutex<Storage<C>>,
}
/// One producer seals batches; one device owner applies/fences them; ticket consumption retires them.
pub struct BatchArena<C> {
    slots: Vec<Slot<C>>,
    batch: usize,
    owner: OwnerId,
}
impl<C> BatchArena<C> {
    /// # Errors
    /// Rejects invalid dimensions or inability to allocate persistent work storage.
    pub fn new(slots: usize, batch: usize) -> Result<Self> {
        if slots == 0 || slots >= u32::MAX as usize || batch == 0 || batch > MAX_SUBMISSION_BATCH {
            return Err(Error::invalid("invalid sealed batch arena dimensions"));
        }
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(slots)
            .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
        for _ in 0..slots {
            let mut tasks = Vec::new();
            tasks
                .try_reserve_exact(batch)
                .map_err(|e| Error::new(ErrorCode::Capacity, e.to_string()))?;
            storage.push(Slot {
                generation: AtomicU32::new(1),
                state: AtomicU8::new(FREE),
                launch: AtomicU8::new(0),
                abandoned: AtomicBool::new(false),
                storage: Mutex::new(Storage {
                    step: None,
                    tasks,
                    descriptor: None,
                    completion: None,
                }),
            });
        }
        Ok(Self {
            slots: storage,
            batch,
            owner: new_owner_id()?,
        })
    }
    fn slot(&self, lease: BatchLease) -> Result<&Slot<C>> {
        if lease.owner != self.owner {
            return Err(Error::invariant("foreign batch owner"));
        }
        let handle = lease.handle;
        let index = u32::try_from(handle.get() & u64::from(u32::MAX))
            .ok()
            .and_then(|n| n.checked_sub(1))
            .ok_or_else(|| Error::invalid("invalid batch handle slot"))?;
        let slot = self
            .slots
            .get(index as usize)
            .ok_or_else(|| Error::invalid("batch handle out of bounds"))?;
        if u64::from(slot.generation.load(Ordering::Acquire))
            != handle.get() >> HANDLE_GENERATION_SHIFT
            || matches!(slot.state.load(Ordering::Acquire), FREE | RETIRED)
        {
            return Err(Error::invariant("stale batch generation"));
        }
        Ok(slot)
    }
    /// # Errors
    /// Rejects mismatched work, stale ownership or an exhausted arena. No driver work exists yet.
    pub fn seal(&self, step: Arc<StepPlan>, tasks: &[ExecutionTask]) -> Result<BatchLease> {
        if tasks.len() > self.batch
            || tasks.len() != step.work.len()
            || tasks
                .iter()
                .zip(&step.work)
                .any(|(task, work)| task.request != work.request || task.state != work.state)
        {
            return Err(Error::invalid("sealed batch task identity mismatch"));
        }
        // Validate before reserving a slot so fallible ABI conversion cannot leak its credit.
        let mut descriptor = SubmissionDescriptor::from_plan(&step, 0)?;
        descriptor.owner = self.owner.get();
        for (work, task) in descriptor.work.iter_mut().zip(tasks) {
            work.computed_frontier = u32::try_from(task.tokens.computed_frontier())
                .map_err(|_| Error::invalid("computed frontier exceeds submission ABI"))?;
            work.readout = match task.tokens.readout() {
                infer_ir::OutputReadout::None => 0,
                infer_ir::OutputReadout::Logits => 1,
                infer_ir::OutputReadout::Full => 2,
            };
        }
        for (index, slot) in self.slots.iter().enumerate() {
            if slot
                .state
                .compare_exchange(FREE, BUILDING, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            let mut storage = slot
                .storage
                .lock()
                .map_err(|_| Error::invariant("batch storage poisoned"))?;
            descriptor.metadata_slot =
                u32::try_from(index).map_err(|_| Error::invalid("batch slot ABI overflow"))?;
            storage.tasks.clear();
            storage.tasks.extend(tasks.iter().cloned());
            storage.step = Some(step);
            descriptor.generation = slot.generation.load(Ordering::Relaxed);
            storage.descriptor = Some(descriptor);
            drop(storage);
            slot.launch.store(0, Ordering::Relaxed);
            slot.abandoned.store(false, Ordering::Relaxed);
            let generation = slot.generation.load(Ordering::Relaxed);
            let handle = BatchHandle::new(
                (u64::from(generation) << HANDLE_GENERATION_SHIFT) | (index as u64 + 1),
            )?;
            slot.state.store(PUBLISHED, Ordering::Release);
            return Ok(BatchLease {
                owner: self.owner,
                handle,
            });
        }
        Err(Error::new(ErrorCode::Capacity, "sealed batch arena full"))
    }
    /// Revoke publication only if the handle has not entered a ring or a device owner.
    /// # Errors
    /// Rejects revocation after application has begun.
    pub fn revoke(&self, handle: BatchLease) -> Result<()> {
        let slot = self.slot(handle)?;
        slot.state
            .compare_exchange(PUBLISHED, BUILDING, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::invariant("batch revoke after device observation"))?;
        Self::recycle(slot)
    }
    /// # Errors
    /// Rejects stale/duplicate application. The callback borrows sealed arrays until encoding returns.
    pub fn apply<T>(
        &self,
        handle: BatchLease,
        launch: impl FnOnce(&StepPlan, &[ExecutionTask]) -> Result<T>,
    ) -> Result<T> {
        let slot = self.slot(handle)?;
        slot.state
            .compare_exchange(PUBLISHED, APPLYING, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::invariant("batch applied more than once"))?;
        let mut storage = slot
            .storage
            .lock()
            .map_err(|_| Error::invariant("batch storage poisoned"))?;
        let step = storage
            .step
            .as_ref()
            .ok_or_else(|| Error::invariant("sealed step missing"))?;
        let result = launch(step, &storage.tasks);
        storage.tasks.clear();
        result
    }
    /// # Errors
    /// The driver must return a ticket before acknowledging acceptance; rejection implies no device readers.
    pub fn acknowledge(&self, handle: BatchLease, accepted: bool) -> Result<()> {
        let slot = self.slot(handle)?;
        slot.state
            .compare_exchange(APPLYING, LAUNCHED, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| Error::invariant("invalid launch acknowledgement"))?;
        slot.launch
            .store(if accepted { 1 } else { 2 }, Ordering::Release);
        Ok(())
    }
    /// # Errors
    /// Rejects stale identity or duplicate completion. Call only after all device readers finish.
    pub fn complete(&self, handle: BatchLease, completion: C) -> Result<()> {
        let slot = self.slot(handle)?;
        if slot.state.load(Ordering::Acquire) != LAUNCHED {
            return Err(Error::invariant("completion before launch acknowledgement"));
        }
        let mut storage = slot
            .storage
            .lock()
            .map_err(|_| Error::invariant("batch storage poisoned"))?;
        if storage.completion.is_some() {
            return Err(Error::invariant("duplicate batch completion"));
        }
        storage.completion = Some(completion);
        drop(storage);
        slot.state.store(COMPLETE, Ordering::Release);
        Ok(())
    }
    /// # Errors
    /// Rejects stale generations. Pending polls never contend with a slow encoding callback.
    pub fn take_completion(&self, handle: BatchLease) -> Result<Option<C>> {
        let slot = self.slot(handle)?;
        if slot
            .state
            .compare_exchange(COMPLETE, BUILDING, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Ok(None);
        }
        let completion = slot
            .storage
            .lock()
            .map_err(|_| Error::invariant("batch storage poisoned"))?
            .completion
            .take();
        if completion.is_none() {
            return Err(Error::invariant("batch completion consumed twice"));
        }
        Self::recycle(slot)?;
        Ok(completion)
    }
    fn recycle(slot: &Slot<C>) -> Result<()> {
        let mut storage = slot
            .storage
            .lock()
            .map_err(|_| Error::invariant("batch storage poisoned"))?;
        storage.step = None;
        storage.descriptor = None;
        storage.tasks.clear();
        drop(storage);
        let generation = slot.generation.load(Ordering::Relaxed);
        if let Some(next) = generation.checked_add(1) {
            slot.generation.store(next, Ordering::Release);
            slot.state.store(FREE, Ordering::Release);
        } else {
            slot.state.store(RETIRED, Ordering::Release);
        }
        Ok(())
    }
    /// # Errors
    /// Returns generation errors; dropping a ticket does not revoke published GPU ownership.
    pub fn abandon(&self, handle: BatchLease) -> Result<()> {
        let slot = self.slot(handle)?;
        slot.abandoned.store(true, Ordering::Release);
        Ok(())
    }
    /// A device owner reclaims abandoned tickets only after publishing the real fence.
    /// # Errors
    /// Rejects poisoned storage. Live tickets and pending driver readers are never reclaimed.
    pub fn reclaim_abandoned(&self) -> Result<()> {
        for slot in &self.slots {
            if slot.abandoned.load(Ordering::Acquire)
                && slot
                    .state
                    .compare_exchange(COMPLETE, BUILDING, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                slot.storage
                    .lock()
                    .map_err(|_| Error::invariant("batch storage poisoned"))?
                    .completion = None;
                Self::recycle(slot)?;
            }
        }
        Ok(())
    }
    /// # Errors
    /// Rejects old generations. None means publication is still waiting for driver application.
    pub fn launch_accepted(&self, handle: BatchLease) -> Result<Option<bool>> {
        Ok(match self.slot(handle)?.launch.load(Ordering::Acquire) {
            1 => Some(true),
            2 => Some(false),
            _ => None,
        })
    }
}

#[cfg(test)]
mod tests;
