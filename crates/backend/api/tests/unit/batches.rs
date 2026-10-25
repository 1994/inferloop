use super::*;
use infer_core::{DecisionId, ProgramId, RequestId, StateId, StepId};
use infer_ir::{CostEstimate, ExecutionRole, PlannedWork};
use std::thread::spawn;
use std::{sync::mpsc, time::Duration};

fn fixture() -> (Arc<StepPlan>, Vec<ExecutionTask>) {
    let step = Arc::new(StepPlan {
        id: StepId::ONE,
        decision: DecisionId::ONE,
        program: ProgramId::ONE,
        role: ExecutionRole::Prefill,
        work: vec![PlannedWork {
            request: RequestId::ONE,
            state: StateId::ONE,
            token_count: 3,
            role: ExecutionRole::Prefill,
        }],
        cost: CostEstimate::default(),
        graph: None,
        quantum_overrun: false,
    });
    let tasks = vec![ExecutionTask {
        plain_greedy: false,
        request: RequestId::ONE,
        state: StateId::ONE,
        tokens: vec![1, 2, 3].into(),

        sampling: None,

        media: None,
    }];
    (step, tasks)
}
#[test]
fn generation_owner_capacity_and_fence_are_independent() -> Result<()> {
    let arena = BatchArena::<u32>::new(1, 1)?;
    let foreign = BatchArena::<u32>::new(1, 1)?;
    let (step, tasks) = fixture();
    let old = arena.seal(step.clone(), &tasks)?;
    assert!(foreign.launch_accepted(old).is_err());
    assert!(arena.seal(step.clone(), &tasks).is_err());
    assert!(arena.complete(old, 1).is_err());
    assert_eq!(arena.launch_accepted(old)?, None);
    arena.apply(old, |borrowed, work| {
        assert_eq!(borrowed.work.as_ptr(), step.work.as_ptr());
        assert_eq!(work.len(), 1);
        Ok(())
    })?;
    assert!(arena.apply(old, |_, _| Ok(())).is_err());
    assert!(arena.revoke(old).is_err());
    arena.acknowledge(old, true)?;
    assert!(arena.acknowledge(old, true).is_err());
    assert_eq!(arena.launch_accepted(old)?, Some(true));
    arena.complete(old, 12)?;
    assert!(arena.complete(old, 13).is_err());
    assert_eq!(arena.take_completion(old)?, Some(12));
    let next = arena.seal(step, &tasks)?;
    assert_ne!(old, next);
    assert!(arena.take_completion(old).is_err());
    arena.revoke(next)?;
    Ok(())
}
#[test]
fn abandoned_readers_are_reclaimed_only_after_device_fence() -> Result<()> {
    let arena = BatchArena::<u32>::new(1, 1)?;
    let (step, tasks) = fixture();
    let old = arena.seal(step.clone(), &tasks)?;
    arena.abandon(old)?;
    arena.reclaim_abandoned()?;
    assert!(arena.seal(step.clone(), &tasks).is_err());
    arena.apply(old, |_, _| Ok(()))?;
    arena.acknowledge(old, true)?;
    arena.reclaim_abandoned()?;
    assert!(arena.seal(step.clone(), &tasks).is_err());
    arena.complete(old, 8)?;
    arena.reclaim_abandoned()?;
    let new = arena.seal(step, &tasks)?;
    assert!(arena.take_completion(old).is_err());
    arena.revoke(new)?;
    Ok(())
}
#[test]
fn pending_poll_never_waits_for_encoding_lock() -> Result<()> {
    let arena = Arc::new(BatchArena::<u32>::new(1, 1)?);
    let (step, tasks) = fixture();
    let handle = arena.seal(step, &tasks)?;
    let (entered, start) = mpsc::channel();
    let (release, gate) = mpsc::channel();
    let driver = arena.clone();
    let owner = spawn(move || -> Result<()> {
        driver.apply(handle, |_, _| {
            entered
                .send(())
                .map_err(|e| Error::invariant(e.to_string()))?;
            gate.recv_timeout(Duration::from_secs(3))
                .map_err(|e| Error::invariant(e.to_string()))?;
            Ok(())
        })?;
        driver.acknowledge(handle, true)?;
        driver.complete(handle, 9)
    });
    start
        .recv_timeout(Duration::from_secs(3))
        .map_err(|e| Error::invariant(e.to_string()))?;
    // The encoder deliberately holds storage. A completion probe must still return immediately.
    let pending = arena.take_completion(handle)?;
    release
        .send(())
        .map_err(|e| Error::invariant(e.to_string()))?;
    owner
        .join()
        .map_err(|_| Error::invariant("driver test panicked"))??;
    assert_eq!(pending, None);
    assert_eq!(arena.take_completion(handle)?, Some(9));
    Ok(())
}
#[test]
fn exhausted_generation_retires_slot_instead_of_aliasing() -> Result<()> {
    let arena = BatchArena::<u32>::new(1, 1)?;
    arena.slots[0].generation.store(u32::MAX, Ordering::Relaxed);
    let (step, tasks) = fixture();
    let handle = arena.seal(step.clone(), &tasks)?;
    arena.revoke(handle)?;
    assert_eq!(arena.slots[0].state.load(Ordering::Acquire), RETIRED);
    assert!(arena.seal(step, &tasks).is_err());
    assert!(arena.launch_accepted(handle).is_err());
    Ok(())
}

#[test]
fn descriptor_layout_and_frontier_are_versioned_before_publication() -> Result<()> {
    assert_eq!(size_of::<crate::WorkDescriptor>(), 32);
    assert_eq!(size_of::<SubmissionDescriptor>(), 2096);
    assert_eq!(align_of::<SubmissionDescriptor>(), 8);
    let arena = BatchArena::<u32>::new(1, 1)?;
    let (mut step, mut tasks) = fixture();
    Arc::get_mut(&mut step)
        .ok_or_else(|| Error::invariant("fixture is shared"))?
        .work[0]
        .token_count = 1;
    tasks[0].tokens = infer_ir::ExecutionInput::Decode {
        position: 65536,
        token: 7,
    };
    let handle = arena.seal(step, &tasks)?;
    let descriptor = arena.slots[0]
        .storage
        .lock()
        .map_err(|e| Error::invariant(e.to_string()))?
        .descriptor
        .ok_or_else(|| Error::invariant("descriptor missing"))?;
    assert_eq!(descriptor.abi_version, crate::SUBMISSION_ABI_VERSION);
    assert_eq!(descriptor.owner, arena.owner.get());
    assert_eq!(
        u64::from(descriptor.generation),
        handle.handle().get() >> 32
    );
    assert_eq!(descriptor.work[0].computed_frontier, 65536);
    assert_eq!(descriptor.work[0].readout, 1);
    arena.revoke(handle)?;
    Ok(())
}
