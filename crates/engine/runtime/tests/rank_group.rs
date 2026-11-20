//! The group protocol a tensor-parallel backend owes the engine, pinned on host doubles before any
//! device code exists (`docs/plans/engineering/parallelism.md`, P0.4).
#[path = "support/rank_group.rs"]
mod rank_group;
mod support;

use infer_core::*;
use infer_ir::*;
use infer_kernel_api::KernelRegistry;
use infer_runtime::*;
use rank_group::RankGroup;
use std::collections::BTreeMap;
use std::sync::atomic::Ordering;
use support::ProtocolBackend;

/// Ticks a scene runs before it calls a stall a hang.
const TICKS: u64 = 1000;

fn rank() -> ProtocolBackend {
    ProtocolBackend::new(16, 8, &support::model(ModelId::ONE)).unwrap()
}

fn engine(group: RankGroup) -> Engine<RankGroup> {
    let mut registry = KernelRegistry::default();
    registry.register(&support::DeclaredKernels).unwrap();
    Engine::new(
        group,
        support::model(ModelId::ONE),
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig::default(),
    )
    .unwrap()
}

fn request(id: u64) -> CanonicalRequest {
    CanonicalRequest {
        id: RequestId::new(id).unwrap(),
        model: ModelId::ONE,
        session: None,
        input: RequestInput::Sequence {
            tokens: vec![1, 2, 3, 4, 5].into(),
            media: vec![],
        },
        workload: Workload::Generate { max_new_tokens: 2 },
        qos: Qos::default(),
        sampling: Sampling::default(),
        prefix_boundary: None,
        extensions: BTreeMap::new(),
    }
}

/// Tick `ticks` times, checking the engine's invariants after each; whether it went idle.
fn run(engine: &mut Engine<RankGroup>, start: u64, ticks: u64) -> bool {
    for tick in start..start + ticks {
        engine.tick(tick * 100).unwrap();
        engine.check_invariants().unwrap();
        if engine.is_idle() {
            return true;
        }
    }
    false
}

fn status(engine: &Engine<RankGroup>, id: u64) -> RequestStatus {
    engine
        .request(RequestId::new(id).unwrap())
        .unwrap()
        .status
        .clone()
}

#[test]
fn a_group_step_completes_only_after_every_rank_fences() {
    let group = RankGroup::new(vec![rank(), rank()]).unwrap();
    let held = group.control(1);
    let in_flight = group.in_flight();
    held.hold_fence.store(true, Ordering::Release);
    let mut engine = engine(group);
    engine.submit(request(1)).unwrap();
    assert!(
        !run(&mut engine, 0, 50),
        "a held rank must keep the group busy"
    );
    assert!(!status(&engine, 1).terminal());
    assert_eq!(
        in_flight.load(Ordering::Acquire),
        1,
        "rank 0 fenced, rank 1 still in flight"
    );
    held.hold_fence.store(false, Ordering::Release);
    assert!(run(&mut engine, 50, TICKS));
    assert!(matches!(
        status(&engine, 1),
        RequestStatus::Finished(FinishReason::Length)
    ));
    assert_eq!(in_flight.load(Ordering::Acquire), 0);
}

#[test]
fn a_reservation_one_rank_refuses_leaves_no_rank_holding_the_state() {
    let group = RankGroup::new(vec![rank(), rank()]).unwrap();
    let refusing = group.control(1);
    refusing.fail_reserve.store(true, Ordering::Release);
    let mut engine = engine(group);
    // Admission reserves before it accepts, so the refusal rejects the submission itself.
    let refused = engine.submit(request(1)).unwrap_err();
    assert!(
        refused.message.contains("rank 1 cannot reserve"),
        "{refused:?}"
    );
    assert!(run(&mut engine, 0, TICKS));
    for index in 0..2 {
        assert_eq!(
            engine.backend().rank(index).reserved_states(),
            0,
            "rank {index}"
        );
    }
    assert_eq!(engine.inspect().state.allocated_pages, 0);
    // The rollback left the group consistent: the next request reserves on every rank.
    refusing.fail_reserve.store(false, Ordering::Release);
    engine.submit(request(2)).unwrap();
    assert!(run(&mut engine, TICKS, TICKS));
    assert!(matches!(
        status(&engine, 2),
        RequestStatus::Finished(FinishReason::Length)
    ));
}

#[test]
fn a_reused_prefix_is_the_shortest_any_rank_holds() {
    let long = rank().with_cached_prefix(4);
    let short = rank().with_cached_prefix(2);
    let (long_log, short_log) = (long.attached_prefixes(), short.attached_prefixes());
    let mut engine = engine(RankGroup::new(vec![long, short]).unwrap());
    engine.submit(request(1)).unwrap();
    assert!(run(&mut engine, 0, TICKS));
    assert!(matches!(
        status(&engine, 1),
        RequestStatus::Finished(FinishReason::Length)
    ));
    assert_eq!(*long_log.lock().unwrap(), [2]);
    assert_eq!(*short_log.lock().unwrap(), [2]);
}

#[test]
fn a_failing_rank_fails_the_group_only_after_every_fence() {
    let group = RankGroup::new(vec![rank(), rank()]).unwrap();
    let (failing, held) = (group.control(0), group.control(1));
    let in_flight = group.in_flight();
    failing.fail_poll.store(true, Ordering::Release);
    held.hold_fence.store(true, Ordering::Release);
    let mut engine = engine(group);
    engine.submit(request(1)).unwrap();
    assert!(!run(&mut engine, 0, 50));
    // Rank 0 already failed, but rank 1's work is still on its device: nothing is reported yet.
    assert!(!status(&engine, 1).terminal());
    assert_eq!(in_flight.load(Ordering::Acquire), 1);
    held.hold_fence.store(false, Ordering::Release);
    assert!(run(&mut engine, 50, TICKS));
    assert!(matches!(
        status(&engine, 1),
        RequestStatus::Finished(FinishReason::Failed(_))
    ));
    assert_eq!(in_flight.load(Ordering::Acquire), 0);
    assert_eq!(engine.inspect().state.allocated_pages, 0);
}
