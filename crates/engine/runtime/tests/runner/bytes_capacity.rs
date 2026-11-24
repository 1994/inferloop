use super::{Fixture, deadline, request};
use infer_core::{Error, ErrorCode, FinishReason, RequestId, Result};
use infer_spi::BackendProvider;
use std::{
    sync::atomic::Ordering,
    time::{Duration, Instant},
};

#[test]
fn occupied_bytes_queue_but_oversized_reservations_are_rejected() -> Result<()> {
    let mut fixture = Fixture::new(false)?;
    fixture.byte_capacity.store(1024, Ordering::Release);
    fixture.reservation_limit.store(1, Ordering::Release);
    fixture.poll_gate.store(false, Ordering::Release);
    fixture.admit(1)?;
    fixture
        .entered
        .recv_timeout(Duration::from_secs(3))
        .map_err(|e| Error::invariant(e.to_string()))?;
    // The first allocation is published; the second quote observes zero free bytes.
    fixture.admit(2)?;
    assert_eq!(fixture.engine.backend().free_state_bytes()?, Some(0));
    assert_eq!(fixture.engine.backend().state_capacity_bytes()?, Some(1024));
    let oversized = fixture.engine.request_preparer()?.prepare(request(3)?)?;
    assert_eq!(
        fixture
            .engine
            .submit_quoted_with_trace(oversized, Some(1025), None)
            .err()
            .map(|e| e.code),
        Some(ErrorCode::Capacity)
    );
    let limit = Instant::now() + Duration::from_secs(3);
    while fixture.reservation_attempts.load(Ordering::Acquire) < 2 {
        fixture.engine.tick(fixture.engine.now_us() + 1)?;
        deadline(limit)?;
    }
    fixture.poll_gate.store(true, Ordering::Release);
    fixture.drain()?;
    for id in [1, 2] {
        let request = fixture.engine.request(RequestId::new(id)?)?;
        assert_eq!(
            request.completed.as_ref().map(|done| &done.reason),
            Some(&FinishReason::Length)
        );
        assert_eq!(request.generated.len(), 2);
    }
    assert_eq!(fixture.owned.load(Ordering::Acquire), 0);
    fixture.engine.check_invariants()
}
