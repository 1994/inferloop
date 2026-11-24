use super::{Fixture, deadline, request};
use infer_core::{Error, FinishReason, RequestId, Result};
use infer_spi::ResourceReply;
use std::{
    sync::atomic::Ordering,
    time::{Duration, Instant},
};

#[test]
fn capacity_wait_renews_only_when_execution_makes_resource_progress() -> Result<()> {
    let mut fixture = Fixture::new(false)?;
    fixture.reservation_limit.store(1, Ordering::Release);
    fixture.poll_gate.store(false, Ordering::Release);
    // Close the compute permit before the gate opens: no poll may consume the
    // default unlimited allowance between the two stores.
    fixture.poll_permits.store(0, Ordering::Release);
    // Keep the first sequence active through several resource progress/retry rounds.
    let mut first = request(1)?;
    first.workload = serde_json::from_value(serde_json::json!({
        "Generate": {"max_new_tokens":4}
    }))
    .map_err(|e| Error::invalid(e.to_string()))?;
    let prepared = fixture.engine.request_preparer()?.prepare(first)?;
    let mut quote = fixture.engine.admission_quote(&prepared)?;
    let limit = Instant::now() + Duration::from_secs(3);
    let bytes = loop {
        if let Some(ResourceReply::ReservationBytes(bytes)) = quote.poll()? {
            break bytes;
        }
        deadline(limit)?;
    };
    fixture
        .engine
        .submit_quoted_with_trace(prepared, bytes, None)?;
    fixture
        .entered
        .recv_timeout(Duration::from_secs(3))
        .map_err(|e| Error::invariant(e.to_string()))?;
    fixture.admit(2)?;
    while fixture.reservation_attempts.load(Ordering::Acquire) < 2 {
        fixture.engine.tick(fixture.engine.now_us() + 1)?;
        deadline(limit)?;
    }
    for _ in 0..50 {
        fixture.engine.tick(fixture.engine.now_us() + 1)?;
        deadline(limit)?;
    }
    let mut before = fixture.reservation_attempts.load(Ordering::Acquire);
    fixture.engine.tick(15_000)?;
    fixture.poll_gate.store(true, Ordering::Release);
    // Allow exactly one compute completion per round. Each completion advances the
    // resource epoch and must produce exactly one renewed reservation attempt.
    for _ in 0..3 {
        let expected = before + 1;
        fixture.poll_permits.store(1, Ordering::Release);
        while fixture.reservation_attempts.load(Ordering::Acquire) < expected {
            fixture.engine.tick(fixture.engine.now_us() + 1)?;
            deadline(limit)?;
        }
        assert_eq!(
            fixture.reservation_attempts.load(Ordering::Acquire),
            expected,
            "one compute completion must renew exactly one reservation attempt"
        );
        before = expected;
    }
    fixture.poll_permits.store(0, Ordering::Release);
    // Consume the rejection of the renewed reservation, without advancing the GPU.
    for _ in 0..50 {
        fixture.engine.tick(fixture.engine.now_us() + 1)?;
        deadline(limit)?;
    }
    assert!(fixture.engine.request(RequestId::ONE)?.completed.is_none());
    let attempts = before;
    // Past the initial 20ms resource deadline, within 20ms of the progress-led retry.
    fixture.engine.tick(29_999)?;
    assert!(
        fixture
            .engine
            .request(RequestId::new(2)?)?
            .completed
            .is_none()
    );
    assert!(
        fixture
            .engine
            .pending_terminal(RequestId::new(2)?)
            .is_none()
    );
    assert_eq!(
        fixture.reservation_attempts.load(Ordering::Acquire),
        attempts
    );
    fixture.engine.check_invariants()?;
    fixture.poll_permits.store(usize::MAX, Ordering::Release);
    fixture.drain()?;
    for (id, tokens) in [(1, 4), (2, 2)] {
        let record = fixture.engine.request(RequestId::new(id)?)?;
        assert_eq!(
            record.completed.as_ref().map(|x| &x.reason),
            Some(&FinishReason::Length)
        );
        assert_eq!(record.generated.len(), tokens);
    }
    assert_eq!(fixture.owned.load(Ordering::Acquire), 0);
    Ok(())
}
