use super::{Fixture, deadline, request};
use infer_core::{Error, FinishReason, RequestId, Result};
use std::{
    sync::atomic::Ordering,
    time::{Duration, Instant},
};

#[test]
fn long_capacity_queue_has_no_implicit_deadline_but_honors_explicit_limits() -> Result<()> {
    for (queue_wait, hard_deadline) in [(None, false), (Some(1_000), false), (None, true)] {
        let mut fixture = Fixture::with_timeouts(false, 0, queue_wait, Duration::from_secs(90))?;
        fixture.reservation_limit.store(1, Ordering::Release);
        fixture.poll_gate.store(false, Ordering::Release);
        // One device submission holds the resource. Finish it at the fence so
        // an explicit queue cap is tested only on the second, waiting request.
        let mut first = request(1)?;
        first.workload = serde_json::from_value(serde_json::json!({
            "Generate": {"max_new_tokens": 1}
        }))
        .map_err(|e| Error::invalid(e.to_string()))?;
        fixture.admit_request(first)?;
        fixture
            .entered
            .recv_timeout(Duration::from_secs(3))
            .map_err(|e| Error::invariant(e.to_string()))?;
        let mut second = request(2)?;
        if hard_deadline {
            second.qos.deadline_us = Some(1_000);
        }
        fixture.admit_request(second)?;
        let limit = Instant::now() + Duration::from_secs(3);
        while fixture.reservation_attempts.load(Ordering::Acquire) < 2 {
            fixture.engine.tick(fixture.engine.now_us() + 1)?;
            deadline(limit)?;
        }
        for _ in 0..50 {
            fixture.engine.tick(fixture.engine.now_us() + 1)?;
            deadline(limit)?;
        }
        let elapsed = if queue_wait.is_none() && !hard_deadline {
            31_000_000
        } else {
            2_000
        };
        // Quotes share the device owner and must precede the held compute flight.
        // Confirm execution, rather than only reservation, before the clock jump.
        let started_limit = Instant::now() + Duration::from_secs(3);
        while !matches!(
            fixture.engine.request(RequestId::ONE)?.status,
            infer_core::RequestStatus::Running { .. }
        ) {
            fixture.engine.tick(fixture.engine.now_us() + 1)?;
            deadline(started_limit)?;
        }
        fixture.engine.tick(elapsed)?;
        // Let every owner finish even if a subsequent assertion finds a regression.
        fixture.poll_gate.store(true, Ordering::Release);
        fixture.drain()?;
        assert_eq!(
            fixture
                .engine
                .request(RequestId::ONE)?
                .completed
                .as_ref()
                .map(|x| &x.reason),
            Some(&FinishReason::Length)
        );
        let second = fixture.engine.request(RequestId::new(2)?)?;
        let reason = second.completed.as_ref().map(|x| &x.reason);
        if hard_deadline {
            assert_eq!(reason, Some(&FinishReason::Deadline));
        } else if queue_wait.is_some() {
            assert_eq!(
                reason,
                Some(&FinishReason::Failed(
                    "scheduler stage queue wait timed out".into()
                ))
            );
        } else {
            assert_eq!(reason, Some(&FinishReason::Length));
            assert_eq!(second.generated.len(), 2);
        }
        assert_eq!(fixture.owned.load(Ordering::Acquire), 0);
    }
    Ok(())
}
