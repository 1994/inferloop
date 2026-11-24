use super::*;

#[test]
fn allowance_scales_with_live_work_without_renewing_duplicate_shapes() -> Result<()> {
    let mut budget = SearchBudget::default();
    assert!(!budget.permits(Duration::ZERO));
    assert!(budget.plan(ProjectionKind::Fp4, [64, 5120, 17408], 13)?);
    let first = budget.allowance();
    assert!(budget.permits(first / 2));
    assert!(!budget.permits(first));
    assert!(!budget.plan(ProjectionKind::Fp4, [64, 5120, 17408], 13)?);
    assert_eq!(budget.allowance(), first);
    assert!(budget.plan(ProjectionKind::Fp4, [1024, 5120, 17408], 13)?);
    assert!(budget.permits(first));
    assert_eq!(budget.snapshot().planned, 2);
    assert!(budget.plan(ProjectionKind::Fp4, [2048, 5120, 17408], 26)?);
    assert_eq!(budget.allowance(), first * 4);
    Ok(())
}

#[test]
fn pending_shapes_exclude_completed_cache_or_measurement_work() -> Result<()> {
    let mut budget = SearchBudget::default();
    let a = [64, 5120, 17408];
    let b = [1024, 5120, 17408];
    budget.plan(ProjectionKind::Fp4, a, 13)?;
    budget.plan(ProjectionKind::Fp4, b, 13)?;
    budget.plan(ProjectionKind::Fp8, a, 7)?;
    budget.complete(ProjectionKind::Fp4, a);
    assert_eq!(
        budget.snapshot().pending,
        vec![(ProjectionKind::Fp4, b), (ProjectionKind::Fp8, a)]
    );
    let allowance = budget.allowance();
    assert!(budget.report_expiration(allowance));
    assert!(!budget.report_expiration(allowance));
    assert!(!budget.plan(ProjectionKind::Fp4, a, 13)?);
    assert_eq!(budget.snapshot().pending.len(), 2);
    assert!(!budget.report_expiration(allowance));
    Ok(())
}

#[test]
fn late_width_adds_its_work_without_restarting_elapsed_time() -> Result<()> {
    let mut budget = SearchBudget::default();
    budget.plan(ProjectionKind::Fp4, [64, 128, 512], 3)?;
    let elapsed = budget.allowance();
    assert!(!budget.permits(elapsed));
    budget.plan(ProjectionKind::Fp4Lt, [1024, 128, 512], 2)?;
    assert!(budget.permits(elapsed));
    assert!(!budget.permits(budget.allowance()));
    assert_eq!(budget.snapshot().pending.len(), 2);
    Ok(())
}

#[test]
fn malformed_work_and_estimate_overflow_leave_the_plan_unchanged() {
    let mut budget = SearchBudget::default();
    assert!(budget.plan(ProjectionKind::Fp4, [0, 128, 512], 1).is_err());
    assert!(budget.plan(ProjectionKind::Fp4, [1, 128, 512], 0).is_err());
    assert!(
        budget
            .plan(ProjectionKind::Fp4, [1, 128, 512], usize::MAX)
            .is_err()
    );
    assert_eq!(budget.snapshot().planned, 0);
    assert_eq!(budget.allowance(), Duration::ZERO);
}

#[test]
fn explicit_test_limits_still_reject_or_bound_searches() -> Result<()> {
    for limit in [Duration::ZERO, Duration::from_secs(600)] {
        let mut budget = SearchBudget::with_limit(limit);
        budget.plan(ProjectionKind::Fp4, [1024, 5120, 17408], 13)?;
        assert_eq!(budget.allowance(), limit);
        assert!(!budget.permits(limit));
        assert_eq!(budget.permits(Duration::ZERO), !limit.is_zero());
    }
    Ok(())
}

#[test]
fn expanded_candidates_add_measurement_windows_and_reopen_only_that_shape() -> Result<()> {
    let mut budget = SearchBudget::default();
    let shape = [64, 5120, 17408];
    budget.plan(ProjectionKind::Fp4, shape, 3)?;
    let original = budget.allowance();
    budget.complete(ProjectionKind::Fp4, shape);
    assert_eq!(budget.snapshot().pending.len(), 0);
    budget.plan(ProjectionKind::Fp4, shape, 6)?;
    assert_eq!(budget.allowance(), original * 2);
    assert_eq!(budget.snapshot().planned, 1);
    assert_eq!(
        budget.snapshot().pending,
        vec![(ProjectionKind::Fp4, shape)]
    );
    Ok(())
}

#[test]
fn only_active_search_time_is_charged_and_early_returns_release_accounting() -> Result<()> {
    let budget = Mutex::new(SearchBudget::default());
    budget
        .lock()
        .map_err(|_| Error::invariant("test budget lock"))?
        .plan(ProjectionKind::Fp4, [64, 128, 512], 13)?;
    assert_eq!(
        budget
            .lock()
            .map_err(|_| Error::invariant("test budget lock"))?
            .elapsed(),
        Duration::ZERO
    );
    let run = || -> Result<()> {
        let _search = SearchScope::begin(&budget)?;
        assert!(SearchScope::begin(&budget).is_err());
        budget
            .lock()
            .map_err(|_| Error::invariant("test budget lock"))?
            .active = Some(
            Instant::now()
                .checked_sub(Duration::from_secs(2))
                .ok_or_else(|| Error::invariant("test clock"))?,
        );
        Err(Error::invalid("cancelled search"))
    };
    assert!(run().is_err());
    let state = budget
        .lock()
        .map_err(|_| Error::invariant("test budget lock"))?;
    assert!(state.active.is_none());
    let charged = state.elapsed();
    assert!(charged >= Duration::from_secs(2));
    assert_eq!(
        state.elapsed(),
        charged,
        "time outside a search cannot consume its budget"
    );
    drop(state);
    Ok(())
}
