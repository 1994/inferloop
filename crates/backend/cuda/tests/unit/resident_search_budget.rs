use super::*;

#[test]
fn one_expired_shape_does_not_starve_later_shapes_or_quantization_families() -> Result<()> {
    let schedule = Schedule::default();
    let now = Instant::now();
    let first = [64, 256, 128];
    let second = [1024, 128, 256];
    schedule.plan(Family::Nvfp4, &[first, second], 2)?;
    assert!(schedule.permits_at(Family::Nvfp4, first, 2, now)?);
    let later = now + measurement_budget(2)? + Duration::from_millis(1);
    assert!(!schedule.permits_at(Family::Nvfp4, first, 2, later)?);
    assert!(schedule.permits_at(Family::Nvfp4, second, 2, later)?);
    assert!(schedule.permits_at(Family::Fp8Channel, first, 2, later)?);
    Ok(())
}

#[test]
fn adding_candidates_expands_work_without_resetting_an_active_timer() -> Result<()> {
    let schedule = Schedule::default();
    let shape = [64, 128, 256];
    let now = Instant::now();
    assert!(schedule.permits_at(Family::Nvfp4, shape, 1, now)?);
    let later = now + measurement_budget(1)? + Duration::from_millis(1);
    assert!(!schedule.permits_at(Family::Nvfp4, shape, 1, later)?);
    assert!(schedule.permits_at(Family::Nvfp4, shape, 4, later)?);
    let exhausted = now + measurement_budget(4)? + Duration::from_millis(1);
    assert!(!schedule.permits_at(Family::Nvfp4, shape, 4, exhausted)?);
    Ok(())
}

#[test]
fn unfinished_work_excludes_completed_shapes_but_includes_unstarted_shapes() -> Result<()> {
    let schedule = Schedule::default();
    let shapes = [[1, 64, 128], [64, 128, 256], [1024, 256, 128]];
    schedule.plan(Family::Nvfp4, &shapes, 4)?;
    schedule.plan(Family::Nvfp4, &shapes, 4)?;
    schedule.complete(Family::Nvfp4, shapes[0])?;
    let state = schedule
        .shapes
        .lock()
        .map_err(|_| Error::invariant("test schedule"))?;
    assert_eq!(state.len(), shapes.len());
    assert_eq!(
        unfinished(&state),
        vec![(Family::Nvfp4, shapes[1]), (Family::Nvfp4, shapes[2])]
    );
    Ok(())
}

#[test]
fn explicit_zero_allowance_forbids_new_measurements() -> Result<()> {
    let schedule = Schedule::with_limit(Duration::ZERO);
    assert!(!schedule.permits(Family::Nvfp4, [64, 128, 256], 12)?);
    assert!(!schedule.permits(Family::Fp8Channel, [64, 128, 256], 24)?);
    Ok(())
}

#[test]
fn unrepresentable_candidate_counts_are_errors() {
    assert!(measurement_budget(usize::MAX).is_err());
}

#[test]
fn completed_shape_does_not_restart_its_allowance() -> Result<()> {
    let schedule = Schedule::default();
    let first = [64, 256, 128];
    let later = [1024, 128, 256];
    let now = Instant::now();
    schedule.plan(Family::Nvfp4, &[first, later], 2)?;
    assert!(schedule.permits_at(Family::Nvfp4, first, 2, now)?);
    schedule.complete(Family::Nvfp4, first)?;
    assert!(!schedule.permits_at(Family::Nvfp4, first, 2, now + Duration::from_millis(1))?);
    assert!(schedule.permits_at(Family::Nvfp4, later, 2, now + Duration::from_millis(1))?);
    Ok(())
}
