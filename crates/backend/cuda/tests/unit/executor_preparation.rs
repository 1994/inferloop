//! Admission and reset must preserve the storage contract chosen for MTP execution.
use super::*;

fn logit_tasks(
    backend: &CudaBackend,
    states: &[StateId],
    round: usize,
) -> TestResult<Vec<ExecutionTask>> {
    let mut tasks = tasks(backend, states, round)?;
    for task in &mut tasks {
        if let ExecutionInput::Prefill { readout, .. } = &mut task.tokens {
            *readout = OutputReadout::Logits;
        }
    }
    Ok(tasks)
}

#[test]
#[ignore = "real MTP2 checkpoint/CUDA; validates private slot binding, admission and reset accounting"]
fn later_admission_keeps_bound_verification_retired_and_reset_restores_its_charge() -> TestResult {
    let (mut backend, program, mut states) = fixture(2, &[OutputReadout::Logits])?;
    let executor = backend.ready.as_mut().unwrap();
    executor.maximum_states = 3;
    executor.slots_disabled = true;
    states.push(StateId::new(2)?);
    // Stay private for this admission; the pool is built below.
    executor.roofline_bound = Some(1);
    backend.reserve_state_for(states[1], 128, OutputReadout::Logits)?;
    let executor = backend.ready.as_mut().unwrap();
    executor.roofline_bound = None;
    executor.slots_disabled = false;
    executor.prepare_execution()?;
    assert!(executor.slots.as_ref().unwrap().verify().is_some());
    backend.view.refresh(executor)?;
    for round in 0..2 {
        let tasks = logit_tasks(&backend, &states, round)?;
        drive(&mut backend, &program, &tasks, false, None)?;
    }
    let before = states
        .iter()
        .map(|id| {
            let state = &backend.ready.as_ref().unwrap().states[id];
            assert!(state.home.is_none() && state.slot.is_some());
            assert!(state.verification_priced);
            assert_eq!(state.program.batch_width(), 1);
            assert!(state.program.released_verification_bytes() > 0);
            (state.budget, state.program.released_verification_bytes())
        })
        .collect::<Vec<_>>();
    let newcomer = StateId::new(3)?;
    backend.reserve_state_for(newcomer, 128, OutputReadout::Full)?;
    for (id, &(budget, credit)) in states.iter().zip(&before) {
        let state = &backend.ready.as_ref().unwrap().states[id];
        assert_eq!(state.program.batch_width(), 1);
        assert_eq!(state.budget, budget);
        assert_eq!(state.program.released_verification_bytes(), credit);
    }
    backend.release_state(newcomer)?;
    backend.reset_state(states[0])?;
    let reset = &backend.ready.as_ref().unwrap().states[&states[0]];
    assert_eq!(reset.budget, before[0].0 + before[0].1);
    assert_eq!(reset.program.batch_width(), 3);
    assert_eq!(reset.program.released_verification_bytes(), 0);
    assert!(reset.slot.is_none() && reset.history.is_empty());
    for round in 0..2 {
        let tasks = logit_tasks(&backend, &states[..1], round)?;
        drive(&mut backend, &program, &tasks, false, None)?;
    }
    for state in states {
        backend.release_state(state)?;
    }
    backend.validate_state_ownership(&[])?;
    Ok(())
}

#[test]
#[ignore = "real MTP2 checkpoint/CUDA; validates pooled quotes without resident home programs"]
fn private_fallback_quotes_cover_verification_when_resident_homes_are_unavailable() -> TestResult {
    check_private_fallback_preparation(32)?;
    check_private_fallback_preparation(1)
}

fn check_private_fallback_preparation(prefill_width: usize) -> TestResult {
    let (mut backend, program, states) =
        fixture_with_width(2, &[OutputReadout::Logits; 2], prefill_width)?;
    for &state in &states {
        backend.release_state(state)?;
    }
    // A supported pool can lack resident home programs after a home capture failed.
    let executor = backend.ready.as_mut().unwrap();
    executor.homes.clear();
    executor.slots = None;
    executor.slots_disabled = true;
    executor.roofline_bound = Some(1);
    for &state in &states {
        let quoted = backend.state_reservation_bytes_for(128, OutputReadout::Logits)?;
        backend.reserve_state_for(state, 128, OutputReadout::Logits)?;
        let state = &backend.ready.as_ref().unwrap().states[&state];
        assert!(state.home.is_none() && state.verification_priced);
        assert_eq!(Some(state.budget), quoted);
        assert_eq!(state.program.batch_width(), 3);
    }
    for round in 0..2 {
        let tasks = logit_tasks(&backend, &states, round)?;
        drive(&mut backend, &program, &tasks, false, None)?;
    }
    backend.reset_state(states[0])?;
    let reset = &backend.ready.as_ref().unwrap().states[&states[0]];
    assert!(reset.slot.is_none() && reset.verification_priced);
    assert_eq!(reset.program.batch_width(), 3);
    for round in 0..2 {
        let tasks = logit_tasks(&backend, &states[..1], round)?;
        drive(&mut backend, &program, &tasks, false, None)?;
    }
    for state in states {
        backend.release_state(state)?;
    }
    backend.validate_state_ownership(&[])?;
    Ok(())
}
