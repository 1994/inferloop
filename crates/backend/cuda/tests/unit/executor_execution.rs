use super::{pool_retry_budget, pooled_envelope};
use std::{env::var, error::Error};

#[test]
fn lazy_pool_preserves_live_private_reservations_and_scratch() {
    // 40 reserved by live sequences and the shared scratch is still owned by them.
    let lazy = pooled_envelope(1_000, 100, 40, 20);
    assert_eq!(lazy, 60);
    assert_eq!(lazy + 40 + 20, 120);
    // At load time that same scratch reservation can instead fund the worker.
    assert_eq!(pooled_envelope(1_000, 100, 0, 20), 120);
}

#[test]
fn pool_envelope_is_capped_by_physical_storage_and_saturates() {
    assert_eq!(pooled_envelope(50, 100, 0, 20), 50);
    assert_eq!(pooled_envelope(50, 100, 40, 20), 50);
    assert_eq!(pooled_envelope(1_000, 100, 101, 20), 0);
    assert_eq!(pooled_envelope(u64::MAX, u64::MAX, 0, 20), u64::MAX);
}

#[test]
#[ignore = "real MTP2 checkpoint and CUDA; run inside safe-run"]
fn lazy_pool_retries_after_a_live_private_reservation_is_released() -> Result<(), Box<dyn Error>> {
    use crate::{
        device::CudaDevice,
        executor::CudaBackend,
        loading::{LoadOptions, LoadedModel},
    };
    use infer_core::{ModelId, StateId};
    use infer_ir::OutputReadout;

    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        var("INFER_TEST_MODEL")?,
        ModelId::new(1)?,
        LoadOptions {
            prefill_width: 256,
            mtp_depth: 2,
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    assert!(loaded.model().max_sequence >= 32768);
    let budget = loaded.device().allocatable_memory_bytes()?;
    // Start without a pool, then admit one long and two short private programs.
    let mut backend = CudaBackend::new(loaded, budget, 1)?;
    backend.maximum_states = 16;
    let long = StateId::new(1)?;
    backend.reserve(long, 32768, OutputReadout::Logits)?;
    backend.reserve(StateId::new(2)?, 512, OutputReadout::Logits)?;
    backend.reserve(StateId::new(3)?, 512, OutputReadout::Logits)?;
    let mut lower = 0;
    let mut upper = budget;
    assert!(backend.pooled_plan(512, 16, upper)?.is_some());
    while lower < upper {
        let middle = lower + (upper - lower) / 2;
        if backend.pooled_plan(512, 16, middle)?.is_some() {
            upper = middle;
        } else {
            lower = middle + 1;
        }
    }
    // Existing reservations are legal; the additional pool is one byte short.
    let active = backend.active_budget()?;
    backend.budget = active.checked_add(lower - 1).ok_or("budget overflow")?;
    backend.ensure_slot_pool(&[true, true]);
    assert!(backend.slots.is_none());
    assert!(!backend.slots_disabled);
    assert_eq!(backend.deferred_pool_plan, Some((512, active)));
    backend.ensure_slot_pool(&[true, true]);
    assert_eq!(backend.deferred_pool_plan, Some((512, active)));

    backend.release(long)?;
    while backend.pool.evict() {}
    backend.ensure_slot_pool(&[true, true]);
    assert!(backend.slots.is_some(), "{:?}", backend.pool_failure_reason);
    assert!(backend.worker.is_some());
    assert!(!backend.slots_disabled);
    assert!(backend.deferred_pool_plan.is_none());
    assert!(backend.homes.iter().all(Option::is_some));
    Ok(())
}

#[test]
fn unquoted_capture_storage_cannot_expand_the_serving_envelope() {
    // A quote nearly fills the envelope; capture materializes another 20 bytes.
    let funding = pool_retry_budget(100, 99, 119, 8);
    assert_eq!(funding, 80);
    assert!(funding + 20 <= 100);
    // Even exact tensor quotes leave one physical allocation quantum of headroom.
    assert_eq!(pool_retry_budget(100, 100, 100, 8), 92);
    assert_eq!(pool_retry_budget(8, 8, 100, 8), 0);
    assert_eq!(pool_retry_budget(100, 100, 100, 0), 99);
}

#[test]
fn retry_quote_accounts_for_storage_retained_by_previous_captures() {
    let envelope = 100;
    let first = pool_retry_budget(envelope, 99, 119, 8);
    assert_eq!(first, 80);
    // The next tensor capture uses 90 bytes but the first left 20 resident:
    // measuring only the second capture would incorrectly publish 110 bytes.
    let total_after_second = 20 + 90;
    assert!(total_after_second > envelope);
    let next = pool_retry_budget(first, 79, total_after_second, 8);
    assert_eq!(next, 49);
    // Both retained increments fit alongside the smaller final capture.
    assert!(next + 20 + 10 <= envelope);
}

#[test]
#[ignore = "real checkpoint and CUDA; run inside safe-run with INFER_TEST_MODEL"]
fn async_decode_commits_only_on_poll_and_matches_synchronous_output() -> Result<(), Box<dyn Error>>
{
    use infer_ir::TaskOutput;
    use infer_spi::BackendProvider;

    let DecodeFixture {
        mut backend,
        program,
        states,
    } = decode_fixture()?;
    let mut reference: Vec<Vec<TaskOutput>> = Vec::new();
    let mut previous = None;
    for asynchronous in [false, true] {
        for state in states {
            backend.reset_state(state)?;
        }
        for position in 0..5 {
            let tasks = decode_test_tasks(states, position)?;
            let (outputs, ticket) = drive_decode_test(
                &mut backend,
                &program,
                &tasks,
                position,
                asynchronous,
                previous.as_mut(),
            )?;
            previous = Some(ticket);
            if asynchronous {
                for (actual, expected) in outputs.iter().zip(&reference[position]) {
                    assert_eq!(actual.request, expected.request);
                    assert_eq!(actual.output.logits, expected.output.logits);
                    assert_eq!(actual.output.next_token, expected.output.next_token);
                }
            } else {
                reference.push(outputs);
            }
        }
    }
    for state in states {
        backend.release_state(state)?;
    }
    backend.validate_state_ownership(&[])?;
    Ok(())
}

fn drive_decode_test(
    backend: &mut super::CudaBackend,
    program: &infer_ir::ExecutionProgram,
    tasks: &[infer_ir::ExecutionTask],
    position: usize,
    asynchronous: bool,
    previous: Option<&mut super::CudaTicket>,
) -> Result<(Vec<infer_ir::TaskOutput>, super::CudaTicket), Box<dyn Error>> {
    use infer_core::{DecisionId, StepId};
    use infer_ir::{CostEstimate, ExecutionRole, PlannedWork, StepPlan};
    use infer_spi::BackendProvider;
    use std::{
        thread::yield_now,
        time::{Duration, Instant},
    };

    let step = StepPlan {
        id: StepId::ONE,
        decision: DecisionId::ONE,
        program: program.id,
        role: ExecutionRole::Forward,
        work: tasks
            .iter()
            .map(|task| PlannedWork {
                request: task.request,
                state: task.state,
                token_count: 1,
                role: ExecutionRole::Forward,
            })
            .collect(),
        cost: CostEstimate::default(),
        graph: None,
        quantum_overrun: false,
    };
    let mut ticket = if asynchronous {
        backend.submit_borrowed(program, &step, tasks)?
    } else {
        backend.check_tasks(program, &step, tasks)?;
        backend.execute_ready(tasks)
    };
    if let Some(previous) = previous {
        assert!(backend.poll(previous).is_err());
    }
    if asynchronous && position > 1 {
        assert!(backend.decode.is_pending());
        for task in tasks {
            assert_eq!(backend.states[&task.state].history.len(), position);
        }
    }
    assert!(!backend.control_ready());
    assert!(backend.release_state(tasks[0].state).is_err());
    assert!(backend.reset_state(tasks[0].state).is_err());
    assert!(
        backend
            .reuse_prefix_shared(tasks[0].state, vec![1_u32].into(), 1)
            .is_err()
    );
    assert!(backend.submit_borrowed(program, &step, tasks).is_err());
    let deadline = Instant::now() + Duration::from_secs(30);
    let outputs = loop {
        if let Some(outputs) = backend.poll(&mut ticket)? {
            break outputs;
        }
        assert!(Instant::now() < deadline);
        yield_now();
    };
    assert!(backend.poll(&mut ticket).is_err());
    assert!(backend.control_ready());
    for task in tasks {
        assert_eq!(backend.states[&task.state].history.len(), position + 1);
    }
    Ok((outputs, ticket))
}

struct DecodeFixture {
    backend: super::CudaBackend,
    program: infer_ir::ExecutionProgram,
    states: [infer_core::StateId; 2],
}

fn decode_fixture() -> Result<DecodeFixture, Box<dyn Error>> {
    use crate::{
        device::CudaDevice,
        executor::CudaBackend,
        loading::{LoadOptions, LoadedModel},
    };
    use infer_core::{ModelId, ProgramId, StateId};
    use infer_ir::OutputReadout;
    use infer_spi::BackendProvider;
    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        var("INFER_TEST_MODEL")?,
        ModelId::ONE,
        LoadOptions {
            prefill_width: 32,
            mtp_depth: 0,
            autotune: false,
            ..LoadOptions::default()
        },
    )?;
    let budget = loaded.device().allocatable_memory_bytes()?;
    let mut backend = CudaBackend::new(loaded, budget, 2)?;
    let program = backend.compile(ProgramId::ONE)?;
    assert!(
        backend
            .slots
            .as_ref()
            .is_some_and(|pool| pool.verify().is_none())
    );
    let states = [StateId::ONE, StateId::new(2)?];
    for state in states {
        backend.reserve_state_for(state, 32, OutputReadout::Logits)?;
    }
    Ok(DecodeFixture {
        backend,
        program,
        states,
    })
}

fn decode_test_tasks(
    states: [infer_core::StateId; 2],
    position: usize,
) -> Result<Vec<infer_ir::ExecutionTask>, Box<dyn Error>> {
    use infer_core::RequestId;
    use infer_ir::{ExecutionInput, ExecutionTask, OutputReadout, TokenBuffer};
    states
        .into_iter()
        .enumerate()
        .map(|(index, state)| {
            Ok(ExecutionTask {
                request: RequestId::new(u64::try_from(index)? + 1)?,
                state,
                tokens: if position == 0 {
                    ExecutionInput::Prefill {
                        span: TokenBuffer::from(vec![1_u32]).span(0..1)?,
                        readout: OutputReadout::Logits,
                    }
                } else {
                    ExecutionInput::Decode {
                        token: u32::try_from(position + index)?,
                        position,
                    }
                },
                plain_greedy: position.is_multiple_of(2),
                sampling: None,
            })
        })
        .collect()
}

#[test]
#[ignore = "real checkpoint and CUDA; run inside safe-run with INFER_TEST_MODEL"]
fn async_decode_enqueue_error_drains_before_maintenance_is_fatal() -> Result<(), Box<dyn Error>> {
    use super::{Completion, CudaTicket};
    use infer_core::{DecisionId, StepId};
    use infer_ir::{CostEstimate, ExecutionRole, PlannedWork, StepPlan};
    use infer_spi::BackendProvider;

    let DecodeFixture {
        mut backend,
        program,
        states,
    } = decode_fixture()?;
    for position in 0..2 {
        let tasks = decode_test_tasks(states, position)?;
        let _ = drive_decode_test(&mut backend, &program, &tasks, position, true, None)?;
    }
    let tasks = decode_test_tasks(states, 2)?;
    let lanes = tasks
        .iter()
        .map(|task| {
            let state = &backend.states[&task.state];
            (state.slot.as_ref().unwrap().slot, 1_u32, 2, 2)
        })
        .collect::<Vec<_>>();
    // A live shared replay forces the following enqueue to fail before publication.
    // Its device owners still need draining before the error reaches maintenance.
    backend
        .slots
        .as_mut()
        .unwrap()
        .submit_lanes(backend.loaded.device(), &lanes, false)?;
    let step = StepPlan {
        id: StepId::ONE,
        decision: DecisionId::ONE,
        program: program.id,
        role: ExecutionRole::Decode,
        work: tasks
            .iter()
            .map(|task| PlannedWork {
                request: task.request,
                state: task.state,
                token_count: 1,
                role: ExecutionRole::Decode,
            })
            .collect(),
        cost: CostEstimate::default(),
        graph: None,
        quantum_overrun: false,
    };
    let mut ticket: CudaTicket = backend.submit_borrowed(&program, &step, &tasks)?;
    assert!(matches!(ticket.completion, Completion::FailedDecode(_)));
    assert!(ticket.drain_required);
    assert!(
        backend.maintenance().is_ok(),
        "maintenance must let the runtime consume the failed ticket"
    );
    assert!(!backend.control_ready());
    assert!(backend.states.values().all(|state| state.poisoned));
    let error = backend.poll(&mut ticket).unwrap_err();
    assert!(!ticket.drain_required);
    assert!(matches!(ticket.completion, Completion::Consumed));
    assert_eq!(backend.maintenance().unwrap_err(), error);
    assert!(backend.release_state(states[0]).is_err());
    Ok(())
}
