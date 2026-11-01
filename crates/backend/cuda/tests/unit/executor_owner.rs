use super::*;
use crate::device::{CudaDevice, asynchronous::block_on, queue_test_gate};
use crate::loading::LoadOptions;
use infer_core::{DecisionId, ModelId, RequestId, StepId};
use infer_ir::{
    CostEstimate, ExecutionInput, ExecutionRole, OutputReadout, PlannedWork, Sampling, StepPlan,
    TaskOutput, TokenBuffer,
};
use infer_spi::BackendProvider;
use std::result::Result as StdResult;
use std::{
    env::var,
    error::Error as StdError,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    thread::yield_now,
    time::{Duration, Instant},
};

type TestResult<T = ()> = StdResult<T, Box<dyn StdError>>;

fn fixture(
    depth: usize,
    readouts: &[OutputReadout],
) -> TestResult<(CudaBackend, ExecutionProgram, Vec<StateId>)> {
    fixture_with_width(depth, readouts, 32)
}

fn fixture_with_width(
    depth: usize,
    readouts: &[OutputReadout],
    prefill_width: usize,
) -> TestResult<(CudaBackend, ExecutionProgram, Vec<StateId>)> {
    CudaDevice::enable_kernel_cache()?;
    let loaded = LoadedModel::open(
        CudaDevice::new(0)?,
        var("INFER_TEST_MODEL")?,
        ModelId::ONE,
        LoadOptions {
            prefill_width,
            mtp_depth: depth,
            autotune: false,
            vision: false,
            ..LoadOptions::default()
        },
    )?;
    assert_eq!(
        loaded.mtp_depth(),
        depth,
        "selected fixture must contain the requested MTP head"
    );
    let budget = loaded.device().allocatable_memory_bytes()?;
    let mut backend = CudaBackend::new(loaded, budget, readouts.len())?;
    let program = backend.compile(ProgramId::ONE)?;
    let states = (0..readouts.len())
        .map(|index| StateId::new(u64::try_from(index).unwrap() + 1))
        .collect::<Result<Vec<_>>>()?;
    for (&state, &readout) in states.iter().zip(readouts) {
        reserve_past_solo(&mut backend, &states, state, readout)?;
    }
    Ok((backend, program, states))
}

/// A second request waits for the solo state to drain before it may fund the pool; a scheduler
/// retries, and the drained state then comes back beside it.
fn reserve_past_solo(
    backend: &mut CudaBackend,
    states: &[StateId],
    state: StateId,
    readout: OutputReadout,
) -> TestResult {
    let Err(error) = backend.reserve_state_for(state, 128, readout) else {
        return Ok(());
    };
    if error.code != ErrorCode::Capacity {
        return Err(error.into());
    }
    let solo = states
        .iter()
        .copied()
        .find(|id| *id != state && backend.release_state(*id).is_ok())
        .ok_or("no solo state to drain")?;
    backend.reserve_state_for(state, 128, readout)?;
    backend.reserve_state_for(solo, 128, readout)?;
    Ok(())
}

#[path = "executor_fixed_batch.rs"]
mod fixed_batch;
#[path = "executor_preparation.rs"]
mod preparation;

fn tasks(
    backend: &CudaBackend,
    states: &[StateId],
    round: usize,
) -> TestResult<Vec<ExecutionTask>> {
    states
        .iter()
        .enumerate()
        .map(|(index, &state)| {
            let view = backend.view.state(state)?;
            Ok(ExecutionTask {
                request: RequestId::new(u64::try_from(index)? + 1)?,
                state,
                tokens: if view.cursor == 0 {
                    let prompt = TokenBuffer::from(vec![u32::try_from(index)? + 1; 64]);
                    ExecutionInput::Prefill {
                        span: prompt.span(0..prompt.len())?,
                        readout: if state == StateId::ONE && backend.view.states.len() > 2 {
                            OutputReadout::Full
                        } else {
                            OutputReadout::Logits
                        },
                    }
                } else {
                    ExecutionInput::Decode {
                        token: u32::try_from(round + index + 1)?,
                        position: view.cursor,
                    }
                },
                sampling: Some(Sampling {
                    temperature: 0.0,
                    ..Sampling::default()
                }),
                plain_greedy: round.is_multiple_of(2),
                publish_prefix: false,

                media: None,
            })
        })
        .collect()
}

fn plan(
    backend: &CudaBackend,
    program: &ExecutionProgram,
    tasks: &[ExecutionTask],
) -> TestResult<StepPlan> {
    let work = tasks
        .iter()
        .map(|task| {
            Ok(PlannedWork {
                request: task.request,
                state: task.state,
                token_count: task
                    .tokens
                    .delta(backend.view.state(task.state)?.cursor)?
                    .len(),
                role: ExecutionRole::Forward,
            })
        })
        .collect::<TestResult<Vec<_>>>()?;
    Ok(StepPlan {
        id: StepId::ONE,
        decision: DecisionId::ONE,
        program: program.id,
        role: ExecutionRole::Forward,
        work,
        cost: CostEstimate::default(),
        graph: None,
        quantum_overrun: false,
    })
}

struct Gate {
    released: Arc<AtomicBool>,
    expired: Arc<AtomicBool>,
}
impl Gate {
    fn new(device: &CudaDevice) -> TestResult<Self> {
        let released = Arc::new(AtomicBool::new(false));
        let expired = Arc::new(AtomicBool::new(false));
        let (gate, timed_out) = (released.clone(), expired.clone());
        queue_test_gate(device, move || {
            let deadline = Instant::now() + Duration::from_secs(5);
            while !gate.load(Ordering::Acquire) {
                if Instant::now() >= deadline {
                    timed_out.store(true, Ordering::Release);
                    break;
                }
                yield_now();
            }
        })?;
        Ok(Self { released, expired })
    }
    fn open(&self) {
        assert!(
            !self.expired.load(Ordering::Acquire),
            "submission waited for the gated GPU"
        );
        self.released.store(true, Ordering::Release);
    }
}
impl Drop for Gate {
    fn drop(&mut self) {
        self.released.store(true, Ordering::Release);
    }
}

fn drive(
    backend: &mut CudaBackend,
    program: &ExecutionProgram,
    tasks: &[ExecutionTask],
    serial: bool,
    previous: Option<&mut CudaTicket>,
) -> TestResult<(Vec<TaskOutput>, Option<CudaTicket>)> {
    let plan = plan(backend, program, tasks)?;
    if serial {
        let executor = backend.ready.as_deref_mut().unwrap();
        executor.check_tasks(program, &plan, tasks)?;
        let result = block_on(executor.execute(tasks));
        block_on(executor.loaded.device().fence())?;
        executor.busy = false;
        backend.view.refresh(executor)?;
        return Ok((result?, None));
    }
    let before = backend
        .view
        .states
        .iter()
        .map(|(id, view)| (*id, view.capacity, view.cursor))
        .collect::<Vec<_>>();
    // Capturing a pool syncs the stream, so it happens before the GPU is gated.
    backend.ready.as_mut().unwrap().prepare_execution()?;
    let gate = Gate::new(backend.ready.as_ref().unwrap().loaded.device())?;
    let mut ticket = backend.submit_borrowed(program, &plan, tasks)?;
    assert!(!gate.expired.load(Ordering::Acquire));
    assert!(backend.poll(&mut ticket)?.is_none());
    let compiled = backend.compile(program.id)?;
    assert_eq!(&compiled, program);
    backend.validate_program(backend.model(), &compiled)?;
    backend.validate_state_ownership(&before)?;
    assert!(!backend.control_ready());
    assert!(backend.reset_state(tasks[0].state).is_err());
    assert!(backend.release_state(tasks[0].state).is_err());
    assert!(
        backend
            .reuse_prefix_shared(tasks[0].state, vec![1_u32].into(), 1)
            .is_err()
    );
    assert!(backend.submit_borrowed(program, &plan, tasks).is_err());
    if let Some(previous) = previous {
        assert!(backend.poll(previous).is_err());
    }
    gate.open();
    let deadline = Instant::now() + Duration::from_secs(30);
    let output = loop {
        if let Some(output) = backend.poll(&mut ticket)? {
            break output;
        }
        assert!(Instant::now() < deadline);
        yield_now();
    };
    assert!(backend.poll(&mut ticket).is_err());
    assert!(backend.control_ready());
    Ok((output, Some(ticket)))
}

fn compare(depth: usize, readouts: &[OutputReadout]) -> TestResult {
    let (mut backend, program, states) = fixture(depth, readouts)?;
    let wakes = Arc::new(AtomicUsize::new(0));
    let wake_count = wakes.clone();
    backend.set_waker(Arc::new(move || {
        wake_count.fetch_add(1, Ordering::Release);
    }));
    // Registration may wake a parked replacement owner; count device notifications below.
    wakes.store(0, Ordering::Release);
    let mut reference: Vec<Vec<TaskOutput>> = Vec::new();
    let mut previous = None;
    for serial in [true, false] {
        for &state in &states {
            backend.reset_state(state)?;
        }
        for round in 0..5 {
            // Leave the other states unprimed in round zero. The next flight mixes
            // one decode with two prefills, then all three continue through decode.
            let active = if states.len() > 2 && round == 0 {
                &states[..1]
            } else {
                &states
            };
            let tasks = tasks(&backend, active, round)?;
            let (actual, ticket) =
                drive(&mut backend, &program, &tasks, serial, previous.as_mut())?;
            if ticket.is_some() {
                previous = ticket;
            }
            if serial {
                reference.push(actual);
            } else {
                assert_eq!(actual.len(), reference[round].len());
                for (actual, expected) in actual.iter().zip(&reference[round]) {
                    assert_eq!(actual.request, expected.request);
                    assert_eq!(actual.output.logits, expected.output.logits);
                    assert_eq!(actual.output.hidden, expected.output.hidden);
                    assert_eq!(actual.output.next_token, expected.output.next_token);
                    assert_eq!(actual.output.tokens, expected.output.tokens);
                }
            }
        }
    }
    assert!(
        wakes.load(Ordering::Acquire) > 0,
        "completion must wake its owner"
    );
    for state in states {
        backend.release_state(state)?;
    }
    backend.validate_state_ownership(&[])?;
    Ok(())
}

#[test]
#[ignore = "real 2B checkpoint and CUDA; run inside safe-run with INFER_TEST_MODEL"]
fn async_prefill_first_binding_decode_and_mixed_readouts_match_serial() -> TestResult {
    compare(
        0,
        &[
            OutputReadout::Full,
            OutputReadout::Logits,
            OutputReadout::Logits,
        ],
    )
}

#[test]
#[ignore = "real MTP2 checkpoint and CUDA; run inside safe-run with INFER_TEST_MODEL"]
fn async_mtp_pooled_and_private_match_serial() -> TestResult {
    compare(2, &[OutputReadout::Logits, OutputReadout::Logits])?;
    compare(2, &[OutputReadout::Logits])
}

#[test]
#[ignore = "real MTP2 checkpoint and CUDA; run inside safe-run with INFER_TEST_MODEL"]
fn async_invalid_mtp_sampling_fences_before_failure() -> TestResult {
    let (mut backend, program, states) =
        fixture(2, &[OutputReadout::Logits, OutputReadout::Logits])?;
    let prefill = tasks(&backend, &states, 0)?;
    let _ = drive(&mut backend, &program, &prefill, false, None)?;
    let mut invalid = tasks(&backend, &states, 1)?;
    for task in &mut invalid {
        task.sampling.as_mut().unwrap().top_p = f32::NAN;
    }
    let plan = plan(&backend, &program, &invalid)?;
    let gate = Gate::new(backend.ready.as_ref().unwrap().loaded.device())?;
    let mut ticket = backend.submit_borrowed(&program, &plan, &invalid)?;
    assert!(backend.poll(&mut ticket)?.is_none());
    assert!(backend.maintenance().is_ok());
    assert!(backend.release_state(states[0]).is_err());
    gate.open();
    let deadline = Instant::now() + Duration::from_secs(30);
    let error = loop {
        match backend.poll(&mut ticket) {
            Ok(None) => {
                assert!(Instant::now() < deadline);
                yield_now();
            }
            Ok(Some(_)) => panic!("invalid sampling must fail"),
            Err(error) => break error,
        }
    };
    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert!(backend.maintenance().is_ok());
    assert!(
        backend
            .ready
            .as_ref()
            .unwrap()
            .states
            .values()
            .all(|state| state.poisoned)
    );
    for state in states {
        backend.release_state(state)?;
    }
    backend.validate_state_ownership(&[])?;
    Ok(())
}

#[test]
#[ignore = "real 2B checkpoint and CUDA; run inside safe-run with INFER_TEST_MODEL"]
fn async_abandoned_ticket_keeps_state_owned_until_backend_drop() -> TestResult {
    let (mut backend, program, states) = fixture(0, &[OutputReadout::Logits])?;
    let tasks = tasks(&backend, &states, 0)?;
    let plan = plan(&backend, &program, &tasks)?;
    let device = backend.ready.as_ref().unwrap().loaded.device().clone();
    let gate = Gate::new(&device)?;
    let ticket = backend.submit_borrowed(&program, &plan, &tasks)?;
    drop(ticket);
    assert!(!backend.control_ready());
    assert!(backend.reset_state(states[0]).is_err());
    assert!(backend.release_state(states[0]).is_err());
    gate.open();
    drop(backend);
    block_on(device.fence())?;
    Ok(())
}

#[test]
#[ignore = "real 2B checkpoint and CUDA; run inside safe-run with INFER_TEST_MODEL"]
fn async_fatal_completion_reports_error_and_retains_flight() -> TestResult {
    let (mut backend, program, states) = fixture(0, &[OutputReadout::Logits])?;
    let tasks = tasks(&backend, &states, 0)?;
    let plan = plan(&backend, &program, &tasks)?;
    let device = backend.ready.as_ref().unwrap().loaded.device().clone();
    let gate = Gate::new(&device)?;
    let mut ticket = backend.submit_borrowed(&program, &plan, &tasks)?;
    // Simulate the owner observing a driver/fence failure while DMA is still pending.
    backend.fatal = Some(Error::new(ErrorCode::Backend, "test completion failure"));
    let error = backend.poll(&mut ticket).unwrap_err();
    assert_eq!(error.code, ErrorCode::Backend);
    assert_eq!(
        backend.poll(&mut ticket).unwrap_err().code,
        ErrorCode::InvalidInput
    );
    assert!(!backend.control_ready());
    assert!(backend.running.is_some());
    assert!(backend.release_state(states[0]).is_err());
    assert!(backend.reset_state(states[0]).is_err());
    gate.open();
    drop(backend);
    block_on(device.fence())?;
    Ok(())
}
