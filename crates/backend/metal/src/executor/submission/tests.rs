use super::*;
use crate::{MetalConfig, MetalKernels};
use infer_core::*;
use infer_ir::*;
use infer_kernel_api::KernelRegistry;
use infer_models::ModelPackage;
use infer_spi::BackendProvider;
use std::thread::sleep;
use std::{
    path::Path,
    time::{Duration, Instant},
};

fn step(program: &ExecutionProgram, states: &[StateId], count: usize) -> Result<StepPlan> {
    Ok(StepPlan {
        id: StepId::ONE,
        decision: DecisionId::ONE,
        program: program.id,
        role: ExecutionRole::Prefill,
        work: states
            .iter()
            .map(|state| {
                Ok(PlannedWork {
                    request: RequestId::new(state.get())?,
                    state: *state,
                    token_count: count,
                    role: ExecutionRole::Prefill,
                })
            })
            .collect::<Result<_>>()?,
        cost: CostEstimate::default(),
        graph: None,
        quantum_overrun: false,
    })
}
fn compile_fixture(backend: &MetalBackend) -> Result<ExecutionProgram> {
    let mut kernels = KernelRegistry::default();
    kernels.register(&MetalKernels)?;
    infer_compiler::compile(
        ProgramId::ONE,
        infer_compiler::lower(
            backend.model(),
            backend.execution_graph(backend.model())?,
            PrecisionPlan::f32(),
        )?,
        &kernels,
        &backend.capabilities(),
        1 << 20,
    )
}

fn plain_task(request: RequestId, state: StateId, tokens: Vec<u32>) -> ExecutionTask {
    ExecutionTask {
        plain_greedy: false,
        publish_prefix: false,
        request,
        state,
        tokens: tokens.into(),
        sampling: None,
        media: None,
    }
}

#[test]
fn encoding_failure_rolls_back_a_batch_and_shared_cow_tails() -> Result<()> {
    if !MetalBackend::available() {
        return Ok(());
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let mut package = ModelPackage::open(root, ModelId::ONE)?;
    let mut backend = MetalBackend::from_package(
        &mut package,
        MetalConfig {
            prefix_cache_bytes: 0,
            block_size: 2,
            probe_bytes: 1 << 20,
            ..Default::default()
        },
    )?;
    let program = compile_fixture(&backend)?;
    backend.reserve_state(StateId::ONE, 8)?;
    let mut ticket = backend.submit(
        &program,
        &step(&program, &[StateId::ONE], 3)?,
        vec![plain_task(RequestId::ONE, StateId::ONE, vec![1, 2, 3])],
    )?;
    let deadline = Instant::now() + Duration::from_secs(10);
    while backend.poll(&mut ticket)?.is_none() {
        assert!(Instant::now() < deadline);
        sleep(Duration::from_micros(100));
    }
    let second = StateId::new(2)?;
    backend.fork_sequence(StateId::ONE, second, 8)?;
    backend.gpu.synchronize();
    let before = backend.capture_execution_state()?;
    let free = backend.kv.free_blocks();
    let original = backend
        .sequences
        .get_mut(&second)
        .ok_or_else(|| Error::invariant("fixture state"))?
        .probes
        .replace(backend.gpu.zeros(1)?);
    let tasks = vec![
        plain_task(RequestId::ONE, StateId::ONE, vec![1, 2, 3, 5]),
        plain_task(RequestId::new(2)?, second, vec![1, 2, 3, 8]),
    ];
    let error = backend
        .submit(
            &program,
            &step(&program, &[StateId::ONE, second], 1)?,
            tasks,
        )
        .err()
        .ok_or_else(|| Error::invariant("encoding failure expected"))?;
    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(backend.capture_execution_state()?, before);
    assert_eq!(backend.kv.free_blocks(), free);
    backend.validate_state_ownership(&[(StateId::ONE, 8, 3), (second, 8, 3)])?;
    backend
        .sequences
        .get_mut(&second)
        .ok_or_else(|| Error::invariant("fixture state"))?
        .probes = original;
    let mut ticket = backend.submit(
        &program,
        &step(&program, &[second], 1)?,
        vec![plain_task(RequestId::new(2)?, second, vec![1, 2, 3, 8])],
    )?;
    assert_eq!(backend.inflight_pins.len(), 1);
    assert_eq!(backend.kv.references(backend.inflight_pins[0])?, 2);
    backend.validate_state_ownership(&[(StateId::ONE, 8, 3), (second, 8, 4)])?;
    while backend.poll(&mut ticket)?.is_none() {
        assert!(Instant::now() < deadline);
        sleep(Duration::from_micros(100));
    }
    assert_eq!(backend.inflight_pins, []);
    backend.validate_state_ownership(&[(StateId::ONE, 8, 3), (second, 8, 4)])?;
    Ok(())
}
