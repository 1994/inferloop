#![cfg(target_os = "macos")]
use infer_backend_metal::{MetalBackend, MetalConfig, MetalKernels};
use infer_core::*;
use infer_ir::*;
use infer_kernel_api::KernelRegistry;
use infer_models::ModelPackage;
use infer_runtime::{Engine, RuntimeConfig};
use infer_spi::BackendProvider;
use infer_workloads::ProjectionWorkloads;
use std::fs::read;
use std::thread::sleep;
use std::{path::Path, path::PathBuf, time::Duration, time::Instant};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny")
}
#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn load(config: MetalConfig, runtime: RuntimeConfig) -> (Engine<MetalBackend>, KernelRegistry) {
    let mut p = ModelPackage::open(root(), ModelId::new(1).unwrap()).unwrap();
    let backend = MetalBackend::from_package(&mut p, config).unwrap();
    let model = backend.model().clone();
    let mut kernels = KernelRegistry::default();
    kernels.register(&MetalKernels).unwrap();
    (
        Engine::new(backend, model, PrecisionPlan::f32(), &kernels, runtime).unwrap(),
        kernels,
    )
}
#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn generate(id: u64) -> CanonicalRequest {
    serde_json::from_value(
        serde_json::json!({"id":id,"model":1,"input":{"Sequence":{"tokens":[1,2,3,5,8,13]}},
        "workload":{"Generate":{"max_new_tokens":5}}}),
    )
    .unwrap()
}
#[expect(
    clippy::panic,
    reason = "This test helper rejects unexpected output variants so an incorrect fixture cannot pass silently"
)]
#[expect(
    clippy::unwrap_used,
    reason = "Integration fixture helpers intentionally fail the test immediately on invalid setup or unexpected runtime output"
)]
fn drain(engine: &mut Engine<MetalBackend>, start: u64) {
    let deadline = Instant::now() + Duration::from_secs(10);
    for now in start..start + 100_000 {
        if engine.is_idle() {
            return;
        }
        engine.tick(now).unwrap();
        engine.drain_events();
        assert!(Instant::now() < deadline, "Metal runtime progress timeout");
        sleep(Duration::from_micros(100));
    }
    panic!("Metal tick limit");
}
#[test]
fn metal_generation_prefix_and_physical_checkpoint_preserve_official_trajectory() {
    if !MetalBackend::available() {
        return;
    }
    let golden: serde_json::Value =
        serde_json::from_slice(&read(root().join("golden.json")).unwrap()).unwrap();
    for chunk in [1, 3, 64] {
        let (mut engine, _) = load(
            MetalConfig {
                block_size: 2,
                ..Default::default()
            },
            RuntimeConfig {
                max_num_batched_tokens: chunk,
                ..Default::default()
            },
        );
        for id in [1, 2] {
            engine.submit(generate(id)).unwrap();
            let start = engine.now_us() + 1;
            drain(&mut engine, start);
            assert_eq!(
                serde_json::to_value(
                    &engine
                        .request(RequestId::new(id).unwrap())
                        .unwrap()
                        .completed
                        .as_ref()
                        .unwrap()
                        .output
                )
                .unwrap(),
                serde_json::json!({"Tokens":golden["greedy_tokens"]})
            );
        }
        assert!(engine.backend().inspect().prefix_hits > 0, "chunk {chunk}");
        assert!(engine.backend().inspect().tokens_executed < 20);
        assert_eq!(engine.backend().inspect().sequences, 0);
        assert!(engine.inspect().cost_model.observations > 0);
        assert_eq!(
            engine.inspect().cost_model.last_source,
            Some(TimingSource::MetalGpu)
        );
    }
    let (mut original, kernels) = load(MetalConfig::default(), RuntimeConfig::default());
    original.submit(generate(1)).unwrap();
    original.tick(0).unwrap();
    let snapshot = loop {
        let (_, snapshot) = original.quiesce(original.now_us() + 1).unwrap();
        if let Some(snapshot) = snapshot {
            break snapshot;
        }
        sleep(Duration::from_micros(100));
    };
    let mut changed = snapshot.clone();
    changed.backend_identity = "host-incorrect-target".into();
    assert!(Engine::restore(original.backend().fresh().unwrap(), &kernels, changed).is_err());
    let mut restored =
        Engine::restore(original.backend().fresh().unwrap(), &kernels, snapshot).unwrap();
    let start = original.now_us() + 1;
    drain(&mut original, start);
    let start = restored.now_us() + 1;
    drain(&mut restored, start);
    assert_eq!(
        original
            .request(RequestId::new(1).unwrap())
            .unwrap()
            .completed
            .as_ref()
            .unwrap()
            .output,
        restored
            .request(RequestId::new(1).unwrap())
            .unwrap()
            .completed
            .as_ref()
            .unwrap()
            .output
    );
    assert_eq!(restored.backend().inspect().allocated_bytes, 0);
}
#[test]
#[expect(
    clippy::cast_possible_truncation,
    reason = "The model stores F32 tensors; intermediate F64 accumulation is intentionally rounded back to F32 at this boundary"
)]
fn metal_projection_readouts_match_independent_torch_for_four_workloads() {
    if !MetalBackend::available() {
        return;
    }
    let golden: serde_json::Value =
        serde_json::from_slice(&read(root().join("readout-golden.json")).unwrap()).unwrap();
    let input: Vec<CanonicalRequest> =
        serde_json::from_slice(&read(root().join("../requests.json")).unwrap()).unwrap();
    for chunk in [1, 64] {
        let (engine, _) = load(
            MetalConfig::default(),
            RuntimeConfig {
                max_num_batched_tokens: chunk,
                ..Default::default()
            },
        );
        let mut engine = engine
            .with_workloads(ProjectionWorkloads::open(root(), 8, 32).unwrap())
            .unwrap();
        for r in input.clone() {
            engine.submit(r).unwrap();
        }
        drain(&mut engine, 0);
        let output = |id| {
            engine
                .request(RequestId::new(id).unwrap())
                .unwrap()
                .completed
                .as_ref()
                .unwrap()
                .output
                .as_ref()
                .unwrap()
        };
        let WorkloadOutput::Embedding(vector) = output(2) else {
            panic!("embedding")
        };
        let expected: Vec<f32> = serde_json::from_value(golden["embedding"].clone()).unwrap();
        assert!(
            infer_quality::compare(&expected, vector, 2e-6, 2e-5)
                .unwrap()
                .passed
        );
        let WorkloadOutput::Ranking(ranks) = output(3) else {
            panic!("rank")
        };
        for rank in ranks {
            assert!(
                (rank.score - golden["rank_scores"][rank.index].as_f64().unwrap() as f32).abs()
                    < 2e-6
            );
        }
        let WorkloadOutput::Decisions(decisions) = output(4) else {
            panic!("decision")
        };
        for (index, result) in decisions.iter().enumerate() {
            let expected: Vec<f32> =
                serde_json::from_value(golden["decision_probabilities"][index].clone()).unwrap();
            if !expected.is_empty() {
                assert!(
                    infer_quality::compare(&expected, &result.probabilities, 2e-6, 2e-5)
                        .unwrap()
                        .passed
                );
            }
        }
        assert!(
            (decisions[3].expected.unwrap()
                - golden["continuous_expected"].as_f64().unwrap() as f32)
                .abs()
                < 2e-6
        );
        assert_eq!(engine.backend().inspect().sequences, 0);
        assert!(
            engine
                .request_records()
                .all(|r| r.completed.as_ref().unwrap().measurement.successful)
        );
    }
}
#[test]
fn metal_cancel_waits_for_owned_completion_and_queued_state_can_release() {
    if !MetalBackend::available() {
        return;
    }
    let (mut engine, _) = load(
        MetalConfig::default(),
        RuntimeConfig {
            max_num_seqs: 1,
            ..Default::default()
        },
    );
    engine.submit(generate(1)).unwrap();
    engine.submit(generate(2)).unwrap();
    engine.tick(0).unwrap();
    assert!(engine.inspect().inflight_step.is_some());
    engine.cancel(RequestId::new(2).unwrap()).unwrap();
    assert_eq!(engine.backend().inspect().sequences, 1);
    engine.cancel(RequestId::new(1).unwrap()).unwrap();
    assert_eq!(engine.backend().inspect().sequences, 1);
    drain(&mut engine, 1);
    assert_eq!(engine.backend().inspect().sequences, 0);
    assert_eq!(engine.inspect().state.allocated_pages, 0);
}

#[test]
fn metal_physical_admission_failure_leaves_no_logical_or_device_state() {
    if !MetalBackend::available() {
        return;
    }
    let mut package = ModelPackage::open(root(), ModelId::new(1).unwrap()).unwrap();
    let memory = package.manifest.host_f32_bytes + package.graph.scratch_elements as u64 * 4 + 4;
    let model = package.imported.model.clone();
    let weights = package.load_host_weights(1024 * 1024).unwrap();
    let backend = MetalBackend::new(
        model.clone(),
        weights,
        MetalConfig {
            memory_bytes: memory,
            prefill_chunk_tokens: 1,
            prefix_cache_bytes: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let mut registry = KernelRegistry::default();
    registry.register(&MetalKernels).unwrap();
    let mut engine = Engine::new(
        backend,
        model,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig::default(),
    )
    .unwrap();
    assert_eq!(
        engine.submit(generate(1)).unwrap_err().code,
        ErrorCode::Capacity
    );
    assert_eq!(engine.inspect().active_requests, 0);
    assert_eq!(engine.inspect().state.allocated_pages, 0);
    assert_eq!(engine.backend().inspect().sequences, 0);
    assert_eq!(engine.backend().inspect().allocated_bytes, 0);
}

#[test]
fn metal_rejects_foreign_program_and_aliased_submission_before_mutating_state() {
    if !MetalBackend::available() {
        return;
    }
    let (mut engine, _) = load(MetalConfig::default(), RuntimeConfig::default());
    let mut program = engine.program().clone();
    let model = engine.model().clone();
    program.backend = BackendKind::Cuda;
    assert!(engine.backend().validate_program(&model, &program).is_err());
    program.backend = BackendKind::Metal;
    let backend = engine.backend_mut();
    let state = StateId::new(1).unwrap();
    backend.reserve_state(state, 16).unwrap();
    let step = StepPlan {
        id: StepId::new(1).unwrap(),
        decision: DecisionId::new(1).unwrap(),
        program: program.id,
        role: ExecutionRole::Prefill,
        work: [1, 2]
            .map(|id| PlannedWork {
                request: RequestId::new(id).unwrap(),
                state,
                token_count: 1,
                role: ExecutionRole::Prefill,
            })
            .to_vec(),
        cost: CostEstimate::default(),
        graph: None,
        quantum_overrun: false,
    };
    let tasks = [1, 2]
        .map(|id| ExecutionTask {
            plain_greedy: false,
            request: RequestId::new(id).unwrap(),
            state,
            tokens: vec![1].into(),

            sampling: None,

            media: None,
        })
        .to_vec();
    assert!(backend.submit(&program, &step, tasks).is_err());
    backend.validate_state_ownership(&[(state, 16, 0)]).unwrap();
    assert_eq!(backend.inspect().tokens_executed, 0);
    backend.release_state(state).unwrap();
}

#[test]
fn metal_completion_ticket_cannot_cross_executor_instances_with_same_step_id() {
    if !MetalBackend::available() {
        return;
    }
    let (mut first, _) = load(MetalConfig::default(), RuntimeConfig::default());
    let (mut second, _) = load(MetalConfig::default(), RuntimeConfig::default());
    let program = first.program().clone();
    let state = StateId::new(1).unwrap();
    let request = RequestId::new(1).unwrap();
    let step = StepPlan {
        id: StepId::new(1).unwrap(),
        decision: DecisionId::new(1).unwrap(),
        program: program.id,
        role: ExecutionRole::Prefill,
        work: vec![PlannedWork {
            request,
            state,
            token_count: 1,
            role: ExecutionRole::Prefill,
        }],
        cost: CostEstimate::default(),
        graph: None,
        quantum_overrun: false,
    };
    let tasks = vec![ExecutionTask {
        plain_greedy: false,
        request,
        state,
        tokens: vec![1].into(),

        sampling: None,

        media: None,
    }];
    let a = first.backend_mut();
    let b = second.backend_mut();
    a.reserve_state(state, 16).unwrap();
    b.reserve_state(state, 16).unwrap();
    let mut ta = a.submit(&program, &step, tasks.clone()).unwrap();
    let mut tb = b.submit(&program, &step, tasks).unwrap();
    assert!(b.poll(&mut ta).is_err());
    assert!(a.poll(&mut tb).is_err());
    // Rejection must leave each owner's matching completion usable.
    let deadline = Instant::now() + Duration::from_secs(10);
    for (backend, ticket) in [(a, &mut ta), (b, &mut tb)] {
        while backend.poll(ticket).unwrap().is_none() {
            assert!(Instant::now() < deadline);
            sleep(Duration::from_millis(1));
        }
        backend.release_state(state).unwrap();
        assert_eq!(backend.inspect().allocated_bytes, 0);
    }
}
