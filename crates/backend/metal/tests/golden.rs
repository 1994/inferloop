#![cfg(target_os = "macos")]
use infer_backend_metal::{MetalBackend, MetalConfig, MetalKernels};
use infer_core::*;
use infer_ir::*;
use infer_kernel_api::KernelRegistry;
use infer_model_recipes::decoder::lower;
use infer_models::ModelPackage;
use infer_spi::BackendProvider;
use std::error::Error;
use std::fs::read;
use std::result::Result;
use std::thread::sleep;
use std::{path::Path, time::Duration, time::Instant};

#[test]
fn actual_metal_gpu_matches_official_layer_hidden_logits_for_both_hybrid_configs() {
    if !MetalBackend::available() {
        eprintln!("No Metal device; GPU test skipped");
        return;
    }
    for name in ["qwen-hybrid-tiny", "qwen-hybrid-grouped"] {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../examples")
            .join(name);
        let golden: serde_json::Value =
            serde_json::from_slice(&read(root.join("golden.json")).unwrap()).unwrap();
        let mut package = ModelPackage::open(&root, ModelId::new(1).unwrap()).unwrap();
        let mut backend = MetalBackend::from_package(&mut package, MetalConfig::default()).unwrap();
        backend.enable_layer_probes(1024 * 1024).unwrap();
        let model = backend.model().clone();
        let mut registry = KernelRegistry::default();
        registry.register(&MetalKernels).unwrap();
        let program = infer_compiler::compile(
            ProgramId::new(1).unwrap(),
            infer_compiler::lower(&model, lower(&model).unwrap(), PrecisionPlan::f32()).unwrap(),
            &registry,
            &backend.capabilities(),
            1 << 20,
        )
        .unwrap();
        let state = StateId::new(1).unwrap();
        backend.reserve_state(state, 32).unwrap();
        for prefix in golden["prefixes"].as_array().unwrap() {
            check_prefix(&mut backend, &program, &model, state, prefix, name).unwrap();
        }

        assert_eq!(backend.inspect().tokens_executed, 6);
        assert!(
            backend.profile()["traceEvents"]
                .as_array()
                .unwrap()
                .iter()
                .any(|e| e["dur"].as_f64().unwrap() > 0.0)
        );
        backend.release_state(state).unwrap();
        assert_eq!(backend.inspect().allocated_bytes, 0);
    }
}

fn check_prefix(
    backend: &mut MetalBackend,
    program: &ExecutionProgram,
    model: &ModelIr,
    state: StateId,
    prefix: &serde_json::Value,
    name: &str,
) -> Result<(), Box<dyn Error>> {
    let tokens: Vec<u32> = serde_json::from_value(prefix["tokens"].clone())?;
    let request = RequestId::new(1)?;
    let step = StepPlan {
        id: StepId::new(tokens.len() as u64)?,
        decision: DecisionId::new(1)?,
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
    let mut ticket = backend.submit(
        program,
        &step,
        vec![ExecutionTask {
            plain_greedy: false,
            request,
            state,
            tokens: tokens.into(),

            sampling: None,

            media: None,
        }],
    )?;
    let deadline = Instant::now() + Duration::from_secs(10);
    let output = loop {
        if let Some(mut result) = backend.poll(&mut ticket)? {
            break result.remove(0).output;
        }
        assert!(Instant::now() < deadline, "Metal completion timeout");
        sleep(Duration::from_millis(1));
    };
    for (label, expected, actual) in [
        (
            "logits",
            serde_json::from_value::<Vec<f32>>(prefix["logits"].clone())?,
            output.logits,
        ),
        (
            "hidden",
            serde_json::from_value::<Vec<Vec<f32>>>(prefix["hidden"].clone())?
                .into_iter()
                .flatten()
                .collect(),
            output.hidden.into_iter().flatten().collect(),
        ),
    ] {
        let metric = infer_quality::compare(&expected, &actual, 2e-6, 2e-5)?;
        assert!(metric.passed, "{name} {label}: {metric:?}");
    }
    let probes = backend.drain_layer_probes();
    assert_eq!(probes.len(), model.mixers.len());
    for probe in probes {
        let expected: Vec<f32> =
            serde_json::from_value(prefix["layers"][probe.layer][probe.position].clone())?;
        let metric = infer_quality::compare(&expected, &probe.hidden, 2e-6, 2e-5)?;
        assert!(metric.passed, "{name} layer {}: {metric:?}", probe.layer);
    }
    Ok(())
}
