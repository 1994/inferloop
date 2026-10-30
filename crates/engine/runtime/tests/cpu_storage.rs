mod support;

use infer_core::event::CpuStage;
use infer_core::{Error, ErrorCode, ModelId, RequestId, Result};
use infer_ir::{CanonicalRequest, PrecisionPlan, WorkloadOutput};
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, EngineOutput, RuntimeConfig};

fn engine(config: RuntimeConfig) -> Result<Engine<support::ProtocolBackend>> {
    let ir = support::model(ModelId::ONE);
    let mut registry = KernelRegistry::default();
    registry.register(&support::DeclaredKernels)?;
    Engine::new(
        // These scenes submit more than a handful of requests, so the double's state map has to
        // be sized for them.
        support::ProtocolBackend::new(64, 16, &ir)?,
        ir,
        PrecisionPlan::f32(),
        &registry,
        config,
    )
}
fn request(id: u64) -> Result<CanonicalRequest> {
    serde_json::from_value(serde_json::json!({"id":id,"model":1,"input":{"Sequence":{"tokens":[1,2,3,4,5]}},"workload":{"Generate":{"max_new_tokens":2}}}))
        .map_err(|error| Error::invalid(error.to_string()))
}
fn drain(
    engine: &mut Engine<support::ProtocolBackend>,
    output: &mut Vec<EngineOutput>,
) -> Result<()> {
    for _ in 0..1000 {
        engine.tick_into(engine.now_us() + 1, output)?;
        if engine.is_idle() {
            return engine.check_invariants();
        }
    }
    Err(Error::invariant("bounded CPU pipeline did not drain"))
}
#[test]
fn result_reader_holds_token_and_byte_credits_after_request_slot_is_freed() -> Result<()> {
    let mut engine = engine(RuntimeConfig {
        max_requests: 1,
        ..RuntimeConfig::default()
    })?;
    engine.submit(request(1)?)?;
    let mut outputs = Vec::with_capacity(4);
    drain(&mut engine, &mut outputs)?;
    let done = engine
        .take_completed(RequestId::ONE)?
        .ok_or_else(|| Error::invariant("missing completion"))?;
    outputs.clear();
    let held = engine.inspect().cpu;
    assert!(held.retained_host_tokens > 0 && held.retained_host_bytes > 0);
    assert_eq!(
        engine.submit(request(2)?).err().map(|e| e.code),
        Some(ErrorCode::Capacity)
    );
    let payload = done
        .output
        .ok_or_else(|| Error::invariant("missing token output"))?;
    let WorkloadOutput::Tokens(tokens) = payload else {
        return Err(Error::invariant("wrong output"));
    };
    assert_eq!(tokens.len(), 2);
    drop(tokens);
    assert_eq!(engine.inspect().cpu.retained_host_tokens, 0);
    assert_eq!(engine.inspect().cpu.retained_host_bytes, 0);
    engine.submit(request(3)?)?;
    drain(&mut engine, &mut outputs)
}

#[test]
fn multi_image_credit_rejection_preserves_admission_and_releases_after_completion() -> Result<()> {
    use infer_core::MediaId;
    use infer_ir::{MediaInput, MediaItem, Modality, RequestInput};
    use std::sync::Arc;

    const FLOATS_PER_IMAGE: usize = 16_384;
    const REQUEST_HEADROOM_BYTES: usize = 192 * 1024;
    let mut config = RuntimeConfig {
        max_requests: 4,
        max_num_seqs: 2,
        candidate_limit: 4,
        history_capacity: 8,
        max_history_bytes: 4096,
        event_capacity: 64,
        ..RuntimeConfig::default()
    };
    config.cpu.host_tokens = 4096;
    let ir = support::model(ModelId::ONE);
    config.cpu.host_bytes =
        config.cpu.fixed_bytes_for(&config, ir.vocab_size)? + REQUEST_HEADROOM_BYTES;
    let mut registry = KernelRegistry::default();
    registry.register(&support::DeclaredKernels)?;
    let mut engine = Engine::new(
        support::ProtocolBackend::new(4, 2, &ir)?.multimodal(),
        ir,
        PrecisionPlan::f32(),
        &registry,
        config,
    )?;
    let image_request = |id| -> Result<CanonicalRequest> {
        let mut request = request(id)?;
        let RequestInput::Sequence { media, .. } = &mut request.input else {
            return Err(Error::invariant("expected a token sequence"));
        };
        for (index, media_id) in [(0, 1), (1, 2)] {
            media.push(MediaInput {
                id: MediaId::new(media_id)?,
                modality: Modality::Image,
                content_digest: format!("image-{id}-{index}"),
                item: Some(MediaItem {
                    modality: Modality::Image,
                    pixels: Arc::from(vec![0.5; FLOATS_PER_IMAGE]),
                    grid: (1, 1, 1),
                    span: index..index + 1,
                }),
            });
        }
        Ok(request)
    };

    engine.submit(image_request(1)?)?;
    let held = engine.inspect().cpu;
    assert!(held.retained_host_bytes >= 2 * FLOATS_PER_IMAGE * size_of::<f32>());
    assert_eq!(
        engine
            .submit(image_request(2)?)
            .err()
            .map(|error| error.code),
        Some(ErrorCode::Capacity)
    );
    let after_rejection = engine.inspect().cpu;
    assert_eq!(
        after_rejection.retained_host_bytes,
        held.retained_host_bytes
    );
    assert_eq!(
        after_rejection.retained_host_tokens,
        held.retained_host_tokens
    );
    assert!(engine.request(RequestId::new(2)?).is_err());
    engine.check_invariants()?;

    let mut outputs = Vec::new();
    drain(&mut engine, &mut outputs)?;
    let completed = engine
        .take_completed(RequestId::ONE)?
        .ok_or_else(|| Error::invariant("missing completion"))?;
    outputs.clear();
    drop(completed);
    assert_eq!(engine.inspect().cpu.retained_host_bytes, 0);
    assert_eq!(engine.inspect().cpu.retained_host_tokens, 0);
    engine.submit(image_request(2)?)?;
    drain(&mut engine, &mut outputs)
}

#[test]
fn growing_runtime_host_limit_admits_sixteen_large_images() -> Result<()> {
    use infer_core::MediaId;
    use infer_ir::{MediaInput, MediaItem, Modality, RequestInput};
    use std::sync::Arc;

    const FLOATS_PER_IMAGE: usize = 4 * 1024 * 1024;
    const INITIAL_LIMIT: usize = 20 * 1024 * 1024;
    const EXPANDED_LIMIT: usize = 320 * 1024 * 1024;
    let mut config = RuntimeConfig {
        max_requests: 16,
        max_num_seqs: 16,
        candidate_limit: 16,
        history_capacity: 8,
        max_history_bytes: 4096,
        event_capacity: 64,
        ..RuntimeConfig::default()
    };
    config.cpu.host_tokens = 4096;
    let ir = support::model(ModelId::ONE);
    config.cpu.host_bytes = config.cpu.fixed_bytes_for(&config, ir.vocab_size)? + EXPANDED_LIMIT;
    let mut registry = KernelRegistry::default();
    registry.register(&support::DeclaredKernels)?;
    let mut engine = Engine::new(
        support::ProtocolBackend::new(32, 16, &ir)?.multimodal(),
        ir,
        PrecisionPlan::f32(),
        &registry,
        config,
    )?;
    let limit = engine.host_byte_limit();
    let pixels: Arc<[f32]> = Arc::from(vec![0.5; FLOATS_PER_IMAGE]);
    let image_request = |id| -> Result<CanonicalRequest> {
        let mut request = request(id)?;
        let RequestInput::Sequence { media, .. } = &mut request.input else {
            return Err(Error::invariant("expected a token sequence"));
        };
        media.push(MediaInput {
            id: MediaId::ONE,
            modality: Modality::Image,
            content_digest: format!("image-{id}"),
            item: Some(MediaItem {
                modality: Modality::Image,
                pixels: pixels.clone(),
                grid: (1, 1, 1),
                span: 0..1,
            }),
        });
        Ok(request)
    };
    limit.set(INITIAL_LIMIT);
    engine.submit(image_request(1)?)?;
    assert_eq!(
        engine
            .submit(image_request(2)?)
            .err()
            .map(|error| error.code),
        Some(ErrorCode::Capacity)
    );
    limit.set(EXPANDED_LIMIT);
    for id in 2..=16 {
        engine.submit(image_request(id)?)?;
    }
    assert!(engine.inspect().cpu.retained_host_bytes >= 16 * FLOATS_PER_IMAGE * size_of::<f32>());
    engine.check_invariants()
}

#[test]
fn bounded_candidate_windows_cover_all_requests_without_full_ready_scan() -> Result<()> {
    let mut engine = engine(RuntimeConfig {
        max_requests: 32,
        candidate_limit: 4,
        max_num_seqs: 2,
        ..RuntimeConfig::default()
    })?;
    for id in 1..=32 {
        engine.submit(request(id)?)?;
    }
    let mut output = Vec::with_capacity(128);
    engine.tick_into(1, &mut output)?;
    let first = engine
        .decisions()
        .back()
        .ok_or_else(|| Error::invariant("no decision"))?;
    let window = first
        .window
        .as_ref()
        .ok_or_else(|| Error::invariant("no candidate evidence"))?;
    assert_eq!((window.inspected, window.ready), (4, 32));
    assert_eq!(
        engine.explain(RequestId::new(32)?)?["latest_window_status"],
        "outside_candidate_window"
    );
    let storage = output.as_ptr();
    drain(&mut engine, &mut output)?;
    assert_eq!(
        output
            .iter()
            .filter(|event| matches!(event, EngineOutput::Finished(_)))
            .count(),
        32
    );
    assert_eq!(output.as_ptr(), storage);
    assert!(
        engine
            .decisions()
            .iter()
            .all(|decision| decision.window.as_ref().is_some_and(|w| w.inspected <= 4))
    );
    Ok(())
}
#[test]
fn generation_uses_separate_preallocated_tail_and_shared_prompt() -> Result<()> {
    let mut engine = engine(RuntimeConfig::default())?;
    let request = request(1)?;
    let infer_ir::RequestInput::Sequence { ref tokens, .. } = request.input else {
        return Err(Error::invariant("fixture input"));
    };
    let prompt = tokens.clone();
    engine.submit(request)?;
    let record = engine.request(RequestId::ONE)?;
    assert!(record.context.prompt.shares_storage(&prompt));
    let tail = record.generated.as_ptr();
    let mut output = Vec::with_capacity(4);
    drain(&mut engine, &mut output)?;
    let record = engine.request(RequestId::ONE)?;
    assert!(record.context.prompt.shares_storage(&prompt));
    assert_eq!(record.generated.as_ptr(), tail);
    assert_eq!(record.context.generated_len(), 2);
    Ok(())
}

#[test]
fn cpu_stage_service_is_exported_separately_from_device_execution() -> Result<()> {
    let mut engine = engine(RuntimeConfig::default())?;
    engine.submit(request(1)?)?;
    let mut output = Vec::with_capacity(4);
    drain(&mut engine, &mut output)?;
    let metrics = engine.prometheus();
    for stage in CpuStage::LABELS {
        assert!(metrics.contains(&format!("infer_cpu_{stage}_seconds_count")));
    }
    Ok(())
}

#[test]
fn replay_byte_budget_counts_decision_schema_even_with_a_short_prompt() -> Result<()> {
    let mut engine = engine(RuntimeConfig {
        max_history_bytes: 1024,
        ..RuntimeConfig::default()
    })?;
    let mut request = request(1)?;
    request.workload = infer_ir::Workload::Decision(infer_ir::DecisionSchema {
        questions: vec![
            infer_ir::DecisionQuestion::Binary {
                negative_token: 1,
                positive_token: 2
            };
            100
        ],
        calibration_temperature: 1.0,
        abstain_below: None,
        readout: infer_ir::DecisionReadout::Projection,
    });
    engine.submit(request)?;
    assert!(engine.inspect().dropped_actions > 0);
    let mut output = Vec::with_capacity(4);
    drain(&mut engine, &mut output)?;
    assert_eq!(
        engine
            .request(RequestId::ONE)?
            .completed
            .as_ref()
            .map(|done| &done.reason),
        Some(&infer_core::FinishReason::Completed)
    );
    Ok(())
}

#[test]
fn tiny_planning_budget_retains_fair_progress_and_does_not_isolate_engine() -> Result<()> {
    let mut engine = engine(RuntimeConfig {
        max_requests: 16,
        max_num_seqs: 8,
        candidate_limit: 8,
        scheduler: infer_ir::SchedulerConfig {
            max_planning_probes: 1,
            ..infer_ir::SchedulerConfig::default()
        },
        ..RuntimeConfig::default()
    })?;
    for id in 1..=16 {
        engine.submit(request(id)?)?;
    }
    let mut output = Vec::with_capacity(64);
    drain(&mut engine, &mut output)?;
    assert_eq!(
        output
            .iter()
            .filter(|event| matches!(event, EngineOutput::Finished(_)))
            .count(),
        16
    );
    assert!(
        engine
            .decisions()
            .iter()
            .flat_map(|decision| &decision.deferred)
            .any(|deferred| deferred.reason == infer_ir::DeferReason::PlanningBudget)
    );
    assert!(engine.inspect().fault.is_none());
    Ok(())
}
