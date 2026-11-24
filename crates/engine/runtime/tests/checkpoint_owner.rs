use infer_test_support::fixtures as support;

use infer_core::{Error, ModelId, RequestId, Result};
use infer_ir::{CanonicalRequest, PrecisionPlan};
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, RuntimeConfig, RuntimeSnapshot};
use std::thread::sleep;
use std::{time::Duration, time::Instant};

fn fixture() -> Result<(support::ProtocolBackend, KernelRegistry)> {
    let ir = support::model(ModelId::ONE);
    let mut registry = KernelRegistry::default();
    registry.register(&support::DeclaredKernels)?;
    Ok((support::ProtocolBackend::new(4, 4, &ir)?, registry))
}
fn request() -> Result<CanonicalRequest> {
    serde_json::from_value(serde_json::json!({"id":1,"model":1,"input":{"Sequence":{"tokens":[1,2,3]}},"workload":{"Generate":{"max_new_tokens":3}}}))
        .map_err(|error| Error::invalid(error.to_string()))
}
fn wait(limit: Instant) -> Result<()> {
    if Instant::now() >= limit {
        return Err(Error::invariant("checkpoint owner did not converge"));
    }
    sleep(Duration::from_micros(100));
    Ok(())
}
#[test]
fn checkpoint_is_acknowledged_by_owner_and_restored_states_rejoin_async_pipeline() -> Result<()> {
    let (backend, registry) = fixture()?;
    let model = support::model(ModelId::ONE);
    let engine = Engine::new(
        backend,
        model,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig::default(),
    )?;
    let mut engine = engine.into_threaded(4, Duration::from_millis(20))?;
    let prepared = engine.request_preparer()?.prepare(request()?)?;
    let mut quote = engine.admission_quote(&prepared)?;
    let limit = Instant::now() + Duration::from_secs(5);
    let bytes = loop {
        if let Some(infer_spi::ResourceReply::ReservationBytes(bytes)) = quote.poll()? {
            break bytes;
        }
        wait(limit)?;
    };
    engine.submit_quoted_with_trace(prepared, bytes, None)?;
    let snapshot = loop {
        let (_, checkpoint) = engine.quiesce(engine.now_us() + 1)?;
        if let Some(checkpoint) = checkpoint {
            break checkpoint;
        }
        wait(limit)?;
    };
    assert!(snapshot.checkpoint_supported);
    assert!(snapshot.inflight.is_none());
    let serialized = serde_json::to_vec(&snapshot).map_err(|e| Error::invalid(e.to_string()))?;
    let snapshot: RuntimeSnapshot =
        serde_json::from_slice(&serialized).map_err(|e| Error::invalid(e.to_string()))?;
    let (backend, registry) = fixture()?;
    let mut resumed = Engine::restore(backend, &registry, snapshot)?
        .into_threaded(4, Duration::from_millis(20))?;
    while !resumed.is_idle() {
        resumed.tick(resumed.now_us() + 1)?;
        wait(limit)?;
    }
    resumed.check_invariants()?;
    let done = resumed
        .take_completed(RequestId::ONE)?
        .ok_or_else(|| Error::invariant("restored completion lost"))?;
    assert_eq!(done.measurement.output_tokens, 3);
    assert!(done.measurement.successful);
    while !engine.is_idle() {
        engine.tick(engine.now_us() + 1)?;
        wait(limit)?;
    }
    Ok(())
}

struct ConstrainedToken;
impl infer_spi::GenerationConstraint for ConstrainedToken {
    fn mask(&mut self) -> Result<&[u32]> {
        Ok(&[8])
    }
    fn commit(&mut self, token: u32) -> Result<()> {
        if token == 3 {
            Ok(())
        } else {
            Err(Error::invariant("unexpected constrained token"))
        }
    }
}
#[test]
fn live_constraints_reject_checkpoint_and_replay_without_losing_generation() -> Result<()> {
    let (backend, registry) = fixture()?;
    let model = support::model(ModelId::ONE);
    let mut engine = Engine::new(
        backend,
        model,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig::default(),
    )?;
    let handle = infer_spi::ConstraintHandle::new(Box::new(ConstrainedToken), 32, 128);
    let prepared = engine
        .request_preparer()?
        .prepare(request()?)?
        .with_constraint(Some(handle))?;
    engine.submit_prepared_with_trace(prepared, None)?;
    assert!(engine.snapshot().is_err());
    let actions = engine.journal()?;
    assert!(
        actions
            .iter()
            .any(|action| matches!(action, infer_runtime::ReplayAction::ConstrainedSubmit(_)))
    );
    let (backend, registry) = fixture()?;
    let mut replay = Engine::new(
        backend,
        support::model(ModelId::ONE),
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig::default(),
    )?;
    assert!(replay.replay(&actions).is_err());
    assert!(replay.is_idle());
    let mut injected = replay.snapshot()?;
    injected
        .requests
        .insert(RequestId::ONE, engine.request(RequestId::ONE)?.clone());
    let (backend, registry) = fixture()?;
    assert!(
        matches!(Engine::restore(backend, &registry, injected), Err(error) if error.code == infer_core::ErrorCode::Unsupported)
    );
    let limit = Instant::now() + Duration::from_secs(5);
    while !engine.is_idle() {
        engine.tick(engine.now_us() + 1)?;
        wait(limit)?;
    }
    engine.check_invariants()?;
    assert!(engine.request(RequestId::ONE)?.constraint.is_none());
    let done = engine.take_completed(RequestId::ONE)?.unwrap();
    assert_eq!(done.measurement.output_tokens, 3);
    assert!(done.measurement.successful);
    Ok(())
}

#[test]
fn constrained_generation_keeps_sampling_on_the_engine_with_a_loaded_draft() -> Result<()> {
    let (backend, registry) = fixture()?;
    let mut engine = Engine::new(
        backend.requiring_engine_sampling(2),
        support::model(ModelId::ONE),
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig::default(),
    )?;
    let mut request = request()?;
    request.sampling.temperature = 0.0;
    let handle = infer_spi::ConstraintHandle::new(Box::new(ConstrainedToken), 32, 128);
    let prepared = engine
        .request_preparer()?
        .prepare(request)?
        .with_constraint(Some(handle))?;
    engine.submit_prepared_with_trace(prepared, None)?;
    let limit = Instant::now() + Duration::from_secs(5);
    while !engine.is_idle() {
        engine.tick(engine.now_us() + 1)?;
        wait(limit)?;
    }
    engine.check_invariants()?;
    let done = engine.take_completed(RequestId::ONE)?.unwrap();
    // The matcher rejects every token except3. The backend rejects either a
    // device-token hint or speculative sampling parameters, on prefill AND decode.
    assert!(done.measurement.successful);
    assert_eq!(done.measurement.output_tokens, 3);
    Ok(())
}
