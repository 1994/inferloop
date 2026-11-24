use super::*;
use crate::RuntimeConfig;
use infer_core::ModelId;
use infer_ir::{CanonicalRequest, PrecisionPlan};
use infer_kernel_api::KernelRegistry;

use infer_test_support::fixtures as support;

#[test]
fn trace_parent_without_retained_events_is_pruned_when_request_is_consumed() -> Result<()> {
    let ir = support::model(ModelId::ONE);
    let mut registry = KernelRegistry::default();
    registry.register(&support::DeclaredKernels)?;
    let mut engine = Engine::new(
        support::ProtocolBackend::new(2, 2, &ir)?,
        ir,
        PrecisionPlan::f32(),
        &registry,
        RuntimeConfig {
            event_capacity: 1,
            history_capacity: 2,
            ..Default::default()
        },
    )?;
    // Fill the ring so request events are dropped; trace ownership is then entirely live-state based.
    engine.event(EventKind::Scheduled, ObjectKind::Step, 1, 0, 0, 0);
    let parent = TraceContext::parse("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")?;
    let request: CanonicalRequest = serde_json::from_value(json!({"id":1,"model":1,
        "input":{"Sequence":{"tokens":[1,2,3]}},"workload":{"Generate":{"max_new_tokens":2}}}))
    .map_err(|e| Error::invalid(e.to_string()))?;
    engine.submit_with_trace(request, Some(parent))?;
    engine.collect_observations();
    assert_eq!(engine.observations.parents.len(), 1);
    assert!(!engine.observations.store.contains_request(1));
    engine.event(EventKind::Scheduled, ObjectKind::Step, 2, 0, 0, 0);
    engine.cancel(RequestId::ONE)?;
    engine.take_completed(RequestId::ONE)?;
    engine.collect_observations();
    assert_eq!(engine.observations.store.history_evicted, 0);
    assert!(engine.observations.parents.is_empty());
    Ok(())
}
