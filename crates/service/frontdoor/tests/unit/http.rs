use axum::body::Body;
use axum::body::to_bytes;
use axum::http::Request;
use axum::http::StatusCode;
use infer_core::{Error, ErrorCode, ModelId, RequestId, Result};
use infer_kernel_api::KernelRegistry;
use infer_observe::ObservationQuery;
use infer_runtime::{CompletedRequest, Engine, EngineOutput, RuntimeConfig};
use infer_spi::BackendProvider;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::{collections::BTreeMap, time::Duration};
use tokio::sync::mpsc as async_mpsc;
use tokio::time::sleep;
use tokio::time::timeout;
use tower::ServiceExt;

use super::*;
use infer_ir::*;

// `actor.rs` includes the same source; both are modules of this crate's test binary, and the
// shared file is written to be included once per target.
#[allow(
    clippy::duplicate_mod,
    reason = "one shared test source, two unit-test modules"
)]
#[path = "../../../../engine/runtime/tests/support/mod.rs"]
pub(super) mod support;

use support::ProtocolBackend;

fn handle() -> RuntimeHandle {
    let ir = support::model(ModelId::ONE);
    let mut registry = KernelRegistry::default();
    registry.register(&support::DeclaredKernels).unwrap();
    RuntimeHandle::start(
        Engine::new(
            ProtocolBackend::new(16, 8, &ir).unwrap(),
            ir,
            PrecisionPlan::f32(),
            &registry,
            RuntimeConfig::default(),
        )
        .unwrap(),
    )
    .unwrap()
}
fn request(id: u64, max_new_tokens: usize) -> CanonicalRequest {
    CanonicalRequest {
        id: RequestId::new(id).unwrap(),
        model: ModelId::ONE,
        session: None,
        input: RequestInput::Sequence {
            tokens: vec![1, 2, 3].into(),
            media: vec![],
        },
        workload: Workload::Generate { max_new_tokens },
        qos: Qos::default(),
        sampling: Sampling::default(),
        prefix_boundary: None,
        extensions: BTreeMap::new(),
    }
}
async fn receive(receiver: &mut async_mpsc::Receiver<Result<EngineOutput>>) -> CompletedRequest {
    loop {
        let event = timeout(Duration::from_secs(2), receiver.recv())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        if let EngineOutput::Finished(done) = event {
            return done;
        }
    }
}
#[tokio::test]
async fn concurrent_streams_route_outputs_and_drain_results() {
    let handle = handle();
    let (a, b) = tokio::join!(handle.submit(request(1, 4)), handle.submit(request(2, 4)));
    let (mut a, mut b) = (a.unwrap(), b.unwrap());
    let (a, b) = tokio::join!(receive(&mut a), receive(&mut b));
    assert_eq!(a.request.get(), 1);
    assert_eq!(b.request.get(), 2);
    assert!(a.measurement.successful && b.measurement.successful);
    let inspection = handle.inspect().await.unwrap();
    assert_eq!(inspection.state.allocated_pages, 0);
    assert_eq!(inspection.completed_requests, 0);
    handle.shutdown().await.unwrap();
}
#[tokio::test]
async fn disconnected_client_is_cancelled_and_freed() {
    let handle = handle();
    let receiver = handle.submit(request(1, 100)).await.unwrap();
    drop(receiver);
    timeout(Duration::from_secs(2), async {
        loop {
            let inspection = handle.inspect().await.unwrap();
            if inspection.active_requests == 0 {
                assert_eq!(inspection.state.allocated_pages, 0);
                break;
            }
            sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap();
    handle.shutdown().await.unwrap();
}
#[tokio::test]
async fn slow_consumer_gets_terminal_cancellation_instead_of_leaking() {
    let handle = handle();
    let mut receiver = handle.submit(request(1, 100)).await.unwrap();
    timeout(Duration::from_secs(2), async {
        loop {
            if handle.inspect().await.unwrap().active_requests == 0 {
                break;
            }
            sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap();
    let result = receive(&mut receiver).await;
    assert_eq!(result.reason, infer_core::FinishReason::Cancelled);
    assert_eq!(handle.inspect().await.unwrap().state.allocated_pages, 0);
    handle.shutdown().await.unwrap();
}
#[tokio::test]
async fn schema_rejection_has_structured_error_and_does_not_enter_runtime() {
    let handle = handle();
    let mut payload = serde_json::to_value(request(1, 4)).unwrap();
    payload["unknown_feature"] = serde_json::json!(true);
    let response = router(handle.clone())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/native/v1/requests")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_vec(&payload).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let error: Error = serde_json::from_slice(&body).unwrap();
    assert_eq!(error.code, ErrorCode::InvalidInput);
    assert_eq!(handle.inspect().await.unwrap().active_requests, 0);
    handle.shutdown().await.unwrap();
}

/// The hybrid fixture's own weights are covered where the package is loaded
/// (`crates/model/package/tests/{loader,package}.rs` and the provider's unit cases); this scene
/// checks that the served text path uses the package's assets and reports its own work.
#[tokio::test]
async fn native_text_uses_package_assets_and_reports_its_work() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let model = support::model(ModelId::ONE);
    let assets = Arc::new(infer_models::TextAssets::open(root, model.max_sequence).unwrap());
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels).unwrap();
    let handle = RuntimeHandle::start(
        Engine::new(
            ProtocolBackend::peaking_at(3, 16, 8, &model).unwrap(),
            model,
            PrecisionPlan::f32(),
            &kernels,
            RuntimeConfig::default(),
        )
        .unwrap(),
    )
    .unwrap();
    let app = router_with_text(handle.clone(), assets);
    let payload = serde_json::json!({"id":1,"model":1,
        "prompt":"hello world system assistant token8 token13",
        "workload":{"Generate":{"max_new_tokens":5}}});
    let post = |body: &serde_json::Value| {
        Request::builder()
            .method("POST")
            .uri("/native/v1/text")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_vec(body).unwrap()))
            .unwrap()
    };
    let response = app.clone().oneshot(post(&payload)).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let result: serde_json::Value = serde_json::from_slice(&body).unwrap();
    // The declared peak is token 3 and the budget is five tokens, so the sequence continues with
    // the fixture's own decoding of that token.
    assert_eq!(
        result["result"]["output"]["Tokens"],
        serde_json::json!([3, 3, 3, 3, 3])
    );
    assert_eq!(result["text"], "system system system system system");
    assert!(result["tokenizer_fingerprint"].as_str().unwrap().len() > 16);
    let mut malformed = payload.clone();
    malformed["unknown"] = serde_json::json!(true);
    assert_eq!(
        app.clone()
            .oneshot(post(&malformed))
            .await
            .unwrap()
            .status(),
        StatusCode::UNPROCESSABLE_ENTITY
    );
    malformed = payload;
    malformed["messages"] = serde_json::json!([{"role":"user","content":"hello"}]);
    assert_eq!(
        app.oneshot(post(&malformed)).await.unwrap().status(),
        StatusCode::BAD_REQUEST
    );
    let inspection = handle.inspect().await.unwrap();
    assert_eq!(inspection.active_requests, 0);
    assert_eq!(inspection.state.allocated_pages, 0);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn metrics_history_and_trace_parent_are_available_after_result_consumption() {
    let handle = handle();
    let app = router(handle.clone());
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/native/v1/requests")
                .header("content-type", "application/json")
                .header(
                    "traceparent",
                    "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
                )
                .body(Body::from(serde_json::to_vec(&request(1, 3)).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let mut observations = BTreeMap::new();
    for path in [
        "/native/v1/observability",
        "/native/v1/events?request=1",
        "/native/v1/traces/otlp",
    ] {
        let response = app
            .clone()
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        observations.insert(
            path,
            serde_json::from_slice::<serde_json::Value>(&bytes).unwrap(),
        );
    }
    let summary = &observations["/native/v1/observability"];
    assert_eq!(summary["metrics"]["successful_requests"], 1);
    assert_eq!(summary["metrics"]["e2e"]["count"], 1);
    assert_eq!(summary["metrics"]["ttft"]["count"], 1);
    assert_eq!(summary["metrics"]["tpot"]["count"], 2);
    assert_eq!(summary["resources"]["logical_pages"], 0);
    let events = &observations["/native/v1/events?request=1"];
    assert_eq!(events["session"], summary["session"]);
    assert!(
        events["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|event| event["event"]["kind"] == "Finished")
    );
    let span =
        &observations["/native/v1/traces/otlp"]["resourceSpans"][0]["scopeSpans"][0]["spans"][0];
    assert_eq!(span["traceId"], "4bf92f3577b34da6a3ce929d0e0e4736");
    assert_eq!(span["parentSpanId"], "00f067aa0ba902b7");
    let metrics = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        metrics.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/plain; version=0.0.4")
    );
    let metrics = String::from_utf8(
        to_bytes(metrics.into_body(), 1 << 20)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(metrics.contains("infer_requests_successful_total{backend=\"cuda\"} 1\n"));
    handle.shutdown().await.unwrap();
}

struct HeldBackend {
    inner: ProtocolBackend,
    complete: Arc<AtomicBool>,
}
impl BackendProvider for HeldBackend {
    type Ticket = <ProtocolBackend as BackendProvider>::Ticket;
    fn identity(&self) -> &str {
        self.inner.identity()
    }
    fn capabilities(&self) -> DeviceCapabilities {
        self.inner.capabilities()
    }
    fn validate_program(&self, model: &ModelIr, program: &ExecutionProgram) -> Result<()> {
        self.inner.validate_program(model, program)
    }
    fn execution_graph(&self, model: &ModelIr) -> Result<DataflowGraph> {
        self.inner.execution_graph(model)
    }
    fn reserve_state_for(
        &mut self,
        state: infer_core::StateId,
        capacity: usize,
        readout: OutputReadout,
    ) -> Result<()> {
        // The engine creates sequences through this entry point; forwarding only `reserve_state`
        // leaves the double's state map empty.
        self.inner.reserve_state_for(state, capacity, readout)
    }
    fn submit(
        &mut self,
        program: &ExecutionProgram,
        step: &StepPlan,
        tasks: Vec<ExecutionTask>,
    ) -> Result<Self::Ticket> {
        self.inner.submit(program, step, tasks)
    }
    fn poll(&mut self, ticket: &mut Self::Ticket) -> Result<Option<Vec<TaskOutput>>> {
        if self.complete.load(Ordering::Acquire) {
            self.inner.poll(ticket)
        } else {
            Ok(None)
        }
    }
}
#[tokio::test]
async fn timed_out_actor_retains_device_ownership_and_serves_diagnostics_until_completion() {
    let ir = support::model(ModelId::ONE);
    let complete = Arc::new(AtomicBool::new(false));
    let backend = HeldBackend {
        inner: ProtocolBackend::new(16, 8, &ir).unwrap(),
        complete: complete.clone(),
    };
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels).unwrap();
    let handle = RuntimeHandle::start(
        Engine::new(
            backend,
            ir,
            PrecisionPlan::f32(),
            &kernels,
            RuntimeConfig {
                submission_timeout_us: 1000,
                ..Default::default()
            },
        )
        .unwrap(),
    )
    .unwrap();
    let mut output = handle.submit(request(1, 3)).await.unwrap();
    let error = timeout(Duration::from_secs(2), output.recv())
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert!(error.message.contains("SubmittedStepTimeout"));
    let inspection = handle.inspect().await.unwrap();
    assert!(!inspection.ready && inspection.resource_release_pending);
    assert!(inspection.state.allocated_pages > 0);
    assert_eq!(
        handle.submit(request(2, 3)).await.unwrap_err().code,
        ErrorCode::Backend
    );
    let diagnostics = handle.observe(ObservationQuery::Diagnostics).await.unwrap();
    assert!(
        diagnostics
            .as_array()
            .unwrap()
            .iter()
            .any(|diagnostic| diagnostic["code"] == "submission_timeout")
    );
    let health = router(handle.clone())
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(health.status(), StatusCode::SERVICE_UNAVAILABLE);
    drop(output);
    complete.store(true, Ordering::Release);
    timeout(Duration::from_secs(2), async {
        loop {
            let inspection = handle.inspect().await.unwrap();
            if !inspection.resource_release_pending {
                assert_eq!(inspection.state.allocated_pages, 0);
                assert_eq!(inspection.completed_requests, 0);
                assert!(!inspection.ready);
                break;
            }
            sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap();
    let (a, b) = tokio::join!(handle.shutdown(), handle.shutdown());
    a.unwrap();
    b.unwrap();
}

#[tokio::test]
async fn invalid_observation_query_has_a_structured_error() {
    let handle = handle();
    let invalid = router(handle.clone())
        .oneshot(
            Request::builder()
                .uri("/native/v1/events?limit=invalid")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(invalid.into_body(), 1 << 20).await.unwrap();
    assert_eq!(
        serde_json::from_slice::<Error>(&bytes).unwrap().code,
        ErrorCode::InvalidInput
    );
    handle.shutdown().await.unwrap();
}
