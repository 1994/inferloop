use axum::body::to_bytes;
use axum::{Router, body::Body, http::Request, http::StatusCode};
use infer_core::ModelId;
use infer_ir::PrecisionPlan;
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, RuntimeConfig};
use serde_json::{Value, json};
use std::env::temp_dir;
use std::fs::copy;
use std::fs::create_dir_all;
use std::fs::read;
use std::fs::read_dir;
use std::fs::write;
use std::path::Path;
use std::process::id;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use tokio::time::timeout;
use tower::ServiceExt;

use crate::http::tests::support;
use crate::{RuntimeHandle, router_with_text};

fn fixture_with(config: RuntimeConfig) -> (RuntimeHandle, Arc<infer_models::TextAssets>) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let model = support::model(ModelId::ONE);
    let assets = Arc::new(infer_models::TextAssets::open(root, model.max_sequence).unwrap());
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels).unwrap();
    let engine = Engine::new(
        support::ProtocolBackend::peaking_at(3, 16, 8, &model).unwrap(),
        model,
        PrecisionPlan::f32(),
        &kernels,
        config,
    )
    .unwrap();
    (RuntimeHandle::start(engine).unwrap(), assets)
}

fn fixture() -> (RuntimeHandle, Arc<infer_models::TextAssets>) {
    fixture_with(RuntimeConfig::default())
}

/// Copy the fixture package with a byte-level decoder so the streamed route is available.
fn streaming_fixture() -> (RuntimeHandle, Arc<infer_models::TextAssets>) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let package = temp_dir().join(format!(
        "infer-responses-stream-{}-{}",
        id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    create_dir_all(&package).unwrap();
    for entry in read_dir(&root).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file() {
            copy(entry.path(), package.join(entry.file_name())).unwrap();
        }
    }
    let mut tokenizer: Value =
        serde_json::from_slice(&read(root.join("tokenizer.json")).unwrap()).unwrap();
    let byte_level = json!({
        "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true
    });
    tokenizer["decoder"] = byte_level.clone();
    tokenizer["pre_tokenizer"] = byte_level;
    write(
        package.join("tokenizer.json"),
        serde_json::to_vec(&tokenizer).unwrap(),
    )
    .unwrap();
    let model = support::model(ModelId::ONE);
    let assets = Arc::new(infer_models::TextAssets::open(&package, model.max_sequence).unwrap());
    assert!(assets.supports_streaming_decode());
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels).unwrap();
    let engine = Engine::new(
        support::ProtocolBackend::peaking_at(3, 16, 8, &model).unwrap(),
        model,
        PrecisionPlan::f32(),
        &kernels,
        RuntimeConfig::default(),
    )
    .unwrap();
    (RuntimeHandle::start(engine).unwrap(), assets)
}

async fn call(app: Router, path: &str, payload: Option<Value>) -> (StatusCode, Value) {
    let request = Request::builder().uri(path);
    let request = if let Some(payload) = payload {
        request
            .method("POST")
            .header("content-type", "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap()
    } else {
        request.body(Body::empty()).unwrap()
    };
    let response = timeout(Duration::from_secs(10), app.oneshot(request))
        .await
        .unwrap()
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

async fn call_stream(app: Router, path: &str, payload: Value) -> (StatusCode, String) {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap();
    let response = timeout(Duration::from_secs(20), app.oneshot(request))
        .await
        .unwrap()
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// The `data:` payloads of an event stream.
fn stream_data(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

#[tokio::test]
async fn responses_reports_typed_items_and_usage() {
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    let (status, body) = call(
        app,
        "/v1/responses",
        Some(json!({
            "model":"1", "input":"hello world", "instructions":"be brief",
            "max_output_tokens":3, "metadata":{"trace":"abc"}
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["object"], "response");
    // The fixture's backend always selects the same token, so a three-token budget stops at
    // the length cap rather than at an end-of-sequence token.
    assert_eq!(body["status"], "incomplete");
    assert_eq!(body["completed_at"], Value::Null);
    assert_eq!(
        body["incomplete_details"],
        json!({"reason":"max_output_tokens"})
    );
    assert_eq!(body["instructions"], "be brief");
    assert_eq!(body["metadata"]["trace"], "abc");
    assert_eq!(body["store"], false);
    assert_eq!(body["tool_choice"], "auto");
    assert!(body["id"].as_str().unwrap().starts_with("resp_"));
    let output = body["output"].as_array().unwrap();
    assert_eq!(output.len(), 1, "{body}");
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["role"], "assistant");
    assert_eq!(output[0]["status"], "incomplete");
    assert_eq!(output[0]["content"][0]["type"], "output_text");
    assert_eq!(output[0]["content"][0]["text"], "system system system");
    let input = body["usage"]["input_tokens"].as_u64().unwrap();
    assert!(input > 0);
    assert_eq!(body["usage"]["output_tokens"], 3);
    assert_eq!(body["usage"]["total_tokens"], input + 3);
    assert_eq!(
        body["usage"]["output_tokens_details"]["reasoning_tokens"],
        0
    );
    assert_eq!(handle.inspect().await.unwrap().active_requests, 0);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn responses_accepts_message_items_and_a_length_stop() {
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    let (status, body) = call(
        app,
        "/v1/responses",
        Some(json!({
            "model":"1",
            "input":[
                {"role":"system","content":"rules"},
                {"role":"user","content":"hello world"}
            ],
            "max_output_tokens":2
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "incomplete");
    assert_eq!(
        body["incomplete_details"],
        json!({"reason":"max_output_tokens"})
    );
    assert_eq!(body["completed_at"], Value::Null);
    assert_eq!(body["output"][0]["content"][0]["text"], "system system");
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn responses_streams_typed_events_that_agree_with_the_complete_reply() {
    let (handle, assets) = streaming_fixture();
    let app = router_with_text(handle.clone(), assets);
    let payload = json!({"model":"1", "input":"hello", "max_output_tokens":3});
    let (status, complete) = call(app.clone(), "/v1/responses", Some(payload.clone())).await;
    assert_eq!(status, StatusCode::OK, "{complete}");
    let mut streamed = payload;
    streamed["stream"] = json!(true);
    let (status, body) = call_stream(app, "/v1/responses", streamed).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let events = stream_data(&body);
    let types: Vec<_> = events
        .iter()
        .map(|event| event["type"].as_str().unwrap())
        .collect();
    for expected in [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.incomplete",
    ] {
        assert!(types.contains(&expected), "{expected}: {types:?}");
    }
    assert_eq!(events.last().unwrap()["type"], "response.incomplete");
    assert!(!body.contains("[DONE]"));
    assert!(body.contains("event: response.created\n"), "{body}");
    assert!(
        body.contains("event: response.output_text.delta\n"),
        "{body}"
    );
    let text: String = events
        .iter()
        .filter_map(|event| {
            (event["type"] == "response.output_text.delta")
                .then(|| event["delta"].as_str().unwrap())
        })
        .collect();
    assert_eq!(text, complete["output"][0]["content"][0]["text"]);
    let item_id = events
        .iter()
        .find(|event| event["type"] == "response.output_item.added")
        .unwrap()["item"]["id"]
        .clone();
    assert_eq!(events[2]["type"], "response.output_item.added");
    assert_eq!(events[3]["item_id"], item_id);
    let completed = &events[events.len() - 1]["response"];
    assert_eq!(completed["status"], "incomplete");
    assert_eq!(completed["usage"]["output_tokens"], 3);
    let sequence: Vec<u64> = events
        .iter()
        .map(|event| event["sequence_number"].as_u64().unwrap())
        .collect();
    assert!(
        sequence.windows(2).all(|pair| pair[0] < pair[1]),
        "{sequence:?}"
    );
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn responses_refuses_unsupported_options_before_admission() {
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    for (option, expected) in [
        (json!({"store":true}), StatusCode::NOT_IMPLEMENTED),
        (
            json!({"previous_response_id":"resp_1"}),
            StatusCode::NOT_IMPLEMENTED,
        ),
        (
            json!({"tools":[{"type":"web_search"}]}),
            StatusCode::NOT_IMPLEMENTED,
        ),
        (
            json!({"text":{"format":{"type":"json_schema","name":"x","schema":{}}}}),
            StatusCode::NOT_IMPLEMENTED,
        ),
        (
            json!({"include":["message.output_text.logprobs"]}),
            StatusCode::NOT_IMPLEMENTED,
        ),
        (
            json!({"input":[{"role":"tool","content":"x"}]}),
            StatusCode::BAD_REQUEST,
        ),
        (
            json!({"input":"hi","unknown":true}),
            StatusCode::BAD_REQUEST,
        ),
    ] {
        let mut payload = json!({"model":"1", "input":"hello"});
        for (key, value) in option.as_object().unwrap() {
            payload[key] = value.clone();
        }
        let (status, body) = call(app.clone(), "/v1/responses", Some(payload)).await;
        assert_eq!(status, expected, "{body}");
        assert!(body["error"]["message"].is_string(), "{body}");
    }
    let (status, body) = call(
        app,
        "/v1/responses",
        Some(json!({"model":"missing", "input":"hello"})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(body["error"]["code"], "model_not_found");
    assert_eq!(handle.inspect().await.unwrap().active_requests, 0);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn responses_reject_a_package_without_a_stable_stream_decode() {
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    let (status, body) = call(
        app,
        "/v1/responses",
        Some(json!({"model":"1", "input":"hello", "stream":true})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn responses_cleanup_after_a_disconnected_stream() {
    let (handle, assets) = streaming_fixture();
    let app = router_with_text(handle.clone(), assets);
    let payload = json!({"model":"1", "input":"hello", "max_output_tokens":8, "stream":true});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/responses")
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    drop(response);
    timeout(Duration::from_secs(10), async {
        loop {
            if handle.inspect().await.unwrap().active_requests == 0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(handle.inspect().await.unwrap().state.allocated_pages, 0);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn responses_media_is_refused_on_a_text_only_deployment() {
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    let (status, body) = call(
        app,
        "/v1/responses",
        Some(json!({
            "model":"1",
            "input":[{"role":"user","content":[
                {"type":"input_image","image_url":"data:image/png;base64,AAAA"}
            ]}],
            "max_output_tokens":1
        })),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    assert_eq!(body["error"]["message"], "this deployment is text-only");
    handle.shutdown().await.unwrap();
}
