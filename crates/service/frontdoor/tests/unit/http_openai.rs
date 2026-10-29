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
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::{sync::Arc, time::Duration};
use tokio::time::sleep;
use tokio::time::timeout;
use tower::ServiceExt;

use crate::cpu::CpuConfig;
use crate::{RuntimeHandle, router_with_text};

// The native HTTP module already mounts the shared protocol double; this module is a sibling
// test and reuses it rather than loading the same file into the crate a second time.
use super::super::tests::support;
use support::ProtocolBackend;

fn tiny_media_app(config: CpuConfig) -> (Router, RuntimeHandle, Arc<AtomicUsize>) {
    use crate::media::MediaService;
    use crate::router_with_text_and_media;
    use infer_models::MediaLimits;

    let package = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/multimodal-tiny");
    let root = package.as_path();
    let multimodal = infer_models::MultimodalAssets::open(root).unwrap().unwrap();
    let assets = Arc::new(infer_models::TextAssets::open(root, 4096).unwrap());
    let mut model = support::model(ModelId::ONE);
    model.max_sequence = 4096;
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels).unwrap();
    let backend = ProtocolBackend::peaking_at(3, 16, 8, &model)
        .unwrap()
        .multimodal();
    let media_tasks = backend.media_tasks();
    let engine = Engine::new(
        backend,
        model,
        PrecisionPlan::f32(),
        &kernels,
        RuntimeConfig::default(),
    )
    .unwrap();
    let handle = RuntimeHandle::start_with_config(engine, config).unwrap();
    let media = Arc::new(MediaService::new(multimodal, MediaLimits::default()));
    let app = router_with_text_and_media(handle.clone(), assets, Some(media));
    (app, handle, media_tasks)
}

#[tokio::test]
async fn concurrent_large_images_share_staging_budget_without_all_failing() {
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD;
    use std::io::Cursor;

    let (app, handle, media_tasks) = tiny_media_app(CpuConfig {
        workers: 2,
        max_jobs: 4,
        max_bytes: 64 << 20,
        ..CpuConfig::default()
    });
    let image = image::RgbImage::from_pixel(1024, 1024, image::Rgb([24, 64, 128]));
    let mut encoded = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut encoded, image::ImageFormat::Png)
        .unwrap();
    let url = format!(
        "data:image/png;base64,{}",
        STANDARD.encode(encoded.into_inner())
    );
    let payload = json!({"model":"1", "max_tokens":1, "messages":[{"role":"user","content":[
        {"type":"text", "text":"Describe the image."},
        {"type":"image_url", "image_url":{"url":url}}
    ]}]});
    let make_request = || {
        Request::builder()
            .method("POST")
            .uri("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(payload.to_string()))
            .unwrap()
    };
    let result = timeout(Duration::from_secs(30), async {
        tokio::join!(
            app.clone().oneshot(make_request()),
            app.oneshot(make_request())
        )
    })
    .await
    .expect("both requests completed");
    for response in [result.0.unwrap(), result.1.unwrap()] {
        let status = response.status();
        let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    }
    assert!(media_tasks.load(Ordering::Relaxed) >= 2);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn inline_image_is_priced_before_its_url_moves_into_preparation() {
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD;

    let (app, handle, media_tasks) = tiny_media_app(CpuConfig {
        workers: 1,
        max_jobs: 2,
        max_bytes: 28 << 20,
        ..CpuConfig::default()
    });
    let bytes = vec![0_u8; 256 * 1024];
    let url = format!("data:image/png;base64,{}", STANDARD.encode(bytes));
    let payload = json!({"model":"1", "max_tokens":1, "messages":[{"role":"user","content":[
        {"type":"image_url", "image_url":{"url":url}}
    ]}]});
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    let body = to_bytes(response.into_body(), 1 << 20).await.unwrap();
    let response: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(response["error"]["code"], "capacity_exceeded");
    assert_eq!(media_tasks.load(Ordering::Relaxed), 0);
    handle.shutdown().await.unwrap();
}

fn fixture() -> (RuntimeHandle, Arc<infer_models::TextAssets>) {
    fixture_with(RuntimeConfig::default())
}

fn fixture_with(config: RuntimeConfig) -> (RuntimeHandle, Arc<infer_models::TextAssets>) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let model = support::model(ModelId::ONE);
    let assets = Arc::new(infer_models::TextAssets::open(root, model.max_sequence).unwrap());
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels).unwrap();
    let engine = Engine::new(
        ProtocolBackend::peaking_at(3, 16, 8, &model).unwrap(),
        model,
        PrecisionPlan::f32(),
        &kernels,
        config,
    )
    .unwrap();
    (RuntimeHandle::start(engine).unwrap(), assets)
}

/// Copy the fixture package with a byte-level tokenizer, which is what the served packages use
/// and the only decoder the streaming path accepts.
fn streaming_fixture() -> (RuntimeHandle, Arc<infer_models::TextAssets>) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let package = temp_dir().join(format!(
        "infer-stream-{}-{}",
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
        ProtocolBackend::peaking_at(3, 16, 8, &model).unwrap(),
        model,
        PrecisionPlan::f32(),
        &kernels,
        RuntimeConfig::default(),
    )
    .unwrap();
    (RuntimeHandle::start(engine).unwrap(), assets)
}

/// Read one SSE body into `(event_data, ...)` strings.
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
        .map(|data| {
            if data.trim() == "[DONE]" {
                Value::String("[DONE]".into())
            } else {
                serde_json::from_str(data).unwrap()
            }
        })
        .collect()
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

#[tokio::test]
async fn completions_follow_the_served_model_and_report_usage() {
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    let (status, models) = call(app.clone(), "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(models["data"][0]["id"], "1");
    let (status, result) = call(
        app,
        "/v1/completions",
        Some(json!({
            "model":"1", "prompt":"hello world system assistant token8 token13", "max_tokens":5
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{result}");
    assert_eq!(result["object"], "text_completion");
    // The declared peak is token 3, which this fixture's tokenizer decodes as "system", and the
    // budget is five tokens. The earlier "token25 ..." was the host executor's arithmetic; the
    // model's own goldens are checked where the fixture's text is registered.
    assert_eq!(
        result["choices"][0]["text"],
        "system system system system system"
    );
    assert_eq!(result["choices"][0]["finish_reason"], "length");
    assert_eq!(
        result["usage"],
        json!({"prompt_tokens":6,"completion_tokens":5,"total_tokens":11})
    );
    let inspection = handle.inspect().await.unwrap();
    assert_eq!(inspection.active_requests, 0);
    assert_eq!(inspection.state.allocated_pages, 0);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn chat_matches_native_template_and_ids_survive_router_rebuilds() {
    let (handle, assets) = fixture();
    let messages = json!([{"role":"user","content":"hello world"}]);
    let native = json!({"id":100,"model":1,"messages":messages,
        "workload":{"Generate":{"max_new_tokens":3}}});
    let app = router_with_text(handle.clone(), assets.clone());
    let (status, expected) = call(app.clone(), "/native/v1/text", Some(native)).await;
    assert_eq!(status, StatusCode::OK);
    let payload = json!({"model":"1","messages":messages,"max_completion_tokens":3});
    let rebuilt = router_with_text(handle.clone(), assets);
    let (a, b) = tokio::join!(
        call(app, "/v1/chat/completions", Some(payload.clone())),
        call(rebuilt, "/v1/chat/completions", Some(payload))
    );
    assert_eq!(a.0, StatusCode::OK, "{}", a.1);
    assert_eq!(b.0, StatusCode::OK, "{}", b.1);
    assert_ne!(a.1["id"], b.1["id"]);
    for result in [a.1, b.1] {
        assert_eq!(result["choices"][0]["message"]["content"], expected["text"]);
        assert_eq!(result["choices"][0]["message"]["role"], "assistant");
        assert_eq!(result["object"], "chat.completion");
        assert_eq!(result["usage"]["prompt_tokens"], 4);
        assert!(result["created"].as_u64().unwrap() > 0);
    }
    assert_eq!(handle.allocate_request_id().unwrap().get(), 103);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn invalid_and_unsupported_requests_return_errors_without_admission() {
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    let payload =
        json!({"model":"1","messages":[{"role":"user","content":"hello"}],"max_tokens":2});
    let cases = [
        ("model", json!("missing"), StatusCode::NOT_FOUND),
        ("stream", json!(true), StatusCode::NOT_IMPLEMENTED),
        ("n", json!(2), StatusCode::NOT_IMPLEMENTED),
        ("top_p", json!(0.0), StatusCode::BAD_REQUEST),
        ("n", json!(0), StatusCode::BAD_REQUEST),
        ("max_tokens", json!(0), StatusCode::BAD_REQUEST),
        ("temperature", json!(-1), StatusCode::BAD_REQUEST),
        ("temperature", json!(3), StatusCode::BAD_REQUEST),
        ("messages", json!([]), StatusCode::BAD_REQUEST),
        (
            "messages",
            json!([{"role":"tool","content":"hello"}]),
            StatusCode::BAD_REQUEST,
        ),
        ("prompt", json!("hello"), StatusCode::BAD_REQUEST),
        // An empty declaration is a valid request with no tools; a malformed one is not.
        ("tools", json!("not-a-list"), StatusCode::BAD_REQUEST),
        (
            "tools",
            json!([{"type": "function"}]),
            StatusCode::BAD_REQUEST,
        ),
        ("tool_choice", json!("required"), StatusCode::BAD_REQUEST),
        ("max_completion_tokens", json!(3), StatusCode::BAD_REQUEST),
    ];
    for (field, value, expected) in cases {
        let mut invalid = payload.clone();
        invalid[field] = value;
        let (status, body) = call(app.clone(), "/v1/chat/completions", Some(invalid)).await;
        assert_eq!(status, expected, "{field}: {body}");
        assert!(body["error"]["message"].is_string());
        assert!(body["error"]["type"].is_string());
        assert!(body["error"]["code"].is_string());
    }
    assert_eq!(handle.inspect().await.unwrap().active_requests, 0);
    assert_eq!(handle.allocate_request_id().unwrap().get(), 1);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn an_explicit_output_budget_over_the_service_cap_is_a_parameter_error() {
    let (handle, assets) = fixture_with(RuntimeConfig {
        max_model_len: Some(8),
        max_output_tokens: Some(3),
        ..Default::default()
    });
    let inspection = handle.inspect().await.unwrap();
    assert_eq!(inspection.lengths.total, 8);
    assert_eq!(inspection.lengths.output_cap, 3);
    let app = router_with_text(handle.clone(), assets);
    let payload = json!({
        "model":"1", "messages":[{"role":"user","content":"hello"}], "max_completion_tokens":4
    });
    let (status, body) = call(app, "/v1/chat/completions", Some(payload)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("output cap"),
        "{body}"
    );
    // The rejected request never consumed a request identity.
    assert_eq!(handle.allocate_request_id().unwrap().get(), 1);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn prompt_plus_output_over_the_total_context_is_rejected_after_tokenization() {
    let (handle, assets) = fixture_with(RuntimeConfig {
        max_model_len: Some(8),
        ..Default::default()
    });
    let app = router_with_text(handle.clone(), assets);
    // Six encoded prompt tokens plus a three-token budget exceeds the eight-token total.
    let payload = json!({
        "model":"1",
        "prompt":"hello world system assistant token8 token13",
        "max_tokens":3
    });
    let (status, body) = call(app.clone(), "/v1/completions", Some(payload.clone())).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("total context"),
        "{body}"
    );
    // The same prompt fits when the budget leaves room, so the check is not a blanket rejection.
    let mut fitting = payload;
    fitting["max_tokens"] = json!(2);
    let (status, body) = call(app, "/v1/completions", Some(fitting)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn malformed_and_oversized_bodies_use_the_compatibility_error_envelope() {
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    for (body, expected) in [
        ("{".into(), StatusCode::BAD_REQUEST),
        (
            " ".repeat(2 * 1024 * 1024 + 1),
            StatusCode::PAYLOAD_TOO_LARGE,
        ),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), expected);
        let bytes = to_bytes(response.into_body(), 1 << 20).await.unwrap();
        let error: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(error["error"]["type"], "invalid_request_error");
    }
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn the_served_name_addresses_the_deployment() {
    let (handle, assets) = fixture_with(RuntimeConfig {
        served_model_name: Some("qwen3-8-27b".into()),
        ..Default::default()
    });
    let app = router_with_text(handle.clone(), assets);
    let (status, models) = call(app.clone(), "/v1/models", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(models["data"][0]["id"], "qwen3-8-27b");
    // The internal identity is not addressable once a deployment name is configured.
    let prompt = "hello world system assistant token8 token13";
    let (status, _) = call(
        app.clone(),
        "/v1/completions",
        Some(json!({"model":"1", "prompt":prompt, "max_tokens":2})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, body) = call(
        app,
        "/v1/completions",
        Some(json!({"model":"qwen3-8-27b", "prompt":prompt, "max_tokens":2})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn streaming_completions_agree_with_the_complete_reply() {
    let (handle, assets) = streaming_fixture();
    let app = router_with_text(handle.clone(), assets);
    let payload = json!({
        "model":"1", "prompt":"hello world system assistant token8 token13", "max_tokens":5
    });
    let (status, complete) = call(app.clone(), "/v1/completions", Some(payload.clone())).await;
    assert_eq!(status, StatusCode::OK, "{complete}");
    let mut streamed = payload;
    streamed["stream"] = json!(true);
    let (status, body) = call_stream(app, "/v1/completions", streamed).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let chunks = stream_data(&body);
    assert_eq!(
        chunks.last(),
        Some(&Value::String("[DONE]".into())),
        "{chunks:?}"
    );
    let text: String = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["text"].as_str())
        .collect();
    assert_eq!(text, complete["choices"][0]["text"]);
    assert_eq!(chunks[0]["object"], "text_completion");
    // The finish reason is published once, immediately before the terminator.
    assert_eq!(
        chunks[chunks.len() - 2]["choices"][0]["finish_reason"],
        complete["choices"][0]["finish_reason"]
    );
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn streaming_chat_publishes_role_content_and_finish_in_order() {
    let (handle, assets) = streaming_fixture();
    let app = router_with_text(handle.clone(), assets);
    let payload = json!({
        "model":"1", "messages":[{"role":"user","content":"hello"}],
        "max_completion_tokens":4, "stream":true
    });
    let (status, body) = call_stream(app, "/v1/chat/completions", payload).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let chunks = stream_data(&body);
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    assert_eq!(chunks[0]["object"], "chat.completion.chunk");
    let content: String = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert!(!content.is_empty(), "{chunks:?}");
    assert_eq!(
        chunks[chunks.len() - 2]["choices"][0]["finish_reason"],
        "length"
    );
    // No content delta may arrive after the finish reason.
    let finish = chunks.len() - 2;
    assert!(
        chunks[..finish]
            .iter()
            .all(|chunk| chunk["choices"][0]["finish_reason"].is_null())
    );
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_streamed_reply_reports_usage_only_when_asked() {
    let (handle, assets) = streaming_fixture();
    let app = router_with_text(handle.clone(), assets);
    let base = json!({
        "model":"1", "messages":[{"role":"user","content":"hello"}],
        "max_completion_tokens":2, "stream":true
    });
    let (_, plain) = call_stream(app.clone(), "/v1/chat/completions", base.clone()).await;
    assert!(
        stream_data(&plain)
            .iter()
            .all(|chunk| chunk["usage"].is_null()),
        "{plain}"
    );
    let mut asked = base;
    asked["stream_options"] = json!({"include_usage": true});
    let (_, with_usage) = call_stream(app, "/v1/chat/completions", asked).await;
    let chunks = stream_data(&with_usage);
    let usage = &chunks[chunks.len() - 2];
    assert_eq!(
        usage["choices"].as_array().unwrap().len(),
        0,
        "{with_usage}"
    );
    // The prompt was tokenized to four tokens by the byte-level fixture.
    assert!(usage["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(usage["usage"]["completion_tokens"], 2);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_package_without_a_streamable_decoder_refuses_streaming() {
    // The plain fixture joins tokens with a separator, so a partial decode would not match.
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    let payload = json!({
        "model":"1", "messages":[{"role":"user","content":"hello"}],
        "max_completion_tokens":2, "stream":true
    });
    let (status, body) = call(app, "/v1/chat/completions", Some(payload)).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn stream_options_without_streaming_are_rejected() {
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    let payload = json!({
        "model":"1", "messages":[{"role":"user","content":"hello"}],
        "max_completion_tokens":2, "stream_options":{"include_usage":true}
    });
    let (status, body) = call(app, "/v1/chat/completions", Some(payload)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_long_chat_history_is_rejected_by_the_prompt_budget() {
    let (handle, assets) = fixture_with(RuntimeConfig {
        // A single short message fits; a many-turn history does not.
        max_model_len: Some(24),
        ..Default::default()
    });
    let app = router_with_text(handle.clone(), assets);
    let short = json!({
        "model":"1", "messages":[{"role":"user","content":"hello world"}], "max_tokens":2
    });
    let (status, body) = call(app.clone(), "/v1/chat/completions", Some(short)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // Tool definitions and a full round both add to the encoded prompt, so extra turns consume
    // the same budget a longer prompt would.
    let mut messages = vec![json!({"role":"user","content":"hello world system"})];
    for _ in 0..8 {
        messages.push(json!({"role":"assistant","content":"token8 token13 hello world"}));
        messages.push(json!({"role":"user","content":"token8 token13 hello world"}));
    }
    let long = json!({"model":"1", "messages":messages, "max_tokens":2});
    let (status, body) = call(app, "/v1/chat/completions", Some(long)).await;
    // An oversized prompt is a capacity rejection, not a silent truncation.
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "{body}");
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn a_disconnected_stream_is_cancelled_and_freed() {
    let (handle, assets) = streaming_fixture();
    let app = router_with_text(handle.clone(), assets);
    let payload = json!({
        "model":"1", "messages":[{"role":"user","content":"hello"}],
        "max_completion_tokens":8, "stream":true
    });
    let request = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(payload.to_string()))
        .unwrap();
    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    // Dropping the body closes the stream while the engine is still generating.
    drop(response);
    timeout(Duration::from_secs(10), async {
        loop {
            if handle.inspect().await.unwrap().active_requests == 0 {
                break;
            }
            sleep(Duration::from_millis(5)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(handle.inspect().await.unwrap().state.allocated_pages, 0);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn tool_history_without_new_tools_returns_a_normal_completion() {
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    // The client declares a tool, returns the result of the call it received, and the model
    // continues in the same conversation.
    let payload = json!({
        "model":"1",
        "messages":[
            {"role":"user","content":"what is the weather?"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"call_1","type":"function",
                 "function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}
            ]},
            {"role":"tool","content":"{\"temp_c\":21}","tool_call_id":"call_1"}
        ],
        "max_completion_tokens":3
    });
    let (status, body) = call(app, "/v1/chat/completions", Some(payload)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(
        body["choices"][0]["message"]["content"].is_string(),
        "{body}"
    );
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn auto_tools_require_a_package_bound_output_dialect() {
    let (handle, assets) = fixture();
    assert!(!assets.supports_tool_calls());
    let app = router_with_text(handle.clone(), assets);
    let payload = json!({"model":"fixture", "messages":[{"role":"user","content":"hi"}],
        "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}]});
    let model = handle.inspect().await.unwrap().model_name;
    let mut payload = payload;
    payload["model"] = json!(model);
    let (status, _) = call(app, "/v1/chat/completions", Some(payload)).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn retrieve_model_matches_list_and_rejects_unknown_names() {
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    let (_, list) = call(app.clone(), "/v1/models", None).await;
    let (status, model) = call(app.clone(), "/v1/models/1", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(model, list["data"][0]);
    let (status, error) = call(app, "/v1/models/missing", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(error["error"]["code"], "model_not_found");
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn completion_stream_uses_sdk_choice_schema_and_reports_usage() {
    let (handle, assets) = streaming_fixture();
    let app = router_with_text(handle.clone(), assets);
    let payload = json!({"model":"1", "prompt":"hello", "max_tokens":2,
        "stream":true, "stream_options":{"include_usage":true}});
    let (status, body) = call_stream(app, "/v1/completions", payload).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let chunks = stream_data(&body);
    for chunk in &chunks[..chunks.len() - 2] {
        assert!(chunk["choices"][0]["text"].is_string(), "{chunk}");
        assert!(chunk["choices"][0].get("delta").is_none());
        assert!(chunk.get("usage").unwrap().is_null());
    }
    let usage = &chunks[chunks.len() - 2];
    assert_eq!(usage["choices"], json!([]));
    assert_eq!(usage["usage"]["completion_tokens"], 2);
    assert_eq!(chunks.last().unwrap(), "[DONE]");
    handle.shutdown().await.unwrap();
}

#[path = "http_strict_tools.rs"]
mod strict_tools;

#[tokio::test]
async fn image_parts_are_rejected_on_a_text_only_deployment() {
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    let (status, body) = call(
        app,
        "/v1/chat/completions",
        Some(json!({
            "model":"1",
            "messages":[{"role":"user","content":[
                {"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}
            ]}],
            "max_tokens":2
        })),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    assert_eq!(body["error"]["code"], "unsupported_feature");
    assert_eq!(handle.inspect().await.unwrap().active_requests, 0);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn public_chat_rejects_server_local_media_paths() {
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    for url in ["file:///etc/passwd", "/etc/passwd"] {
        let (status, body) = call(
            app.clone(),
            "/v1/chat/completions",
            Some(json!({
                "model":"1",
                "messages":[{"role":"user","content":[
                    {"type":"image_url","image_url":{"url":url}}
                ]}],
                "max_tokens":2
            })),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{url}: {body}");
        assert_eq!(body["error"]["code"], "unsupported_feature");
    }
    assert_eq!(handle.inspect().await.unwrap().active_requests, 0);
    handle.shutdown().await.unwrap();
}

#[path = "http_openai_decisions.rs"]
mod decisions;
