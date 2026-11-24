use super::{call, fixture, support};
use axum::http::StatusCode;
use infer_core::ModelId;
use infer_ir::PrecisionPlan;
use infer_kernel_api::KernelRegistry;
use infer_runtime::{Engine, RuntimeConfig};
use serde_json::{Value, json};
use std::env::temp_dir;
use std::f64::consts::E;
use std::fs::{copy, create_dir_all, read, read_dir, write};
use std::path::Path;
use std::process::id;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use support::ProtocolBackend;

use crate::{RuntimeHandle, router_with_text};

/// The fixture vocabulary's tokens 6..=31, renamed to the 26 labels a reply can start with.
const FIRST_LABEL_TOKEN: usize = 6;

/// A copy of the fixture package whose words include the option letters and whose reply prompt
/// ends at a word boundary, with a backend whose logits peak at `peak`.
fn labelled(peak: usize) -> (RuntimeHandle, Arc<infer_models::TextAssets>) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let package = temp_dir().join(format!(
        "infer-decisions-{}-{}",
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
    for (offset, letter) in (b'A'..=b'Z').enumerate() {
        let vocab = tokenizer["model"]["vocab"].as_object_mut().unwrap();
        let token = FIRST_LABEL_TOKEN + offset;
        vocab.remove(&format!("token{token}"));
        vocab.insert(char::from(letter).to_string(), json!(token));
    }
    write(
        package.join("tokenizer.json"),
        serde_json::to_vec(&tokenizer).unwrap(),
    )
    .unwrap();
    let mut config: Value =
        serde_json::from_slice(&read(root.join("tokenizer_config.json")).unwrap()).unwrap();
    config["chat_template"] = json!(
        "{% for message in messages %}{{ message['role'] }} {{ message['content'].strip() }} \
         {% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    );
    write(
        package.join("tokenizer_config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let model = support::model(ModelId::ONE);
    let assets = Arc::new(infer_models::TextAssets::open(&package, model.max_sequence).unwrap());
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels).unwrap();
    let engine = Engine::new(
        ProtocolBackend::peaking_at(peak, 16, 8, &model).unwrap(),
        model,
        PrecisionPlan::f32(),
        &kernels,
        RuntimeConfig::default(),
    )
    .unwrap();
    (RuntimeHandle::start(engine).unwrap(), assets)
}

fn close(value: &Value, expected: f64) -> bool {
    value
        .as_f64()
        .is_some_and(|value| (value - expected).abs() < 1e-5)
}

#[tokio::test]
async fn decisions_read_the_label_the_model_prefers_for_every_question_type() {
    // Peak at "B": the second option of every question.
    let (handle, assets) = labelled(FIRST_LABEL_TOKEN + 1);
    let app = router_with_text(handle.clone(), assets);
    let (status, body) = call(
        app,
        "/v1/decisions",
        Some(json!({
            "model": "1",
            "input": "hello world",
            "questions": [
                {"type": "predicate", "name": "p", "instructions": "hello"},
                {"type": "choice", "name": "c", "instructions": "world", "choices": [
                    {"value": "billing"}, {"value": false}, {"value": "other"},
                ]},
                {"type": "score", "instructions": "hello", "levels": [
                    {"label": "low"}, {"label": "mid"}, {"label": "high"},
                ]},
            ],
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["model"], "1");
    let answers = body["answers"].as_array().unwrap();
    assert_eq!(answers.len(), 3);

    // Two labels, logits 0 and 1 over a vocabulary of 32 whose other entries are 0.
    assert_eq!(answers[0]["type"], "predicate");
    assert_eq!(answers[0]["name"], "p");
    assert!(close(&answers[0]["probability"], E / (1.0 + E)));

    let three = E / (2.0 + E);
    let mass = (2.0 + E) / (31.0 + E);
    assert_eq!(answers[1]["choice"], json!(false));
    assert!(close(&answers[1]["probabilities"][1]["probability"], three));
    assert!(close(&answers[1]["confidence"], three * mass));

    assert_eq!(answers[2]["name"], Value::Null);
    assert!(close(&answers[2]["score"], 1.0));
    assert_eq!(answers[2]["probabilities"][1]["label"], "mid");

    let usage = &body["usage"];
    assert!(usage["input_tokens"].as_u64().unwrap() > 0);
    assert_eq!(usage["output_tokens"], 0);
    assert_eq!(usage["total_tokens"], usage["input_tokens"]);
    assert_eq!(handle.inspect().await.unwrap().active_requests, 0);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn decisions_reject_a_package_whose_labels_are_not_reply_tokens() {
    // The plain fixture has no letter tokens, so no label can be read.
    let (handle, assets) = fixture();
    let app = router_with_text(handle.clone(), assets);
    let (status, body) = call(
        app.clone(),
        "/v1/decisions",
        Some(json!({"model": "1", "input": "hello", "questions": [
            {"type": "predicate", "instructions": "hello"},
        ]})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    assert_eq!(body["error"]["code"], "unsupported_feature");

    let (status, body) = call(
        app,
        "/v1/decisions",
        Some(json!({"model": "other", "input": "hello", "questions": [
            {"type": "predicate", "instructions": "hello"},
        ]})),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert_eq!(handle.inspect().await.unwrap().completed_requests, 0);
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn generation_and_decisions_share_one_deployment() {
    let (handle, assets) = labelled(FIRST_LABEL_TOKEN);
    let app = router_with_text(handle.clone(), assets);
    let (status, body) = call(
        app.clone(),
        "/v1/completions",
        Some(json!({"model": "1", "prompt": "hello world", "max_tokens": 2})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (status, body) = call(
        app,
        "/v1/decisions",
        Some(json!({"model": "1", "input": "hello", "questions": [
            {"type": "choice", "instructions": "", "choices": [{"value": "a"}, {"value": "b"}]},
        ]})),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["answers"][0]["choice"], "a");
    handle.shutdown().await.unwrap();
}
