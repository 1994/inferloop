use super::*;
use std::fs::remove_dir_all;

fn strict_fixture(text: &str) -> (RuntimeHandle, Arc<infer_models::TextAssets>) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let package = temp_dir().join(format!(
        "infer-strict-{}-{}",
        id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    create_dir_all(&package).unwrap();
    let mut vocab = serde_json::Map::new();
    for byte in 32u8..127 {
        let key = if byte == b' ' {
            "Ġ".into()
        } else {
            char::from(byte).to_string()
        };
        vocab.insert(key, json!(byte - 32));
    }
    vocab.insert("<eos>".into(), json!(95));
    let byte_level =
        json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true});
    write(package.join("tokenizer.json"), serde_json::to_vec(&json!({
        "version":"1.0", "truncation":null, "padding":null,
        "added_tokens":[{"id":95,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],
        "normalizer":null,"pre_tokenizer":byte_level,"post_processor":null,"decoder":byte_level,
        "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"vocab":vocab,"merges":[]}
    })).unwrap()).unwrap();
    write(
        package.join("chat_template.jinja"),
        "{# <tool_call> name arguments #}x",
    )
    .unwrap();
    write(
        package.join("generation_config.json"),
        r#"{"eos_token_id":95,"temperature":0.0,"enable_thinking":false}"#,
    )
    .unwrap();
    let mut model = support::model(ModelId::ONE);
    model.vocab_size = 96;
    model.max_sequence = 512;
    let assets = Arc::new(infer_models::TextAssets::open(&package, model.max_sequence).unwrap());
    remove_dir_all(package).unwrap();
    let mut kernels = KernelRegistry::default();
    kernels.register(&support::DeclaredKernels).unwrap();
    let tokens = text
        .bytes()
        .map(|byte| usize::from(byte - 32))
        .chain([95])
        .collect();
    let engine = Engine::new(
        ProtocolBackend::scripted(tokens, &model).unwrap(),
        model,
        PrecisionPlan::f32(),
        &kernels,
        RuntimeConfig::default(),
    )
    .unwrap();
    (RuntimeHandle::start(engine).unwrap(), assets)
}
fn request(stream: bool) -> Value {
    json!({"model":"1","messages":[{"role":"user","content":"x"}],"max_tokens":128,"temperature":0.0,
        "stream":stream,"tool_choice":"required","parallel_tool_calls":false,
        "tools":[{"type":"function","function":{"name":"ping","strict":true,"parameters":{"type":"object","properties":{},"required":[],"additionalProperties":false}}}]})
}
#[tokio::test]
async fn strict_tools_round_trip_through_runtime_and_stream() {
    let text = r#"<tool_call>{"name":"ping","arguments":{}}</tool_call>"#;
    let (handle, assets) = strict_fixture(text);
    let app = router_with_text(handle.clone(), assets);
    // Repeated requests also exercise cached grammars with independent matcher positions.
    for _ in 0..2 {
        let (status, body) = call(app.clone(), "/v1/chat/completions", Some(request(false))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["choices"][0]["finish_reason"], "tool_calls", "{body}");
        assert_eq!(
            body["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{}"
        );
    }
    let (status, body) = call_stream(app, "/v1/chat/completions", request(true)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let events = stream_data(&body);
    assert!(
        events
            .iter()
            .any(|event| event["choices"][0]["finish_reason"] == "tool_calls"),
        "{body}"
    );
    assert_eq!(events.last().unwrap(), "[DONE]");
    let state = handle.inspect().await.unwrap();
    assert_eq!(state.active_requests, 0);
    assert_eq!(state.state.allocated_pages, 0);
    handle.shutdown().await.unwrap();
}
#[tokio::test]
async fn strict_tool_truncation_does_not_claim_a_complete_call() {
    let (handle, assets) = strict_fixture("<tool_call>");
    let mut payload = request(false);
    payload["max_tokens"] = json!(5);
    let (status, body) = call(
        router_with_text(handle.clone(), assets),
        "/v1/chat/completions",
        Some(payload),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["choices"][0]["finish_reason"], "length");
    assert!(body["choices"][0]["message"]["tool_calls"].is_null());
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn unsupported_strict_schema_is_rejected_before_admission() {
    let (handle, assets) = strict_fixture("x");
    let mut payload = request(false);
    payload["tools"][0]["function"]["parameters"]["patternProperties"] = json!({});
    let (status, body) = call(
        router_with_text(handle.clone(), assets),
        "/v1/chat/completions",
        Some(payload),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    assert_eq!(handle.inspect().await.unwrap().active_requests, 0);
    handle.shutdown().await.unwrap();
}

fn responses_request(stream: bool) -> Value {
    json!({"model":"1","input":"x","max_output_tokens":128,"temperature":0.0,
        "stream":stream,"tool_choice":"required","parallel_tool_calls":false,
        "tools":[{"type":"function","name":"ping","strict":true,
            "parameters":{"type":"object","properties":{},"required":[],
                "additionalProperties":false}}]})
}

#[tokio::test]
async fn responses_publish_function_call_items_and_their_round_trip() {
    let text = r#"<tool_call>{"name":"ping","arguments":{}}</tool_call>"#;
    let (handle, assets) = strict_fixture(text);
    let app = router_with_text(handle.clone(), assets);
    let (status, body) = call(app.clone(), "/v1/responses", Some(responses_request(false))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "completed");
    let output = body["output"].as_array().unwrap();
    assert_eq!(output.len(), 1, "{body}");
    assert_eq!(output[0]["type"], "function_call");
    assert_eq!(output[0]["name"], "ping");
    assert_eq!(output[0]["arguments"], "{}");
    assert_eq!(output[0]["status"], "completed");
    let call_id = output[0]["call_id"].as_str().unwrap();
    assert!(call_id.starts_with("call_"));

    let (status, streamed) =
        call_stream(app.clone(), "/v1/responses", responses_request(true)).await;
    assert_eq!(status, StatusCode::OK, "{streamed}");
    let events = stream_data(&streamed);
    let done = events
        .iter()
        .find(|event| event["type"] == "response.function_call_arguments.done")
        .unwrap();
    assert_eq!(done["arguments"], "{}");
    let completed = events.last().unwrap();
    assert_eq!(completed["type"], "response.completed");
    assert!(
        completed["response"]["usage"]["output_tokens"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(completed["response"]["output"][0]["type"], "function_call");

    // A client sends the call item back with its output; the next turn is a normal history.
    let (status, body) = call(
        app,
        "/v1/responses",
        Some(json!({
            "model":"1",
            "input":[
                {"role":"user","content":"x"},
                {"type":"function_call","call_id":call_id,"name":"ping","arguments":"{}"},
                {"type":"function_call_output","call_id":call_id,"output":"pong"}
            ],
            "max_output_tokens":128
        })),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "completed");
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn responses_complete_text_and_stream_end_with_completed() {
    let (handle, assets) = strict_fixture("hello");
    let app = router_with_text(handle.clone(), assets);
    let payload = json!({"model":"1","input":"x","max_output_tokens":16});
    let (status, body) = call(app.clone(), "/v1/responses", Some(payload.clone())).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["status"], "completed");
    assert!(body["completed_at"].as_u64().unwrap() > 0);
    assert_eq!(body["incomplete_details"], Value::Null);
    assert_eq!(body["output"][0]["content"][0]["text"], "hello");

    let mut streamed = payload;
    streamed["stream"] = json!(true);
    let (status, body) = call_stream(app, "/v1/responses", streamed).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let events = stream_data(&body);
    assert_eq!(events.last().unwrap()["type"], "response.completed");
    let text: String = events
        .iter()
        .filter(|event| event["type"] == "response.output_text.delta")
        .map(|event| event["delta"].as_str().unwrap())
        .collect();
    assert_eq!(text, "hello");
    assert_eq!(events.last().unwrap()["response"]["status"], "completed");
    handle.shutdown().await.unwrap();
}

#[tokio::test]
async fn responses_map_reasoning_text_to_a_reasoning_item() {
    let (handle, assets) = strict_fixture("<think>why</think>hello");
    let app = router_with_text(handle.clone(), assets);
    let payload = json!({"model":"1","input":"x","max_output_tokens":64,
        "reasoning":{"effort":"low"}});
    let (status, body) = call(app.clone(), "/v1/responses", Some(payload.clone())).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let output = body["output"].as_array().unwrap();
    assert_eq!(output.len(), 2, "{body}");
    assert_eq!(output[0]["type"], "reasoning");
    assert_eq!(output[0]["content"][0]["type"], "reasoning_text");
    assert_eq!(output[0]["content"][0]["text"], "why");
    assert_eq!(output[1]["type"], "message");
    assert_eq!(output[1]["content"][0]["text"], "hello");
    assert_eq!(body["status"], "completed");
    assert_eq!(body["reasoning"]["effort"], "low");

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
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.output_text.delta",
        "response.completed",
    ] {
        assert!(types.contains(&expected), "{expected}: {types:?}");
    }
    let reasoning: String = events
        .iter()
        .filter(|event| event["type"] == "response.reasoning_text.delta")
        .map(|event| event["delta"].as_str().unwrap())
        .collect();
    assert_eq!(reasoning, "why");
    let completed = &events.last().unwrap()["response"];
    assert_eq!(completed["output"][0]["type"], "reasoning");
    assert_eq!(completed["output"][1]["type"], "message");
    handle.shutdown().await.unwrap();
}
