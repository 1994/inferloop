use super::*;
use serde_json::json;

fn parse(value: Value) -> ResponsesRequest {
    serde_json::from_value(value).expect("fixture request must deserialize")
}

fn prepared(value: Value) -> Prepared {
    parse(value)
        .prepare(StoreOptions::default(), None)
        .expect("fixture request must prepare")
}

#[test]
fn input_string_and_instructions_become_messages() {
    let mut prepared = prepared(json!({
        "model":"1", "input":"hello", "instructions":"be brief"
    }));
    let (messages, media) = prepared.generation.chat_messages().unwrap();
    assert_eq!(media.len(), 0);
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "system");
    assert_eq!(messages[0].content, "be brief");
    assert_eq!(messages[1].role, "user");
    assert_eq!(messages[1].content, "hello");
}

#[test]
fn an_item_list_accepts_text_and_developer_messages() {
    let mut prepared = prepared(json!({
        "model":"1",
        "input":[
            {"role":"developer","content":"rules"},
            {"type":"message","role":"user","content":[
                {"type":"input_text","text":"hello "},
                {"type":"input_text","text":"world"}
            ]}
        ]
    }));
    let (messages, media) = prepared.generation.chat_messages().unwrap();
    assert_eq!(media.len(), 0);
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "system");
    assert_eq!(messages[0].content, "rules");
    assert_eq!(messages[1].content, "hello world");
}

#[test]
fn function_call_items_round_trip_into_tool_history() {
    let mut prepared = prepared(json!({
        "model":"1",
        "input":[
            {"role":"user","content":"weather?"},
            {"type":"function_call","call_id":"call_1","name":"get_weather","arguments":"{\"city\":\"Paris\"}"},
            {"type":"function_call_output","call_id":"call_1","output":"{\"temp_c\":21}"}
        ]
    }));
    let (messages, _) = prepared.generation.chat_messages().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1].role, "assistant");
    assert_eq!(messages[1].tool_calls.len(), 1);
    assert_eq!(messages[1].tool_calls[0].id, "call_1");
    assert_eq!(messages[1].tool_calls[0].name, "get_weather");
    assert_eq!(messages[1].tool_calls[0].arguments, "{\"city\":\"Paris\"}");
    assert_eq!(messages[2].role, "tool");
    assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_1"));
    assert_eq!(messages[2].content, "{\"temp_c\":21}");
}

#[test]
fn output_message_items_round_trip_with_their_content_fields() {
    let mut prepared = prepared(json!({
        "model":"1",
        "input":[
            {"role":"user","content":"hello"},
            {"type":"message","id":"msg_1_1","status":"completed","role":"assistant",
             "content":[{"type":"output_text","text":"answer","annotations":[],"logprobs":[]}]},
            {"role":"user","content":"again"}
        ]
    }));
    let (messages, _) = prepared.generation.chat_messages().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[1].role, "assistant");
    assert_eq!(messages[1].content, "answer");
}

#[test]
fn a_reasoning_item_is_accepted_for_a_full_output_round_trip() {
    let mut prepared = prepared(json!({
        "model":"1",
        "input":[
            {"role":"user","content":"hello"},
            {"type":"reasoning","id":"rs_1","summary":[],"content":[
                {"type":"reasoning_text","text":"thinking"}
            ]}
        ]
    }));
    let (messages, _) = prepared.generation.chat_messages().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].content, "hello");
}

#[test]
fn function_output_must_answer_an_earlier_call() {
    let mut prepared = prepared(json!({
        "model":"1",
        "input":[
            {"role":"user","content":"hello"},
            {"type":"function_call_output","call_id":"missing","output":"x"}
        ]
    }));
    let error = prepared.generation.chat_messages().unwrap_err();
    assert!(error.message.contains("answer an earlier"));
}

#[test]
fn unsupported_current_options_are_refused() {
    let cases = [
        json!({"model":"1","input":"hi","store":true}),
        json!({"model":"1","input":"hi","previous_response_id":"resp_1"}),
        json!({"model":"1","input":"hi","conversation":"conv_1"}),
        json!({"model":"1","input":"hi","background":true}),
        json!({"model":"1","input":"hi","include":["message.output_text.logprobs"]}),
        json!({"model":"1","input":"hi","truncation":"auto"}),
        json!({"model":"1","input":"hi","service_tier":"flex"}),
        json!({"model":"1","input":"hi","text":{"format":{"type":"json_schema","name":"x","schema":{}}}}),
        json!({"model":"1","input":"hi","text":{"verbosity":"high"}}),
        json!({"model":"1","input":"hi","reasoning":{"summary":"auto"}}),
        json!({"model":"1","input":"hi","tools":[{"type":"web_search"}]}),
        json!({"model":"1","input":"hi","stream":true,"stream_options":{"include_obfuscation":true}}),
        json!({"model":"1","input":"hi","top_logprobs":1}),
        json!({"model":"1","input":"hi","max_tool_calls":1}),
        json!({"model":"1","input":"hi","prompt":"p"}),
    ];
    for case in cases {
        let error = parse(case.clone())
            .prepare(StoreOptions::default(), None)
            .err()
            .unwrap();
        assert_eq!(error.code, infer_core::ErrorCode::Unsupported, "{case}");
    }
}

#[test]
fn null_optional_options_are_treated_as_absent() {
    let prepared = prepared(json!({
        "model":"1", "input":"hi",
        "store":null, "previous_response_id":null, "conversation":null,
        "background":null, "include":null, "truncation":null, "service_tier":null,
        "prompt":null, "context_management":null, "moderation":null,
        "access_programs":null, "top_logprobs":null, "max_tool_calls":null,
        "tools":null, "reasoning":null, "text":null, "metadata":null,
        "stream_options":null,
        "prompt_cache_key":null, "safety_identifier":null
    }));
    assert_eq!(prepared.spec.tool_choice, json!("auto"));
    assert_eq!(prepared.spec.tools.len(), 0);
}

#[test]
fn malformed_input_items_are_parameter_errors() {
    for case in [
        json!({"model":"1"}),
        json!({"model":"1","input":[]}),
        json!({"model":"1","input":[{"role":"tool","content":"x"}]}),
        json!({"model":"1","input":[{"role":"user","content":"x","unknown":true}]}),
        json!({"model":"1","input":[{"type":"function_call","name":"f","arguments":"{}"}]}),
        json!({"model":"1","input":[{"role":"user","content":"x"}],"metadata":{"a":1}}),
        json!({"model":"1","input":"x","text":{"format":{"type":"unknown"}}}),
    ] {
        let error = parse(case.clone())
            .prepare(StoreOptions::default(), None)
            .err()
            .unwrap();
        assert_eq!(error.code, infer_core::ErrorCode::InvalidInput, "{case}");
    }
}

#[test]
fn function_tools_are_translated_and_echoed() {
    let prepared = prepared(json!({
        "model":"1", "input":"hi", "tool_choice":{"type":"function","name":"ping"},
        "tools":[{"type":"function","name":"ping","description":"d","strict":true,
            "output_schema":null,"allowed_callers":null,
            "defer_loading":false,"async":false,
            "parameters":{"type":"object","properties":{}}}]
    }));
    assert_eq!(
        prepared.spec.tool_choice,
        json!({"type":"function","name":"ping"})
    );
    assert_eq!(prepared.spec.tools.len(), 1);
    assert_eq!(prepared.spec.tools[0]["strict"], json!(true));
    assert_eq!(prepared.generation.tools.as_ref().unwrap().len(), 1);
}

#[test]
fn advisory_hints_and_metadata_are_accepted() {
    let prepared = prepared(json!({
        "model":"1", "input":"hi", "store":false, "service_tier":"auto",
        "safety_identifier":"user-hash", "prompt_cache_key":"cache-key",
        "prompt_cache_retention":"24h", "prompt_cache_options":{"mode":"implicit"},
        "metadata":{"trace":"abc"}
    }));
    assert_eq!(prepared.spec.metadata["trace"], json!("abc"));
    assert_eq!(prepared.spec.user, None);
}

#[test]
fn reasoning_effort_selects_thinking() {
    for (effort, thinking) in [
        ("none", Some(false)),
        ("minimal", Some(false)),
        ("low", Some(true)),
        ("medium", Some(true)),
        ("high", Some(true)),
        ("xhigh", Some(true)),
        ("max", Some(true)),
    ] {
        let prepared = prepared(json!({
            "model":"1", "input":"hi", "reasoning":{"effort":effort}
        }));
        assert_eq!(prepared.generation.enable_thinking, thinking, "{effort}");
    }
    let error = parse(json!({"model":"1","input":"hi","reasoning":{"effort":"extreme"}}))
        .prepare(StoreOptions::default(), None)
        .err()
        .unwrap();
    assert_eq!(error.code, infer_core::ErrorCode::InvalidInput);
}

fn installed() -> StoreOptions {
    StoreOptions {
        installed: true,
        capabilities: ResponseStoreCapabilities {
            retrieve: true,
            previous_response_id: true,
            background: true,
            cancel: true,
            event_replay: true,
            persistent: false,
            shared: false,
            max_record_bytes: 1024,
        },
    }
}

#[test]
fn an_installed_store_enables_stateful_fields() {
    let prepared = parse(json!({
        "model":"1", "input":"hello", "store":true, "background":false
    }))
    .prepare(installed(), None)
    .unwrap();
    assert!(prepared.spec.store);
    assert!(!prepared.spec.background);
    assert_eq!(prepared.spec.previous_response_id, None);

    let prepared = parse(json!({"model":"1", "input":"hello"}))
        .prepare(installed(), None)
        .unwrap();
    assert!(prepared.spec.store, "store defaults to true when installed");

    let prepared = parse(json!({"model":"1", "input":"hello", "store":false}))
        .prepare(installed(), None)
        .unwrap();
    assert!(!prepared.spec.store);
}

#[test]
fn previous_response_id_combines_prior_input_and_output() {
    let prior = PriorState {
        input: vec![json!({"role":"user","content":"first"})],
        output: vec![json!({
            "type":"message","role":"assistant","status":"completed",
            "content":[{"type":"output_text","text":"answer","annotations":[]}]
        })],
        model: "1".to_string(),
    };
    let mut prepared = parse(json!({
        "model":"1", "input":"second", "previous_response_id":"resp_1"
    }))
    .prepare(installed(), Some(prior))
    .unwrap();
    let (messages, media) = prepared.generation.chat_messages().unwrap();
    assert_eq!(media.len(), 0);
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].content, "first");
    assert_eq!(messages[1].content, "answer");
    assert_eq!(messages[2].content, "second");
    let input: Value = serde_json::from_slice(&prepared.request_input).unwrap();
    assert_eq!(input.as_array().unwrap().len(), 3);
    assert_eq!(
        prepared.spec.previous_response_id.as_deref(),
        Some("resp_1")
    );
}

#[test]
fn background_needs_store_and_rejects_streaming() {
    let error = parse(json!({
        "model":"1", "input":"hello", "background":true, "store":false
    }))
    .prepare(installed(), None)
    .err()
    .unwrap();
    assert_eq!(error.code, infer_core::ErrorCode::InvalidInput);

    let error = parse(json!({
        "model":"1", "input":"hello", "background":true, "stream":true
    }))
    .prepare(installed(), None)
    .err()
    .unwrap();
    assert_eq!(error.code, infer_core::ErrorCode::Unsupported);

    let error = parse(json!({"model":"1", "input":"hello", "background":true}))
        .prepare(StoreOptions::default(), None)
        .err()
        .unwrap();
    assert_eq!(error.code, infer_core::ErrorCode::Unsupported);
}

#[test]
fn stateful_options_without_a_store_stay_unsupported() {
    for request in [
        json!({"model":"1", "input":"hello", "store":true}),
        json!({"model":"1", "input":"hello", "previous_response_id":"resp_1"}),
        json!({"model":"1", "input":"hello", "background":true}),
    ] {
        let error = parse(request)
            .prepare(StoreOptions::default(), None)
            .err()
            .unwrap();
        assert_eq!(error.code, infer_core::ErrorCode::Unsupported);
    }
}
