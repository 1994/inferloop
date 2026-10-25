use super::*;
use serde_json::json;

fn parse(value: Value) -> GenerationRequest {
    serde_json::from_value(value).expect("fixture request must deserialize")
}

fn chat(messages: &Value, extra: &Value) -> GenerationRequest {
    let mut object = json!({"model": "1", "messages": messages});
    for (key, value) in extra.as_object().expect("extra must be an object") {
        object[key] = value.clone();
    }
    parse(object)
}

fn weather_tool() -> Value {
    json!([{"type": "function", "function": {
        "name": "get_weather",
        "description": "Look up the weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
    }}])
}

#[test]
fn a_plain_chat_request_declares_no_tools() {
    let request = chat(&json!([{"role": "user", "content": "hi"}]), &json!({}));
    request.validate(true).unwrap();
    assert!(!request.tool_policy(true).unwrap().parses());
    let messages = request.chat_messages().unwrap().0;
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].content, "hi");
    assert_eq!(messages[0].tool_calls.len(), 0);
}

#[test]
fn declared_tools_enable_auto_parsing_and_reach_the_template() {
    let request = chat(
        &json!([{"role": "user", "content": "weather?"}]),
        &json!({"tools": weather_tool()}),
    );
    request.validate(true).unwrap();
    let policy = request.tool_policy(true).unwrap();
    assert!(policy.parses());
    assert!(policy.declared().unwrap().contains("get_weather"));
    assert_eq!(policy.template().unwrap()[0]["type"], "function");
    assert_eq!(
        policy.template().unwrap()[0]["function"]["name"],
        "get_weather"
    );
}

#[test]
fn tool_choice_none_keeps_the_declaration_but_disables_publishing() {
    let request = chat(
        &json!([{"role": "user", "content": "weather?"}]),
        &json!({"tools": weather_tool(), "tool_choice": "none"}),
    );
    request.validate(true).unwrap();
    assert!(!request.tool_policy(true).unwrap().parses());
}

#[test]
fn constrained_modes_request_a_generation_matcher() {
    for extra in [
        json!({"tools": weather_tool(), "tool_choice": "required"}),
        json!({"tools": weather_tool(), "parallel_tool_calls": false}),
        json!({"tools": weather_tool(), "tool_choice": {
            "type": "function", "function": {"name": "get_weather"}}}),
        json!({"tools": [{"type": "function", "function": {
            "name": "get_weather", "strict": true}}]}),
    ] {
        let request = chat(&json!([{"role": "user", "content": "x"}]), &extra);
        assert!(
            request.tool_policy(true).unwrap().constraint().is_some(),
            "{extra}"
        );
    }
}

#[test]
fn malformed_tool_declarations_are_parameter_errors() {
    for extra in [
        json!({"tools": [{"type": "function", "function": {"name": "  "}}]}),
        json!({"tools": [
            {"type": "function", "function": {"name": "f"}},
            {"type": "function", "function": {"name": "f"}}
        ]}),
        json!({"tools": weather_tool(), "tool_choice": "sometimes"}),
    ] {
        let request = chat(&json!([{"role": "user", "content": "x"}]), &extra);
        let error = request.tool_policy(true).unwrap_err();
        assert_eq!(error.code, infer_core::ErrorCode::InvalidInput, "{extra}");
    }
    // A tool kind the service does not implement is reported as unsupported, not invalid.
    let other = chat(
        &json!([{"role": "user", "content": "x"}]),
        &json!({"tools": [{"type": "other", "function": {"name": "f"}}]}),
    );
    assert_eq!(
        other.tool_policy(true).unwrap_err().code,
        infer_core::ErrorCode::Unsupported
    );
}

#[test]
fn a_tool_round_trips_into_internal_history() {
    let request = chat(
        &json!([
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}
            ]},
            {"role": "tool", "content": "{\"temp_c\":21}", "tool_call_id": "call_1"}
        ]),
        &json!({"tools": weather_tool()}),
    );
    request.validate(true).unwrap();
    let messages = request.chat_messages().unwrap().0;
    assert_eq!(messages.len(), 3);
    // A null assistant content is stored as empty text, not rejected.
    assert_eq!(messages[1].content, "");
    assert_eq!(messages[1].tool_calls.len(), 1);
    assert_eq!(messages[1].tool_calls[0].name, "get_weather");
    assert_eq!(messages[1].tool_calls[0].arguments, "{\"city\":\"Paris\"}");
    assert_eq!(messages[2].tool_call_id.as_deref(), Some("call_1"));
    // The template view needs an argument object, not the wire string.
    assert_eq!(
        messages[1].tool_calls[0].argument_object().unwrap()["city"],
        "Paris"
    );
}

#[test]
fn a_result_that_answers_no_earlier_call_is_rejected() {
    let request = chat(
        &json!([
            {"role": "user", "content": "x"},
            {"role": "tool", "content": "r", "tool_call_id": "call_missing"}
        ]),
        &json!({}),
    );
    assert!(request.chat_messages().is_err());
}

#[test]
fn duplicate_call_ids_and_misplaced_fields_are_rejected() {
    let duplicate = json!([
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "function": {"name": "f", "arguments": "{}"}},
            {"id": "call_1", "function": {"name": "g", "arguments": "{}"}}
        ]}
    ]);
    assert!(chat(&duplicate, &json!({})).chat_messages().is_err());
    let on_user = json!([{"role": "user", "content": "x", "tool_calls": [
        {"id": "call_1", "function": {"name": "f", "arguments": "{}"}}
    ]}]);
    assert!(chat(&on_user, &json!({})).chat_messages().is_err());
    let id_on_assistant = json!([
        {"role": "assistant", "content": "x", "tool_call_id": "call_1"}
    ]);
    assert!(chat(&id_on_assistant, &json!({})).chat_messages().is_err());
    let tool_without_id = json!([{"role": "tool", "content": "r"}]);
    assert!(chat(&tool_without_id, &json!({})).chat_messages().is_err());
}

#[test]
fn text_content_parts_are_joined_and_other_parts_are_refused() {
    let parts = json!([{"role": "user", "content": [
        {"type": "text", "text": "hello "}, {"type": "text", "text": "world"}
    ]}]);
    assert_eq!(
        chat(&parts, &json!({})).chat_messages().unwrap().0[0].content,
        "hello world"
    );
    let image = json!([{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
    ]}]);
    let (messages, media) = chat(&image, &json!({})).chat_messages().unwrap();
    assert_eq!(
        messages[0].content,
        "<|vision_start|><|image_pad|><|vision_end|>"
    );
    assert_eq!(media.len(), 1);
    assert_eq!(media[0].modality, Modality::Image);
    let missing_url =
        json!([{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]);
    assert!(chat(&missing_url, &json!({})).chat_messages().is_err());
    let video = json!([{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "file:///tmp/clip.mp4"}}
    ]}]);
    let (messages, media) = chat(&video, &json!({})).chat_messages().unwrap();
    assert_eq!(
        messages[0].content,
        "<|vision_start|><|video_pad|><|vision_end|>"
    );
    assert_eq!(media[0].modality, Modality::Video);
    let assistant_media = json!([{"role": "assistant", "content": [
        {"type": "image_url", "image_url": {"url": "file:///tmp/a.png"}}
    ]}]);
    assert!(chat(&assistant_media, &json!({})).chat_messages().is_err());
}

#[test]
fn completions_refuse_the_chat_only_tool_fields() {
    for extra in [
        json!({"tools": weather_tool()}),
        json!({"tool_choice": "auto"}),
        json!({"parallel_tool_calls": true}),
    ] {
        let mut object = json!({"model": "1", "prompt": "hi"});
        for (key, value) in extra.as_object().unwrap() {
            object[key] = value.clone();
        }
        assert!(parse(object).validate(false).is_err(), "{extra}");
    }
}

#[test]
fn nullable_tool_calls_from_sdk_history_are_accepted() {
    let request = chat(
        &json!([
            {"role":"user", "content":"hello"},
            {"role":"assistant", "content":"hi", "tool_calls":null}
        ]),
        &json!({}),
    );
    request.validate(true).unwrap();
    let messages = request.chat_messages().unwrap().0;
    assert_eq!(messages[1].tool_calls.len(), 0);
    assert_eq!(messages[1].content, "hi");
}

#[test]
fn invalid_function_names_and_choices_are_parameter_errors() {
    for name in [
        "with space",
        " leading",
        "trailing ",
        "a.b",
        "中文",
        &"a".repeat(65),
    ] {
        let request = chat(
            &json!([{"role":"user", "content":"hi"}]),
            &json!({"tools":[{"type":"function", "function":{"name":name}}]}),
        );
        assert_eq!(
            request.tool_policy(true).unwrap_err().code,
            infer_core::ErrorCode::InvalidInput
        );
    }
    for choice in [
        json!({"type":"function", "function":{"name":"missing"}}),
        json!({"type":"invalid", "function":{"name":"get_weather"}}),
    ] {
        let request = chat(
            &json!([{"role":"user", "content":"hi"}]),
            &json!({"tools":weather_tool(), "tool_choice":choice}),
        );
        assert_eq!(
            request.tool_policy(true).unwrap_err().code,
            infer_core::ErrorCode::InvalidInput
        );
    }
}
