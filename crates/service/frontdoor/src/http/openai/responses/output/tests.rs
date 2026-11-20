use super::super::request::{TextConfigEcho, TextFormatEcho, ToolChoiceEcho};
use super::*;
use infer_core::{FinishReason, RequestId};
use infer_models::{ParsedOutput, ParsedToolCall};
use serde_json::{Value, json};
use std::collections::BTreeMap;

fn request() -> RequestId {
    RequestId::new(7).unwrap()
}

fn parsed(content: &str, reasoning: Option<&str>, calls: Vec<ParsedToolCall>) -> ParsedOutput {
    ParsedOutput {
        content: content.to_string(),
        reasoning: reasoning.map(str::to_string),
        calls,
        truncated: false,
    }
}

fn items_json(namespace: &str, parsed: &ParsedOutput, status: &str) -> Value {
    serde_json::to_value(items(namespace, parsed, status)).unwrap()
}

fn call(name: &str, arguments: &str) -> ParsedToolCall {
    ParsedToolCall {
        name: name.to_string(),
        arguments: arguments.to_string(),
    }
}

#[test]
fn items_are_ordered_reasoning_message_and_calls() {
    let output = items_json(
        "7",
        &parsed("hello", Some("thinking"), vec![call("ping", "{}")]),
        "completed",
    );
    assert_eq!(output.as_array().unwrap().len(), 3);
    assert_eq!(output[0]["type"], "reasoning");
    assert_eq!(output[0]["id"], "rs_7_0");
    assert_eq!(output[0]["content"][0]["text"], "thinking");
    assert_eq!(output[1]["type"], "message");
    assert_eq!(output[1]["id"], "msg_7_1");
    assert_eq!(output[1]["content"][0]["text"], "hello");
    assert_eq!(output[2]["type"], "function_call");
    assert_eq!(output[2]["call_id"], "call_7_0");
    assert_eq!(output[2]["name"], "ping");
    assert_eq!(output[2]["arguments"], "{}");
}

#[test]
fn a_call_only_response_has_no_message_item() {
    let output = items_json(
        "7",
        &parsed("", None, vec![call("ping", "{}")]),
        "completed",
    );
    assert_eq!(output.as_array().unwrap().len(), 1);
    assert_eq!(output[0]["type"], "function_call");
}

#[test]
fn an_empty_response_still_has_a_message_item() {
    let output = items_json("7", &parsed("", None, Vec::new()), "completed");
    assert_eq!(output.as_array().unwrap().len(), 1);
    assert_eq!(output[0]["type"], "message");
    assert_eq!(output[0]["content"], json!([]));
}

#[test]
fn length_is_an_incomplete_terminal_state() {
    let (status, details) =
        terminal(&FinishReason::Length, &parsed("x", None, Vec::new())).unwrap();
    assert_eq!(status, "incomplete");
    assert_eq!(
        details.map(|details| details.reason),
        Some("max_output_tokens".to_string())
    );
    let (status, details) = terminal(&FinishReason::Eos, &parsed("x", None, Vec::new())).unwrap();
    assert_eq!(status, "completed");
    assert!(details.is_none());
}

#[test]
fn an_unterminated_region_without_length_is_a_backend_error() {
    let mut output = parsed("x", None, Vec::new());
    output.truncated = true;
    let error = terminal(&FinishReason::Eos, &output).unwrap_err();
    assert_eq!(error.code, infer_core::ErrorCode::Backend);
}

#[test]
fn the_response_object_reports_usage_and_the_requested_limits() {
    let spec = Spec {
        response_id: None,
        store: false,
        background: false,
        previous_response_id: None,
        instructions: Some("be brief".to_string()),
        tools: Vec::new(),
        tool_choice: ToolChoiceEcho::Mode("auto".to_string()),
        parallel_tool_calls: true,
        reasoning_effort: Some("low".to_string()),
        text: TextConfigEcho {
            format: TextFormatEcho {
                kind: "text".to_string(),
            },
        },
        metadata: BTreeMap::new(),
        max_output_tokens: Some(32),
        user: Some("user-1".to_string()),
    };
    let output = items("7", &parsed("hello", None, Vec::new()), "completed");
    let usage = Usage {
        input: 5,
        output: 2,
        reasoning: 0,
    };
    let value = serde_json::to_value(
        Snapshot {
            request: request(),
            created: 100,
            completed_at: Some(101),
            model: "1",
            status: "completed",
            incomplete: None,
            error: None,
            output: &output,
            spec: &spec,
            temperature: 0.5,
            top_p: 0.9,
            usage: Some(&usage),
        }
        .value(),
    )
    .unwrap();
    assert_eq!(value["object"], "response");
    assert_eq!(value["id"], "resp_7");
    assert_eq!(value["status"], "completed");
    assert_eq!(value["completed_at"], 101);
    assert_eq!(value["instructions"], "be brief");
    assert_eq!(value["max_output_tokens"], 32);
    assert_eq!(value["usage"]["input_tokens"], 5);
    assert_eq!(value["usage"]["output_tokens"], 2);
    assert_eq!(
        value["usage"]["input_tokens_details"],
        json!({"cached_tokens":0,"cache_write_tokens":0})
    );
    assert_eq!(value["temperature"], 0.5);
    assert_eq!(value["reasoning"]["effort"], "low");
    assert_eq!(value["store"], false);
    assert_eq!(value["previous_response_id"], Value::Null);
    assert_eq!(value["access_programs"], Value::Null);
}
