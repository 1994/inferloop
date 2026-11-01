use super::*;
use infer_models::ToolDialect;
use serde_json::Value;
use std::collections::BTreeSet;

fn policy(names: &[&str]) -> ToolPolicy {
    ToolPolicy::Auto {
        constraint: None,
        declared: names
            .iter()
            .map(|name| (*name).to_string())
            .collect::<BTreeSet<_>>(),
        template: OpaqueJson(serde_json::json!([])),
    }
}

fn value(choice: ChatChoice) -> Value {
    serde_json::to_value(choice).unwrap()
}

fn choice(text: &str, policy: &ToolPolicy, reason: &'static str) -> Value {
    value(
        chat_choice(
            RequestId::new(7).unwrap(),
            text,
            ToolDialect::FunctionParameters,
            policy,
            false,
            reason,
            false,
        )
        .unwrap(),
    )
}

#[test]
fn only_successful_terminal_states_are_completion_choices() {
    assert_eq!(finish_reason(&FinishReason::Length).unwrap(), "length");
    assert_eq!(finish_reason(&FinishReason::Eos).unwrap(), "stop");
    for reason in [
        FinishReason::Cancelled,
        FinishReason::Deadline,
        FinishReason::Failed("device fault".into()),
    ] {
        assert_eq!(finish_reason(&reason).unwrap_err().code, ErrorCode::Backend);
    }
}

#[test]
fn a_declared_call_becomes_a_structured_choice_with_a_stable_id() {
    let choice = choice(
        "Sure.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
        &policy(&["get_weather"]),
        "stop",
    );
    assert_eq!(choice["finish_reason"], "tool_calls");
    assert_eq!(choice["message"]["content"], "Sure.");
    let call = &choice["message"]["tool_calls"][0];
    assert_eq!(call["id"], "call_7_0");
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "get_weather");
    let arguments: Value =
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(arguments["city"], "Paris");
}

#[test]
fn a_call_only_response_carries_null_content() {
    let choice = choice(
        "<tool_call><function=ping>\n</function></tool_call>",
        &policy(&["ping"]),
        "stop",
    );
    assert!(choice["message"]["content"].is_null());
    assert_eq!(
        choice["message"]["tool_calls"][0]["function"]["arguments"],
        "{}"
    );
}

#[test]
fn a_length_stop_keeps_length_even_with_a_published_call() {
    let choice = choice(
        "<tool_call><function=ping>\n</function></tool_call>",
        &policy(&["ping"]),
        "length",
    );
    assert_eq!(choice["finish_reason"], "length");
    assert_eq!(choice["message"]["tool_calls"].as_array().unwrap().len(), 1);
}

#[test]
fn an_undeclared_call_is_not_published() {
    let choice = choice(
        "<tool_call><function=invented>\n</function></tool_call>",
        &policy(&["declared"]),
        "stop",
    );
    assert_eq!(choice["finish_reason"], "stop");
    assert!(choice["message"]["tool_calls"].is_null());
    assert!(
        choice["message"]["content"]
            .as_str()
            .unwrap()
            .contains("invented")
    );
}

#[test]
fn disabled_tool_publishing_leaves_the_markers_in_the_text() {
    let choice = choice(
        "<tool_call><function=ping>\n</function></tool_call>",
        &ToolPolicy::Disabled,
        "stop",
    );
    assert!(choice["message"]["tool_calls"].is_null());
    assert!(
        choice["message"]["content"]
            .as_str()
            .unwrap()
            .contains("<tool_call>")
    );
}

#[test]
fn thinking_is_split_out_and_reasoning_is_published_separately() {
    let text = "\u{3c}think\u{3e}weigh the options\u{3c}/think\u{3e}\n\nThe answer is 17.";
    let parsed =
        infer_models::parse_model_output_with(text, ToolDialect::FunctionParameters, None).unwrap();
    let choice = value(
        chat_choice(
            RequestId::new(7).unwrap(),
            text,
            ToolDialect::FunctionParameters,
            &ToolPolicy::Disabled,
            true,
            "stop",
            false,
        )
        .unwrap(),
    );
    assert_eq!(parsed.reasoning.as_deref(), Some("weigh the options"));
    assert_eq!(choice["message"]["content"], "The answer is 17.");
    assert_eq!(choice["message"]["reasoning_content"], "weigh the options");
}

#[test]
fn an_unterminated_generated_call_is_an_error_unless_length_capped() {
    let text = "<tool_call><function=ping>";
    assert!(
        chat_choice(
            RequestId::ONE,
            text,
            ToolDialect::FunctionParameters,
            &policy(&["ping"]),
            false,
            "stop",
            false,
        )
        .is_err()
    );
    let choice = value(
        chat_choice(
            RequestId::ONE,
            text,
            ToolDialect::FunctionParameters,
            &policy(&["ping"]),
            false,
            "length",
            false,
        )
        .unwrap(),
    );
    assert_eq!(choice["finish_reason"], "length");
}

#[test]
fn thinking_with_tool_choice_none_preserves_tool_like_text() {
    let text = "<think>why</think><tool_call><function=ping></function></tool_call>";
    let choice = value(
        chat_choice(
            RequestId::ONE,
            text,
            ToolDialect::FunctionParameters,
            &ToolPolicy::Disabled,
            true,
            "stop",
            false,
        )
        .unwrap(),
    );
    assert!(
        choice["message"]["content"]
            .as_str()
            .unwrap()
            .contains("<tool_call>")
    );
    assert!(choice["message"]["tool_calls"].is_null());
}

#[test]
fn a_prompt_opened_reasoning_region_is_not_assistant_content() {
    let choice = value(
        chat_choice(
            RequestId::ONE,
            "weigh the options</think>\n\nThe answer is 17.",
            ToolDialect::FunctionParameters,
            &ToolPolicy::Disabled,
            true,
            "stop",
            true,
        )
        .unwrap(),
    );
    assert_eq!(choice["message"]["reasoning_content"], "weigh the options");
    assert_eq!(choice["message"]["content"], "The answer is 17.");
}
