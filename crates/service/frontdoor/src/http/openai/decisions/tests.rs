use super::request::{Answering, ChoiceValue, DecisionRequest, ReadPlan};
use super::{answer, shared_prefix};
use infer_core::ErrorCode;
use serde_json::{Value, json};
use std::path::Path;

fn parse(body: Value) -> DecisionRequest {
    serde_json::from_value(body).unwrap()
}

fn plan(body: Value) -> infer_core::Result<ReadPlan> {
    parse(body).plan()
}

fn error(body: Value) -> infer_core::Error {
    plan(body).unwrap_err()
}

fn predicate() -> Value {
    json!({"type": "predicate", "name": "damaged", "instructions": "Is it damaged?"})
}

#[test]
fn string_input_is_the_evidence_and_options_are_lettered_in_order() {
    let plan = plan(json!({
        "model": "m",
        "input": "The screen is cracked.",
        "questions": [
            predicate(),
            {"type": "choice", "instructions": "Which team?", "choices": [
                {"value": "billing", "description": "Payments."},
                {"value": true},
            ]},
            {"type": "score", "instructions": "How bad?", "levels": [
                {"label": "Cosmetic"}, {"label": "Blocked", "description": "No workaround."},
            ]},
        ],
    }))
    .unwrap();
    assert_eq!(plan.system, None);
    assert_eq!(plan.evidence, "The screen is cracked.");
    let [first, second, third] = plan.questions.as_slice() else {
        panic!("three questions expected");
    };
    assert_eq!(first.name.as_deref(), Some("damaged"));
    assert_eq!(first.answering, Answering::Predicate);
    assert_eq!(
        first.text,
        "Question: Is it damaged?\nA: false\nB: true\nAnswer with the letter of one option only."
    );
    assert_eq!(second.name, None);
    assert_eq!(
        second.text,
        "Question: Which team?\nA: \"billing\" - Payments.\nB: true\nAnswer with the letter of one option only."
    );
    assert_eq!(
        second.answering,
        Answering::Choice(vec![
            ChoiceValue::Text("billing".into()),
            ChoiceValue::Bool(true)
        ])
    );
    assert_eq!(third.options, 2);
    assert!(third.text.contains("B: Blocked - No workaround.\n"));
}

#[test]
fn conversation_keeps_speakers_and_developer_turns_become_the_system_prompt() {
    let plan = plan(json!({
        "model": "m",
        "input": [
            {"role": "system", "content": "Be strict."},
            {"role": "developer", "content": [{"type": "input_text", "text": "Judge refunds."}]},
            {"role": "user", "content": [
                {"type": "input_text", "text": "Refund me."},
                {"type": "input_json", "data": {"charges": [49, 49]}},
            ]},
            {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{\"id\":7}"},
            {"type": "function_call_output", "call_id": "c1", "output": "two charges"},
            {"role": "assistant", "type": "message", "phase": "final_answer",
             "content": [{"type": "output_text", "text": "Refunded."}]},
        ],
        "questions": [predicate()],
    }))
    .unwrap();
    assert_eq!(plan.system.as_deref(), Some("Be strict.\n\nJudge refunds."));
    assert_eq!(
        plan.evidence,
        "User: Refund me.\n{\"charges\":[49,49]}\n\nFunction call lookup (c1): {\"id\":7}\n\n\
         Function output (c1): two charges\n\nAssistant: Refunded."
    );
}

#[test]
fn user_only_messages_read_as_plain_evidence() {
    let plan = plan(json!({
        "model": "m",
        "input": [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}],
        "questions": [predicate()],
    }))
    .unwrap();
    assert_eq!(plan.evidence, "one\n\ntwo");
}

#[test]
fn malformed_requests_are_parameter_errors() {
    let base = |questions: Value| json!({"model": "m", "input": "x", "questions": questions});
    for body in [
        base(json!([])),
        base(json!([{"type": "choice", "instructions": "", "choices": [{"value": "a"}]}])),
        base(json!([{"type": "choice", "instructions": "", "choices": [
            {"value": "a"}, {"value": "a"},
        ]}])),
        base(
            json!([{"type": "score", "instructions": "", "levels": (0..11)
            .map(|i| json!({"label": i.to_string()})).collect::<Vec<_>>()}]),
        ),
        // Wider than the label alphabet the backend reads.
        base(
            json!([{"type": "choice", "instructions": "", "choices": (0..27)
            .map(|i| json!({"value": i.to_string()})).collect::<Vec<_>>()}]),
        ),
        json!({"model": "m", "input": [
            {"type": "function_call_output", "call_id": "missing", "output": "x"},
        ], "questions": [predicate()]}),
        json!({"model": "m", "input": [
            {"role": "developer", "content": [{"type": "input_json", "data": {}}]},
        ], "questions": [predicate()]}),
        json!({"model": "m", "input": [
            {"role": "user", "content": [{"type": "input_json", "data": [1]}]},
        ], "questions": [predicate()]}),
        json!({"model": "m", "input": [{"role": "user", "content": "x", "extra": 1}],
            "questions": [predicate()]}),
        json!({"model": "m", "input": 7, "questions": [predicate()]}),
        json!({"model": "m", "input": "x", "questions": [predicate()],
            "prompt_cache_options": {"ttl": "1h"}}),
    ] {
        assert_eq!(error(body).code, ErrorCode::InvalidInput);
    }
    let unknown = serde_json::from_value::<DecisionRequest>(
        json!({"model": "m", "input": "x", "questions": [predicate()], "stream": true}),
    );
    assert!(unknown.is_err());
}

#[test]
fn json_input_respects_the_published_depth_bound() {
    let mut deep = json!({});
    for _ in 0..40 {
        deep = json!({"a": deep});
    }
    let body = json!({"model": "m", "input": [
        {"role": "user", "content": [{"type": "input_json", "data": deep}]},
    ], "questions": [predicate()]});
    assert_eq!(error(body).code, ErrorCode::InvalidInput);
}

#[test]
fn images_are_explicitly_unsupported_in_input_and_instructions() {
    let image = json!({"type": "input_image", "image_url": "https://example.com/a.png"});
    for body in [
        json!({"model": "m", "input": [{"role": "user", "content": [image]}],
            "questions": [predicate()]}),
        json!({"model": "m", "input": "x", "questions": [
            {"type": "predicate", "instructions": [{"type": "input_text", "text": "Same?"}, image]},
        ]}),
    ] {
        assert_eq!(error(body).code, ErrorCode::Unsupported);
    }
}

#[test]
fn cache_hints_are_accepted_without_changing_the_prompt() {
    let plan = plan(json!({
        "model": "m",
        "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "doc", "prompt_cache_breakpoint": {"mode": "explicit"}},
        ]}],
        "questions": [predicate()],
        "prompt_cache_key": "tenant-1",
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        "safety_identifier": "user-1",
    }))
    .unwrap();
    assert_eq!(plan.evidence, "doc");
}

#[test]
fn answers_follow_the_protocol_shapes() {
    let predicate = answer(Some("p"), &Answering::Predicate, &[0.25, 0.75], 0.5).unwrap();
    let predicate = serde_json::to_value(predicate).unwrap();
    assert_eq!(
        predicate,
        json!({"type": "predicate", "name": "p", "probability": 0.75})
    );

    let values = vec![
        ChoiceValue::Text("billing".into()),
        ChoiceValue::Bool(false),
    ];
    let choice = answer(None, &Answering::Choice(values), &[0.25, 0.75], 0.5).unwrap();
    let choice = serde_json::to_value(choice).unwrap();
    assert_eq!(choice["choice"], json!(false));
    assert_eq!(choice["name"], Value::Null);
    assert_eq!(choice["confidence"], json!(0.375));
    assert_eq!(
        choice["probabilities"],
        json!([{"value": "billing", "probability": 0.25}, {"value": false, "probability": 0.75}])
    );

    let levels = vec!["low".to_owned(), "mid".to_owned(), "high".to_owned()];
    let score = answer(
        Some("s"),
        &Answering::Score(levels),
        &[0.25, 0.5, 0.25],
        1.0,
    )
    .unwrap();
    let score = serde_json::to_value(score).unwrap();
    assert_eq!(score["score"], json!(1.0));
    assert_eq!(
        score["probabilities"][2],
        json!({"value": 2, "label": "high", "probability": 0.25})
    );
    assert_eq!(score["confidence"], json!(0.5));

    let mismatch = answer(None, &Answering::Predicate, &[1.0], 1.0).unwrap_err();
    assert_eq!(mismatch.code, ErrorCode::Invariant);
}

#[test]
fn the_shared_prefix_covers_the_evidence_only_when_questions_can_share_it() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/qwen-hybrid-tiny");
    let assets = infer_models::TextAssets::open(root, 128).unwrap();
    let body = |input: &str, count: usize| json!({"model": "m", "input": input, "questions": vec![predicate(); count]});
    let long = "hello world ".repeat(40);
    let shared = shared_prefix(&assets, &plan(body(&long, 2)).unwrap()).unwrap();
    // At least the evidence every question starts from.
    assert!(shared.is_some_and(|tokens| tokens >= 80), "{shared:?}");
    assert_eq!(
        shared_prefix(&assets, &plan(body(&long, 1)).unwrap()).unwrap(),
        None
    );
    // A prefix within the narrow prompt graph is not worth running one question first.
    assert_eq!(
        shared_prefix(&assets, &plan(body("hello world", 2)).unwrap()).unwrap(),
        None
    );
    assert_eq!(
        shared_prefix(&assets, &plan(body("", 2)).unwrap()).unwrap(),
        None
    );
}
