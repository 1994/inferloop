//! Response objects and output items of the `OpenAI` Responses protocol.
//!
//! The complete and the streamed route build the same item shapes from the same parser events;
//! only the surrounding envelope differs.
use super::super::execution::finish_reason;
use super::request::Spec;
use infer_core::{FinishReason, RequestId, Result};
use infer_models::ParsedOutput;
use serde_json::{Value, json};

/// Token usage reported by one response.
pub(super) struct Usage {
    pub(super) input: usize,
    pub(super) output: usize,
    pub(super) reasoning: usize,
}

/// Everything needed to serialize a response at one point in its lifecycle.
pub(super) struct Snapshot<'a> {
    pub(super) request: RequestId,
    pub(super) created: u64,
    pub(super) completed_at: Option<u64>,
    pub(super) model: &'a str,
    pub(super) status: &'a str,
    pub(super) incomplete: Option<Value>,
    pub(super) error: Option<Value>,
    pub(super) output: &'a [Value],
    pub(super) spec: &'a Spec,
    pub(super) temperature: f32,
    pub(super) top_p: f32,
    pub(super) usage: Option<&'a Usage>,
}

impl Snapshot<'_> {
    /// Serialize the response object.
    pub(super) fn value(&self) -> Value {
        let usage = self.usage.map(|usage| {
            json!({
                "input_tokens": usage.input,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": usage.output,
                "output_tokens_details": {"reasoning_tokens": usage.reasoning},
                "total_tokens": usage.input + usage.output,
            })
        });
        json!({
            "id": self
                .spec
                .response_id
                .clone()
                .unwrap_or_else(|| response_id(self.request)),
            "object": "response",
            "created_at": self.created,
            "status": self.status,
            "completed_at": self.completed_at,
            "error": self.error,
            "incomplete_details": self.incomplete,
            "instructions": self.spec.instructions,
            "max_output_tokens": self.spec.max_output_tokens,
            "model": self.model,
            "output": self.output,
            "parallel_tool_calls": self.spec.parallel_tool_calls,
            "previous_response_id": self.spec.previous_response_id,
            "reasoning": reasoning_echo(self.spec),
            "store": self.spec.store,
            "temperature": self.temperature,
            "text": self.spec.text,
            "tool_choice": self.spec.tool_choice,
            "tools": self.spec.tools,
            "top_p": self.top_p,
            "truncation": "disabled",
            "usage": usage,
            "user": self.spec.user,
            "metadata": self.spec.metadata,
            "access_programs": null,
            "background": self.spec.background,
            "conversation": null,
            "service_tier": "default",
        })
    }
}

fn reasoning_echo(spec: &Spec) -> Value {
    json!({"effort": spec.reasoning_effort, "summary": null, "context": null})
}

/// The response identifier derived from the request identity, stable across streamed events.
pub(super) fn response_id(request: RequestId) -> String {
    format!("resp_{}", request.get())
}

/// Publish the completed output items in the order the parser reports them.
///
/// `status` is the response's terminal status: a message cut off by the output budget keeps the
/// `incomplete` status, while reasoning and fully parsed calls are complete on their own.
pub(super) fn items(namespace: &str, parsed: &ParsedOutput, status: &str) -> Vec<Value> {
    let mut items = Vec::new();
    let mut position = 0;
    if let Some(reasoning) = parsed.reasoning.as_deref().filter(|text| !text.is_empty()) {
        items.push(reasoning_item(namespace, position, reasoning, "completed"));
        position += 1;
    }
    if !parsed.content.is_empty() || parsed.calls.is_empty() {
        items.push(message_item(namespace, position, &parsed.content, status));
    }
    for (index, call) in parsed.calls.iter().enumerate() {
        items.push(function_call_item(
            namespace,
            index,
            &call.name,
            &call.arguments,
            "completed",
        ));
    }
    items
}

/// One `output_text` content part.
pub(super) fn output_text_part(text: &str) -> Value {
    json!({"type":"output_text","text":text,"annotations":[],"logprobs":[]})
}

/// A reasoning item carrying the generated reasoning as reasoning text content.
pub(super) fn reasoning_item(namespace: &str, position: usize, text: &str, status: &str) -> Value {
    let content = if text.is_empty() {
        Vec::new()
    } else {
        vec![json!({"type":"reasoning_text","text":text})]
    };
    json!({
        "id": format!("rs_{namespace}_{position}"),
        "type": "reasoning",
        "status": status,
        "summary": [],
        "content": content,
    })
}

/// An assistant message item. An in-progress item publishes no content part yet.
pub(super) fn message_item(namespace: &str, position: usize, text: &str, status: &str) -> Value {
    let content = if text.is_empty() {
        Vec::new()
    } else {
        vec![output_text_part(text)]
    };
    json!({
        "id": format!("msg_{namespace}_{position}"),
        "type": "message",
        "status": status,
        "role": "assistant",
        "content": content,
    })
}

/// One round-trip function call. The `call_id` is the identifier a later output item answers.
pub(super) fn function_call_item(
    namespace: &str,
    index: usize,
    name: &str,
    arguments: &str,
    status: &str,
) -> Value {
    json!({
        "id": call_item_id(namespace, index),
        "type": "function_call",
        "status": status,
        "call_id": call_id(namespace, index),
        "name": name,
        "arguments": arguments,
    })
}

/// The id of one function-call output item.
pub(super) fn call_item_id(namespace: &str, index: usize) -> String {
    format!("fc_{namespace}_{index}")
}

/// The stable identifier shared by a call item and its output item.
pub(super) fn call_id(namespace: &str, index: usize) -> String {
    format!("call_{namespace}_{index}")
}

/// The terminal status and details of one finished generation.
///
/// # Errors
/// Returns a backend error when the parser withheld a structured region for any reason other
/// than the engine's length cap.
pub(super) fn terminal(
    reason: &FinishReason,
    parsed: &ParsedOutput,
) -> Result<(&'static str, Option<Value>)> {
    let reason = finish_reason(reason)?;
    if parsed.truncated && reason != "length" {
        return Err(infer_core::Error::new(
            infer_core::ErrorCode::Backend,
            "model output ended inside a structured region",
        ));
    }
    if reason == "length" {
        Ok(("incomplete", Some(json!({"reason":"max_output_tokens"}))))
    } else {
        Ok(("completed", None))
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/http_responses_output.rs"]
mod tests;
