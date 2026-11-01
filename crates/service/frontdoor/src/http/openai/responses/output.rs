//! Response objects and output items of the `OpenAI` Responses protocol.
//!
//! The complete and the streamed route build the same typed item shapes from the same parser
//! events; only the surrounding envelope differs.
use super::super::execution::finish_reason;
use super::request::{FunctionToolEcho, Spec, TextConfigEcho, ToolChoiceEcho};
use infer_core::{FinishReason, RequestId, Result};
use infer_models::ParsedOutput;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Token usage reported by one response.
pub(super) struct Usage {
    pub(super) input: usize,
    pub(super) output: usize,
    pub(super) reasoning: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub(super) struct ResponseErrorBody {
    pub(super) code: String,
    pub(super) message: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub(super) struct IncompleteDetails {
    pub(super) reason: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct ReasoningEcho {
    effort: Option<String>,
    summary: Option<()>,
    context: Option<()>,
}

#[derive(Serialize, Deserialize, Clone)]
pub(super) struct InputTokensDetails {
    pub(super) cached_tokens: usize,
    pub(super) cache_write_tokens: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub(super) struct OutputTokensDetails {
    pub(super) reasoning_tokens: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub(super) struct ResponseUsageBody {
    pub(super) input_tokens: usize,
    pub(super) input_tokens_details: InputTokensDetails,
    pub(super) output_tokens: usize,
    pub(super) output_tokens_details: OutputTokensDetails,
    pub(super) total_tokens: usize,
}

#[derive(Serialize, Deserialize, Clone)]
pub(super) struct ReasoningText {
    #[serde(rename = "type")]
    kind: String,
    text: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub(super) struct ReasoningItem {
    id: String,
    status: String,
    summary: Vec<()>,
    content: Vec<ReasoningText>,
}

#[derive(Serialize, Deserialize, Clone)]
pub(super) struct OutputTextPart {
    #[serde(rename = "type")]
    kind: String,
    text: String,
    annotations: Vec<()>,
    logprobs: Vec<()>,
}

#[derive(Serialize, Deserialize, Clone)]
pub(super) struct MessageItem {
    id: String,
    status: String,
    role: String,
    content: Vec<OutputTextPart>,
}

#[derive(Serialize, Deserialize, Clone)]
pub(super) struct FunctionCallItem {
    id: String,
    status: String,
    call_id: String,
    name: String,
    arguments: String,
}

/// One typed output item.
#[derive(Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
pub(super) enum ResponseItem {
    #[serde(rename = "reasoning")]
    Reasoning(ReasoningItem),
    #[serde(rename = "message")]
    Message(MessageItem),
    #[serde(rename = "function_call")]
    FunctionCall(FunctionCallItem),
}

/// Complete Responses object.
#[derive(Serialize, Deserialize, Clone)]
pub struct ResponseBody {
    pub(super) id: String,
    object: String,
    created_at: u64,
    pub(super) status: String,
    pub(super) completed_at: Option<u64>,
    pub(super) error: Option<ResponseErrorBody>,
    pub(super) incomplete_details: Option<IncompleteDetails>,
    instructions: Option<String>,
    max_output_tokens: Option<usize>,
    model: String,
    output: Vec<ResponseItem>,
    parallel_tool_calls: bool,
    previous_response_id: Option<String>,
    reasoning: ReasoningEcho,
    store: bool,
    temperature: f32,
    text: TextConfigEcho,
    tool_choice: ToolChoiceEcho,
    tools: Vec<FunctionToolEcho>,
    top_p: f32,
    truncation: String,
    pub(super) usage: Option<ResponseUsageBody>,
    user: Option<String>,
    metadata: BTreeMap<String, String>,
    access_programs: Option<()>,
    background: bool,
    conversation: Option<()>,
    service_tier: String,
}

impl ResponseBody {
    /// The output items in response order.
    #[must_use]
    pub(super) fn output(&self) -> &[ResponseItem] {
        &self.output
    }
}

/// Everything needed to serialize a response at one point in its lifecycle.
pub(super) struct Snapshot<'a> {
    pub(super) request: RequestId,
    pub(super) created: u64,
    pub(super) completed_at: Option<u64>,
    pub(super) model: &'a str,
    pub(super) status: &'a str,
    pub(super) incomplete: Option<IncompleteDetails>,
    pub(super) error: Option<ResponseErrorBody>,
    pub(super) output: &'a [ResponseItem],
    pub(super) spec: &'a Spec,
    pub(super) temperature: f32,
    pub(super) top_p: f32,
    pub(super) usage: Option<&'a Usage>,
}

impl Snapshot<'_> {
    /// Build the typed response object.
    pub(super) fn value(&self) -> ResponseBody {
        ResponseBody {
            id: self
                .spec
                .response_id
                .clone()
                .unwrap_or_else(|| response_id(self.request)),
            object: "response".to_string(),
            created_at: self.created,
            status: self.status.to_string(),
            completed_at: self.completed_at,
            error: self.error.clone(),
            incomplete_details: self.incomplete.clone(),
            instructions: self.spec.instructions.clone(),
            max_output_tokens: self.spec.max_output_tokens,
            model: self.model.to_string(),
            output: self.output.to_vec(),
            parallel_tool_calls: self.spec.parallel_tool_calls,
            previous_response_id: self.spec.previous_response_id.clone(),
            reasoning: reasoning_echo(self.spec),
            store: self.spec.store,
            temperature: self.temperature,
            text: self.spec.text.clone(),
            tool_choice: self.spec.tool_choice.clone(),
            tools: self.spec.tools.clone(),
            top_p: self.top_p,
            truncation: "disabled".to_string(),
            usage: self.usage.map(usage_body),
            user: self.spec.user.clone(),
            metadata: self.spec.metadata.clone(),
            access_programs: None,
            background: self.spec.background,
            conversation: None,
            service_tier: "default".to_string(),
        }
    }
}

fn reasoning_echo(spec: &Spec) -> ReasoningEcho {
    ReasoningEcho {
        effort: spec.reasoning_effort.clone(),
        summary: None,
        context: None,
    }
}

const fn usage_body(usage: &Usage) -> ResponseUsageBody {
    ResponseUsageBody {
        input_tokens: usage.input,
        input_tokens_details: InputTokensDetails {
            cached_tokens: 0,
            cache_write_tokens: 0,
        },
        output_tokens: usage.output,
        output_tokens_details: OutputTokensDetails {
            reasoning_tokens: usage.reasoning,
        },
        total_tokens: usage.input + usage.output,
    }
}

/// The response identifier derived from the request identity, stable across streamed events.
pub(super) fn response_id(request: RequestId) -> String {
    format!("resp_{}", request.get())
}

/// Publish the completed output items in the order the parser reports them.
///
/// `status` is the response's terminal status: a message cut off by the output budget keeps the
/// `incomplete` status, while reasoning and fully parsed calls are complete on their own.
pub(super) fn items(namespace: &str, parsed: &ParsedOutput, status: &str) -> Vec<ResponseItem> {
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
pub(super) fn output_text_part(text: &str) -> OutputTextPart {
    OutputTextPart {
        kind: "output_text".to_string(),
        text: text.to_string(),
        annotations: Vec::new(),
        logprobs: Vec::new(),
    }
}

/// A reasoning item carrying the generated reasoning as reasoning text content.
pub(super) fn reasoning_item(
    namespace: &str,
    position: usize,
    text: &str,
    status: &str,
) -> ResponseItem {
    let content = if text.is_empty() {
        Vec::new()
    } else {
        vec![ReasoningText {
            kind: "reasoning_text".to_string(),
            text: text.to_string(),
        }]
    };
    ResponseItem::Reasoning(ReasoningItem {
        id: format!("rs_{namespace}_{position}"),
        status: status.to_string(),
        summary: Vec::new(),
        content,
    })
}

/// An assistant message item. An in-progress item publishes no content part yet.
pub(super) fn message_item(
    namespace: &str,
    position: usize,
    text: &str,
    status: &str,
) -> ResponseItem {
    let content = if text.is_empty() {
        Vec::new()
    } else {
        vec![output_text_part(text)]
    };
    ResponseItem::Message(MessageItem {
        id: format!("msg_{namespace}_{position}"),
        status: status.to_string(),
        role: "assistant".to_string(),
        content,
    })
}

/// One round-trip function call. The `call_id` is the identifier a later output item answers.
pub(super) fn function_call_item(
    namespace: &str,
    index: usize,
    name: &str,
    arguments: &str,
    status: &str,
) -> ResponseItem {
    ResponseItem::FunctionCall(FunctionCallItem {
        id: call_item_id(namespace, index),
        status: status.to_string(),
        call_id: call_id(namespace, index),
        name: name.to_string(),
        arguments: arguments.to_string(),
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
) -> Result<(&'static str, Option<IncompleteDetails>)> {
    let reason = finish_reason(reason)?;
    if parsed.truncated && reason != "length" {
        return Err(infer_core::Error::new(
            infer_core::ErrorCode::Backend,
            "model output ended inside a structured region",
        ));
    }
    if reason == "length" {
        Ok((
            "incomplete",
            Some(IncompleteDetails {
                reason: "max_output_tokens".to_string(),
            }),
        ))
    } else {
        Ok(("completed", None))
    }
}

#[cfg(test)]
#[path = "../../../../tests/unit/http_responses_output.rs"]
mod tests;
