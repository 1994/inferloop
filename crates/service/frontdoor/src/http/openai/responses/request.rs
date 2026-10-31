//! Wire form, validation and conversion of an `OpenAI` Responses request.
//!
//! The Responses protocol describes the same generation the chat route runs, with a different
//! request and output vocabulary. Validation therefore happens here, once, and the result is the
//! internal [`GenerationRequest`] the existing admission and encoding path already understands.
use super::super::request::{
    GenerationRequest, WireMessage, WireTool, WireToolCall, WireToolCallFunction, WireToolChoice,
    WireToolFunction, WireToolName,
};
use infer_core::{Error, Result};
use serde::Deserialize;
use serde_json::{Map, Value, json};

/// Metadata entries one request may attach.
const MAX_METADATA_ENTRIES: usize = 16;
/// Longest metadata key, as the published contract allows.
const MAX_METADATA_KEY_BYTES: usize = 64;
/// Longest metadata value, as the published contract allows.
const MAX_METADATA_VALUE_BYTES: usize = 512;
/// Longest `prompt_cache_key` the contract accepts.
const MAX_CACHE_KEY_BYTES: usize = 64;
/// Longest `safety_identifier` the contract accepts.
const MAX_SAFETY_IDENTIFIER_BYTES: usize = 128;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ResponsesRequest {
    pub model: String,
    /// A string or a list of input items.
    pub input: Option<Value>,
    pub instructions: Option<Value>,
    pub max_output_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    pub store: Option<bool>,
    pub tools: Option<Vec<ResponsesTool>>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub reasoning: Option<ReasoningRequest>,
    pub text: Option<TextRequest>,
    pub metadata: Option<Map<String, Value>>,
    pub previous_response_id: Option<String>,
    pub background: Option<bool>,
    pub include: Option<Vec<String>>,
    pub truncation: Option<String>,
    pub service_tier: Option<String>,
    pub user: Option<String>,
    pub safety_identifier: Option<String>,
    pub prompt_cache_key: Option<String>,
    pub prompt_cache_retention: Option<String>,
    pub prompt_cache_options: Option<Value>,
    pub top_logprobs: Option<usize>,
    pub max_tool_calls: Option<usize>,
    pub prompt: Option<Value>,
    pub conversation: Option<Value>,
    pub context_management: Option<Value>,
    pub moderation: Option<Value>,
    pub access_programs: Option<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StreamOptions {
    pub include_obfuscation: Option<bool>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ReasoningRequest {
    pub effort: Option<String>,
    pub summary: Option<String>,
    pub generate_summary: Option<String>,
    pub context: Option<String>,
    pub mode: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TextRequest {
    pub format: Option<Value>,
    pub verbosity: Option<String>,
}

/// One declared tool. Only the `function` family has an execution path here; the remaining
/// current tool families are parsed enough to reject them explicitly.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ResponsesTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub name: Option<String>,
    pub description: Option<String>,
    pub parameters: Option<Value>,
    pub strict: Option<bool>,
    pub output_schema: Option<Value>,
    pub defer_loading: Option<bool>,
    #[serde(rename = "allowed_callers")]
    pub allowed_callers: Option<Value>,
    #[serde(rename = "async")]
    pub asynchronous: Option<bool>,
}

/// The Responses-specific values a generated response echoes and prices.
#[derive(Clone)]
pub(super) struct Spec {
    pub(super) instructions: Option<String>,
    /// Declared tools in Responses shape, for the response object.
    pub(super) tools: Vec<Value>,
    /// The normalized `tool_choice` value for the response object.
    pub(super) tool_choice: Value,
    pub(super) parallel_tool_calls: bool,
    /// The requested reasoning effort, echoed when present.
    pub(super) reasoning_effort: Option<String>,
    pub(super) text: Value,
    pub(super) metadata: Map<String, Value>,
    pub(super) max_output_tokens: Option<usize>,
    pub(super) user: Option<String>,
}

/// A validated request split into the internal generation and the response metadata.
pub(super) struct Prepared {
    pub(super) generation: GenerationRequest,
    pub(super) spec: Spec,
}

impl ResponsesRequest {
    /// Validate the request and convert it into the internal chat-shaped generation.
    ///
    /// # Errors
    /// Rejects malformed input items and any option this deployment cannot honor. Unsupported
    /// options are refused rather than ignored.
    pub(super) fn prepare(self) -> Result<Prepared> {
        self.reject_unimplemented()?;
        self.validate_hints()?;
        let instructions = instructions(self.instructions)?;
        let messages = input_messages(
            self.input
                .ok_or_else(|| Error::invalid("responses requires an input"))?,
            instructions.as_deref(),
        )?;
        let stream = self.stream.unwrap_or(false);
        if self.stream_options.is_some() && !stream {
            return Err(Error::invalid("stream_options requires stream=true"));
        }
        let (tools, declared) = declared_tools(self.tools)?;
        let tool_choice = tool_choice(self.tool_choice.as_ref())?;
        let reasoning_effort = self
            .reasoning
            .as_ref()
            .and_then(|reasoning| reasoning.effort.clone());
        let text = text_config(self.text.as_ref())?;
        let metadata = metadata(self.metadata)?;
        let spec = Spec {
            instructions,
            tool_choice: tool_choice.echo(),
            tools,
            parallel_tool_calls: self.parallel_tool_calls.unwrap_or(true),
            reasoning_effort: reasoning_effort.clone(),
            text,
            metadata,
            max_output_tokens: self.max_output_tokens,
            user: self.user.clone(),
        };
        let generation = GenerationRequest {
            model: self.model,
            messages: Some(messages),
            prompt: None,
            max_tokens: None,
            max_completion_tokens: self.max_output_tokens,
            temperature: self.temperature,
            seed: None,
            stream: Some(stream),
            n: Some(1),
            top_p: self.top_p,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            repetition_penalty: None,
            enable_thinking: thinking_override(reasoning_effort.as_deref())?,
            tools: (!declared.is_empty()).then_some(declared),
            tool_choice: Some(tool_choice.wire()),
            parallel_tool_calls: self.parallel_tool_calls,
            stream_options: None,
            mm_processor_kwargs: None,
        };
        Ok(Prepared { generation, spec })
    }

    /// Refuse current fields whose behavior this deployment does not implement.
    fn reject_unimplemented(&self) -> Result<()> {
        if self.previous_response_id.is_some() {
            return Err(Error::unsupported(
                "previous_response_id needs server-side response storage; this deployment does not store responses",
            ));
        }
        if present(self.conversation.as_ref()) {
            return Err(Error::unsupported(
                "conversation needs server-side response storage",
            ));
        }
        if self.background == Some(true) {
            return Err(Error::unsupported("background responses are not supported"));
        }
        if self.store == Some(true) {
            return Err(Error::unsupported(
                "store=true needs server-side response storage; this deployment does not store responses",
            ));
        }
        if self
            .include
            .as_ref()
            .is_some_and(|include| !include.is_empty())
        {
            return Err(Error::unsupported("include is not supported"));
        }
        match self.truncation.as_deref() {
            None | Some("disabled") => {}
            Some("auto") => {
                return Err(Error::unsupported(
                    "truncation=auto needs input truncation; this deployment rejects oversized input",
                ));
            }
            Some(_) => return Err(Error::invalid("truncation must be auto or disabled")),
        }
        match self.service_tier.as_deref() {
            None | Some("auto" | "default") => {}
            Some(_) => {
                return Err(Error::unsupported(
                    "only the default service tier is supported",
                ));
            }
        }
        for (field, value) in [
            ("prompt", &self.prompt),
            ("context_management", &self.context_management),
            ("moderation", &self.moderation),
            ("access_programs", &self.access_programs),
        ] {
            if present(value.as_ref()) {
                return Err(Error::unsupported(format!("{field} is not supported")));
            }
        }
        for (field, set) in [
            ("top_logprobs", self.top_logprobs.is_some()),
            ("max_tool_calls", self.max_tool_calls.is_some()),
        ] {
            if set {
                return Err(Error::unsupported(format!("{field} is not supported")));
            }
        }
        let unsupported_reasoning = self.reasoning.as_ref().is_some_and(|reasoning| {
            reasoning.summary.is_some()
                || reasoning.generate_summary.is_some()
                || reasoning.context.is_some()
                || reasoning.mode.is_some()
        });
        if unsupported_reasoning {
            return Err(Error::unsupported(
                "reasoning summaries and reasoning context are not supported; reasoning text is returned as a reasoning item",
            ));
        }
        if self
            .text
            .as_ref()
            .is_some_and(|text| text.verbosity.is_some())
        {
            return Err(Error::unsupported("text.verbosity is not supported"));
        }
        Ok(())
    }

    /// Validate advisory identifiers that do not change the generation.
    fn validate_hints(&self) -> Result<()> {
        if self
            .prompt_cache_options
            .as_ref()
            .is_some_and(|options| !options.is_object() && !options.is_null())
        {
            return Err(Error::invalid("prompt_cache_options must be an object"));
        }
        if self
            .stream_options
            .as_ref()
            .is_some_and(|options| options.include_obfuscation == Some(true))
        {
            return Err(Error::unsupported(
                "stream obfuscation is not supported; set include_obfuscation to false",
            ));
        }
        if self
            .prompt_cache_key
            .as_ref()
            .is_some_and(|key| key.len() > MAX_CACHE_KEY_BYTES)
        {
            return Err(Error::invalid(format!(
                "prompt_cache_key must be at most {MAX_CACHE_KEY_BYTES} bytes"
            )));
        }
        if self
            .safety_identifier
            .as_ref()
            .is_some_and(|identifier| identifier.len() > MAX_SAFETY_IDENTIFIER_BYTES)
        {
            return Err(Error::invalid(format!(
                "safety_identifier must be at most {MAX_SAFETY_IDENTIFIER_BYTES} bytes"
            )));
        }
        match self.prompt_cache_retention.as_deref() {
            None | Some("in_memory" | "24h") => {}
            Some(_) => {
                return Err(Error::invalid(
                    "prompt_cache_retention must be in_memory or 24h",
                ));
            }
        }
        Ok(())
    }
}

/// Instructions are a system message in front of the input.
fn instructions(value: Option<Value>) -> Result<Option<String>> {
    match value {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(text)) => Ok(Some(text)),
        Some(Value::Array(_)) => Err(Error::unsupported(
            "instructions as a list of input items is not supported",
        )),
        Some(_) => Err(Error::invalid("instructions must be a string")),
    }
}

/// One complete request's messages: instructions, then the input items.
fn input_messages(input: Value, instructions: Option<&str>) -> Result<Vec<WireMessage>> {
    let mut messages = Vec::new();
    if let Some(text) = instructions.filter(|text| !text.is_empty()) {
        messages.push(message("system", Value::String(text.to_string())));
    }
    match input {
        Value::String(text) => messages.push(message("user", Value::String(text))),
        Value::Array(items) => {
            if items.is_empty() {
                return Err(Error::invalid("input requires at least one item"));
            }
            for (index, item) in items.iter().enumerate() {
                push_item(item, index, &mut messages)?;
            }
        }
        _ => {
            return Err(Error::invalid(
                "input must be a string or a list of input items",
            ));
        }
    }
    if !messages.iter().any(|message| message.role != "system") {
        return Err(Error::invalid(
            "input requires at least one user, assistant or tool item",
        ));
    }
    Ok(messages)
}

fn message(role: &str, content: Value) -> WireMessage {
    WireMessage {
        role: role.to_string(),
        content: Some(content),
        tool_calls: Vec::new(),
        tool_call_id: None,
    }
}

/// Dispatch one input item by its `type`. A missing type is the plain message form.
fn push_item(item: &Value, index: usize, messages: &mut Vec<WireMessage>) -> Result<()> {
    let object = item
        .as_object()
        .ok_or_else(|| Error::invalid(format!("input item {index} must be an object")))?;
    match object.get("type") {
        None | Some(Value::Null) => push_message(object, index, messages),
        Some(Value::String(kind)) => match kind.as_str() {
            "message" => push_message(object, index, messages),
            "function_call" => push_function_call(object, index, messages),
            "function_call_output" => push_function_call_output(object, index, messages),
            "reasoning" => check_reasoning(object, index),
            "item_reference" => Err(Error::unsupported(
                "item_reference needs server-side response storage",
            )),
            other => Err(Error::unsupported(format!(
                "input item type {other:?} is not supported"
            ))),
        },
        Some(_) => Err(Error::invalid(format!(
            "input item {index} type must be a string"
        ))),
    }
}

fn push_message(
    object: &Map<String, Value>,
    index: usize,
    messages: &mut Vec<WireMessage>,
) -> Result<()> {
    check_keys(
        object,
        &["type", "role", "content", "id", "status", "phase"],
        index,
    )?;
    let role = match string_field(object, "role", index)? {
        "developer" => "system",
        role @ ("system" | "user" | "assistant") => role,
        _ => {
            return Err(Error::invalid(format!(
                "input item {index} role must be user, assistant, system or developer"
            )));
        }
    };
    let content = message_content(object.get("content"), index)?;
    messages.push(message(role, content));
    Ok(())
}

/// A message's content as chat wire content: a string, or text and media parts.
fn message_content(content: Option<&Value>, index: usize) -> Result<Value> {
    match content {
        None | Some(Value::Null) => Ok(Value::String(String::new())),
        Some(Value::String(text)) => Ok(Value::String(text.clone())),
        Some(Value::Array(parts)) => content_parts(parts, index),
        Some(_) => Err(Error::invalid(format!(
            "input item {index} content must be a string, null or a list of parts"
        ))),
    }
}

fn content_parts(parts: &[Value], item: usize) -> Result<Value> {
    let mut converted = Vec::with_capacity(parts.len());
    let mut text = String::new();
    let mut has_media = false;
    for (index, part) in parts.iter().enumerate() {
        let object = part.as_object().ok_or_else(|| {
            Error::invalid(format!(
                "input item {item} content part {index} must be an object"
            ))
        })?;
        let kind = string_field(object, "type", index)?;
        match kind {
            "output_text" => {
                // A message item copied from `response.output` carries these two fields.
                check_keys(object, &["type", "text", "annotations", "logprobs"], index)?;
                let value = string_field(object, "text", index)?;
                text.push_str(value);
                converted.push(json!({"type":"text","text":value}));
            }
            "input_text" | "text" => {
                check_keys(object, &["type", "text"], index)?;
                let value = string_field(object, "text", index)?;
                text.push_str(value);
                converted.push(json!({"type":"text","text":value}));
            }
            "input_image" => {
                has_media = true;
                converted.push(image_part(object, index)?);
            }
            "input_file" | "input_audio" | "input_video" => {
                return Err(Error::unsupported(format!(
                    "input content type {kind:?} is not supported"
                )));
            }
            other => {
                return Err(Error::unsupported(format!(
                    "input content type {other:?} is not supported"
                )));
            }
        }
    }
    if has_media {
        Ok(Value::Array(converted))
    } else {
        Ok(Value::String(text))
    }
}

/// Normalize a Responses image part into the chat image part the media path accepts.
fn image_part(object: &Map<String, Value>, index: usize) -> Result<Value> {
    check_keys(object, &["type", "image_url", "detail", "file_id"], index)?;
    if object.get("file_id").is_some_and(|value| !value.is_null()) {
        return Err(Error::unsupported("input_image.file_id is not supported"));
    }
    let url = match object.get("image_url") {
        Some(Value::String(url)) => url.as_str(),
        Some(Value::Object(detail)) => detail
            .get("url")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::invalid("input_image.image_url requires a URL"))?,
        _ => return Err(Error::invalid("input_image requires an image_url")),
    };
    let detail = match object.get("detail") {
        None | Some(Value::Null) => None,
        Some(Value::String(detail)) => Some(detail.as_str()),
        Some(_) => return Err(Error::invalid("input_image.detail must be a string")),
    };
    let image = detail.map_or_else(
        || json!({"url":url}),
        |detail| json!({"url":url, "detail":detail}),
    );
    Ok(json!({"type":"image_url","image_url":image}))
}

fn push_function_call(
    object: &Map<String, Value>,
    index: usize,
    messages: &mut Vec<WireMessage>,
) -> Result<()> {
    check_keys(
        object,
        &[
            "type",
            "id",
            "call_id",
            "name",
            "arguments",
            "status",
            "caller",
        ],
        index,
    )?;
    if object.get("caller").is_some_and(|caller| !caller.is_null()) {
        return Err(Error::unsupported("function_call.caller is not supported"));
    }
    let call_id = string_field(object, "call_id", index)?;
    let name = string_field(object, "name", index)?;
    if call_id.trim().is_empty() || name.trim().is_empty() {
        return Err(Error::invalid(format!(
            "function_call item {index} requires a call_id and name"
        )));
    }
    let arguments = string_field(object, "arguments", index)?;
    let call = WireToolCall {
        id: call_id.to_string(),
        kind: Some("function".to_string()),
        function: WireToolCallFunction {
            name: name.to_string(),
            arguments: arguments.to_string(),
        },
    };
    // Calls belong to the assistant turn that produced them. A preceding assistant message is
    // that turn, even when it carried content before the call.
    if let Some(last) = messages.last_mut().filter(|last| last.role == "assistant") {
        last.tool_calls.push(call);
    } else {
        let mut assistant = message("assistant", Value::String(String::new()));
        assistant.tool_calls.push(call);
        messages.push(assistant);
    }
    Ok(())
}

fn push_function_call_output(
    object: &Map<String, Value>,
    index: usize,
    messages: &mut Vec<WireMessage>,
) -> Result<()> {
    check_keys(
        object,
        &[
            "type", "id", "call_id", "output", "status", "name", "caller",
        ],
        index,
    )?;
    if object.get("caller").is_some_and(|caller| !caller.is_null()) {
        return Err(Error::unsupported(
            "function_call_output.caller is not supported",
        ));
    }
    let call_id = string_field(object, "call_id", index)?;
    if call_id.trim().is_empty() {
        return Err(Error::invalid(format!(
            "function_call_output item {index} requires a call_id"
        )));
    }
    let output = match object.get("output") {
        Some(Value::String(output)) => output.clone(),
        Some(Value::Array(parts)) => output_text(parts, index)?,
        Some(_) => {
            return Err(Error::invalid(format!(
                "function_call_output item {index} output must be a string or text parts"
            )));
        }
        None => {
            return Err(Error::invalid(format!(
                "function_call_output item {index} requires an output"
            )));
        }
    };
    let mut result = message("tool", Value::String(output));
    result.tool_call_id = Some(call_id.to_string());
    messages.push(result);
    Ok(())
}

fn output_text(parts: &[Value], item: usize) -> Result<String> {
    let mut text = String::new();
    for (index, part) in parts.iter().enumerate() {
        let object = part.as_object().ok_or_else(|| {
            Error::invalid(format!(
                "function_call_output item {item} output part {index} must be an object"
            ))
        })?;
        match string_field(object, "type", index)? {
            "input_text" | "output_text" | "text" => {
                check_keys(object, &["type", "text"], index)?;
                text.push_str(string_field(object, "text", index)?);
            }
            other => {
                return Err(Error::unsupported(format!(
                    "function_call_output content type {other:?} is not supported"
                )));
            }
        }
    }
    Ok(text)
}

/// A reasoning item carries no prompt content for this model family. It is accepted for a full
/// output round trip and skipped, because the package templates have no reasoning-history slot.
fn check_reasoning(object: &Map<String, Value>, index: usize) -> Result<()> {
    check_keys(
        object,
        &[
            "type",
            "id",
            "summary",
            "content",
            "encrypted_content",
            "status",
        ],
        index,
    )?;
    for field in ["summary", "content"] {
        if object
            .get(field)
            .is_some_and(|value| !value.is_array() && !value.is_null())
        {
            return Err(Error::invalid(format!(
                "reasoning item {index} {field} must be a list"
            )));
        }
    }
    if object
        .get("encrypted_content")
        .is_some_and(|value| !value.is_string() && !value.is_null())
    {
        return Err(Error::invalid(format!(
            "reasoning item {index} encrypted_content must be a string"
        )));
    }
    Ok(())
}

/// Convert the declared tools into the internal form and the response echo.
fn declared_tools(tools: Option<Vec<ResponsesTool>>) -> Result<(Vec<Value>, Vec<WireTool>)> {
    let mut echo = Vec::new();
    let mut wire = Vec::new();
    for (index, tool) in tools.unwrap_or_default().into_iter().enumerate() {
        if tool.kind != "function" {
            return Err(Error::unsupported(format!(
                "tool type {:?} is not supported; only function tools are supported",
                tool.kind
            )));
        }
        if tool
            .output_schema
            .as_ref()
            .is_some_and(|value| !value.is_null())
            || tool.defer_loading == Some(true)
            || tool
                .allowed_callers
                .as_ref()
                .is_some_and(|value| !value.is_null())
            || tool.asynchronous == Some(true)
        {
            return Err(Error::unsupported(
                "function tool output_schema, defer_loading, allowed_callers and async are not supported",
            ));
        }
        let name = tool.name.ok_or_else(|| {
            Error::invalid(format!("declared tool {index} requires a function name"))
        })?;
        let strict = tool.strict;
        echo.push(json!({
            "type": "function",
            "name": name,
            "description": tool.description,
            "parameters": tool.parameters,
            "strict": strict.unwrap_or(false),
        }));
        wire.push(WireTool {
            kind: "function".to_string(),
            function: WireToolFunction {
                name,
                description: tool.description,
                parameters: tool.parameters,
                strict,
            },
        });
    }
    Ok((echo, wire))
}

/// The request's tool choice after validation.
enum Choice {
    /// Explicitly disabled by `"none"`.
    Disabled,
    /// Omitted or `"auto"`.
    Auto,
    Required,
    Named(String),
}

impl Choice {
    fn echo(&self) -> Value {
        match self {
            Self::Named(name) => json!({"type":"function","name":name}),
            Self::Disabled => Value::String("none".to_string()),
            Self::Auto => Value::String("auto".to_string()),
            Self::Required => Value::String("required".to_string()),
        }
    }
    fn wire(&self) -> WireToolChoice {
        match self {
            Self::Disabled => WireToolChoice::Mode("none".to_string()),
            Self::Auto => WireToolChoice::Mode("auto".to_string()),
            Self::Required => WireToolChoice::Mode("required".to_string()),
            Self::Named(name) => WireToolChoice::Named {
                kind: "function".to_string(),
                function: WireToolName { name: name.clone() },
            },
        }
    }
}

/// Validate `tool_choice`. The declaration itself is checked together with the tools by the
/// shared chat policy.
fn tool_choice(value: Option<&Value>) -> Result<Choice> {
    match value {
        None | Some(Value::Null) => Ok(Choice::Auto),
        Some(Value::String(mode)) => match mode.as_str() {
            "none" => Ok(Choice::Disabled),
            "auto" => Ok(Choice::Auto),
            "required" => Ok(Choice::Required),
            _ => Err(Error::invalid("unknown tool_choice mode")),
        },
        Some(Value::Object(object)) => {
            check_keys(object, &["type", "name", "allowed_tools"], 0)?;
            if object
                .get("allowed_tools")
                .is_some_and(|allowed| !allowed.is_null())
            {
                return Err(Error::unsupported(
                    "tool_choice.allowed_tools is not supported",
                ));
            }
            if string_field(object, "type", 0)? != "function" {
                return Err(Error::unsupported(
                    "only a named function tool_choice is supported",
                ));
            }
            Ok(Choice::Named(string_field(object, "name", 0)?.to_string()))
        }
        Some(_) => Err(Error::invalid(
            "tool_choice must be a mode string or a function object",
        )),
    }
}

fn thinking_override(effort: Option<&str>) -> Result<Option<bool>> {
    match effort {
        None => Ok(None),
        Some("none" | "minimal") => Ok(Some(false)),
        Some("low" | "medium" | "high" | "xhigh" | "max") => Ok(Some(true)),
        Some(_) => Err(Error::invalid(
            "reasoning.effort must be none, minimal, low, medium, high, xhigh or max",
        )),
    }
}

fn text_config(value: Option<&TextRequest>) -> Result<Value> {
    let Some(text) = value else {
        return Ok(json!({"format":{"type":"text"}}));
    };
    let format = match text.format.as_ref() {
        None | Some(Value::Null) => json!({"type":"text"}),
        Some(Value::Object(object)) => {
            check_keys(
                object,
                &["type", "name", "schema", "strict", "description"],
                0,
            )?;
            match string_field(object, "type", 0)? {
                "text" => json!({"type":"text"}),
                "json_object" | "json_schema" => {
                    return Err(Error::unsupported(
                        "structured text formats are not supported; only text is available",
                    ));
                }
                _ => return Err(Error::invalid("unknown text.format type")),
            }
        }
        Some(_) => return Err(Error::invalid("text.format must be an object")),
    };
    Ok(json!({"format":format}))
}

fn metadata(value: Option<Map<String, Value>>) -> Result<Map<String, Value>> {
    let Some(metadata) = value else {
        return Ok(Map::new());
    };
    if metadata.len() > MAX_METADATA_ENTRIES {
        return Err(Error::invalid(format!(
            "metadata allows at most {MAX_METADATA_ENTRIES} entries"
        )));
    }
    for (key, value) in &metadata {
        if key.len() > MAX_METADATA_KEY_BYTES {
            return Err(Error::invalid(format!(
                "metadata keys must be at most {MAX_METADATA_KEY_BYTES} bytes"
            )));
        }
        let value = value
            .as_str()
            .ok_or_else(|| Error::invalid("metadata values must be strings"))?;
        if value.len() > MAX_METADATA_VALUE_BYTES {
            return Err(Error::invalid(format!(
                "metadata values must be at most {MAX_METADATA_VALUE_BYTES} bytes"
            )));
        }
    }
    Ok(metadata)
}

fn string_field<'a>(object: &'a Map<String, Value>, field: &str, index: usize) -> Result<&'a str> {
    object.get(field).and_then(Value::as_str).ok_or_else(|| {
        Error::invalid(format!(
            "input item {index} requires a string {field} field"
        ))
    })
}

/// Whether an optional JSON field was actually supplied.
fn present(value: Option<&Value>) -> bool {
    value.is_some_and(|value| !value.is_null())
}

fn check_keys(object: &Map<String, Value>, allowed: &[&str], index: usize) -> Result<()> {
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(Error::invalid(format!(
            "input item {index} has unknown field {key:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "../../../../tests/unit/http_responses_request.rs"]
mod tests;
