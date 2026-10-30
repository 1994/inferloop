use crate::constants::GENERATED_TOKEN_STAGING_BYTES;
use crate::constants::TEXT_STAGING_EXPANSION;
use crate::constants::TEXT_STAGING_OVERHEAD_BYTES;
use crate::media::MediaRequest;
use infer_core::{Error, Result};
use infer_ir::Modality;
use infer_models::{ChatMessage, ToolCall};
use serde_json::Value;
use std::collections::BTreeSet;
use std::result::Result as StdResult;

/// Maximum chat messages accepted in one generation request.
const MAX_CHAT_MESSAGES: usize = 256;
/// Default completion-token budget when a request omits an explicit limit.
const DEFAULT_MAX_COMPLETION_TOKENS: usize = 16;
/// Largest number of function tools accepted in one request.
const MAX_TOOLS: usize = 128;
/// `OpenAI` function names contain at most 64 ASCII characters.
const MAX_TOOL_NAME_BYTES: usize = 64;

/// The generated-token budget a request asked for, before the service cap and remaining
/// context apply.
pub(super) struct RequestedOutput {
    pub(super) tokens: usize,
    /// True when the client set `max_tokens` or `max_completion_tokens` explicitly.
    pub(super) explicit: bool,
}

/// Tool calling enabled by one request.
///
/// Strict schemas, required/named choices and single-call requests carry a generation matcher.
#[derive(Debug)]
pub(super) enum ToolPolicy {
    /// No `tools`, or `tool_choice: "none"`. Structured calls are not published.
    Disabled,
    /// Calls are parsed from the completed response and published when they name a declared tool.
    Auto {
        declared: BTreeSet<String>,
        /// The declared tools as the chat template expects them.
        template: Value,
        constraint: Option<infer_models::ToolConstraints>,
    },
}

impl ToolPolicy {
    pub(super) fn constraint(&self) -> Option<infer_models::ToolConstraints> {
        match self {
            Self::Disabled => None,
            Self::Auto { constraint, .. } => constraint.clone(),
        }
    }

    /// Declared names for the parser, if any call may be published.
    pub(super) const fn declared(&self) -> Option<&BTreeSet<String>> {
        match self {
            Self::Disabled => None,
            Self::Auto { declared, .. } => Some(declared),
        }
    }
    /// Declared tools for the chat template.
    pub(super) fn template(&self) -> Option<Value> {
        match self {
            Self::Disabled => None,
            Self::Auto { template, .. } => Some(template.clone()),
        }
    }
    /// Whether a completed response should be parsed for calls.
    pub(super) const fn parses(&self) -> bool {
        matches!(self, Self::Auto { .. })
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GenerationRequest {
    pub model: String,
    pub messages: Option<Vec<WireMessage>>,
    pub prompt: Option<String>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub seed: Option<u64>,
    pub stream: Option<bool>,
    pub n: Option<usize>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    pub enable_thinking: Option<bool>,
    pub tools: Option<Vec<WireTool>>,
    pub tool_choice: Option<WireToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    pub mm_processor_kwargs: Option<crate::ImageProcessingOptions>,
}

/// Streaming modifiers accepted alongside `stream=true`.
#[derive(Debug, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

impl GenerationRequest {
    pub(super) fn validate(&self, chat: bool) -> Result<RequestedOutput> {
        if let Some(options) = &self.mm_processor_kwargs {
            options.validate()?;
        }
        if self.model.is_empty() {
            return Err(Error::invalid("model must not be empty"));
        }
        if chat {
            if self.prompt.is_some() || self.messages.as_ref().is_none_or(Vec::is_empty) {
                return Err(Error::invalid(
                    "chat requires nonempty messages and no prompt",
                ));
            }
            if self.messages.as_ref().is_some_and(|messages| {
                messages.len() > MAX_CHAT_MESSAGES
                    || messages.iter().any(|m| {
                        !["system", "user", "assistant", "tool"].contains(&m.role.as_str())
                    })
            }) {
                return Err(Error::invalid(
                    "at most 256 text messages with system/user/assistant/tool roles are supported",
                ));
            }
        } else if self.messages.is_some()
            || self.prompt.as_ref().is_none_or(String::is_empty)
            || self.max_completion_tokens.is_some()
            || self.tools.is_some()
            || self.tool_choice.is_some()
            || self.parallel_tool_calls.is_some()
            || self.mm_processor_kwargs.is_some()
        {
            return Err(Error::invalid(
                "completions require a nonempty string prompt; use max_tokens",
            ));
        }
        if self.max_tokens.is_some() && self.max_completion_tokens.is_some() {
            return Err(Error::invalid("provide only one token limit"));
        }
        let explicit = self.max_tokens.is_some() || self.max_completion_tokens.is_some();
        let tokens = self
            .max_completion_tokens
            .or(self.max_tokens)
            .unwrap_or(DEFAULT_MAX_COMPLETION_TOKENS);
        if tokens == 0
            || self.n == Some(0)
            || self
                .temperature
                .is_some_and(|t| !t.is_finite() || !(0.0..=2.0).contains(&t))
            || self
                .top_p
                .is_some_and(|p| !p.is_finite() || !(0.0..=1.0).contains(&p))
        {
            return Err(Error::invalid("invalid generation parameters"));
        }
        if self.n.is_some_and(|n| n != 1) {
            return Err(Error::unsupported("only n=1 is supported"));
        }
        if self.stream_options.is_some() && self.stream != Some(true) {
            return Err(Error::invalid("stream_options requires stream=true"));
        }
        Ok(RequestedOutput { tokens, explicit })
    }

    /// Resolve the request's tool-calling mode.
    ///
    /// # Errors
    /// Rejects a malformed declaration or a
    /// `tool_choice` that names a tool the request did not declare.
    pub(super) fn tool_policy(&self, chat: bool) -> Result<ToolPolicy> {
        if !chat {
            return Ok(ToolPolicy::Disabled);
        }
        let declared_tools = self.tools.as_deref().unwrap_or_default();
        if declared_tools.len() > MAX_TOOLS {
            return Err(Error::invalid(format!(
                "at most {MAX_TOOLS} tools may be declared"
            )));
        }
        let mut names = BTreeSet::new();
        let mut template = Vec::with_capacity(declared_tools.len());
        for tool in declared_tools {
            if tool.kind != "function" {
                return Err(Error::unsupported("only function tools are supported"));
            }
            let name = &tool.function.name;
            if name.is_empty()
                || name.len() > MAX_TOOL_NAME_BYTES
                || !name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
            {
                return Err(Error::invalid(
                    "tool function name must be 1..64 ASCII letters, digits, underscores or dashes",
                ));
            }
            if !names.insert(name.clone()) {
                return Err(Error::invalid("tool function names must be unique"));
            }
            template.push(serde_json::json!({
                "type": "function",
                "function": tool.function,
            }));
        }
        let parallel = self.parallel_tool_calls.unwrap_or(true);
        let mut required = false;
        let mut selected = None;
        let enabled = match &self.tool_choice {
            None => !declared_tools.is_empty(),
            Some(WireToolChoice::Mode(mode)) => match mode.as_str() {
                "auto" => !declared_tools.is_empty(),
                "none" => false,
                "required" => {
                    required = true;
                    true
                }
                _ => return Err(Error::invalid("unknown tool_choice mode")),
            },
            Some(WireToolChoice::Named { kind, function }) => {
                if kind != "function" || !names.contains(&function.name) {
                    return Err(Error::invalid(
                        "tool_choice must name a declared function tool",
                    ));
                }
                selected = Some(function.name.clone());
                true
            }
        };
        if !enabled {
            return Ok(ToolPolicy::Disabled);
        }
        if names.is_empty() && required {
            return Err(Error::invalid("required tool choice needs tools"));
        }
        let template = Value::Array(template);
        let constrained = required
            || selected.is_some()
            || !parallel
            || declared_tools
                .iter()
                .any(|tool| tool.function.strict == Some(true));
        let constraint = constrained.then(|| infer_models::ToolConstraints {
            tools: template.clone(),
            required,
            named: selected,
            parallel,
        });
        Ok(ToolPolicy::Auto {
            declared: names,
            template,
            constraint,
        })
    }

    /// Convert the wire messages into the internal history, checking result association.
    ///
    /// # Errors
    /// Rejects unsupported content parts, an assistant call without an identifier, a duplicated
    /// call identifier, and a `tool` result that does not answer an earlier assistant call.
    pub(super) fn chat_messages(&mut self) -> Result<(Vec<ChatMessage>, Vec<MediaRequest>)> {
        let wire = self.messages.take().unwrap_or_default();
        let mut call_ids = BTreeSet::new();
        let mut out = Vec::with_capacity(wire.len());
        let mut media_requests = Vec::new();
        for mut message in wire {
            let (text, media) = message.take_rendered()?;
            if !media.is_empty() && message.role != "user" {
                return Err(Error::invalid("only a user message may carry media"));
            }
            media_requests.extend(media);
            let mut history = ChatMessage::new(message.role.as_str(), text);
            if !message.tool_calls.is_empty() && message.role != "assistant" {
                return Err(Error::invalid(
                    "only an assistant message may carry tool_calls",
                ));
            }
            for call in &message.tool_calls {
                if call.id.trim().is_empty() {
                    return Err(Error::invalid("tool call id must not be empty"));
                }
                if call.function.name.trim().is_empty() {
                    return Err(Error::invalid("tool call name must not be empty"));
                }
                if call.kind.as_deref().is_some_and(|kind| kind != "function") {
                    return Err(Error::unsupported("only function tool calls are supported"));
                }
                if !call_ids.insert(call.id.clone()) {
                    return Err(Error::invalid("tool call ids must be unique"));
                }
                history.tool_calls.push(ToolCall {
                    id: call.id.clone(),
                    name: call.function.name.clone(),
                    arguments: call.function.arguments.clone(),
                });
            }
            if let Some(id) = &message.tool_call_id {
                if message.role != "tool" {
                    return Err(Error::invalid("only a tool message may carry tool_call_id"));
                }
                // A result must answer a call the same history contains. History may reference
                // tools that are no longer declared; it just cannot invent an association.
                if !call_ids.contains(id) {
                    return Err(Error::invalid(
                        "tool result must answer an earlier assistant tool call",
                    ));
                }
                history.tool_call_id = Some(id.clone());
            } else if message.role == "tool" {
                return Err(Error::invalid("a tool message requires tool_call_id"));
            }
            out.push(history);
        }
        if out.is_empty() {
            return Err(Error::invalid("chat requires nonempty messages"));
        }
        Ok((out, media_requests))
    }

    pub(super) fn preparation_bytes(&self, generated: usize) -> Result<usize> {
        let text = self
            .messages
            .as_ref()
            .map_or(Some(0), |messages| {
                messages
                    .iter()
                    .try_fold(0usize, |sum, message| sum.checked_add(message.text_len()))
            })
            .and_then(|bytes| bytes.checked_add(self.prompt.as_ref().map_or(0, String::len)))
            .and_then(|bytes| {
                bytes.checked_add(self.tools.as_deref().map_or(0, |tools| {
                    tools.iter().fold(0usize, |sum, tool| {
                        sum.saturating_add(serde_json::to_string(tool).map_or(0, |s| s.len()))
                    })
                }))
            });
        text.and_then(|n| n.checked_mul(TEXT_STAGING_EXPANSION))
            .and_then(|n| n.checked_add(TEXT_STAGING_OVERHEAD_BYTES))
            .and_then(|n| {
                generated
                    .checked_mul(GENERATED_TOKEN_STAGING_BYTES)
                    .and_then(|tail| n.checked_add(tail))
            })
            .ok_or_else(|| Error::invalid("text preparation budget overflow"))
    }
}

/// One assistant or tool message on the wire.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Value>,
    #[serde(default, deserialize_with = "nullable_tool_calls")]
    pub tool_calls: Vec<WireToolCall>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
}

/// SDK message serialization can include `tool_calls: null`.
fn nullable_tool_calls<'de, D>(deserializer: D) -> StdResult<Vec<WireToolCall>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    use serde::Deserialize;
    Ok(Option::<Vec<WireToolCall>>::deserialize(deserializer)?.unwrap_or_default())
}

impl WireMessage {
    /// The message parts in order, validating each one.
    fn parts(&self) -> Result<Vec<WirePart<'_>>> {
        match &self.content {
            None | Some(Value::Null) => Ok(Vec::new()),
            Some(Value::String(text)) => Ok(vec![WirePart::Text(text.as_str())]),
            Some(Value::Array(parts)) => parts
                .iter()
                .enumerate()
                .map(|(index, part)| wire_part(part, index))
                .collect(),
            Some(_) => Err(Error::invalid(
                "message content must be a string, null, or a list of text parts",
            )),
        }
    }
    /// Total text, URL and call bytes used to estimate staging.
    fn text_len(&self) -> usize {
        let content = self.parts().map_or(0, |parts| {
            parts.iter().fold(0usize, |sum, part| match part {
                WirePart::Text(text) => sum.saturating_add(text.len()),
                WirePart::Media(media) => sum.saturating_add(media.url.len()),
            })
        });
        self.tool_calls.iter().fold(content, |sum, call| {
            sum.saturating_add(call.id.len())
                .saturating_add(call.function.name.len())
                .saturating_add(call.function.arguments.len())
        })
    }
    /// Rendered text with media placeholders plus the media requests in content order.
    fn take_rendered(&mut self) -> Result<(String, Vec<MediaRequest>)> {
        let mut text = String::new();
        let mut media = Vec::new();
        let parts = match self.content.take() {
            None | Some(Value::Null) => return Ok((text, media)),
            Some(Value::String(value)) => return Ok((value, media)),
            Some(Value::Array(parts)) => parts,
            Some(_) => {
                return Err(Error::invalid(
                    "message content must be a string, null, or a list of text parts",
                ));
            }
        };
        for (index, mut part) in parts.into_iter().enumerate() {
            let modality = match wire_part(&part, index)? {
                WirePart::Text(value) => {
                    text.push_str(value);
                    continue;
                }
                WirePart::Media(request) => request.modality,
            };
            let field = match modality {
                Modality::Image => "image_url",
                Modality::Video => "video_url",
                _ => return Err(Error::invariant("validated media modality changed")),
            };
            let url = part
                .as_object_mut()
                .and_then(|object| object.get_mut(field))
                .and_then(Value::as_object_mut)
                .and_then(|object| object.remove("url"));
            if let Some(Value::String(url)) = url {
                text.push_str(placeholder(&modality));
                media.push(MediaRequest { modality, url });
            } else {
                return Err(Error::invariant("validated media URL changed"));
            }
        }
        Ok((text, media))
    }
}

/// One accepted content part.
enum WirePart<'a> {
    Text(&'a str),
    Media(WireMedia<'a>),
}

struct WireMedia<'a> {
    modality: Modality,
    url: &'a str,
}

const fn placeholder(modality: &Modality) -> &'static str {
    match modality {
        Modality::Image => "<|vision_start|><|image_pad|><|vision_end|>",
        Modality::Video => "<|vision_start|><|video_pad|><|vision_end|>",
        Modality::Text | Modality::Audio | Modality::Extension(_) => "",
    }
}

/// Validate one content part and return its text or media reference.
fn wire_part(part: &Value, index: usize) -> Result<WirePart<'_>> {
    let object = part
        .as_object()
        .ok_or_else(|| Error::invalid(format!("message content part {index} must be an object")))?;
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::invalid(format!("message content part {index} requires a type")))?;
    match kind {
        "text" => {
            if let Some(key) = object
                .keys()
                .find(|key| !["type", "text"].contains(&key.as_str()))
            {
                return Err(Error::invalid(format!(
                    "unknown message content part field {key:?}"
                )));
            }
            let text = object.get("text").and_then(Value::as_str).ok_or_else(|| {
                Error::invalid(format!(
                    "text content part {index} requires a string text field"
                ))
            })?;
            Ok(WirePart::Text(text))
        }
        "image_url" | "input_image" => Ok(WirePart::Media(image_request(object, index)?)),
        "video_url" | "input_video" => Ok(WirePart::Media(video_request(object, index)?)),
        "audio" | "input_audio" => Err(Error::unsupported("audio input is not supported")),
        _ => Err(Error::unsupported(
            "only text, image_url and video_url message content parts are supported",
        )),
    }
}

fn validate_public_media_url(url: &str) -> Result<()> {
    if url.starts_with("data:") || url.starts_with("http://") || url.starts_with("https://") {
        Ok(())
    } else {
        Err(Error::unsupported(
            "public media URLs must be data:, http:// or https://; server-local paths are not allowed",
        ))
    }
}

fn image_request(object: &serde_json::Map<String, Value>, index: usize) -> Result<WireMedia<'_>> {
    let url = if let Some(value) = object.get("image_url") {
        let detail = value
            .as_object()
            .ok_or_else(|| Error::invalid(format!("image_url part {index} must be an object")))?;
        if let Some(key) = detail
            .keys()
            .find(|key| !["url", "detail"].contains(&key.as_str()))
        {
            return Err(Error::invalid(format!("unknown image_url field {key:?}")));
        }
        if let Some(detail) = detail.get("detail") {
            let detail = detail
                .as_str()
                .ok_or_else(|| Error::invalid("image detail must be a string"))?;
            if !["auto", "low", "high"].contains(&detail) {
                return Err(Error::invalid("image detail must be auto, low or high"));
            }
        }
        detail.get("url")
    } else {
        object.get("image_url")
    };
    let url = url
        .and_then(Value::as_str)
        .ok_or_else(|| Error::invalid(format!("image_url part {index} requires a URL")))?;
    validate_public_media_url(url)?;
    Ok(WireMedia {
        modality: Modality::Image,
        url,
    })
}

fn video_request(object: &serde_json::Map<String, Value>, index: usize) -> Result<WireMedia<'_>> {
    let value = object
        .get("video_url")
        .ok_or_else(|| Error::invalid(format!("video_url part {index} requires a video_url")))?;
    let detail = value
        .as_object()
        .ok_or_else(|| Error::invalid(format!("video_url part {index} must be an object")))?;
    if let Some(key) = detail.keys().find(|key| !["url"].contains(&key.as_str())) {
        return Err(Error::invalid(format!("unknown video_url field {key:?}")));
    }
    let url = detail
        .get("url")
        .and_then(Value::as_str)
        .filter(|url| !url.is_empty())
        .ok_or_else(|| Error::invalid(format!("video_url part {index} requires a URL")))?;
    validate_public_media_url(url)?;
    Ok(WireMedia {
        modality: Modality::Video,
        url,
    })
}

/// An assistant tool call in the history.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireToolCall {
    pub id: String,
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    pub function: WireToolCallFunction,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireToolCallFunction {
    pub name: String,
    /// Raw JSON text exactly as the caller received it.
    pub arguments: String,
}

/// One declared function tool.
#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: WireToolFunction,
}

#[derive(serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireToolFunction {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// `tool_choice` as either a mode string or a named function.
#[derive(serde::Deserialize)]
#[serde(untagged)]
pub(super) enum WireToolChoice {
    Mode(String),
    Named {
        #[serde(rename = "type")]
        kind: String,
        function: WireToolName,
    },
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireToolName {
    pub name: String,
}

#[cfg(test)]
#[path = "../../../tests/unit/http_openai_request.rs"]
mod tests;
