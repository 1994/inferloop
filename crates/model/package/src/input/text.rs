use super::constraint::GrammarCache;
use crate::constants::MIB;
use crate::constants::MIB_U64;
use crate::input::tool::MAX_TOOL_CALLS;
use crate::input::tool::ToolCall;
use crate::input::tool::ToolDialect;
use crate::package_path;
use infer_core::{Error, ErrorCode, Result};
use minijinja_contrib::pycompat::unknown_method_callback;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::mem::take;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex};
use std::{collections::BTreeMap, path::Path};
use tokenizers::Tokenizer;

/// Maximum accepted `tokenizer.json` asset size.
const TOKENIZER_JSON_MAX_BYTES: u64 = 64 * MIB_U64;
/// Maximum accepted tokenizer-config or chat-template asset size.
const TEXT_ASSET_MAX_BYTES: u64 = 2 * MIB_U64;
/// Maximum accepted chat or input text size.
const TEXT_MAX_BYTES: usize = 2 * MIB;
/// Maximum chat messages accepted by the template renderer.
const MAX_CHAT_MESSAGES: usize = 256;
/// Template evaluation fuel bounding render work per chat request.
const TEMPLATE_FUEL: u64 = 100_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: String,
    /// Message text. `null` is accepted and stored as empty so an assistant message that only
    /// calls a tool keeps its nullable content contract.
    #[serde(default, deserialize_with = "nullable_text")]
    pub content: String,
    /// Assistant calls, kept so a full tool round can be replayed into the template.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Identifier associating a `tool` result with the assistant call it answers.
    #[serde(default)]
    pub tool_call_id: Option<String>,
}
impl ChatMessage {
    /// A message carrying text only, which is the common case for callers that do not use tools.
    #[must_use]
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            tool_calls: Vec::new(),
            tool_call_id: None,
        }
    }
}
/// Deserialize a text field that may be `null`.
fn nullable_text<'de, D>(deserializer: D) -> StdResult<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<String>::deserialize(deserializer)?.unwrap_or_default())
}

/// Build the template view of the message history.
///
/// The wire protocol carries tool arguments as a JSON string, but both shipped templates require
/// an object: they iterate its entries to render `<parameter=>` blocks. Decoding here keeps the
/// prompt format and the parser dialect in agreement without changing the published contract.
fn template_messages(messages: &[ChatMessage]) -> Result<serde_json::Value> {
    let mut out = Vec::with_capacity(messages.len());
    for message in messages {
        let mut calls = Vec::with_capacity(message.tool_calls.len());
        for call in &message.tool_calls {
            let arguments = call.argument_object()?;
            calls.push(serde_json::json!({
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": arguments},
            }));
        }
        out.push(serde_json::json!({
            "role": message.role,
            "content": message.content,
            "tool_calls": calls,
            "tool_call_id": message.tool_call_id,
        }));
    }
    Ok(serde_json::Value::Array(out))
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ChatOptions {
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    pub tools: Option<serde_json::Value>,
}
impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: false,
            tools: None,
        }
    }
}
pub struct TextAssets {
    pub(super) constraint_cache: Mutex<GrammarCache>,
    pub(super) tokenizer: Arc<Tokenizer>,
    template: Option<String>,
    special: BTreeMap<String, serde_json::Value>,
    pub fingerprint: String,
    max_tokens: usize,
    pub generation: crate::GenerationDefaults,
    /// Tool-call dialect this package's template asks the model to emit.
    tool_dialect: ToolDialect,
    /// Whether the tokenizer's decoder can be applied to a token sub-list independently.
    streaming_decode: bool,
    tool_supported: bool,
}
fn read(path: &Path, budget: u64) -> Result<Vec<u8>> {
    use std::io::Read;
    let file = File::open(path).map_err(|e| Error::invalid(e.to_string()))?;
    if file
        .metadata()
        .map_err(|e| Error::invalid(e.to_string()))?
        .len()
        > budget
    {
        return Err(Error::new(
            ErrorCode::Capacity,
            "tokenizer/template asset exceeds budget",
        ));
    }
    let mut bytes = vec![];
    file.take(budget + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| Error::invalid(e.to_string()))?;
    if bytes.len() as u64 > budget {
        return Err(Error::new(
            ErrorCode::Capacity,
            "tokenizer asset grew beyond budget",
        ));
    }
    Ok(bytes)
}
impl TextAssets {
    /// Resolve chat generation settings, including the tokenizer's end-of-message token.
    /// Explicit EOS overrides replace the defaults, including the tokenizer token.
    ///
    /// # Errors
    /// Returns an error if sampling settings are invalid or the declared tokenizer EOS is
    /// malformed or absent from the tokenizer vocabulary.
    pub fn resolve_chat_generation(
        &self,
        overrides: &crate::SamplingOverrides,
    ) -> Result<crate::ResolvedGeneration> {
        let mut resolved = self.generation.resolve(overrides)?;
        if overrides.eos_token.is_some() || overrides.eos_tokens.is_some() {
            return Ok(resolved);
        }
        let Some(value) = self
            .special
            .get("eos_token")
            .filter(|value| !value.is_null())
        else {
            return Ok(resolved);
        };
        let token = value
            .as_str()
            .ok_or_else(|| Error::invalid("invalid tokenizer EOS token"))?;
        let id = self
            .tokenizer
            .token_to_id(token)
            .ok_or_else(|| Error::invalid("tokenizer EOS token is absent from vocabulary"))?;
        if !resolved.sampling.is_eos(id) {
            resolved.sampling.eos_tokens.push(id);
            let source = resolved
                .sources
                .get("eos_tokens")
                .map_or("runtime fallback", String::as_str);
            resolved.sources.insert(
                "eos_tokens".into(),
                format!("{source} and tokenizer_config.json"),
            );
        }
        Ok(resolved)
    }

    /// # Errors
    /// Returns an I/O or invalid-input error for missing, malformed, unsupported, or oversized assets.
    pub fn open(root: impl AsRef<Path>, max_tokens: usize) -> Result<Self> {
        if max_tokens == 0 {
            return Err(Error::invalid("empty text token budget"));
        }
        let root = root.as_ref();
        let bytes = read(
            &package_path(root, "tokenizer.json")?,
            TOKENIZER_JSON_MAX_BYTES,
        )?;
        let mut tokenizer =
            Tokenizer::from_bytes(&bytes).map_err(|e| Error::invalid(e.to_string()))?;
        tokenizer
            .with_truncation(None)
            .map_err(|e| Error::invalid(e.to_string()))?;
        tokenizer.with_padding(None);
        let mut hash = Sha256::new();
        hash.update(&bytes);
        let mut special = BTreeMap::new();
        let mut template = None;
        if root.join("tokenizer_config.json").exists() {
            let config_bytes = read(
                &package_path(root, "tokenizer_config.json")?,
                TEXT_ASSET_MAX_BYTES,
            )?;
            hash.update(&config_bytes);
            let value: serde_json::Value =
                serde_json::from_slice(&config_bytes).map_err(|e| Error::invalid(e.to_string()))?;
            for key in ["bos_token", "eos_token", "pad_token", "unk_token"] {
                if let Some(v) = value.get(key) {
                    special.insert(key.into(), v.get("content").unwrap_or(v).clone());
                }
            }
            if let Some(v) = value.get("chat_template") {
                if let Some(s) = v.as_str() {
                    template = Some(s.to_owned());
                } else if let Some(list) = v.as_array() {
                    template = list
                        .iter()
                        .find(|v| v.get("name").and_then(|v| v.as_str()) == Some("default"))
                        .and_then(|v| v.get("template"))
                        .and_then(|v| v.as_str())
                        .map(str::to_owned);
                    if template.is_none() {
                        return Err(Error::unsupported(
                            "named chat templates require a default template",
                        ));
                    }
                } else if !v.is_null() {
                    return Err(Error::invalid("invalid chat template asset"));
                }
            }
        }
        if root.join("chat_template.jinja").exists() {
            let bytes = read(
                &package_path(root, "chat_template.jinja")?,
                TEXT_ASSET_MAX_BYTES,
            )?;
            hash.update(&bytes);
            template = Some(String::from_utf8(bytes).map_err(|e| Error::invalid(e.to_string()))?);
        }
        let tool_supported = template
            .as_deref()
            .is_some_and(|template| ToolDialect::from_template(template).is_some());
        let tool_dialect = template
            .as_deref()
            .map_or(ToolDialect::JsonBlock, ToolDialect::detect);
        // Word-piece style decoders insert a separator between consecutive tokens, so decoding
        // pieces independently loses it. Only byte-level concatenation is decomposable.
        let streaming_decode = byte_level_decoder(&bytes);
        Ok(Self {
            constraint_cache: Mutex::new(GrammarCache::default()),
            generation: crate::GenerationDefaults::open(root)?,
            tokenizer: Arc::new(tokenizer),
            template,
            special,
            fingerprint: format!("{:x}", hash.finalize()),
            max_tokens,
            tool_dialect,
            streaming_decode,
            tool_supported,
        })
    }
    /// Whether the package declares a recognized tool-call envelope.
    #[must_use]
    pub const fn supports_tool_calls(&self) -> bool {
        self.tool_supported
    }
    /// Tool-call dialect bound to this package's template.
    #[must_use]
    pub const fn tool_dialect(&self) -> ToolDialect {
        self.tool_dialect
    }
    /// Whether this package can publish a stable incremental decode.
    ///
    /// False for a tokenizer whose decoder inserts separators between tokens: pieces decoded
    /// separately would not match the one-shot decode, so the service refuses to stream rather
    /// than publish text it would have to rewrite.
    #[must_use]
    pub const fn supports_streaming_decode(&self) -> bool {
        self.streaming_decode
    }
    /// Create an incremental decoder for one streamed response.
    #[must_use]
    pub fn stream_decoder(&self) -> TextStreamDecoder {
        TextStreamDecoder {
            tokenizer: Arc::clone(&self.tokenizer),
            pending: Vec::new(),
        }
    }
    /// # Errors
    /// Returns an invalid-input or capacity error if tokenization fails or the token limit is exceeded.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        if text.len() > TEXT_MAX_BYTES {
            return Err(Error::new(ErrorCode::Capacity, "text input exceeds 2 MiB"));
        }
        let encoded = self
            .tokenizer
            .encode(text, add_special_tokens)
            .map_err(|e| Error::invalid(e.to_string()))?;
        let tokens = encoded.get_ids().to_vec();
        if tokens.is_empty() || tokens.len() > self.max_tokens {
            return Err(Error::new(
                ErrorCode::Capacity,
                "text token count exceeds context budget",
            ));
        }
        Ok(tokens)
    }
    /// # Errors
    /// Returns an invalid-input error if token IDs cannot be decoded.
    pub fn decode(&self, tokens: &[u32], skip_special: bool) -> Result<String> {
        if tokens.len() > self.max_tokens {
            return Err(Error::new(
                ErrorCode::Capacity,
                "decode exceeds token budget",
            ));
        }
        self.tokenizer
            .decode(tokens, skip_special)
            .map_err(|e| Error::invalid(e.to_string()))
    }
    /// # Errors
    /// Returns an invalid-input or unsupported error for invalid messages, template errors, or exhausted template fuel.
    pub fn render_chat(&self, messages: &[ChatMessage], options: &ChatOptions) -> Result<String> {
        let template = self
            .template
            .as_ref()
            .ok_or_else(|| Error::unsupported("package has no chat template"))?;
        let bytes = messages
            .iter()
            .try_fold(0usize, |sum, m| {
                let calls = m.tool_calls.iter().try_fold(0usize, |calls, call| {
                    calls
                        .checked_add(call.id.len())
                        .and_then(|n| n.checked_add(call.name.len()))
                        .and_then(|n| n.checked_add(call.arguments.len()))
                })?;
                sum.checked_add(m.content.len())
                    .and_then(|n| n.checked_add(calls))
                    .and_then(|n| n.checked_add(m.tool_call_id.as_ref().map_or(0, String::len)))
            })
            .ok_or_else(|| Error::invalid("chat length overflow"))?;
        if messages.is_empty()
            || messages.len() > MAX_CHAT_MESSAGES
            || bytes > TEXT_MAX_BYTES
            || messages.iter().any(|m| {
                !["system", "user", "assistant", "tool"].contains(&m.role.as_str())
                    // Calls belong to the assistant turn that produced them, and only a `tool`
                    // result carries the identifier it answers.
                    || (!m.tool_calls.is_empty() && m.role != "assistant")
                    || (m.tool_call_id.is_some() && m.role != "tool")
                    || m.tool_calls.len() > MAX_TOOL_CALLS
            })
        {
            return Err(Error::invalid("invalid/budget-exceeding chat messages"));
        }
        let mut env = minijinja::Environment::new();
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
        env.set_fuel(Some(TEMPLATE_FUEL));
        env.set_unknown_method_callback(unknown_method_callback);
        env.add_function(
            "raise_exception",
            |message: String| -> StdResult<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    message,
                ))
            },
        );
        env.add_template("chat", template)
            .map_err(|e| Error::invalid(e.to_string()))?;
        let mut context = self.special.clone();
        context.insert("messages".into(), template_messages(messages)?);
        context.insert(
            "add_generation_prompt".into(),
            options.add_generation_prompt.into(),
        );
        context.insert("enable_thinking".into(), options.enable_thinking.into());
        context.insert(
            "tools".into(),
            options.tools.clone().unwrap_or(serde_json::Value::Null),
        );
        let rendered = env
            .get_template("chat")
            .map_err(|e| Error::invalid(e.to_string()))?
            .render(context)
            .map_err(|e| Error::invalid(e.to_string()))?;
        if rendered.len() > TEXT_MAX_BYTES {
            return Err(Error::new(
                ErrorCode::Capacity,
                "rendered chat exceeds budget",
            ));
        }
        Ok(rendered)
    }
    /// # Errors
    /// Returns a template, tokenization, or capacity error if chat rendering or encoding fails.
    pub fn encode_chat(&self, messages: &[ChatMessage], options: &ChatOptions) -> Result<Vec<u32>> {
        self.encode(&self.render_chat(messages, options)?, false)
    }
}

/// Whether the tokenizer's decoder is a pure byte-level concatenation.
fn byte_level_decoder(bytes: &[u8]) -> bool {
    #[derive(serde::Deserialize)]
    struct Probe {
        #[serde(default)]
        decoder: Option<Kind>,
    }
    #[derive(serde::Deserialize)]
    struct Kind {
        #[serde(rename = "type")]
        kind: String,
    }
    serde_json::from_slice::<Probe>(bytes)
        .ok()
        .and_then(|probe| probe.decoder)
        .is_some_and(|decoder| decoder.kind == "ByteLevel")
}

/// Incremental detokenizer over committed tokens.
///
/// A response is published token by token, and a token boundary can fall inside a UTF-8
/// sequence. The decoder therefore holds the tail of an incomplete sequence until more tokens
/// arrive, which keeps every published prefix stable: nothing already sent is ever rewritten.
/// The decode cost is bounded by the bytes of one character rather than by the whole response.
pub struct TextStreamDecoder {
    pub(super) tokenizer: Arc<Tokenizer>,
    /// Tokens whose decoded text is not yet known to be complete.
    pending: Vec<u32>,
}

/// Tokens retained before an unfinished character is published anyway.
///
/// Four bytes are enough for any UTF-8 sequence; the bound stops a tokenizer that legitimately
/// emits a replacement character from stalling the stream.
const MAX_PENDING_TOKENS: usize = 8;

impl TextStreamDecoder {
    /// Feed one committed token, returning the text that is stable now.
    ///
    /// # Errors
    /// Returns an invalid-input error if the token cannot be decoded.
    pub fn push(&mut self, token: u32) -> Result<String> {
        self.pending.push(token);
        let text = self.decode(&self.pending)?;
        // A trailing replacement character means the sequence is still incomplete.
        if text.ends_with('\u{FFFD}') && self.pending.len() < MAX_PENDING_TOKENS {
            return Ok(String::new());
        }
        self.pending.clear();
        Ok(text)
    }
    /// Flush the last tokens at end of stream.
    ///
    /// # Errors
    /// Returns an invalid-input error if the remaining tokens cannot be decoded.
    pub fn finish(&mut self) -> Result<String> {
        if self.pending.is_empty() {
            return Ok(String::new());
        }
        let pending = take(&mut self.pending);
        self.decode(&pending)
    }
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| Error::invalid(e.to_string()))
    }
}
