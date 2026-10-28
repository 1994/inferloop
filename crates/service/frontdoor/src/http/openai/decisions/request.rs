//! Wire form and validation of an `OpenAI` Decisions request.
//!
//! Validation turns the request into the text a language model reads: the shared evidence once,
//! and each question as the options it must label. Nothing here touches the tokenizer, so a
//! malformed request is rejected before any request identity or CPU credit is spent.
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeSet;

/// Questions one request may ask, as the published contract allows.
const MAX_QUESTIONS: usize = 400;
/// Fewest options a choice or score question offers.
const MIN_OPTIONS: usize = 2;
/// Most choices the published contract allows; this backend's label alphabet is narrower.
const MAX_PROTOCOL_CHOICES: usize = 255;
/// Most ordered levels one score question defines.
const MAX_LEVELS: usize = 10;
/// Most bytes in a `prompt_cache_key`.
const MAX_CACHE_KEY_BYTES: usize = 64;
/// Most bytes in a `safety_identifier`.
const MAX_SAFETY_IDENTIFIER_BYTES: usize = 128;
/// Deepest container nesting one `input_json` part may use.
const MAX_JSON_DEPTH: usize = 32;
/// Most JSON values one `input_json` part may hold.
const MAX_JSON_VALUES: usize = 10_000;
/// Most compact UTF-8 bytes one `input_json` part may render to.
const MAX_JSON_BYTES: usize = 256 * 1024;
/// The only prompt cache lifetime the contract defines.
const CACHE_TTL: &str = "30m";

/// Labels a reply starts with, one per option in order. Each is checked to be a single token
/// at the start of this package's reply before a request uses it.
pub(super) const LABELS: [&str; 26] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z",
];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in super::super) struct DecisionRequest {
    pub model: String,
    pub input: Value,
    pub questions: Vec<WireQuestion>,
    /// Accepted for compatibility; reuse follows identical prefixes rather than a caller key.
    pub prompt_cache_key: Option<String>,
    pub prompt_cache_options: Option<CacheOptions>,
    /// Opaque caller metadata; it is never an identity and does not change the answers.
    pub safety_identifier: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in super::super) struct CacheOptions {
    pub mode: Option<CacheMode>,
    pub ttl: Option<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub(in super::super) enum CacheMode {
    Implicit,
    Explicit,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(in super::super) enum WireQuestion {
    Predicate {
        instructions: Instructions,
        name: Option<String>,
    },
    Choice {
        instructions: Instructions,
        choices: Vec<WireChoice>,
        name: Option<String>,
    },
    Score {
        instructions: Instructions,
        levels: Vec<WireLevel>,
        name: Option<String>,
    },
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(in super::super) enum Instructions {
    Text(String),
    Parts(Vec<Value>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in super::super) struct WireChoice {
    pub value: ChoiceValue,
    pub description: Option<String>,
}

/// A choice value keeps its JSON type: the string `"true"` and the boolean `true` differ.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(untagged)]
pub(in super::super) enum ChoiceValue {
    Bool(bool),
    Text(String),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(in super::super) struct WireLevel {
    pub label: String,
    pub description: Option<String>,
}

/// What a question returns once its labels are read.
#[derive(Debug, PartialEq, Eq)]
pub(in super::super) enum Answering {
    /// The probability of the second option, `true`.
    Predicate,
    /// The value of the most probable option.
    Choice(Vec<ChoiceValue>),
    /// The expected level index, with each level's label.
    Score(Vec<String>),
}

/// One question ready to render after the shared evidence.
#[derive(Debug)]
pub(in super::super) struct ReadQuestion {
    pub name: Option<String>,
    pub answering: Answering,
    /// The question, its labeled options and the reply instruction.
    pub text: String,
    pub options: usize,
}

/// The request as a model reads it.
#[derive(Debug)]
pub(in super::super) struct ReadPlan {
    /// Developer and system messages, kept as the chat's system turn.
    pub system: Option<String>,
    /// Shared evidence every question is asked about.
    pub evidence: String,
    pub questions: Vec<ReadQuestion>,
}

impl DecisionRequest {
    /// Whether the caller asked for explicit cache boundaries only.
    ///
    /// Reuse here follows identical token prefixes automatically, so the mode is recorded rather
    /// than enforced.
    pub(in super::super) fn explicit_cache(&self) -> bool {
        self.prompt_cache_options
            .as_ref()
            .is_some_and(|options| matches!(options.mode, Some(CacheMode::Explicit)))
    }

    /// Validate the request and render what each question reads.
    ///
    /// # Errors
    /// Returns an invalid-input error for a malformed request or one over this backend's label
    /// limit, and an unsupported error for image inputs, which this release does not read.
    pub(in super::super) fn plan(&self) -> Result<ReadPlan> {
        if self.model.is_empty() {
            return Err(Error::invalid("model must not be empty"));
        }
        if self
            .prompt_cache_key
            .as_ref()
            .is_some_and(|key| key.len() > MAX_CACHE_KEY_BYTES)
        {
            return Err(Error::invalid(format!(
                "prompt_cache_key exceeds {MAX_CACHE_KEY_BYTES} bytes"
            )));
        }
        if self
            .prompt_cache_options
            .as_ref()
            .and_then(|options| options.ttl.as_deref())
            .is_some_and(|ttl| ttl != CACHE_TTL)
        {
            return Err(Error::invalid(format!(
                "prompt_cache_options.ttl must be {CACHE_TTL:?}"
            )));
        }
        if self
            .safety_identifier
            .as_ref()
            .is_some_and(|id| id.len() > MAX_SAFETY_IDENTIFIER_BYTES)
        {
            return Err(Error::invalid(format!(
                "safety_identifier exceeds {MAX_SAFETY_IDENTIFIER_BYTES} bytes"
            )));
        }
        if self.questions.is_empty() || self.questions.len() > MAX_QUESTIONS {
            return Err(Error::invalid(format!(
                "questions must contain 1 to {MAX_QUESTIONS} items"
            )));
        }
        let (system, evidence) = evidence(&self.input)?;
        let questions = self
            .questions
            .iter()
            .enumerate()
            .map(|(index, question)| read_question(index, question))
            .collect::<Result<_>>()?;
        Ok(ReadPlan {
            system,
            evidence,
            questions,
        })
    }
}

fn read_question(index: usize, question: &WireQuestion) -> Result<ReadQuestion> {
    let (name, instructions, answering, options) = match question {
        WireQuestion::Predicate { instructions, name } => (
            name,
            instructions,
            Answering::Predicate,
            vec![("false".to_owned(), None), ("true".to_owned(), None)],
        ),
        WireQuestion::Choice {
            instructions,
            choices,
            name,
        } => {
            if !(MIN_OPTIONS..=MAX_PROTOCOL_CHOICES).contains(&choices.len()) {
                return Err(Error::invalid(format!(
                    "question {index}: choices must contain {MIN_OPTIONS} to {MAX_PROTOCOL_CHOICES} items"
                )));
            }
            let mut seen = BTreeSet::new();
            let mut options = Vec::with_capacity(choices.len());
            for choice in choices {
                if !seen.insert(&choice.value) {
                    return Err(Error::invalid(format!(
                        "question {index}: choice values must be unique"
                    )));
                }
                // The value is shown as JSON so a string and a boolean of the same text stay
                // distinguishable to the model as they are to the caller.
                let shown = serde_json::to_string(&choice.value)
                    .map_err(|error| Error::invalid(error.to_string()))?;
                options.push((shown, choice.description.clone()));
            }
            (
                name,
                instructions,
                Answering::Choice(choices.iter().map(|c| c.value.clone()).collect()),
                options,
            )
        }
        WireQuestion::Score {
            instructions,
            levels,
            name,
        } => {
            if !(MIN_OPTIONS..=MAX_LEVELS).contains(&levels.len()) {
                return Err(Error::invalid(format!(
                    "question {index}: levels must contain {MIN_OPTIONS} to {MAX_LEVELS} items"
                )));
            }
            (
                name,
                instructions,
                Answering::Score(levels.iter().map(|l| l.label.clone()).collect()),
                levels
                    .iter()
                    .map(|l| (l.label.clone(), l.description.clone()))
                    .collect(),
            )
        }
    };
    if options.len() > LABELS.len() {
        return Err(Error::invalid(format!(
            "question {index}: this deployment labels at most {} options",
            LABELS.len()
        )));
    }
    let instructions = instruction_text(index, instructions)?;
    let mut text = String::new();
    if !instructions.is_empty() {
        text.push_str("Question: ");
        text.push_str(&instructions);
        text.push('\n');
    }
    for (label, (option, description)) in LABELS.iter().zip(&options) {
        text.push_str(label);
        text.push_str(": ");
        text.push_str(option);
        if let Some(description) = description.as_deref().filter(|d| !d.is_empty()) {
            text.push_str(" - ");
            text.push_str(description);
        }
        text.push('\n');
    }
    text.push_str("Answer with the letter of one option only.");
    Ok(ReadQuestion {
        name: name.clone(),
        answering,
        text,
        options: options.len(),
    })
}

fn instruction_text(index: usize, instructions: &Instructions) -> Result<String> {
    match instructions {
        Instructions::Text(text) => Ok(text.clone()),
        Instructions::Parts(parts) => {
            if parts.is_empty() {
                return Err(Error::invalid(format!(
                    "question {index}: instructions parts must not be empty"
                )));
            }
            let texts = parts
                .iter()
                .map(|part| {
                    let object = part_object(part)?;
                    match part_type(object)? {
                        "input_text" => {
                            known_fields(object, &["type", "text"])?;
                            string_field(object, "text")
                        }
                        "input_image" => Err(image_unsupported()),
                        other => Err(Error::invalid(format!(
                            "question {index}: unknown instructions part type {other:?}"
                        ))),
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(texts.join("\n"))
        }
    }
}

/// One conversation entry, rendered for the transcript.
enum Entry {
    User(String),
    Assistant(String),
    Call {
        kind: &'static str,
        name: String,
        id: String,
        body: String,
    },
    Output {
        kind: &'static str,
        id: String,
        body: String,
    },
}

/// The system turn and the evidence text from `input`.
fn evidence(input: &Value) -> Result<(Option<String>, String)> {
    let items = match input {
        Value::String(text) => return Ok((None, text.clone())),
        Value::Array(items) => items,
        _ => {
            return Err(Error::invalid(
                "input must be a string or an array of input items",
            ));
        }
    };
    let mut system = Vec::new();
    let mut entries = Vec::with_capacity(items.len());
    let mut calls = BTreeSet::new();
    for (index, item) in items.iter().enumerate() {
        let object = item
            .as_object()
            .ok_or_else(|| Error::invalid(format!("input item {index} must be an object")))?;
        let kind = match object.get("type") {
            Some(Value::String(kind)) => kind.as_str(),
            None if object.contains_key("role") => "message",
            _ => {
                return Err(Error::invalid(format!(
                    "input item {index} requires a string type"
                )));
            }
        };
        match kind {
            "message" => message(index, object, &mut system, &mut entries)?,
            "function_call" | "custom_tool_call" => {
                entries.push(tool_call(
                    index,
                    object,
                    kind == "custom_tool_call",
                    &mut calls,
                )?);
            }
            "function_call_output" | "custom_tool_call_output" => {
                let custom = kind == "custom_tool_call_output";
                entries.push(tool_result(index, object, custom, &calls)?);
            }
            "item_reference" => {
                return Err(Error::unsupported(
                    "item references are not supported by /v1/decisions",
                ));
            }
            other => {
                return Err(Error::invalid(format!(
                    "input item {index} has unknown type {other:?}"
                )));
            }
        }
    }
    let system = (!system.is_empty()).then(|| system.join("\n\n"));
    Ok((system, transcript(&entries)))
}

/// One tool call, whose identifier later outputs must name.
fn tool_call(
    index: usize,
    object: &Map<String, Value>,
    custom: bool,
    calls: &mut BTreeSet<(bool, String)>,
) -> Result<Entry> {
    let body = if custom { "input" } else { "arguments" };
    known_fields(
        object,
        &["type", "call_id", "name", body, "id", "namespace", "status"],
    )?;
    let id = string_field(object, "call_id")?;
    if id.is_empty() || !calls.insert((custom, id.clone())) {
        return Err(Error::invalid(format!(
            "input item {index}: call_id must be nonempty and unique"
        )));
    }
    Ok(Entry::Call {
        kind: if custom {
            "Custom tool call"
        } else {
            "Function call"
        },
        name: string_field(object, "name")?,
        id,
        body: string_field(object, body)?,
    })
}

/// One tool output, which must answer an earlier call of the same kind.
fn tool_result(
    index: usize,
    object: &Map<String, Value>,
    custom: bool,
    calls: &BTreeSet<(bool, String)>,
) -> Result<Entry> {
    known_fields(object, &["type", "call_id", "output", "id", "status"])?;
    let id = string_field(object, "call_id")?;
    if !calls.contains(&(custom, id.clone())) {
        return Err(Error::invalid(format!(
            "input item {index}: output call_id {id:?} matches no earlier call"
        )));
    }
    Ok(Entry::Output {
        kind: if custom {
            "Custom tool output"
        } else {
            "Function output"
        },
        id,
        body: tool_output(index, object.get("output"))?,
    })
}

/// Plain user evidence reads as written; any other turn needs its speaker to be legible.
fn transcript(entries: &[Entry]) -> String {
    if entries.iter().all(|entry| matches!(entry, Entry::User(_))) {
        return entries
            .iter()
            .filter_map(|entry| match entry {
                Entry::User(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n");
    }
    entries
        .iter()
        .map(|entry| match entry {
            Entry::User(text) => format!("User: {text}"),
            Entry::Assistant(text) => format!("Assistant: {text}"),
            Entry::Call {
                kind,
                name,
                id,
                body,
            } => format!("{kind} {name} ({id}): {body}"),
            Entry::Output { kind, id, body } => format!("{kind} ({id}): {body}"),
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn message(
    index: usize,
    object: &Map<String, Value>,
    system: &mut Vec<String>,
    entries: &mut Vec<Entry>,
) -> Result<()> {
    let role = string_field(object, "role")?;
    let allowed: &[&str] = if role == "assistant" {
        &["type", "role", "content", "phase", "id", "status"]
    } else {
        &["type", "role", "content"]
    };
    known_fields(object, allowed)?;
    let content = object
        .get("content")
        .ok_or_else(|| Error::invalid(format!("input item {index} requires content")))?;
    match role.as_str() {
        "user" => entries.push(Entry::User(content_text(
            index,
            content,
            ContentRole::User,
        )?)),
        // System messages are treated as developer messages, which instruct the reader.
        "developer" | "system" => {
            system.push(content_text(index, content, ContentRole::Developer)?);
        }
        "assistant" => entries.push(Entry::Assistant(content_text(
            index,
            content,
            ContentRole::Assistant,
        )?)),
        other => {
            return Err(Error::invalid(format!(
                "input item {index} has unsupported role {other:?}"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ContentRole {
    User,
    Developer,
    Assistant,
}

fn content_text(index: usize, content: &Value, role: ContentRole) -> Result<String> {
    let parts = match content {
        Value::String(text) => return Ok(text.clone()),
        Value::Array(parts) => parts,
        _ => {
            return Err(Error::invalid(format!(
                "input item {index} content must be a string or an array of parts"
            )));
        }
    };
    let texts = parts
        .iter()
        .map(|part| {
            let object = part_object(part)?;
            match (role, part_type(object)?) {
                (ContentRole::User | ContentRole::Developer, "input_text") => {
                    known_fields(object, &["type", "text", "prompt_cache_breakpoint"])?;
                    breakpoint(object)?;
                    string_field(object, "text")
                }
                (ContentRole::User, "input_json") => {
                    known_fields(object, &["type", "data"])?;
                    json_text(index, object.get("data"))
                }
                (ContentRole::User, "input_image") => Err(image_unsupported()),
                (ContentRole::Assistant, "output_text") => {
                    known_fields(object, &["type", "text", "annotations", "logprobs"])?;
                    string_field(object, "text")
                }
                (ContentRole::Assistant, "refusal") => {
                    known_fields(object, &["type", "refusal"])?;
                    Ok(format!("(refused) {}", string_field(object, "refusal")?))
                }
                (_, kind) => Err(Error::invalid(format!(
                    "input item {index}: content part {kind:?} is not accepted for this role"
                ))),
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(texts.join("\n"))
}

fn tool_output(index: usize, output: Option<&Value>) -> Result<String> {
    match output {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => Ok(parts
            .iter()
            .map(|part| {
                let object = part_object(part)?;
                match part_type(object)? {
                    "input_text" => {
                        known_fields(object, &["type", "text", "prompt_cache_breakpoint"])?;
                        breakpoint(object)?;
                        string_field(object, "text")
                    }
                    "input_image" => Err(image_unsupported()),
                    other => Err(Error::invalid(format!(
                        "input item {index}: tool output part {other:?} is not accepted"
                    ))),
                }
            })
            .collect::<Result<Vec<_>>>()?
            .join("\n")),
        _ => Err(Error::invalid(format!(
            "input item {index} output must be a string or an array of parts"
        ))),
    }
}

/// Render `input_json` data as compact text, within the contract's size bounds.
fn json_text(index: usize, data: Option<&Value>) -> Result<String> {
    let data = data
        .ok_or_else(|| Error::invalid(format!("input item {index}: input_json requires data")))?;
    let objects = match data {
        Value::Object(_) => true,
        Value::Array(items) => items.iter().all(Value::is_object),
        _ => false,
    };
    if !objects {
        return Err(Error::invalid(format!(
            "input item {index}: input_json data must be an object or an array of objects"
        )));
    }
    let mut values = 0usize;
    if !json_within(data, 1, &mut values) {
        return Err(Error::invalid(format!(
            "input item {index}: input_json exceeds {MAX_JSON_DEPTH} levels or {MAX_JSON_VALUES} values"
        )));
    }
    let text = serde_json::to_string(data).map_err(|error| Error::invalid(error.to_string()))?;
    if text.len() > MAX_JSON_BYTES {
        return Err(Error::invalid(format!(
            "input item {index}: input_json exceeds {MAX_JSON_BYTES} bytes"
        )));
    }
    Ok(text)
}

/// Whether `value` stays within the depth and value-count bounds, counting as it walks.
fn json_within(value: &Value, depth: usize, values: &mut usize) -> bool {
    *values += 1;
    if *values > MAX_JSON_VALUES {
        return false;
    }
    match value {
        Value::Array(items) => {
            depth <= MAX_JSON_DEPTH && items.iter().all(|v| json_within(v, depth + 1, values))
        }
        Value::Object(fields) => {
            depth <= MAX_JSON_DEPTH && fields.values().all(|v| json_within(v, depth + 1, values))
        }
        _ => true,
    }
}

/// A breakpoint is accepted where the contract allows one; it marks no boundary here.
fn breakpoint(object: &Map<String, Value>) -> Result<()> {
    match object.get("prompt_cache_breakpoint") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::Object(fields))
            if fields.len() == 1 && fields.get("mode") == Some(&Value::from("explicit")) =>
        {
            Ok(())
        }
        Some(_) => Err(Error::invalid(
            "prompt_cache_breakpoint must be {\"mode\":\"explicit\"}",
        )),
    }
}

fn part_object(part: &Value) -> Result<&Map<String, Value>> {
    part.as_object()
        .ok_or_else(|| Error::invalid("content parts must be objects"))
}

fn part_type(object: &Map<String, Value>) -> Result<&str> {
    object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::invalid("content parts require a string type"))
}

fn string_field(object: &Map<String, Value>, field: &str) -> Result<String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| Error::invalid(format!("{field} must be a string")))
}

fn known_fields(object: &Map<String, Value>, allowed: &[&str]) -> Result<()> {
    object
        .keys()
        .find(|key| !allowed.contains(&key.as_str()))
        .map_or(Ok(()), |key| {
            Err(Error::invalid(format!("unknown field {key:?}")))
        })
}

fn image_unsupported() -> Error {
    Error::unsupported("image inputs are not supported by /v1/decisions in this release")
}
