//! Strict function/parameter envelope; values become append-only JSON at parameter boundaries.
use super::types::{FUNCTION_MARKER, PARAMETER_MARKER, ParsedToolCall, TOOL_CALL_CLOSE};
use std::collections::BTreeSet;
const FUNCTION_CLOSE: &str = "</function>";
const PARAMETER_CLOSE: &str = "</parameter>";
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Start,
    Name,
    Between,
    Key,
    Value,
    Closed,
    Invalid,
}
#[derive(Debug, Clone)]
pub(super) struct FunctionScanner {
    phase: Phase,
    cursor: usize,
    start: usize,
    name: Option<String>,
    key: String,
    keys: BTreeSet<String>,
    arguments: String,
    json_value: Option<bool>,
    quoted: bool,
    escaped: bool,
    abandoned_close: Option<usize>,
}
impl FunctionScanner {
    pub(super) const fn new() -> Self {
        Self {
            phase: Phase::Start,
            cursor: 0,
            start: 0,
            name: None,
            key: String::new(),
            keys: BTreeSet::new(),
            arguments: String::new(),
            json_value: None,
            quoted: false,
            escaped: false,
            abandoned_close: None,
        }
    }
    pub(super) fn scan(&mut self, body: &str, flush: bool) -> Option<usize> {
        while self.cursor < body.len() {
            let byte = body.as_bytes()[self.cursor];
            if byte == b'<' && self.phase != Phase::Value {
                let tail = &body[self.cursor..];
                if tail.starts_with(TOOL_CALL_CLOSE) {
                    return Some(self.cursor);
                }
                if TOOL_CALL_CLOSE.starts_with(tail) {
                    return None;
                }
            }
            match self.phase {
                Phase::Start | Phase::Between | Phase::Closed => {
                    if !self.boundary(body) {
                        return None;
                    }
                }
                Phase::Name | Phase::Key => self.header(body, byte),
                Phase::Value => {
                    if !self.value(body, byte) {
                        return None;
                    }
                }
                Phase::Invalid => self.cursor += 1,
            }
        }
        if flush { self.abandoned_close } else { None }
    }
    fn boundary(&mut self, body: &str) -> bool {
        let tail = &body[self.cursor..];
        if body.as_bytes()[self.cursor].is_ascii_whitespace() {
            self.cursor += 1;
            return true;
        }
        match self.phase {
            Phase::Start => {
                if FUNCTION_MARKER.starts_with(tail) {
                    return false;
                }
                if tail.starts_with(FUNCTION_MARKER) {
                    self.cursor += FUNCTION_MARKER.len();
                    self.start = self.cursor;
                    self.phase = Phase::Name;
                } else {
                    self.phase = Phase::Invalid;
                }
                true
            }
            Phase::Between if tail.starts_with(FUNCTION_CLOSE) => {
                self.arguments.push('}');
                self.phase = Phase::Closed;
                self.cursor += FUNCTION_CLOSE.len();
                true
            }
            Phase::Between if FUNCTION_CLOSE.starts_with(tail) => false,
            Phase::Between => self.parameter(tail),
            _ => {
                self.phase = Phase::Invalid;
                true
            }
        }
    }
    /// Open one parameter inside `<function=...>`.
    ///
    /// The shipped `Qwen3.8` template asks for `<parameter=name>`, but the model sometimes
    /// drops or swaps the marker and writes `<name>` or `<function=name>`. Both are
    /// unambiguous here: the tag sits between parameters, and its value still ends at
    /// `</parameter>`. A tag that cannot name a parameter keeps the block invalid.
    fn parameter(&mut self, tail: &str) -> bool {
        if tail.starts_with(PARAMETER_MARKER) {
            self.cursor += PARAMETER_MARKER.len();
            self.start = self.cursor;
            self.phase = Phase::Key;
            return true;
        }
        if PARAMETER_MARKER.starts_with(tail) {
            return false;
        }
        let Some(close) = tail.find('>') else {
            // The opener may still complete in a later chunk.
            return false;
        };
        if close == 0 || !tail.starts_with('<') {
            self.phase = Phase::Invalid;
            return true;
        }
        let tag = tail[1..close].trim();
        let name = tag.strip_prefix("function=").unwrap_or(tag);
        if !names_a_parameter(name) || !self.keys.insert(name.to_string()) {
            self.phase = Phase::Invalid;
            return true;
        }
        self.key = name.to_string();
        self.json_value = None;
        self.quoted = false;
        self.escaped = false;
        self.phase = Phase::Value;
        self.cursor += close + 1;
        self.start = self.cursor;
        true
    }
    fn header(&mut self, body: &str, byte: u8) {
        if byte == b'>' {
            let name = body[self.start..self.cursor].trim();
            if name.is_empty() {
                self.phase = Phase::Invalid;
            } else if self.phase == Phase::Name {
                self.name = Some(name.to_string());
                self.arguments.push('{');
                self.phase = Phase::Between;
            } else if self.keys.insert(name.to_string()) {
                self.key = name.to_string();
                self.phase = Phase::Value;
                self.start = self.cursor + 1;
                self.json_value = None;
                self.quoted = false;
                self.escaped = false;
            } else {
                self.phase = Phase::Invalid;
            }
        }
        self.cursor += 1;
    }
    fn value(&mut self, body: &str, byte: u8) -> bool {
        if byte == b'<' && !self.quoted {
            let tail = &body[self.cursor..];
            if tail.len() < PARAMETER_CLOSE.len() && PARAMETER_CLOSE.starts_with(tail) {
                return false;
            }
            if tail.starts_with(PARAMETER_CLOSE) {
                self.abandoned_close = None;
                self.append_value(&body[self.start..self.cursor]);
                self.cursor += PARAMETER_CLOSE.len();
                self.phase = Phase::Between;
                return true;
            }
            // A missing parameter closer must terminate as an invalid block, not swallow following calls.
            if tail.starts_with(TOOL_CALL_CLOSE) {
                self.abandoned_close.get_or_insert(self.cursor);
                self.cursor += TOOL_CALL_CLOSE.len();
                return true;
            }
            if TOOL_CALL_CLOSE.starts_with(tail) {
                return false;
            }
        }
        if self.json_value.is_none() && !byte.is_ascii_whitespace() {
            self.json_value = Some(matches!(byte, b'{' | b'[' | b'"'));
        }
        // Free text has no quote state. Skip its ordinary bytes with the string searcher's
        // optimized delimiter scan instead of branching on every byte of long parameters.
        if self.json_value == Some(false) && byte != b'<' {
            let tail = &body[self.cursor..];
            self.cursor += tail.find('<').unwrap_or(tail.len());
            return true;
        }
        if self.json_value == Some(true) {
            if self.escaped {
                self.escaped = false;
            } else if self.quoted && byte == b'\\' {
                self.escaped = true;
            } else if byte == b'"' {
                self.quoted = !self.quoted;
            }
        }
        self.cursor += 1;
        true
    }
    fn append_value(&mut self, raw: &str) {
        let raw = raw.strip_prefix('\n').unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        let value = serde_json::from_str::<serde_json::Value>(raw)
            .unwrap_or_else(|_| serde_json::Value::String(raw.to_string()));
        if self.keys.len() > 1 {
            self.arguments.push(',');
        }
        if let Ok(key) = serde_json::to_string(&self.key) {
            self.arguments.push_str(&key);
        }
        self.arguments.push(':');
        if let Ok(value) = serde_json::to_string(&value) {
            self.arguments.push_str(&value);
        }
    }
    pub(super) fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }
    pub(super) fn arguments(&self) -> Option<&str> {
        self.name.as_ref().map(|_| self.arguments.as_str())
    }
    pub(super) fn complete(&self) -> Option<ParsedToolCall> {
        if self.phase != Phase::Closed {
            return None;
        }
        Some(ParsedToolCall {
            name: self.name.clone()?,
            arguments: self.arguments.clone(),
        })
    }
}

/// Whether a tolerant opener tag (`<name>` or `<function=name>`) names exactly one parameter.
///
/// The marker words themselves are rejected: `<parameter>` without a name is a broken opener,
/// not a parameter called `parameter`.
fn names_a_parameter(name: &str) -> bool {
    !name.is_empty()
        && name != "parameter"
        && name != "function"
        && name
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
}
