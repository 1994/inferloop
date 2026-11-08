//! Per-dialect lexical state; neither scanner knows about an HTTP response.
use super::{
    function::FunctionScanner,
    json::JsonScanner,
    types::{ParsedToolCall, ToolDialect},
};
#[derive(Debug, Clone)]
pub(super) enum BlockScanner {
    Json(JsonScanner),
    Function(FunctionScanner),
}
impl BlockScanner {
    pub(super) const fn new(dialect: ToolDialect) -> Self {
        match dialect {
            ToolDialect::JsonBlock => Self::Json(JsonScanner::new()),
            ToolDialect::FunctionParameters => Self::Function(FunctionScanner::new()),
        }
    }
    pub(super) fn scan(&mut self, body: &str, flush: bool) -> Option<usize> {
        match self {
            Self::Json(scan) => scan.scan(body),
            Self::Function(scan) => scan.scan(body, flush),
        }
    }
    pub(super) fn name(&self) -> Option<&str> {
        match self {
            Self::Json(scan) => scan.name(),
            Self::Function(scan) => scan.name(),
        }
    }
    pub(super) fn arguments<'a>(&'a self, body: &'a str) -> Option<&'a str> {
        match self {
            Self::Json(scan) => scan.arguments(body),
            Self::Function(scan) => scan.arguments(),
        }
    }
    /// Finish a block at `</tool_call>`; the JSON dialect parses its own envelope.
    pub(super) fn close(&mut self) {
        if let Self::Function(scan) = self {
            scan.close();
        }
    }
    pub(super) fn complete(&self, body: &str) -> Option<ParsedToolCall> {
        match self {
            Self::Json(_) => super::json::complete(body),
            Self::Function(scan) => scan.complete(),
        }
    }
}
