//! Package-bound output dialects, independent of HTTP and device execution.
mod block;
mod function;
mod json;
mod parser;
mod types;
pub use parser::{OutputStreamParser, assemble, parse, parse_with};
#[cfg(test)]
use types::{
    MAX_TOOL_ARGUMENT_BYTES, REASONING_CLOSE, REASONING_OPEN, TOOL_CALL_CLOSE, TOOL_CALL_OPEN,
};
pub use types::{MAX_TOOL_CALLS, ParsedOutput, ParsedToolCall, StreamEvent, ToolCall, ToolDialect};
#[cfg(test)]
#[path = "../../tests/unit/input_tool.rs"]
mod tests;
