//! Service limits of the agentic layer.

/// Background Responses jobs one deployment may run concurrently.
pub const MAX_BACKGROUND_RESPONSES: usize = 8;

/// Model rounds one tool-loop turn may run. Matches the gateway round cap of vLLM agentic-api
/// (`MAX_GATEWAY_TOOL_ROUNDS`); a turn that needs more ends `incomplete` and can be continued.
pub const MAX_TOOL_ROUNDS: u32 = 10;

/// Engine tool calls one turn may execute, before any smaller request `max_tool_calls`. The round
/// cap times a generous parallel fan-out per round.
pub const MAX_TOOL_CALLS: u32 = 64;

/// Engine tool calls running at once across the process; later calls wait for a permit.
pub const MAX_CONCURRENT_TOOL_CALLS: usize = 16;

/// Largest tool output fed back to the model, before any smaller executor limit. Bounds how much
/// one call can grow the context and the stored history.
pub const MAX_TOOL_OUTPUT_BYTES: usize = 1024 * 1024;
