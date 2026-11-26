//! Named control-plane limits and JSON-RPC error codes shared across the agent crate.

/// Maximum number of requests accepted in one JSON-RPC batch.
pub const MAX_BATCH_REQUESTS: usize = 64;

/// Retained call records in one agent control session.
pub const MAX_CALL_HISTORY: usize = 256;

/// Retained experiment results in one agent control session.
pub const MAX_EXPERIMENT_HISTORY: usize = 16;

/// Maximum requests accepted in one benchmark or experiment workload.
pub const MAX_BENCHMARK_REQUESTS: usize = 256;

/// Default semantic-event count when an event query omits `limit`.
pub const DEFAULT_EVENT_LIMIT: usize = 256;

/// Largest semantic-event count an event query may request.
pub const MAX_EVENT_LIMIT: usize = 4096;

/// Default absolute tolerance for numeric accuracy constraints.
pub const DEFAULT_ACCURACY_ATOL: f64 = 1e-5;

/// Default relative tolerance for numeric accuracy constraints.
pub const DEFAULT_ACCURACY_RTOL: f64 = 1e-5;

/// Default tolerated P99 latency regression ratio for experiments.
pub const DEFAULT_MAX_P99_REGRESSION: f64 = 0.05;

/// Wall-clock seconds an isolated benchmark may run before it is aborted.
pub const BENCHMARK_TIMEOUT_SECS: u64 = 30;

/// Maximum tick iterations an isolated benchmark may execute.
pub const MAX_BENCHMARK_TICKS: u32 = 1_000_000;

/// Method-name characters retained in one call record.
pub const MAX_RECORDED_METHOD_CHARS: usize = 128;

/// Initial capacity, in bytes, of the NDJSON frame buffer.
pub const INITIAL_FRAME_CAPACITY: usize = 4096;

/// JSON-RPC invalid request error code.
pub const JSONRPC_INVALID_REQUEST: i32 = -32600;
/// JSON-RPC method not found error code.
pub const JSONRPC_METHOD_NOT_FOUND: i32 = -32601;
/// JSON-RPC invalid params error code.
pub const JSONRPC_INVALID_PARAMS: i32 = -32602;
/// JSON-RPC parse error code.
pub const JSONRPC_PARSE_ERROR: i32 = -32700;
/// JSON-RPC server error code used for backend and framing failures.
pub const JSONRPC_SERVER_ERROR: i32 = -32000;

/// How long a queued submission waits for device capacity before it is dropped.
pub const QUEUE_WAIT_US: u64 = 30_000_000;
