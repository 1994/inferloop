use infer_core::{KernelId, OpId, RequestId, StateId, StepId, TensorId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpTrace {
    pub request: RequestId,
    pub step: StepId,
    pub state: StateId,
    pub op: OpId,
    pub kernel: KernelId,
    pub position: usize,
    /// Number of contiguous token rows represented by this dispatch.
    #[serde(default = "single_trace_token")]
    pub token_count: usize,
    pub elapsed_ns: u64,
    pub timestamp_ns: u64,
    pub inputs: Vec<TensorId>,
    pub outputs: Vec<TensorId>,
    pub states: Vec<TensorId>,
}
const fn single_trace_token() -> usize {
    1
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerProbe {
    pub request: RequestId,
    pub step: StepId,
    pub state: StateId,
    pub op: OpId,
    pub layer: usize,
    pub position: usize,
    pub hidden: Vec<f32>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    pub sequences: usize,
    pub reserved_bytes: u64,
    pub allocated_bytes: u64,
    pub tokens_executed: u64,
    pub trace_dropped: u64,
    pub prefix_hits: u64,
    pub prefix_entries: usize,
    pub prefix_bytes: u64,
    pub probe_samples: usize,
    pub probe_dropped: u64,
    #[serde(default)]
    pub kv_cache: Option<KvCacheInspection>,
}
/// Effective load-time execution choices, reported so a benchmark can prove which graph
/// geometry and speculation depth actually ran instead of restating the request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionProfileInspection {
    /// Default prompt chunk width for private programs. A pooled worker may split it
    /// into smaller captured replays without changing long-context scheduling.
    pub prefill_width: usize,
    /// Actual pooled worker width, when a shared worker is available.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pooled_prefill_width: Option<usize>,
    /// Candidate rows one captured verification graph processes per replay.
    pub batch_width: usize,
    /// Resolved speculative draft depth, whichever algorithm supplies the proposals.
    pub draft_depth: usize,
    /// Whether the prompt graph width came from automatic device policy.
    pub automatic_prefill: bool,
    /// Activation arena budget the prompt width was validated against.
    pub arena_budget_bytes: u64,
    /// Numerical attention profile selected at load. Backends without a selectable profile
    /// leave this empty.
    #[serde(default)]
    pub attention_profile: String,
    /// Canonical prompt positions retained before bulk attention; zero means the opt-in full-FA4 path.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub precise_prefix_positions: Option<usize>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheInspection {
    /// Pages retained by submitted device copies until their completion fence.
    #[serde(default)]
    pub pinned_blocks: usize,
    pub block_size: usize,
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub active_blocks: usize,
    pub cached_blocks: usize,
    pub shared_blocks: usize,
    pub available_blocks: usize,
    pub pool_bytes: u64,
    pub bytes_per_block: u64,
    pub allocations: u64,
    pub releases: u64,
    pub cow_copies: u64,
    pub cache_evictions: u64,
    pub reused_tokens: u64,
}
