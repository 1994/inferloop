use crate::{CapabilityRequirements, MediaPlan, PrecisionPlan};
use infer_core::{DecisionId, KernelId, ModelId, OpId, ProgramId, RequestId, StateId, StepId};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Stable event code for `DeferReason::BatchLimit`.
const DEFER_CODE_BATCH_LIMIT: u32 = 3;
/// Stable event code for `DeferReason::IncompatibleProgram`.
const DEFER_CODE_INCOMPATIBLE_PROGRAM: u32 = 4;
/// Stable event code for `DeferReason::Deadline`.
const DEFER_CODE_DEADLINE: u32 = 5;
/// Stable event code for `DeferReason::StateCapacity`.
const DEFER_CODE_STATE_CAPACITY: u32 = 6;
/// Stable event code for `DeferReason::TransferBudget`.
const DEFER_CODE_TRANSFER_BUDGET: u32 = 7;
/// Stable event code for `DeferReason::EncoderBudget`.
const DEFER_CODE_ENCODER_BUDGET: u32 = 8;
/// Stable event code for `DeferReason::ChunkLimit`.
const DEFER_CODE_CHUNK_LIMIT: u32 = 9;
/// Stable event code for `DeferReason::AtomicCostLimit`.
const DEFER_CODE_ATOMIC_COST_LIMIT: u32 = 10;
/// Stable event code for `DeferReason::PreemptionFocus`.
const DEFER_CODE_PREEMPTION_FOCUS: u32 = 11;
/// Stable event code for `DeferReason::Preparation`.
const DEFER_CODE_PREPARATION: u32 = 12;
/// Stable event code for `DeferReason::PlanningBudget`.
const DEFER_CODE_PLANNING_BUDGET: u32 = 13;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Operation {
    TokenEmbedding,
    RmsNorm,
    Attention,
    LinearAttention,
    Convolution,
    MatMul,
    Silu,
    Residual,
    LmHead,
    Pool,
    Rank,
    Decision,
    Split,
    Rope,
    Sigmoid,
    Multiply,
    GatedNorm,
    MoeRoute,
    ExpertMatMul,
    MoeCombine,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperationIr {
    pub id: OpId,
    pub operation: Operation,
    pub shape: Vec<usize>,
    pub layer: Option<usize>,
    pub requirements: CapabilityRequirements,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionIr {
    pub model: ModelId,
    pub precision: PrecisionPlan,
    pub operations: Vec<OperationIr>,
    pub dataflow: crate::DataflowGraph,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledOp {
    pub op: OperationIr,
    pub kernel: KernelId,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionProgram {
    pub backend: crate::BackendKind,
    pub id: ProgramId,
    pub model: ModelId,
    pub precision: PrecisionPlan,
    pub operations: Vec<CompiledOp>,
    pub workspace_bytes: u64,
    pub dataflow: crate::DataflowGraph,
}
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum ExecutionRole {
    #[default]
    Prefill,
    Decode,
    Forward,
    /// A step contains compatible work from more than one execution phase.
    Mixed,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphVariant {
    pub batch_capacity: usize,
    pub max_tokens: usize,
    #[serde(default)]
    pub program: Option<ProgramId>,
    #[serde(default)]
    pub workspace_bytes: u64,
}
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostEstimate {
    pub gpu_us: u64,
    pub workspace_bytes: u64,
    pub num_gpu_blocks: usize,
    #[serde(default)]
    pub logical_pages: usize,
    #[serde(default)]
    pub state_bytes: u64,
    pub transfer_us: u64,
    pub encoder_us: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadyWork {
    pub request: RequestId,
    pub program: ProgramId,
    pub state: StateId,
    pub role: ExecutionRole,
    pub remaining_tokens: usize,
    pub tenant: String,
    pub weight: u32,
    pub deadline_us: Option<u64>,
    pub virtual_finish: u64,
    pub cost_per_token: CostEstimate,
    pub cost_query: crate::CostQuery,
    pub latency_deadline_us: Option<u64>,
    pub remaining_latency_us: u64,
    pub remaining_completion_us: u64,
    pub last_service_us: u64,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlannedWork {
    pub request: RequestId,
    pub state: StateId,
    pub token_count: usize,
    #[serde(default)]
    pub role: ExecutionRole,
}
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepPlan {
    pub id: StepId,
    pub decision: DecisionId,
    pub program: ProgramId,
    pub role: ExecutionRole,
    pub work: Vec<PlannedWork>,
    pub cost: CostEstimate,
    pub graph: Option<GraphVariant>,
    #[serde(default)]
    pub quantum_overrun: bool,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeferReason {
    TokenBudget,
    GpuBudget,
    Workspace,
    BatchLimit,
    IncompatibleProgram,
    Deadline,
    StateCapacity,
    TransferBudget,
    EncoderBudget,
    ChunkLimit,
    AtomicCostLimit,
    PreemptionFocus { request: RequestId },
    Preparation,
    PlanningBudget,
}
impl DeferReason {
    pub const LABELS: [&'static str; 14] = [
        "max_num_batched_tokens",
        "gpu_budget",
        "workspace",
        "batch_limit",
        "incompatible_program",
        "deadline",
        "state_capacity",
        "transfer_budget",
        "encoder_budget",
        "chunk_limit",
        "atomic_cost_limit",
        "preemption_focus",
        "preparation",
        "planning_budget",
    ];
    #[must_use]
    pub const fn code(&self) -> u32 {
        match self {
            Self::TokenBudget => 0,
            Self::GpuBudget => 1,
            Self::Workspace => 2,
            Self::BatchLimit => DEFER_CODE_BATCH_LIMIT,
            Self::IncompatibleProgram => DEFER_CODE_INCOMPATIBLE_PROGRAM,
            Self::Deadline => DEFER_CODE_DEADLINE,
            Self::StateCapacity => DEFER_CODE_STATE_CAPACITY,
            Self::TransferBudget => DEFER_CODE_TRANSFER_BUDGET,
            Self::EncoderBudget => DEFER_CODE_ENCODER_BUDGET,
            Self::ChunkLimit => DEFER_CODE_CHUNK_LIMIT,
            Self::AtomicCostLimit => DEFER_CODE_ATOMIC_COST_LIMIT,
            Self::PreemptionFocus { .. } => DEFER_CODE_PREEMPTION_FOCUS,
            Self::Preparation => DEFER_CODE_PREPARATION,
            Self::PlanningBudget => DEFER_CODE_PLANNING_BUDGET,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeferredWork {
    pub request: RequestId,
    pub reason: DeferReason,
    pub required: u64,
    pub available: u64,
}
/// Frozen bounded planning coverage. Unvisited requests have no resource rejection evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionWindow {
    pub epoch: u64,
    pub cost_epoch: u64,
    pub resource_epoch: u64,
    pub inspected: usize,
    pub ready: usize,
}
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedulingDecision {
    #[serde(default)]
    pub window: Option<DecisionWindow>,
    pub id: DecisionId,
    pub step: Option<StepPlan>,
    pub deferred: Vec<DeferredWork>,
    #[serde(default)]
    pub selected: Vec<crate::SelectionEvidence>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub gpu_budget_us: u64,
    pub workspace_bytes: u64,
    pub free_state_pages: usize,
    pub free_logical_pages: usize,
    pub graphs: Vec<GraphVariant>,
    pub free_state_bytes: u64,
    pub transfer_budget_us: u64,
    pub encoder_budget_us: u64,
    pub scheduler: crate::SchedulerConfig,
}
#[derive(Debug, Clone)]
pub struct ExecutionTask {
    /// Allows a backend to replace a generation logits readout with an exact plain-greedy token.
    /// This flag does not change speculative acceptance or the committed input timeline.
    pub plain_greedy: bool,
    /// This prompt chunk ends at the request's prefix boundary: the backend may keep the
    /// covered prefix as a reusable snapshot for later requests that repeat it.
    pub publish_prefix: bool,
    pub request: RequestId,
    pub state: StateId,
    pub tokens: crate::ExecutionInput,
    /// Sampling parameters for backends that decide tokens themselves (speculative decode);
    /// `None` leaves sampling to the engine output stage.
    pub sampling: Option<crate::Sampling>,
    /// Prepared media for a prompt-prefill step. Retained by the request until the backend
    /// completes the media prefill; `None` for text and decode steps.
    pub media: Option<Arc<MediaPlan>>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelOutput {
    /// A backend's plain-greedy decision for the position after the accepted tokens.
    /// The producer validates every logit as finite and preserves F32 total-order/lowest-index ties.
    /// When present, logits are omitted; the engine validates eligibility and vocabulary bounds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_token: Option<u32>,
    pub logits: Vec<f32>,
    pub hidden: Vec<Vec<f32>>,
    /// Tokens the backend already decided for this step; empty when the engine samples.
    #[serde(default)]
    pub tokens: Vec<u32>,
}
#[derive(Debug, Clone)]
pub struct TaskOutput {
    pub request: RequestId,
    pub output: ModelOutput,
}

impl Clone for StepPlan {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            decision: self.decision,
            program: self.program,
            role: self.role,
            work: self.work.clone(),
            cost: self.cost,
            graph: self.graph.clone(),
            quantum_overrun: self.quantum_overrun,
        }
    }
    fn clone_from(&mut self, source: &Self) {
        self.id = source.id;
        self.decision = source.decision;
        self.program = source.program;
        self.role = source.role;
        self.cost = source.cost;
        self.quantum_overrun = source.quantum_overrun;
        self.work.clone_from(&source.work);
        self.graph.clone_from(&source.graph);
    }
}
impl Clone for SchedulingDecision {
    fn clone(&self) -> Self {
        Self {
            window: self.window.clone(),
            id: self.id,
            step: self.step.clone(),
            deferred: self.deferred.clone(),
            selected: self.selected.clone(),
        }
    }
    fn clone_from(&mut self, source: &Self) {
        self.id = source.id;
        self.window.clone_from(&source.window);
        self.step.clone_from(&source.step);
        self.deferred.clone_from(&source.deferred);
        self.selected.clone_from(&source.selected);
    }
}
