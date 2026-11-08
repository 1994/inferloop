//! Scheduling inputs, evidence and cold-path cost calibration contracts.
use crate::{BackendKind, CostEstimate, ExecutionRole};
use infer_core::{ProgramId, RequestId, StepId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Default prefill chunk size in tokens when a scheduler is configured without overrides.
const DEFAULT_PREFILL_CHUNK_TOKENS: usize = 32;
/// Default decode/forward chunk size in tokens when a scheduler is configured without overrides.
const DEFAULT_FORWARD_CHUNK_TOKENS: usize = 64;
/// Default weighted-fair quantum in tokens granted to a ready request per turn.
const DEFAULT_FAIR_QUANTUM_TOKENS: usize = 8;
/// Default maximum time a request may wait before aging applies, in microseconds.
const DEFAULT_MAX_WAIT_US: u64 = 20_000;
/// Default share of the planning budget reserved for urgent-SLO requests, in percent.
const DEFAULT_URGENT_BUDGET_PERCENT: u32 = 75;
/// Default upper bound on a soft execution quantum for an otherwise feasible singleton token,
/// in microseconds.
const DEFAULT_MAX_SINGLETON_GPU_US: u64 = 1_000_000;
/// Default number of deterministic packing trials attempted per scheduling decision.
const DEFAULT_MAX_PLANNING_PROBES: usize = 4096;
/// Default maximum number of retained cost profiles.
const DEFAULT_MAX_PROFILES: usize = 512;
/// Default EWMA alpha applied to new cost observations, in percent.
const DEFAULT_EWMA_ALPHA_PERCENT: u32 = 25;
/// Default safety margin added to predicted cost, in percent.
const DEFAULT_SAFETY_MARGIN_PERCENT: u32 = 20;
/// Unconfigured tenants share the runtime request capacity without an additional quota.
const DEFAULT_MAX_ACTIVE_REQUESTS: usize = usize::MAX;
/// Token capacity follows model and resource admission unless a tenant quota is explicit.
const DEFAULT_MAX_RESERVED_TOKENS: usize = usize::MAX;
/// Page capacity follows the backend pool unless a tenant quota is explicit.
const DEFAULT_MAX_STATE_PAGES: usize = usize::MAX;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SchedulerConfig {
    pub prefill_chunk_tokens: usize,
    pub forward_chunk_tokens: usize,
    pub fair_quantum_tokens: usize,
    pub max_wait_us: u64,
    pub urgent_budget_percent: u32,
    /// Execution-time quantum is soft only for an otherwise feasible singleton token.
    pub max_singleton_gpu_us: u64,
    /// Deterministic packing trials per decision; yields a partial batch when exhausted.
    pub max_planning_probes: usize,
}
impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            prefill_chunk_tokens: DEFAULT_PREFILL_CHUNK_TOKENS,
            forward_chunk_tokens: DEFAULT_FORWARD_CHUNK_TOKENS,
            fair_quantum_tokens: DEFAULT_FAIR_QUANTUM_TOKENS,
            max_wait_us: DEFAULT_MAX_WAIT_US,
            urgent_budget_percent: DEFAULT_URGENT_BUDGET_PERCENT,
            max_singleton_gpu_us: DEFAULT_MAX_SINGLETON_GPU_US,
            max_planning_probes: DEFAULT_MAX_PLANNING_PROBES,
        }
    }
}
/// Captured prompt shapes used by one resident state. Zero narrow width means no tail graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillGeometry {
    pub width: usize,
    pub narrow_width: usize,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostQuery {
    pub program: ProgramId,
    pub backend: BackendKind,
    pub role: ExecutionRole,
    pub tokens: usize,
    pub context_tokens: usize,
    /// GPU execution per token, or CPU execution per token on the Host backend.
    pub fallback_per_token_us: u64,
    pub workspace_bytes: u64,
    /// Additional resources only: already-reserved KV must not be charged twice.
    pub num_gpu_blocks: usize,
    pub state_bytes: u64,
    pub transfer_us: u64,
    pub encoder_us: u64,
    #[serde(default)]
    pub page_growth: Option<PageGrowth>,
    #[serde(default)]
    pub logical_growth: Option<PageGrowth>,
    /// Advisory replay geometry; resource and progress limits still take precedence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefill_geometry: Option<PrefillGeometry>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PageGrowth {
    pub block_size: usize,
    pub allocated_pages: usize,
    pub bytes_per_page: u64,
    pub cow_tail: bool,
}
impl PageGrowth {
    #[must_use]
    pub fn required_pages(&self, end_tokens: usize) -> Option<usize> {
        (self.block_size > 0).then(|| {
            end_tokens
                .div_ceil(self.block_size)
                .saturating_sub(self.allocated_pages)
                .saturating_add(usize::from(self.cow_tail))
        })
    }
}
impl CostQuery {
    #[must_use]
    pub const fn from_unit(
        program: ProgramId,
        backend: BackendKind,
        role: ExecutionRole,
        tokens: usize,
        context_tokens: usize,
        unit: CostEstimate,
    ) -> Self {
        Self {
            program,
            backend,
            role,
            tokens,
            context_tokens,
            fallback_per_token_us: unit.gpu_us,
            workspace_bytes: unit.workspace_bytes,
            num_gpu_blocks: unit.num_gpu_blocks,
            state_bytes: unit.state_bytes,
            transfer_us: unit.transfer_us,
            encoder_us: unit.encoder_us,
            page_growth: None,
            logical_growth: None,
            prefill_geometry: None,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimingSource {
    CpuWall,
    MetalGpu,
    CudaGpu,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionTiming {
    pub elapsed_us: u64,
    pub source: TimingSource,
}
impl ExecutionTiming {
    #[must_use]
    pub const fn matches_backend(&self, backend: BackendKind) -> bool {
        // The CPU test backends E1 deleted had their own `CpuWall` source; the device backends are
        // the only ones left to match.
        matches!(
            (backend, self.source),
            (BackendKind::Cuda, TimingSource::CudaGpu)
                | (BackendKind::Metal, TimingSource::MetalGpu)
        )
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostObservation {
    pub step: StepId,
    pub work: crate::SharedOutput<CostQuery>,
    pub timing: ExecutionTiming,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CostModelConfig {
    pub adaptive: bool,
    pub max_profiles: usize,
    pub ewma_alpha_percent: u32,
    pub safety_margin_percent: u32,
}
impl Default for CostModelConfig {
    fn default() -> Self {
        Self {
            adaptive: true,
            max_profiles: DEFAULT_MAX_PROFILES,
            ewma_alpha_percent: DEFAULT_EWMA_ALPHA_PERCENT,
            safety_margin_percent: DEFAULT_SAFETY_MARGIN_PERCENT,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostModelInspection {
    pub provider: String,
    pub adaptive: bool,
    pub profiles: usize,
    pub observations: u64,
    pub evictions: u64,
    pub last_source: Option<TimingSource>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionReason {
    UrgentSlo,
    Aging,
    DecodeLatency,
    WeightedFair,
    MinimumQuantum,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectionEvidence {
    pub request: RequestId,
    pub role: ExecutionRole,
    pub tokens: usize,
    pub target_us: Option<u64>,
    pub slack_us: Option<i64>,
    pub waiting_us: u64,
    pub virtual_finish: u64,
    pub predicted_us: u64,
    pub reason: SelectionReason,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TenantQuota {
    pub max_active_requests: usize,
    pub max_reserved_tokens: usize,
    pub max_state_pages: usize,
    pub weight: Option<u32>,
}
impl Default for TenantQuota {
    fn default() -> Self {
        Self {
            max_active_requests: DEFAULT_MAX_ACTIVE_REQUESTS,
            max_reserved_tokens: DEFAULT_MAX_RESERVED_TOKENS,
            max_state_pages: DEFAULT_MAX_STATE_PAGES,
            weight: None,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AdmissionConfig {
    pub default_tenant: TenantQuota,
    pub tenants: BTreeMap<String, TenantQuota>,
    /// Reject predicted impossible latency targets; this is not an SLO guarantee.
    pub reject_infeasible_slo: bool,
}
#[derive(Debug, Clone)]
pub struct AdmissionInput<'a> {
    pub tenant: &'a str,
    pub weight: u32,
    pub reserved_tokens: usize,
    pub required_pages: usize,
    pub initial_pages: usize,
    pub required_bytes: Option<u64>,
    pub tenant_active: usize,
    pub tenant_tokens: usize,
    pub tenant_pages: usize,
    pub free_pages: usize,
    pub free_bytes: Option<u64>,
    /// Single-reservation byte budget when an adapter owns queued reservation intent.
    /// Physical allocation still uses current free bytes after earlier leases finish.
    pub state_capacity_bytes: Option<u64>,
    /// Stable ceiling the state budget can ever fund for one request.
    /// A request above it is rejected outright; one that only exceeds the momentary
    /// capacity waits for other leases to drain instead.
    pub state_budget_bytes: Option<u64>,
    pub now_us: u64,
    pub target_us: Option<u64>,
    pub predicted_latency_us: u64,
    pub minimum_execution_us: u64,
    pub max_atomic_us: u64,
    pub resources_ready: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdmissionReason {
    Capacity,
    TenantRequests,
    TenantTokens,
    TenantPages,
    TenantWeight,
    StateBytes,
    /// The request fits the state budget but not the capacity available right now.
    StateBytesBusy,
    MediaNotReady,
    SloInfeasible,
    AtomicExecution,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdmissionDecision {
    pub rejection: Option<AdmissionReason>,
    pub required: u64,
    pub available: u64,
    pub predicted_latency_us: u64,
    pub target_us: Option<u64>,
    pub slo_at_risk: bool,
}
