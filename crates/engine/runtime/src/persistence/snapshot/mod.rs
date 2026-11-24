mod capture;
mod restore;
mod validation;
use crate::preparation::request_bytes;
use crate::{Engine, RequestRecord, RuntimeConfig, engine::TenantShare};
use infer_core::map::BoundedMap;
use infer_core::{Error, IdAllocator, RequestId, Result};
use infer_ir::{
    CanonicalRequest, CostObservation, ExecutionProgram, ModelIr, SchedulingDecision, StepPlan,
    Workload,
};
use infer_quality::ProgressGuard;
use infer_state::SequenceStates;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::{collections::BTreeMap, collections::BTreeSet, collections::VecDeque};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ReplayAction {
    Submit(Arc<CanonicalRequest>),
    ConstrainedSubmit(RequestId),
    Tick { now_us: u64 },
    Poll { now_us: u64 },
    Quiesce { now_us: u64 },
    Cancel(RequestId),
    Drain(RequestId),
    CostFeedback(CostObservation),
}
impl ReplayAction {
    pub(crate) fn retained_bytes(&self) -> usize {
        match self {
            Self::Submit(request) => request_bytes(request),
            Self::CostFeedback(observation) => observation
                .work
                .len()
                .saturating_mul(size_of::<infer_ir::CostQuery>()),
            _ => 0,
        }
    }
    pub(crate) fn retained_tokens(&self) -> usize {
        match self {
            Self::Submit(request) => match &request.input {
                infer_ir::RequestInput::Sequence { tokens, .. } => tokens.len(),
                infer_ir::RequestInput::Pairs { query, documents } => {
                    documents.iter().fold(query.len(), |count, tokens| {
                        count.saturating_add(tokens.len())
                    })
                }
            },
            _ => 0,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSnapshot {
    #[serde(default)]
    pub fault: Option<crate::EngineFault>,
    pub backend_identity: String,
    pub policy_identity: String,
    pub workload_identity: String,
    pub checkpoint_supported: bool,
    pub model: ModelIr,
    pub program: ExecutionProgram,
    pub config: RuntimeConfig,
    pub state: SequenceStates,
    pub requests: BTreeMap<RequestId, RequestRecord>,
    pub queues: infer_scheduler::RequestQueue,
    pub resource_epoch: u64,
    pub cost_epoch: u64,
    /// Completed and active calibration sweeps; the cursor may refer to a retired owner.
    pub priority_epoch: u64,
    pub priority_sweep_epoch: u64,
    pub priority_cursor: Option<RequestId>,
    pub seen_requests: BTreeSet<RequestId>,
    pub retired_request_floor: u64,
    pub(crate) tenants: BoundedMap<Arc<str>, TenantShare>,
    pub(crate) ids: IdAllocator,
    pub global_progress_epoch: u64,
    pub now_us: u64,
    pub(crate) guard: ProgressGuard,
    pub decisions: VecDeque<SchedulingDecision>,
    pub actions: VecDeque<ReplayAction>,
    pub dropped_actions: u64,
    pub inflight: Option<StepPlan>,
    pub execution_state: Option<Vec<u8>>,
    #[serde(default)]
    pub cost_provider_identity: String,
    #[serde(default)]
    pub admission_provider_identity: String,
    #[serde(default)]
    pub cost_state: Option<Vec<u8>>,
    #[serde(default)]
    pub pending_cost_observations: VecDeque<CostObservation>,
    pub preemption_focus: Option<RequestId>,
}
fn validate_active_record(record: &RequestRecord) -> Result<()> {
    if record.tenant.as_ref() != record.request.qos.tenant
        || record.unit >= record.plan.units.len()
        || record.prefill_done > record.prefill_target
        || record.prefill_target > record.context.len()
        || record.prefill_target < record.plan.units[record.unit].len()
    {
        return Err(Error::invalid("invalid active unit cursor"));
    }
    let initial = &record.plan.units[record.unit];
    if record.context.prompt.as_slice() != initial.as_slice()
        || record.context.generated_len() != record.generated.len()
    {
        return Err(Error::invalid(
            "snapshot context does not match input/generated tokens",
        ));
    }
    match record.request.workload {
        Workload::Generate { max_new_tokens }
            if record.generated.len() >= max_new_tokens || !record.outputs.is_empty() =>
        {
            return Err(Error::invalid(
                "invalid generated token count in checkpoint",
            ));
        }
        Workload::Generate { .. } => {}
        _ if !record.generated.is_empty() || record.outputs.len() != record.unit => {
            return Err(Error::invalid("invalid forward outputs in checkpoint"));
        }
        _ => {}
    }
    Ok(())
}
