use crate::Engine;
use crate::stages::prefill::retained_readout;
use infer_core::{Error, ErrorCode, Result};
use infer_ir::{
    AdmissionDecision, AdmissionInput, AdmissionReason, CanonicalRequest, ExecutionRole, MediaPlan,
    RequestInput, Workload, WorkloadPlan,
};
use infer_spi::{AdmissionPolicy, BackendProvider, SchedulingPolicy};

/// Deadline targets a request can carry: completion deadline, TTFT and TPOT.
const MAX_DEADLINE_TARGETS: usize = 3;

impl<B: BackendProvider, P: SchedulingPolicy> Engine<B, P> {
    /// Install a planning provider before any admission.
    /// # Errors
    /// Rejects provider changes after request or replay history exists.
    pub fn with_admission_policy(
        mut self,
        policy: impl AdmissionPolicy + Send + Sync + 'static,
    ) -> Result<Self> {
        if !self.host.requests.is_empty() || !self.actions.is_empty() {
            return Err(Error::new(
                ErrorCode::Conflict,
                "install admission policy before admission",
            ));
        }
        self.admission_policy = Box::new(policy);
        Ok(self)
    }
    pub(crate) fn check_admission(
        &self,
        request: &CanonicalRequest,
        plan: &WorkloadPlan,
    ) -> Result<AdmissionDecision> {
        let media = match &request.input {
            RequestInput::Sequence { tokens, media } if !media.is_empty() => Some(
                MediaPlan::from_sequence(tokens, media)?
                    .ok_or_else(|| Error::invariant("media request has no prepared media plan"))?,
            ),
            _ => None,
        };
        let readout = retained_readout(&request.workload);
        let bytes = if let Some(media) = &media {
            self.backend
                .state_reservation_bytes_for_media(plan.reserved_tokens, readout, media)?
        } else {
            self.backend
                .state_reservation_bytes_for(plan.reserved_tokens, readout)?
        };
        self.check_admission_with_bytes(request, plan, bytes)
    }
    pub(crate) fn check_admission_with_bytes(
        &self,
        request: &CanonicalRequest,
        plan: &WorkloadPlan,
        bytes: Option<u64>,
    ) -> Result<AdmissionDecision> {
        let generate = matches!(request.workload, Workload::Generate { .. });
        let peak_tokens = plan.reserved_tokens.saturating_sub(usize::from(generate));
        if peak_tokens.div_ceil(self.config.block_size) > self.config.num_gpu_blocks
            || self
                .backend
                .kv_cache()
                .is_some_and(|c| peak_tokens.div_ceil(c.block_size) > c.total_blocks)
        {
            return Err(Error::new(
                ErrorCode::Capacity,
                "request cannot fit the KV/logical page pool even in isolation",
            ));
        }
        let role = if generate {
            ExecutionRole::Prefill
        } else {
            ExecutionRole::Forward
        };
        let (first, minimum, complete) = self.admission_costs(request, plan, role)?;
        // Work already in flight cannot be preempted. Queued work is not assumed to run first.
        let blocking = self.inflight.as_ref().map_or(0, |s| {
            s.step
                .cost
                .gpu_us
                .saturating_sub(self.now_us.saturating_sub(s.submitted_us))
        });
        let mut targets = Vec::with_capacity(MAX_DEADLINE_TARGETS);
        if let Some(deadline) = request.qos.deadline_us {
            targets.push((deadline, complete.saturating_add(blocking)));
        }
        if let Some(ttft) = request.qos.ttft_slo_us {
            targets.push((
                self.now_us.saturating_add(ttft),
                first.saturating_add(blocking),
            ));
        }
        if let Some(tpot) = request.qos.tpot_slo_us {
            let decode = self
                .costs
                .estimate(&[self.base_query(ExecutionRole::Decode, 1, plan.reserved_tokens)])?
                .gpu_us;
            targets.push((self.now_us.saturating_add(tpot), decode));
        }
        let target = targets
            .into_iter()
            .min_by_key(|(t, c)| i128::from(*t) - i128::from(self.now_us) - i128::from(*c));
        let (active, tokens, pages) = self.tenant_usage(&request.qos.tenant);
        let resources_ready = match &request.input {
            RequestInput::Sequence { media, .. } if !media.is_empty() => {
                if !self.backend.supports_multimodal() {
                    return Err(Error::unsupported(
                        "backend has no bound multimodal encoder for media input",
                    ));
                }
                true
            }
            RequestInput::Sequence { .. } | RequestInput::Pairs { .. } => true,
        };
        let decision = self.admission_policy.check(&AdmissionInput {
            tenant: &request.qos.tenant,
            weight: request.qos.weight,
            reserved_tokens: plan.reserved_tokens,
            required_pages: plan.reserved_tokens.div_ceil(self.config.block_size),
            initial_pages: 1,
            required_bytes: bytes,
            tenant_active: active,
            tenant_tokens: tokens,
            tenant_pages: pages,
            free_pages: self.state.free_pages(),
            free_bytes: self.backend.free_state_bytes()?,
            state_capacity_bytes: if self.backend.tracks_reservation_intent() {
                self.backend.state_capacity_bytes()?
            } else {
                None
            },
            now_us: self.now_us,
            target_us: target.map(|t| t.0),
            predicted_latency_us: target.map_or(complete, |t| t.1),
            minimum_execution_us: minimum,
            max_atomic_us: self.config.scheduler.max_singleton_gpu_us,
            resources_ready,
        })?;
        if let Some(reason) = decision.rejection {
            return Err(Error::new(
                if reason == AdmissionReason::TenantWeight {
                    ErrorCode::InvalidInput
                } else {
                    ErrorCode::Capacity
                },
                format!(
                    "admission {reason:?}: required={}, available={}",
                    decision.required, decision.available
                ),
            ));
        }
        Ok(decision)
    }
    pub(crate) fn tenant_usage(&self, tenant: &str) -> (usize, usize, usize) {
        self.tenants
            .get(tenant)
            .map_or((0, 0, 0), |usage| (usage.active, usage.tokens, usage.pages))
    }
    pub(crate) fn admission_costs(
        &self,
        request: &CanonicalRequest,
        plan: &WorkloadPlan,
        role: ExecutionRole,
    ) -> Result<(u64, u64, u64)> {
        let first = self
            .costs
            .estimate(&[self.base_query(role, plan.units[0].len(), plan.units[0].len())])?
            .gpu_us;
        let minimum = self.costs.estimate(&[self.base_query(role, 1, 1)])?.gpu_us;
        let complete = if let Workload::Generate { max_new_tokens } = request.workload {
            let decode = self
                .costs
                .estimate(&[self.base_query(ExecutionRole::Decode, 1, plan.reserved_tokens)])?
                .gpu_us;
            first.saturating_add(decode.saturating_mul(max_new_tokens.saturating_sub(1) as u64))
        } else {
            plan.units.iter().try_fold(0u64, |sum, tokens| {
                let cost = self
                    .costs
                    .estimate(&[self.base_query(role, tokens.len(), tokens.len())])?
                    .gpu_us;
                sum.checked_add(cost)
                    .ok_or_else(|| Error::invalid("admission latency overflow"))
            })?
        };
        Ok((first, minimum, complete))
    }
}
