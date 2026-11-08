use infer_core::{Error, Result};
use infer_ir::{AdmissionConfig, AdmissionDecision, AdmissionInput, AdmissionReason};
use infer_spi::AdmissionPolicy;
use std::iter::once;

pub struct ResourceAdmission {
    pub config: AdmissionConfig,
}
impl ResourceAdmission {
    /// # Errors
    /// Returns an invalid-input error for invalid configuration, or a capacity error if the requested resources cannot be reserved.
    pub fn new(config: AdmissionConfig) -> Result<Self> {
        for quota in once(&config.default_tenant).chain(config.tenants.values()) {
            if quota.max_active_requests == 0
                || quota.max_reserved_tokens == 0
                || quota.max_state_pages == 0
                || quota.weight == Some(0)
            {
                return Err(Error::invalid("positive tenant quotas required"));
            }
        }
        if config.tenants.keys().any(String::is_empty) {
            return Err(Error::invalid("empty quota tenant"));
        }
        Ok(Self { config })
    }
}
impl AdmissionPolicy for ResourceAdmission {
    fn check(&self, i: &AdmissionInput<'_>) -> Result<AdmissionDecision> {
        let quota = self
            .config
            .tenants
            .get(i.tenant)
            .unwrap_or(&self.config.default_tenant);
        let mut d = AdmissionDecision {
            rejection: None,
            required: 0,
            available: 0,
            predicted_latency_us: i.predicted_latency_us,
            target_us: i.target_us,
            slo_at_risk: i
                .target_us
                .is_some_and(|t| i.now_us.saturating_add(i.predicted_latency_us) > t),
        };
        let limits = [
            (
                AdmissionReason::TenantRequests,
                i.tenant_active as u128 + 1,
                quota.max_active_requests as u128,
            ),
            (
                AdmissionReason::TenantTokens,
                i.tenant_tokens as u128 + i.reserved_tokens as u128,
                quota.max_reserved_tokens as u128,
            ),
            (
                AdmissionReason::TenantPages,
                i.tenant_pages as u128 + i.required_pages as u128,
                quota.max_state_pages as u128,
            ),
            (
                AdmissionReason::Capacity,
                i.initial_pages as u128,
                i.free_pages as u128,
            ),
            (
                AdmissionReason::AtomicExecution,
                u128::from(i.minimum_execution_us),
                u128::from(i.max_atomic_us),
            ),
        ];
        let mut reject = |reason, required: u128, available: u128| {
            d.rejection = Some(reason);
            d.required = u64::try_from(required).unwrap_or(u64::MAX);
            d.available = u64::try_from(available).unwrap_or(u64::MAX);
        };
        if !i.resources_ready {
            reject(AdmissionReason::MediaNotReady, 1, 0);
        } else if let Some(weight) = quota.weight
            && weight != i.weight
        {
            reject(
                AdmissionReason::TenantWeight,
                u128::from(i.weight),
                u128::from(weight),
            );
        } else if let Some((reason, required, available)) =
            limits.into_iter().find(|(_, r, a)| r > a)
        {
            reject(reason, required, available);
        } else if let (Some(required), Some(available)) =
            (i.required_bytes, i.state_capacity_bytes.or(i.free_bytes))
            && required > available
        {
            // Above the stable budget the request can never be admitted; below it the
            // shortfall is momentary and the caller waits instead of failing the turn.
            let busy = i
                .state_budget_bytes
                .is_some_and(|budget| required <= budget);
            reject(
                if busy {
                    AdmissionReason::StateBytesBusy
                } else {
                    AdmissionReason::StateBytes
                },
                u128::from(required),
                u128::from(available),
            );
        } else if self.config.reject_infeasible_slo
            && let Some(target_us) = i.target_us
            && i.now_us.saturating_add(i.predicted_latency_us) > target_us
        {
            reject(
                AdmissionReason::SloInfeasible,
                u128::from(i.predicted_latency_us),
                u128::from(target_us.saturating_sub(i.now_us)),
            );
        }
        Ok(d)
    }
}
