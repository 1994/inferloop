use infer_core::*;
use infer_ir::*;
use infer_scheduler::*;
use infer_spi::{AdmissionPolicy, CostModelProvider};
use std::collections::BTreeMap;

#[cfg(test)]
mod fixtures {
    use super::*;

    pub fn query(tokens: usize) -> CostQuery {
        CostQuery::from_unit(
            ProgramId::new(1).unwrap(),
            BackendKind::Metal,
            ExecutionRole::Prefill,
            tokens,
            16,
            CostEstimate {
                gpu_us: 1,
                ..Default::default()
            },
        )
    }

    pub fn observation(id: u64, tokens: usize, us: u64) -> CostObservation {
        CostObservation {
            step: StepId::new(id).unwrap(),
            work: vec![query(tokens)].into(),
            timing: ExecutionTiming {
                elapsed_us: us,
                source: TimingSource::MetalGpu,
            },
        }
    }
}

use fixtures::*;

#[test]
fn timing_samples_change_batch_estimates_and_checkpoint_restores_exactly() {
    let mut costs =
        CalibratedCosts::new("weights-and-program".into(), CostModelConfig::default()).unwrap();
    assert_eq!(costs.estimate(&[query(8)]).unwrap().gpu_us, 8);
    costs.observe(&observation(1, 8, 800)).unwrap();
    assert_eq!(costs.estimate(&[query(8)]).unwrap().gpu_us, 960);
    assert_eq!(costs.estimate(&[query(4)]).unwrap().gpu_us, 480);
    let saved = costs.capture_state().unwrap();
    let mut restored =
        CalibratedCosts::new("weights-and-program".into(), CostModelConfig::default()).unwrap();
    restored.restore_state(saved.as_deref()).unwrap();
    assert_eq!(costs.inspect(), restored.inspect());
    assert_eq!(
        costs.estimate(&[query(8)]).unwrap(),
        restored.estimate(&[query(8)]).unwrap()
    );
    let mut wrong =
        CalibratedCosts::new("other-weights".into(), CostModelConfig::default()).unwrap();
    assert!(wrong.restore_state(saved.as_deref()).is_err());
}
#[test]
fn corrupt_profiles_and_calibrated_overflow_are_rejected_without_panicking() {
    let mut costs = CalibratedCosts::new("binding".into(), CostModelConfig::default()).unwrap();
    costs.observe(&observation(1, 1, 100)).unwrap();
    let saved = costs.capture_state().unwrap().unwrap();
    for field in ["samples", "updated", "mean_us", "reference_us"] {
        let mut corrupted: serde_json::Value = serde_json::from_slice(&saved).unwrap();
        corrupted["profiles"][0][field] = serde_json::json!(0);
        assert!(
            costs
                .restore_state(Some(&serde_json::to_vec(&corrupted).unwrap()))
                .is_err()
        );
        assert_eq!(costs.capture_state().unwrap().unwrap(), saved);
    }
    let mut corrupted: serde_json::Value = serde_json::from_slice(&saved).unwrap();
    corrupted["profiles"][0]["mean_us"] = serde_json::json!(u64::MAX);
    corrupted["profiles"][0]["reference_us"] = serde_json::json!(1);
    costs
        .restore_state(Some(&serde_json::to_vec(&corrupted).unwrap()))
        .unwrap();
    let mut q = query(1);
    q.fallback_per_token_us = u64::MAX;
    assert!(costs.estimate(&[q]).is_err());
}
#[test]
fn calibration_is_bounded_and_rejects_wrong_timing_source_and_invalid_data() {
    let mut costs = CalibratedCosts::new(
        "binding".into(),
        CostModelConfig {
            max_profiles: 2,
            ..Default::default()
        },
    )
    .unwrap();
    for (id, tokens) in [(1, 1), (2, 4), (3, 16)] {
        costs.observe(&observation(id, tokens, 100)).unwrap();
    }
    assert_eq!(costs.inspect().profiles, 2);
    assert_eq!(costs.inspect().evictions, 1);
    let before = costs.capture_state().unwrap();
    let mut bad = observation(4, 8, 200);
    bad.timing.source = TimingSource::CpuWall;
    assert!(costs.observe(&bad).is_err());
    assert_eq!(before, costs.capture_state().unwrap());
    bad.timing.source = TimingSource::MetalGpu;
    bad.timing.elapsed_us = 0;
    assert!(costs.observe(&bad).is_err());
    assert!(costs.estimate(&[query(0)]).is_err());
}
#[test]
fn static_cost_mode_does_not_learn_from_wall_measurements() {
    let mut costs = CalibratedCosts::new(
        "binding".into(),
        CostModelConfig {
            adaptive: false,
            ..Default::default()
        },
    )
    .unwrap();
    costs.observe(&observation(1, 8, 10000)).unwrap();
    assert_eq!(costs.estimate(&[query(8)]).unwrap().gpu_us, 8);
    assert_eq!(costs.inspect().observations, 0);
}
const fn admission() -> AdmissionInput<'static> {
    AdmissionInput {
        tenant: "a",
        weight: 1,
        reserved_tokens: 16,
        required_pages: 1,
        initial_pages: 1,
        required_bytes: Some(1024),
        tenant_active: 0,
        tenant_tokens: 0,
        tenant_pages: 0,
        free_pages: 4,
        free_bytes: Some(4096),
        state_capacity_bytes: None,
        state_budget_bytes: None,
        now_us: 10,
        target_us: Some(100),
        predicted_latency_us: 20,
        minimum_execution_us: 2,
        max_atomic_us: 1000,
        resources_ready: true,
    }
}
#[test]
fn admission_checks_tenant_physical_capacity_readiness_and_predicted_slo() {
    let policy = ResourceAdmission::new(AdmissionConfig {
        default_tenant: TenantQuota {
            max_active_requests: 1,
            ..Default::default()
        },
        reject_infeasible_slo: true,
        ..Default::default()
    })
    .unwrap();
    let mut input = admission();
    assert!(policy.check(&input).unwrap().rejection.is_none());
    input.tenant_active = 1;
    assert_eq!(
        policy.check(&input).unwrap().rejection,
        Some(AdmissionReason::TenantRequests)
    );
    input.tenant_active = 0;
    input.free_bytes = Some(512);
    assert_eq!(
        policy.check(&input).unwrap().rejection,
        Some(AdmissionReason::StateBytes)
    );
    input.free_bytes = Some(4096);
    input.resources_ready = false;
    assert_eq!(
        policy.check(&input).unwrap().rejection,
        Some(AdmissionReason::MediaNotReady)
    );
    input.resources_ready = true;
    input.target_us = Some(20);
    assert_eq!(
        policy.check(&input).unwrap().rejection,
        Some(AdmissionReason::SloInfeasible)
    );
}
#[test]
fn a_state_byte_shortfall_waits_when_the_request_fits_the_budget() {
    let policy = ResourceAdmission::new(AdmissionConfig::default()).unwrap();
    let mut input = admission();
    input.required_bytes = Some(2048);
    input.state_capacity_bytes = Some(1024);
    input.state_budget_bytes = Some(8192);
    assert_eq!(
        policy.check(&input).unwrap().rejection,
        Some(AdmissionReason::StateBytesBusy)
    );
    // Above the budget the request can never fit, so it is refused outright.
    input.state_budget_bytes = Some(1024);
    assert_eq!(
        policy.check(&input).unwrap().rejection,
        Some(AdmissionReason::StateBytes)
    );
    // Without a reported budget there is nothing to wait for.
    input.state_budget_bytes = None;
    assert_eq!(
        policy.check(&input).unwrap().rejection,
        Some(AdmissionReason::StateBytes)
    );
}
#[test]
fn soft_slo_admission_records_risk_and_tenant_weight_override_is_enforced() {
    let mut config = AdmissionConfig::default();
    config.tenants.insert(
        "a".into(),
        TenantQuota {
            weight: Some(2),
            ..Default::default()
        },
    );
    let policy = ResourceAdmission::new(config).unwrap();
    let mut input = admission();
    input.target_us = Some(20);
    assert_eq!(
        policy.check(&input).unwrap().rejection,
        Some(AdmissionReason::TenantWeight)
    );
    input.weight = 2;
    let decision = policy.check(&input).unwrap();
    assert!(decision.slo_at_risk);
    assert!(decision.rejection.is_none());
}

#[test]
fn long_context_admission_uses_resources_and_explicit_tenant_quotas() {
    let mut input = admission();
    input.tenant_active = 300;
    input.tenant_tokens = 3 * 262_144;
    input.reserved_tokens = 262_144;
    input.tenant_pages = input.tenant_tokens / 16;
    input.required_pages = input.reserved_tokens / 16;
    let policy = ResourceAdmission::new(AdmissionConfig::default()).unwrap();
    assert!(policy.check(&input).unwrap().rejection.is_none());
    input.free_bytes = Some(512);
    assert_eq!(
        policy.check(&input).unwrap().rejection,
        Some(AdmissionReason::StateBytes)
    );
    input.free_bytes = Some(4096);
    input.free_pages = 0;
    assert_eq!(
        policy.check(&input).unwrap().rejection,
        Some(AdmissionReason::Capacity)
    );
    input.free_pages = 4;
    for (quota, expected) in [
        (
            TenantQuota {
                max_active_requests: 256,
                ..Default::default()
            },
            AdmissionReason::TenantRequests,
        ),
        (
            TenantQuota {
                max_reserved_tokens: 65536,
                ..Default::default()
            },
            AdmissionReason::TenantTokens,
        ),
        (
            TenantQuota {
                max_state_pages: 4096,
                ..Default::default()
            },
            AdmissionReason::TenantPages,
        ),
    ] {
        let policy = ResourceAdmission::new(AdmissionConfig {
            tenants: BTreeMap::from([("a".into(), quota)]),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(policy.check(&input).unwrap().rejection, Some(expected));
    }
}

#[test]
fn queued_byte_admission_distinguishes_occupied_from_insufficient_capacity() {
    let policy = ResourceAdmission::new(AdmissionConfig::default()).unwrap();
    let mut input = admission();
    input.free_bytes = Some(0);
    assert_eq!(
        policy.check(&input).unwrap().rejection,
        Some(AdmissionReason::StateBytes)
    );
    input.state_capacity_bytes = Some(1024);
    assert!(policy.check(&input).unwrap().rejection.is_none());
    input.required_bytes = Some(1025);
    assert_eq!(
        policy.check(&input).unwrap().rejection,
        Some(AdmissionReason::StateBytes)
    );
}
