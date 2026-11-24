use super::*;
use serde_json::json;

fn plan(policy: AcceptancePolicy) -> SpeculationPlan {
    SpeculationPlan {
        proposal: ProgramId::new(1).expect("valid id"),
        verification: ProgramId::new(2).expect("valid id"),
        hidden_state_taps: vec![3],
        max_candidates: 4,
        acceptance_policy: policy,
    }
}

#[test]
fn the_acceptance_policy_round_trips_by_name() {
    for (policy, name) in [
        (AcceptancePolicy::Greedy, "greedy"),
        (AcceptancePolicy::Residual, "residual"),
    ] {
        let encoded = serde_json::to_value(plan(policy)).expect("plan serializes");
        assert_eq!(encoded["acceptance_policy"], json!(name));
        let decoded: SpeculationPlan = serde_json::from_value(encoded).expect("plan deserializes");
        assert_eq!(decoded.acceptance_policy, policy);
    }
}

#[test]
fn an_unknown_policy_is_refused_rather_than_carried() {
    let mut encoded =
        serde_json::to_value(plan(AcceptancePolicy::Greedy)).expect("plan serializes");
    encoded["acceptance_policy"] = json!("greedy_typo");
    assert!(serde_json::from_value::<SpeculationPlan>(encoded).is_err());
}

#[test]
fn greedy_is_what_a_plan_defaults_to() {
    assert_eq!(AcceptancePolicy::default(), AcceptancePolicy::Greedy);
}
