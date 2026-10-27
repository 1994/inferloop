use super::*;
use crate::tuning::{MIN_REPS, TilingTable};

fn config() -> GemmConfig {
    GemmConfig {
        tile_m: 16,
        tile_n: 64,
        main_loop: super::super::MainLoop::Plain,
    }
}
fn times() -> PairedTimes {
    PairedTimes {
        baseline_ms: vec![1.0; MIN_REPS],
        candidate_ms: vec![0.8; MIN_REPS],
        baseline_median_ms: 1.0,
        candidate_median_ms: 0.8,
        speedup: 1.25,
    }
}
fn trial(order: MeasurementOrder) -> GemmTrial {
    GemmTrial {
        config: config(),
        order,
        cold: times(),
        warm: times(),
    }
}
fn entry() -> Entry {
    Entry {
        key: ResidentGemmKey {
            hardware: "synthetic-profile".into(),
            driver_version: 1,
            driver_build: Some("test-build".into()),
            kernel_schema: "unit-fixture".into(),
            kernel_fingerprint: "frozen-IR".into(),
            activation_mode: "nvfp4-swiglu".into(),
            m: 128,
            n: 256,
            k: 768,
        },
        selected: config(),
        bitwise_verified: true,
        trials: vec![trial(MeasurementOrder::BaselineFirst)],
        rejected: vec![],
        confirmation: trial(MeasurementOrder::CandidateFirst),
    }
}
#[test]
fn candidates_respect_each_runtime_geometry() {
    for shape in [[64, 128, 256], [128, 256, 768], [1024, 512, 512]] {
        let choices: Vec<_> = candidates(shape).collect();
        assert!(choices.contains(&baseline()));
        assert!(choices.iter().any(|choice| *choice != baseline()));
        for choice in choices {
            assert!(shape[0].is_multiple_of(choice.tile_m));
            assert!(shape[1].is_multiple_of(choice.tile_n));
            assert_eq!(choice.main_loop, super::super::MainLoop::Plain);
        }
    }
    for shape in [
        [0, 128, 256],
        [65, 128, 256],
        [64, 127, 256],
        [64, 128, 255],
    ] {
        assert!(candidates(shape).next().is_none());
    }
}
#[test]
fn cache_requires_numeric_and_both_measurement_orders() {
    let valid = entry();
    assert!(valid.valid());
    let mut missing = valid.clone();
    missing.bitwise_verified = false;
    assert!(!missing.valid());
    let mut missing = valid.clone();
    missing.confirmation.order = MeasurementOrder::Unspecified;
    assert!(!missing.valid());
    let mut missing = valid;
    missing.trials.clear();
    assert!(!missing.valid());
}
#[test]
fn cache_rejects_tampered_samples_and_unsupported_kernel_variants() {
    let mut changed = entry();
    changed.confirmation.cold.candidate_median_ms = 0.6;
    assert!(!changed.valid());
    let mut changed = entry();
    changed.confirmation.warm.speedup = 2.0;
    assert!(!changed.valid());
    let mut changed = entry();
    changed.selected.main_loop = super::super::MainLoop::Prefetch8;
    assert!(!changed.valid());
    let mut changed = entry();
    changed.rejected.push((config(), String::new()));
    assert!(!changed.valid());
}
#[test]
fn old_cache_migrates_and_fused_evidence_round_trips() {
    let old: TilingTable = serde_json::from_str(r#"{"device":"legacy","entries":[]}"#).unwrap();
    assert!(old.resident_swiglu.is_empty());
    let mut current = old;
    current.resident_swiglu.push(entry());
    let encoded = serde_json::to_string(&current).unwrap();
    let restored: TilingTable = serde_json::from_str(&encoded).unwrap();
    assert_eq!(restored.resident_swiglu.len(), 1);
    assert!(restored.resident_swiglu[0].valid());
    assert_eq!(restored.resident_swiglu[0].selected, config());
}
#[test]
fn fused_search_budget_is_distinct_and_not_renewed_by_repeated_layers() {
    let session = ResidentTuningSession::default();
    let shape = [64, 256, 768];
    session
        .plan_projection(ProjectionKind::Fp4, shape, search_trials(shape))
        .unwrap();
    session
        .plan_projection(ProjectionKind::Swiglu, shape, search_trials(shape))
        .unwrap();
    let before = session.projection_budget.lock().unwrap().snapshot();
    session
        .plan_projection(ProjectionKind::Swiglu, shape, search_trials(shape))
        .unwrap();
    let after = session.projection_budget.lock().unwrap().snapshot();
    assert_eq!(before.planned, 2);
    assert_eq!(before.allowance, after.allowance);
    session
        .complete_projection(ProjectionKind::Swiglu, shape)
        .unwrap();
    let final_state = session.projection_budget.lock().unwrap().snapshot();
    assert_eq!(final_state.pending, vec![(ProjectionKind::Fp4, shape)]);
}

#[test]
fn baseline_reference_scope_restores_after_nested_early_return() {
    let original = config();
    assert_eq!(recorded_config(original), original);
    let failed = with_baseline(|| {
        assert_eq!(recorded_config(original), baseline());
        with_baseline(|| assert_eq!(recorded_config(original), baseline()));
        Err::<(), _>("early return")
    });
    assert!(failed.is_err());
    assert_eq!(recorded_config(original), original);
}
