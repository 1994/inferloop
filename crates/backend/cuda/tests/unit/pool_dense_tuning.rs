use super::*;
use crate::tuning::{MIN_REPS, TilingTable, projection_budget::SearchBudget};

fn times() -> resident_gemm::PairedTimes {
    resident_gemm::PairedTimes {
        baseline_ms: vec![1.0; MIN_REPS],
        candidate_ms: vec![0.8; MIN_REPS],
        baseline_median_ms: 1.0,
        candidate_median_ms: 0.8,
        speedup: 1.25,
    }
}
use crate::tuning::resident_gemm;
fn trial(choice: Choice, order: MeasurementOrder) -> Trial {
    Trial {
        choice,
        order,
        cold: times(),
        warm: times(),
    }
}
fn entry() -> Entry {
    Entry {
        key: Key {
            native: ResidentGemmKey {
                hardware: "synthetic-device".into(),
                driver_version: 1,
                driver_build: Some("test-driver".into()),
                kernel_schema: "pool-dense-test".into(),
                kernel_fingerprint: "frozen-kernel-identity".into(),
                activation_mode: "f32-bf16-pooled-dense".into(),
                m: 24,
                n: 64,
                k: 4096,
            },
            tile: [4, 256],
            baseline: Choice::Native,
        },
        selected: Choice::Gemv,
        exact_candidates_verified: true,
        trials: vec![
            trial(Choice::Gemv, MeasurementOrder::BaselineFirst),
            trial(Choice::Shared, MeasurementOrder::BaselineFirst),
        ],
        rejected: vec![],
        confirmation: trial(Choice::Gemv, MeasurementOrder::CandidateFirst),
    }
}
#[test]
fn cache_requires_full_numeric_and_bidirectional_evidence() {
    assert!(entry().valid());
    let mut missing = entry();
    missing.exact_candidates_verified = false;
    assert!(!missing.valid());
    let mut missing = entry();
    missing.confirmation.order = MeasurementOrder::Unspecified;
    assert!(!missing.valid());
    let mut missing = entry();
    missing.trials.pop();
    assert!(!missing.valid());
    let mut repeated = entry();
    repeated.trials[1] = repeated.trials[0].clone();
    assert!(!repeated.valid());
}
#[test]
fn cache_rejects_timing_tampering_and_single_cache_state_winners() {
    let mut changed = entry();
    changed.confirmation.cold.candidate_median_ms = 0.6;
    assert!(!changed.valid());
    let mut changed = entry();
    changed.confirmation.warm.speedup = 2.0;
    assert!(!changed.valid());
    let mut changed = entry();
    changed.confirmation.warm = resident_gemm::PairedTimes {
        baseline_ms: vec![1.0; MIN_REPS],
        candidate_ms: vec![1.1; MIN_REPS],
        baseline_median_ms: 1.0,
        candidate_median_ms: 1.1,
        speedup: 1.0 / 1.1,
    };
    assert!(!changed.valid());
}
#[test]
fn legacy_fallback_never_newly_promotes_tensor_core_arithmetic() {
    let mut small = entry();
    small.key.native.m = 4;
    small.key.baseline = Choice::Gemv;
    small.selected = Choice::Shared;
    small.trials.remove(0);
    small.confirmation.choice = Choice::Shared;
    assert!(small.valid());
    small.selected = Choice::Native;
    small.confirmation.choice = Choice::Native;
    assert!(!small.valid());
    let mut fallback = entry();
    fallback.selected = Choice::Native;
    fallback.confirmation.choice = Choice::Native;
    assert!(fallback.valid());
    fallback.key.tile[0] = 3;
    assert!(!fallback.valid());
}
#[test]
fn rejected_candidates_are_complete_and_cannot_also_count_as_measured() {
    let mut rejected = entry();
    rejected.trials.pop();
    rejected
        .rejected
        .push((Choice::Shared, "complete output mismatch".into()));
    assert!(rejected.valid());
    rejected.rejected[0].1.clear();
    assert!(!rejected.valid());
    let mut duplicate = entry();
    duplicate
        .rejected
        .push((Choice::Shared, "duplicate proof".into()));
    assert!(!duplicate.valid());
}
#[test]
fn old_cache_migrates_and_crossover_evidence_round_trips() {
    let mut table: TilingTable =
        serde_json::from_str(r#"{"device":"legacy","entries":[]}"#).unwrap();
    assert!(table.resident_pool_dense.is_empty());
    table.resident_pool_dense.push(entry());
    let restored: TilingTable =
        serde_json::from_str(&serde_json::to_string(&table).unwrap()).unwrap();
    assert!(restored.resident_pool_dense[0].valid());
    assert_eq!(restored.resident_pool_dense[0].selected, Choice::Gemv);
}
#[test]
fn budget_counts_actual_tile_geometries_once_and_tracks_completion() {
    let shape = [24, 64, 4096];
    let a = LinearTiling::new(4, 256).unwrap();
    let b = LinearTiling::new(4, 512).unwrap();
    let mut budget = SearchBudget::default();
    assert!(budget.plan(kind(a), shape, search_trials(shape)).unwrap());
    let first = budget.snapshot();
    assert!(!budget.plan(kind(a), shape, search_trials(shape)).unwrap());
    assert_eq!(first.allowance, budget.snapshot().allowance);
    assert!(budget.plan(kind(b), shape, search_trials(shape)).unwrap());
    assert_eq!(budget.snapshot().planned, 2);
    budget.complete(kind(a), shape);
    assert_eq!(budget.snapshot().pending, vec![(kind(b), shape)]);
}
#[test]
fn independent_numeric_verifier_rejects_truncation_nonfinite_and_changed_bits() {
    assert!(verify(&[1.0, -0.0], &[1.0, -0.0]).is_ok());
    assert!(verify(&[], &[]).is_err());
    assert!(verify(&[1.0], &[1.0, 2.0]).is_err());
    assert!(verify(&[f32::NAN], &[f32::NAN]).is_err());
    assert!(verify(&[0.0], &[-0.0]).is_err());
}

#[test]
fn private_reference_scope_restores_after_nested_failure() {
    assert_eq!(recorded_choice(Choice::Native), Choice::Native);
    let failed = with_private_reference(|| {
        assert_eq!(recorded_choice(Choice::Native), Choice::Gemv);
        with_private_reference(|| assert_eq!(recorded_choice(Choice::Shared), Choice::Gemv));
        Err::<(), _>("early return")
    });
    assert!(failed.is_err());
    assert_eq!(recorded_choice(Choice::Native), Choice::Native);
}
