use super::*;
use crate::tuning::search::Evidence;
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
        evidence: Evidence {
            selected: Choice::Gemv,
            exact_candidates_verified: true,
            trials: vec![
                trial(Choice::Gemv, MeasurementOrder::BaselineFirst),
                trial(Choice::Shared, MeasurementOrder::BaselineFirst),
            ],
            rejected: Some(vec![]),
            confirmation: trial(Choice::Gemv, MeasurementOrder::CandidateFirst),
        },
    }
}
#[test]
fn cache_requires_full_numeric_and_bidirectional_evidence() {
    assert!(entry().valid());
    let mut missing = entry();
    missing.evidence.exact_candidates_verified = false;
    assert!(!missing.valid());
    let mut missing = entry();
    missing.evidence.confirmation.order = MeasurementOrder::Unspecified;
    assert!(!missing.valid());
    let mut missing = entry();
    missing.evidence.trials.pop();
    assert!(!missing.valid());
    let mut repeated = entry();
    repeated.evidence.trials[1] = repeated.evidence.trials[0].clone();
    assert!(!repeated.valid());
}
#[test]
fn cache_rejects_timing_tampering_and_single_cache_state_winners() {
    let mut changed = entry();
    changed.evidence.confirmation.cold.candidate_median_ms = 0.6;
    assert!(!changed.valid());
    let mut changed = entry();
    changed.evidence.confirmation.warm.speedup = 2.0;
    assert!(!changed.valid());
    let mut changed = entry();
    changed.evidence.confirmation.warm = resident_gemm::PairedTimes {
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
    small.evidence.selected = Choice::Shared;
    small.evidence.trials.remove(0);
    small.evidence.confirmation.choice = Choice::Shared;
    assert!(small.valid());
    small.evidence.selected = Choice::Native;
    small.evidence.confirmation.choice = Choice::Native;
    assert!(!small.valid());
    let mut fallback = entry();
    fallback.evidence.selected = Choice::Native;
    fallback.evidence.confirmation.choice = Choice::Native;
    assert!(fallback.valid());
    fallback.key.tile[0] = 3;
    assert!(!fallback.valid());
}
#[test]
fn rejected_candidates_are_complete_and_cannot_also_count_as_measured() {
    let mut rejected = entry();
    rejected.evidence.trials.pop();
    rejected
        .evidence
        .rejected
        .get_or_insert_default()
        .push((Choice::Shared, "complete output mismatch".into()));
    assert!(rejected.valid());
    rejected.evidence.rejected.as_mut().unwrap()[0].1.clear();
    assert!(!rejected.valid());
    let mut duplicate = entry();
    duplicate
        .evidence
        .rejected
        .get_or_insert_default()
        .push((Choice::Shared, "duplicate proof".into()));
    assert!(!duplicate.valid());
}
#[test]
fn old_cache_migrates_and_crossover_evidence_round_trips() {
    let mut table: TilingTable =
        serde_json::from_str(r#"{"device":"legacy","entries":[]}"#).unwrap();
    assert!(search::records::<Key, Choice>(&table, FAMILY).is_empty());
    table
        .records
        .entry(FAMILY.into())
        .or_default()
        .push(serde_json::to_value(entry()).unwrap());
    let restored: TilingTable =
        serde_json::from_str(&serde_json::to_string(&table).unwrap()).unwrap();
    let found = search::lookup(&restored, FAMILY, &entry().key, Entry::valid).unwrap();
    assert_eq!(found.evidence.selected, Choice::Gemv);
}
#[test]
fn records_from_before_the_shared_protocol_keep_their_selection() {
    // The former section stored the same fields, so its entries load as records unchanged.
    let legacy = serde_json::json!({ "resident_pool_dense": [entry()] });
    let table: TilingTable = serde_json::from_value(legacy).unwrap();
    let found = search::lookup(&table, FAMILY, &entry().key, Entry::valid).unwrap();
    assert_eq!(found.evidence.selected, Choice::Gemv);
    let rewritten = serde_json::to_value(&table).unwrap();
    assert!(rewritten.get("resident_pool_dense").is_none());
    assert_eq!(rewritten["records"][FAMILY][0]["selected"], "Gemv");
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
#[test]
fn every_tensor_core_geometry_is_searchable_by_shape_alone() {
    // Outside the qualified family: a narrow reduction, and a wide output at many rows.
    for shape in [[1, 6144, 2048], [16, 16, 2048], [16, 248_320, 2048]] {
        assert!(!supported(shape));
        assert!(searchable(shape));
        assert_eq!(baseline(shape), Choice::Native);
    }
    // A very wide output at few rows keeps its dedicated vocabulary-head GEMV.
    assert!(!searchable([1, 248_320, 2048]));
    assert!(!searchable([0, 16, 2048]));
}
#[test]
fn qualified_family_keeps_its_baseline_and_candidates() {
    assert_eq!(baseline([16, 48, 5120]), Choice::Gemv);
    assert_eq!(baseline([24, 48, 5120]), Choice::Native);
    assert_eq!(
        candidates([16, 48, 5120]).collect::<Vec<_>>(),
        [Choice::Shared]
    );
    assert_eq!(
        candidates([24, 48, 5120]).collect::<Vec<_>>(),
        [Choice::Gemv, Choice::Shared]
    );
}
#[test]
fn extended_geometry_searches_gemv_and_wider_tensor_core_tiles() {
    assert_eq!(
        candidates([1, 6144, 2048]).collect::<Vec<_>>(),
        [
            Choice::Gemv,
            Choice::Shared,
            Choice::NativeN64,
            Choice::NativeN128
        ]
    );
    assert!(Choice::NativeN64.native_family() && Choice::NativeN128.native_family());
    assert!(!Choice::Gemv.native_family());
}
