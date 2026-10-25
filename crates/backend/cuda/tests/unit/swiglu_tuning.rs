use super::swiglu::{Entry, budget_for};
use crate::resident::nvfp4_gemm::workspace::{SwigluTile, select_swiglu_tile, swiglu_candidates};
use std::time::Duration;

#[test]
fn swiglu_candidates_are_legal_and_conservative() {
    let candidates = swiglu_candidates(1024, 5120, 17408);
    assert_eq!(candidates[0], SwigluTile::FALLBACK);
    assert!(candidates.contains(&SwigluTile {
        rows: 128,
        columns: 64
    }));
    assert!(candidates.contains(&SwigluTile {
        rows: 128,
        columns: 128
    }));
    assert_eq!(
        select_swiglu_tile(1024, 5120, 17408),
        Some(SwigluTile::FALLBACK)
    );

    for (rows, columns, hidden) in [(96, 5120, 17408), (1024, 5137, 17408), (1024, 5120, 17409)] {
        assert_eq!(select_swiglu_tile(rows, columns, hidden), None);
        assert_eq!(swiglu_candidates(rows, columns, hidden), Vec::new());
    }
}

#[test]
fn swiglu_table_entry_requires_device_kernel_geometry_and_real_proof() {
    let entry = Entry {
        device: "dev|sm_120|24576MiB".into(),
        kernel: "packed-swiglu-v1".into(),
        rows: 1024,
        columns: 5120,
        hidden: 17408,
        weight_proof: "proof".into(),
        tile_rows: 64,
        tile_columns: 64,
        exact: true,
        real_weights: true,
        baseline_us: 400.0,
        candidate_us: 390.0,
        speedup: 1.025,
    };
    assert!(entry.matches("dev|sm_120|24576MiB", "proof", 1024, 5120, 17408));
    assert!(!entry.matches("other", "proof", 1024, 5120, 17408));
    assert!(!entry.matches("dev|sm_120|24576MiB", "proof", 128, 5120, 17408));
    let mut bad = entry.clone();
    bad.exact = false;
    assert!(!bad.matches("dev|sm_120|24576MiB", "proof", 1024, 5120, 17408));
    let mut bad = entry;
    bad.real_weights = false;
    assert!(!bad.matches("dev|sm_120|24576MiB", "proof", 1024, 5120, 17408));
}

#[test]
fn swiglu_shape_budget_is_finite_and_candidate_scaled() {
    let one = budget_for(1);
    let four = budget_for(4);
    assert!(one > Duration::ZERO);
    assert!(four > one);
    assert_eq!(budget_for(0), one);
}
