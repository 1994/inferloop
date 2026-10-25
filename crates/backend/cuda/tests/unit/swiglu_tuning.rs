use super::swiglu::{Entry, EntryKey};
use crate::resident::nvfp4_gemm::workspace::{SwigluTile, select_swiglu_tile, swiglu_candidates};

const DEVICE: &str = "dev|sm_120|24576MiB";
const DRIVER: &str = "NVRM version: test";
const HARDWARE: &str = "hardware";
const FINGERPRINT: &str = "fp";
const PROOF: &str = "proof";

fn key() -> EntryKey<'static> {
    EntryKey {
        device: DEVICE,
        weight_proof: PROOF,
        driver_build: DRIVER,
        hardware: HARDWARE,
        kernel_fingerprint: FINGERPRINT,
        scales: (1.0, 2.0),
        rows: 1024,
        columns: 5120,
        hidden: 17408,
    }
}

fn entry() -> Entry {
    Entry {
        device: DEVICE.into(),
        kernel: concat!("packed-swiglu-v1/", env!("CARGO_PKG_VERSION")).into(),
        rows: 1024,
        columns: 5120,
        hidden: 17408,
        weight_proof: PROOF.into(),
        driver_build: DRIVER.into(),
        hardware: HARDWARE.into(),
        kernel_fingerprint: FINGERPRINT.into(),
        input_scale: 1.0,
        down_scale: 2.0,
        tile_rows: 64,
        tile_columns: 64,
        exact: true,
        real_weights: true,
        baseline_us: 400.0,
        candidate_us: 390.0,
        cold_base_us: 401.0,
        cold_trial_us: 391.0,
        warm_base_us: 399.0,
        warm_trial_us: 389.0,
        cold_ab_base_samples: vec![401.0],
        cold_ab_trial_samples: vec![391.0],
        cold_ba_base_samples: vec![401.0],
        cold_ba_trial_samples: vec![391.0],
        warm_ab_base_samples: vec![399.0],
        warm_ab_trial_samples: vec![389.0],
        warm_ba_base_samples: vec![399.0],
        warm_ba_trial_samples: vec![389.0],
        ab_base_us: 401.0,
        ab_trial_us: 391.0,
        ba_base_us: 399.0,
        ba_trial_us: 389.0,
        speedup: 1.025,
    }
}

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
    let entry = entry();
    assert!(entry.matches(&key()));
    for key in [
        EntryKey {
            device: "other",
            ..key()
        },
        EntryKey {
            weight_proof: "other",
            ..key()
        },
        EntryKey {
            driver_build: "other",
            ..key()
        },
        EntryKey {
            hardware: "other",
            ..key()
        },
        EntryKey {
            kernel_fingerprint: "other",
            ..key()
        },
        EntryKey {
            scales: (1.0, 3.0),
            ..key()
        },
        EntryKey { rows: 128, ..key() },
    ] {
        assert!(!entry.matches(&key));
    }
    let mut bad = entry.clone();
    bad.exact = false;
    assert!(!bad.matches(&key()));
    let mut bad = entry.clone();
    bad.real_weights = false;
    assert!(!bad.matches(&key()));
    let mut bad = entry;
    bad.cold_trial_us = f64::NAN;
    assert!(!bad.matches(&key()));
}
