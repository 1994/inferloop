//! Cache integrity and actual automatic profiling of native gated norms.
use super::*;
use crate::device::device_error;
use std::fs::read;
#[test]
#[ignore = "requires CUDA/safe-run and a private frozen tuning table; seeds measured profile"]
fn gated_norm_profile_preserves_all_other_entries() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let path = super::super::table_path()
        .ok_or_else(|| Error::invalid("private tuning table required"))?;
    let before: serde_json::Value =
        serde_json::from_slice(&read(&path).map_err(device_error)?).map_err(device_error)?;
    let device = CudaDevice::new(0)?;
    let session = ResidentTuningSession::default();
    for rows in [3072, 12288] {
        let shape = Shape::new(rows, 128, 1e-6f32).ok_or_else(|| Error::invalid("norm fixture"))?;
        resolve(&device, shape, false, &session)?;
        assert_eq!(
            options(device.stream.device().ordinal(), 128, rows, 1e-6).occupancy,
            Some(UNMEASURED)
        );
        resolve(&device, shape, true, &session)?;
    }
    let after: serde_json::Value =
        serde_json::from_slice(&read(&path).map_err(device_error)?).map_err(device_error)?;
    let before = before
        .as_object()
        .ok_or_else(|| Error::invariant("tuning table object"))?;
    for (key, value) in before {
        if key != "resident_gated_norm" {
            assert_eq!(after[key], *value, "old tuning field {key}");
        }
    }
    let table = super::super::TilingTable::load(&path)?;
    assert_eq!(table.resident_gated_norm.len(), 2);
    for entry in table.resident_gated_norm {
        assert!(entry.valid());
        assert_ne!(
            entry.selected, BASELINE,
            "known profiled prefill should improve"
        );
        let mut bad = entry.clone();
        bad.confirmation.verified_elements = 0;
        assert!(!bad.valid());
        let mut bad = entry.clone();
        bad.confirmation.candidate_first = false;
        assert!(!bad.valid());
        let mut bad = entry.clone();
        bad.key.shape.rows = usize::MAX;
        assert!(!bad.valid());
        let mut bad = entry.clone();
        bad.search.pop();
        assert!(!bad.valid());
        let mut bad = entry.clone();
        bad.confirmation.warm.candidate_ms[0] = f32::NAN;
        assert!(!bad.valid());
        println!(
            "{}",
            serde_json::json!({"norm_profile":true,"shape":entry.key.shape,
            "selected_occupancy":entry.selected,"complete_evidence":true,
            "other_tuning_fields_unchanged":true})
        );
    }
    Ok(())
}
#[test]
fn malformed_norm_shapes_and_partial_trials_are_rejected() {
    assert!(Shape::new(0, 128, 1e-6).is_none());
    assert!(Shape::new(3072, 256, 1e-6).is_none());
    assert!(Shape::new(usize::MAX, 128, 1e-6).is_none());
    assert!(Shape::new(3072, 128, f32::NAN).is_none());
    assert!(Shape::new(3072, 128, 0.0).is_none());
    let trial = evidence::Trial {
        occupancy: 4,
        candidate_first: true,
        verified_elements: 3072 * 128,
        cold: evidence::Times {
            baseline_ms: vec![2.0; 10],
            candidate_ms: vec![1.0; 10],
        },
        warm: evidence::Times {
            baseline_ms: vec![2.0; 10],
            candidate_ms: vec![1.0; 10],
        },
    };
    assert!(trial.confirmed());
    let mut bad = trial.clone();
    bad.cold.baseline_ms.pop();
    assert!(!bad.confirmed());
    let mut bad = trial.clone();
    bad.warm.candidate_ms[0] = 0.0;
    assert!(!bad.confirmed());
    let mut bad = trial;
    bad.occupancy = 32;
    assert!(!bad.confirmed());
}
