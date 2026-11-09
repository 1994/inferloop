//! Cache integrity and actual automatic profiling of native gated norms.
use super::*;
use crate::device::device_error;
use crate::tuning::resident_gemm::MeasurementOrder;
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
            options(device.stream().device().ordinal(), 128, rows, 1e-6).occupancy_hint(),
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
    // A private table may already hold other geometries; these two must be among them.
    let measured: Vec<_> = search::records::<Key, i32>(&table, FAMILY)
        .into_iter()
        .filter(|entry| {
            [3072, 12288].contains(&entry.key.shape.rows) && entry.key.shape.dimension == 128
        })
        .collect();
    assert!(measured.len() >= 2, "both geometries measured");
    for entry in measured {
        assert!(entry.valid());
        assert_ne!(
            entry.evidence.selected, BASELINE,
            "known profiled prefill should improve"
        );
        let mut bad = entry.clone();
        bad.evidence.exact_candidates_verified = false;
        assert!(!bad.valid());
        let mut bad = entry.clone();
        bad.evidence.confirmation.order = MeasurementOrder::BaselineFirst;
        assert!(!bad.valid());
        let mut bad = entry.clone();
        bad.key.shape.rows = usize::MAX;
        assert!(!bad.valid());
        let mut bad = entry.clone();
        bad.evidence.trials.pop();
        assert!(!bad.valid());
        let mut bad = entry.clone();
        bad.evidence.confirmation.warm.candidate_ms[0] = f32::NAN;
        assert!(!bad.valid());
        println!(
            "{}",
            serde_json::json!({"norm_profile":true,"shape":entry.key.shape,
            "selected_occupancy":entry.evidence.selected,"complete_evidence":true,
            "other_tuning_fields_unchanged":true})
        );
    }
    Ok(())
}
#[test]
fn malformed_norm_shapes_are_rejected() {
    assert!(Shape::new(0, 128, 1e-6).is_none());
    assert!(Shape::new(3072, 256, 1e-6).is_none());
    assert!(Shape::new(usize::MAX, 128, 1e-6).is_none());
    assert!(Shape::new(3072, 128, f32::NAN).is_none());
    assert!(Shape::new(3072, 128, 0.0).is_none());
}
/// A record in the former section's shape, selecting occupancy 4 over every candidate.
fn former(selected: i32) -> serde_json::Value {
    let times =
        serde_json::json!({ "baseline_ms": vec![2.0_f32; 10], "candidate_ms": vec![1.0_f32; 10] });
    let trial = |occupancy: i32, candidate_first: bool| {
        serde_json::json!({ "occupancy": occupancy, "candidate_first": candidate_first,
            "verified_elements": 3072 * 128, "cold": times, "warm": times })
    };
    serde_json::json!({
        "key": { "hardware": "device", "driver_version": 13000, "driver_build": "build",
            "kernel_fingerprint": "kernels",
            "shape": Shape::new(3072, 128, 1e-6) },
        "selected": selected,
        "search": CANDIDATES.map(|occupancy| trial(occupancy, false)),
        "confirmation": trial(selected, true),
    })
}
#[test]
fn former_norm_records_keep_their_choice_and_incomplete_ones_are_rejected() {
    let decode = |stored| serde_json::from_value::<Entry>(legacy(stored)).ok();
    let migrated = decode(former(4)).unwrap();
    assert_eq!(migrated.evidence.selected, 4);
    assert!(migrated.valid());
    let mut short = migrated.clone();
    short.evidence.confirmation.cold.baseline_ms.pop();
    assert!(!short.valid());
    let mut slow = migrated.clone();
    slow.evidence.confirmation.warm.candidate_ms[0] = 0.0;
    assert!(!slow.valid());
    let mut partial = former(4);
    partial["confirmation"]["verified_elements"] = 1.into();
    assert!(!decode(partial).unwrap().valid());
    let unknown = former(32);
    assert!(!decode(unknown).unwrap().valid());
    let mut unreadable = former(4);
    unreadable["search"] = "lost".into();
    assert_eq!(legacy(unreadable.clone()), unreadable);
}
