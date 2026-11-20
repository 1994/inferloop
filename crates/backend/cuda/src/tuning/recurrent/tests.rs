use super::*;
use std::time::Instant;
fn times(candidate: f32) -> PairedTimes {
    paired_times(&[2.0; MIN_REPS], &[candidate; MIN_REPS]).unwrap()
}
fn entry() -> Entry {
    let trial = |order| Trial {
        choice: 64,
        order,
        cold: times(1.0),
        warm: times(1.0),
    };
    Entry {
        key: Key {
            hardware: "test-device".into(),
            driver_version: 13000,
            driver_build: Some("test-driver".into()),
            kernel_fingerprint: "test-ir".into(),
            schema: SCHEMA.into(),
            shape: [16, 48, 128, 64],
        },
        evidence: Evidence {
            selected: 64,
            exact_candidates_verified: true,
            trials: vec![trial(MeasurementOrder::BaselineFirst)],
            rejected: Some(Vec::new()),
            confirmation: trial(MeasurementOrder::CandidateFirst),
            max_error: Some(1e-7),
        },
    }
}
#[test]
fn rejected_cache_does_not_encode_a_valid_hardware_decision() {
    let good = entry();
    assert!(good.valid());
    let mut bad = good.clone();
    bad.evidence.confirmation.order = MeasurementOrder::BaselineFirst;
    assert!(!bad.valid());
    let mut bad = good.clone();
    bad.evidence.trials.clear();
    assert!(!bad.valid());
    let mut bad = good.clone();
    bad.evidence.max_error = Some(2e-5);
    assert!(!bad.valid());
    let mut bad = good.clone();
    bad.evidence.max_error = None;
    assert!(!bad.valid());
    let mut bad = good.clone();
    bad.evidence.confirmation.cold.candidate_ms[0] = f32::NAN;
    assert!(!bad.valid());
    let mut bad = good.clone();
    bad.evidence.confirmation.cold = paired_times(&[2.0; 2], &[1.0; 2]).unwrap();
    assert!(!bad.valid());
    let mut bad = good.clone();
    bad.evidence.selected = 32;
    assert!(!bad.valid());
    let mut bad = good.clone();
    bad.evidence.selected = 128;
    bad.evidence.confirmation.choice = 128;
    bad.evidence.trials.clear();
    assert!(!bad.valid());
    let mut bad = good.clone();
    bad.evidence.trials.push(bad.evidence.trials[0].clone());
    assert!(!bad.valid());
    let mut bad = good;
    bad.key.schema = "old-neumann".into();
    assert!(!bad.valid());
}
#[test]
fn former_recurrent_records_keep_their_choice_and_deviation() {
    let times = serde_json::json!({ "baseline_ms": vec![2.0_f32; MIN_REPS],
        "candidate_ms": vec![1.0_f32; MIN_REPS] });
    let trial = |candidate_first: bool, error: f64| {
        serde_json::json!({ "value_block": 64, "candidate_first": candidate_first,
            "max_f64_error": error, "cold": times, "warm": times })
    };
    let former = serde_json::json!({
        "key": entry().key, "selected": 64,
        "search": [trial(false, 1e-7)], "confirmation": trial(true, 4e-7),
    });
    let migrated: Entry = serde_json::from_value(legacy(former)).unwrap();
    assert_eq!(migrated.evidence.selected, 64);
    assert_eq!(migrated.evidence.max_error, Some(4e-7));
    assert!(migrated.valid());
}
#[test]
fn supports_algorithm_geometry_without_device_names() {
    assert!(supported([2, 4, 32, 32]));
    assert!(supported([16, 48, 128, 256]));
    assert!(!supported([0, 48, 128, 64]));
    assert!(!supported([16, 47, 128, 64]));
    assert!(!supported([16, 48, 128, 3]));
    assert!(!supported([16, 48, 64, 64]));
}
#[test]
fn numerical_gate_rejects_nonfinite_or_partial_results() {
    assert!(reference::maximum_error(&[1.0], &[1.0]).is_ok());
    assert!(reference::maximum_error(&[1.0], &[1.0, 1.0]).is_err());
    assert!(reference::maximum_error(&[f32::NAN], &[1.0]).is_err());
    assert!(reference::maximum_error(&[1.0], &[1.1]).is_err());
}
#[test]
#[ignore = "requires CUDA hardware; finite hardware tuning with independent F64 references"]
fn hardware_search_passes_independent_reference_and_reverse_confirmation() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let shape = [16, 48, 128, 64];
    let selected = select(&device, shape, true, &ResidentTuningSession::default())
        .ok_or_else(|| Error::invalid("hardware selection unexpectedly rejected"))?;
    let entry = memo()
        .lock()
        .map_err(|_| Error::invariant("test tuning cache poisoned"))?
        .values()
        .find(|x| x.key.shape == shape)
        .cloned()
        .ok_or_else(|| Error::invalid("hardware search did not complete evidence"))?;
    assert_eq!(entry.evidence.selected, selected);
    assert!(entry.valid());
    let started = Instant::now();
    for _ in 0..48 {
        assert_eq!(
            select(&device, shape, false, &ResidentTuningSession::default()),
            Some(selected)
        );
    }
    eprintln!(
        "RECURRENT_IDENTITY_48_CACHED_MS {}",
        started.elapsed().as_secs_f64() * 1000.0
    );
    eprintln!(
        "RECURRENT_TUNING {}",
        serde_json::to_string(&entry).map_err(device_error)?
    );
    Ok(())
}
