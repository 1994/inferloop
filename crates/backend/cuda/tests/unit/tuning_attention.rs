use super::*;
use std::time::Instant;
fn shape() -> Shape {
    Shape {
        heads: 24,
        kv_heads: 4,
        dim: 256,
        lanes: 64,
        window: 0,
    }
}
fn times() -> PairedTimes {
    times_at(1.0)
}
fn times_at(candidate: f32) -> PairedTimes {
    paired_times(&[2.0; MIN_REPS], &[candidate; MIN_REPS]).unwrap()
}
fn entry() -> Entry {
    let trial = |query_tile, order| Trial {
        choice: Choice::Tiled { query_tile },
        order,
        cold: times(),
        warm: times(),
    };
    Entry {
        key: Key {
            hardware: "test-device".into(),
            driver_api: 13000,
            driver_build: Some("test-driver".into()),
            fingerprint: "test-kernel".into(),
            schema: SCHEMA.into(),
            shape: shape(),
        },
        evidence: Evidence {
            selected: Choice::Tiled { query_tile: 16 },
            exact_candidates_verified: true,
            trials: QUERY_TILES
                .map(|tile| trial(tile, MeasurementOrder::BaselineFirst))
                .to_vec(),
            rejected: Some(Vec::new()),
            confirmation: trial(16, MeasurementOrder::CandidateFirst),
            max_error: Some(1e-7),
        },
    }
}
#[test]
fn cache_requires_numerical_proof_all_candidates_and_reverse_confirmation() {
    let good = entry();
    assert!(good.valid());
    for variant in 0..12 {
        let mut bad = good.clone();
        let evidence = &mut bad.evidence;
        match variant {
            0 => bad.key.schema = "stale-bound".into(),
            1 => bad.key.fingerprint.clear(),
            2 => bad.key.hardware.clear(),
            3 => evidence.exact_candidates_verified = false,
            4 => evidence.trials.pop().map_or((), |_| ()),
            5 => evidence.trials[1] = evidence.trials[0].clone(),
            6 => evidence.confirmation.order = MeasurementOrder::BaselineFirst,
            7 => evidence.max_error = Some(MAX_ERROR),
            8 => evidence.max_error = Some(f64::NAN),
            9 => evidence.confirmation.cold = times_at(2.0),
            10 => evidence
                .confirmation
                .warm
                .baseline_ms
                .pop()
                .map_or((), |_| ()),
            _ => evidence.selected = Choice::Tiled { query_tile: 64 },
        }
        assert!(!bad.valid(), "variant {variant}");
    }
}
#[test]
fn fallback_can_have_complete_evidence_without_a_speedup() {
    let mut value = entry();
    let evidence = &mut value.evidence;
    evidence.selected = Choice::Simt;
    evidence.confirmation.choice = Choice::Simt;
    evidence.confirmation.cold = times_at(2.0);
    evidence.confirmation.warm = evidence.confirmation.cold.clone();
    assert!(value.valid());
    value.evidence.trials.clear();
    assert!(!value.valid());
}
#[test]
fn former_attention_records_keep_their_choice_and_deviation() -> Result<()> {
    let times = serde_json::json!({ "baseline_ms": vec![2.0_f32; MIN_REPS],
        "candidate_ms": vec![1.0_f32; MIN_REPS] });
    let trial = |query_tile: usize, candidate_first: bool, error: f64| {
        serde_json::json!({ "choice": Choice::Tiled { query_tile },
            "candidate_first": candidate_first, "max_f64_error": error,
            "cold": times, "warm": times })
    };
    let good = entry();
    let former = |fixtures: usize| {
        serde_json::json!({
            "key": good.key, "selected": Choice::Tiled { query_tile: 16 },
            "search": [trial(16, false, 1e-7), trial(32, false, 2e-7)],
            "confirmation": trial(16, true, 3e-7), "fixtures": fixtures,
        })
    };
    let decode = |stored| serde_json::from_value::<Entry>(legacy(stored)).map_err(device_error);
    let migrated = decode(former(reference::FIXTURES))?;
    assert_eq!(migrated.evidence.selected, Choice::Tiled { query_tile: 16 });
    assert_eq!(migrated.evidence.max_error, Some(3e-7));
    assert!(migrated.valid());
    assert!(!decode(former(reference::FIXTURES - 1))?.valid());
    Ok(())
}
#[test]
fn unsupported_geometry_never_selects_a_tensor_core_configuration() {
    assert!(shape().supported());
    for s in [
        Shape {
            heads: 0,
            ..shape()
        },
        Shape {
            kv_heads: 0,
            ..shape()
        },
        Shape {
            heads: 23,
            ..shape()
        },
        Shape {
            dim: 128,
            ..shape()
        },
        Shape {
            lanes: 51,
            ..shape()
        },
        Shape {
            window: usize::MAX,
            ..shape()
        },
    ] {
        assert!(!s.supported());
    }
}
#[test]
fn independent_reference_preserves_inactive_rows_and_quantized_append() -> Result<()> {
    let f = Fixture::new(shape(), 2)?;
    assert!(f.expected.iter().all(|v| *v == 0.0));
    assert!(f.keys.iter().zip(&f.expected_keys).all(|(a, b)| a.0 == b.0));
    assert!(
        f.values
            .iter()
            .zip(&f.expected_values)
            .all(|(a, b)| a.0 == b.0)
    );
    let f = Fixture::new(shape(), 0)?;
    let inactive = (f.shape.lanes - 1) * f.shape.heads * f.shape.dim;
    assert!(f.expected[inactive..].iter().all(|v| *v == 0.0));
    assert!(f.expected[..inactive].iter().all(|v| v.is_finite()));
    Ok(())
}
#[test]
fn attention_cache_round_trip_preserves_hardware_scope_and_old_tables() -> Result<()> {
    let mut table = super::super::TilingTable::default();
    let stored = serde_json::to_value(entry()).map_err(device_error)?;
    table.records.entry(FAMILY.into()).or_default().push(stored);
    let json = serde_json::to_string(&table).map_err(device_error)?;
    let decoded: super::super::TilingTable = serde_json::from_str(&json).map_err(device_error)?;
    let found = search::lookup(&decoded, FAMILY, &entry().key, Entry::valid);
    assert!(found.is_some());
    let legacy: super::super::TilingTable = serde_json::from_str("{}").map_err(device_error)?;
    assert!(legacy.records.is_empty());
    Ok(())
}
#[test]
#[ignore = "requires CUDA; full F64/cache numerical gate and hardware AB/BA tuning"]
fn hardware_query_sharing_search_and_cached_reuse() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let session = ResidentTuningSession::default();
    for lanes in [64, 256] {
        let shape = Shape { lanes, ..shape() };
        let selected = select(&device, shape, true, &session);
        let entry = memo()
            .lock()
            .map_err(|_| Error::invariant("test attention memo poisoned"))?
            .values()
            .find(|e| e.key.shape == shape)
            .cloned()
            .ok_or_else(|| Error::invalid("attention hardware proof did not complete"))?;
        assert!(entry.valid());
        assert_eq!(entry.evidence.selected, selected);
        let start = Instant::now();
        for _ in 0..16 {
            assert_eq!(select(&device, shape, false, &session), selected);
        }
        eprintln!(
            "ATTENTION_CACHED_16_MS {}",
            start.elapsed().as_secs_f64() * 1000.0
        );
        eprintln!(
            "ATTENTION_TUNING {}",
            serde_json::to_string(&entry).map_err(device_error)?
        );
    }
    Ok(())
}
