use super::*;
use crate::tuning::MIN_REPS;
use crate::tuning::TilingTable;
use crate::tuning::qualify::Times;
fn shape() -> Shape {
    Shape {
        heads: 24,
        kv_heads: 4,
        dim: 256,
        lanes: 64,
        window: 0,
    }
}
fn entry() -> Entry {
    let cases = || {
        TIMED_FIXTURES
            .into_iter()
            .map(|case| Case {
                case,
                cold: Times {
                    baseline_ms: vec![2.0; MIN_REPS],
                    candidate_ms: vec![1.0; MIN_REPS],
                },
                warm: Times {
                    baseline_ms: vec![2.0; MIN_REPS],
                    candidate_ms: vec![1.0; MIN_REPS],
                },
            })
            .collect()
    };
    Entry {
        key: Key {
            hardware: "test-device".into(),
            driver_api: 13040,
            driver_build: Some("test-driver".into()),
            fingerprint: "0".repeat(64),
            schema: SCHEMA.into(),
            shape: shape(),
        },
        selected: Choice::RustSimt,
        verified_fixtures: reference::FIXTURES,
        verified_elements: reference::FIXTURES * 64 * 24 * 256,
        max_f64_error: 1e-7,
        forward: cases(),
        reverse: cases(),
        history_verified_elements: 0,
    }
}
#[test]
fn cache_rejects_missing_bit_proof_history_and_reverse_confirmation() {
    let good = entry();
    assert!(valid(&good));
    for index in 0..10 {
        let mut bad = good.clone();
        match index {
            0 => bad.verified_elements -= 1,
            1 => bad.verified_fixtures -= 1,
            2 => bad.reverse.clear(),
            3 => bad.forward[1].case = 0,
            4 => bad.reverse[1].cold.candidate_ms = vec![2.0; MIN_REPS],
            5 => bad.forward[0].warm.candidate_ms = vec![f32::NAN; MIN_REPS],
            6 => bad.max_f64_error = super::super::MAX_ERROR,
            7 => bad.key.schema = "stale-contract".into(),
            8 => bad.key.fingerprint = "missing-artifact".into(),
            9 => bad.selected = Choice::Tiled { query_tile: 16 },
            _ => unreachable!(),
        }
        assert!(!valid(&bad), "variant {index}");
    }
}
#[test]
fn legacy_tables_and_exact_fallback_evidence_round_trip() -> Result<()> {
    let old: TilingTable = serde_json::from_str("{}").map_err(device_error)?;
    assert!(old.records.is_empty());
    let mut fallback = entry();
    fallback.selected = Choice::Simt;
    fallback.reverse[0].cold.candidate_ms = vec![2.0; MIN_REPS];
    assert!(valid(&fallback));
    let stored = fallback.clone();
    let table: TilingTable =
        serde_json::from_value(serde_json::json!({ "resident_exact_attention": [fallback] }))
            .map_err(device_error)?;
    let text = serde_json::to_string(&table).map_err(device_error)?;
    let restored: TilingTable = serde_json::from_str(&text).map_err(device_error)?;
    assert!(valid(
        &search::find::<Key, Entry>(&restored, FAMILY, &stored.key, |_| true).unwrap()
    ));
    Ok(())
}
#[test]
fn unqualified_geometry_stays_on_the_general_kernel() {
    for s in [
        Shape {
            dim: 128,
            ..shape()
        },
        Shape {
            heads: 12,
            ..shape()
        },
        Shape {
            lanes: 512,
            ..shape()
        },
        Shape {
            window: usize::MAX,
            ..shape()
        },
    ] {
        assert!(!supported(s));
    }
}
#[test]
#[ignore = "requires CUDA and safe-run; exact full-output/KV and cold/warm AB/BA tuning"]
fn hardware_exact_search_and_cache_reuse() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let session = ResidentTuningSession::default();
    for lanes in [64, 256] {
        let shape = Shape { lanes, ..shape() };
        let selected = select(&device, shape, true, &session);
        let entry = memo()
            .lock()
            .map_err(|_| Error::invariant("test exact attention memo poisoned"))?
            .values()
            .find(|entry| entry.key.shape == shape)
            .cloned()
            .ok_or_else(|| Error::invalid("exact attention tuning did not complete"))?;
        assert!(valid(&entry));
        assert_eq!(selected, Choice::RustSimt);
        assert_eq!(select(&device, shape, false, &session), selected);
        println!(
            "EXACT_ATTENTION_TUNING {}",
            serde_json::to_string(&entry).map_err(device_error)?
        );
    }
    Ok(())
}
