use super::*;
use crate::device::device_error;
use crate::tuning::MIN_REPS;
use crate::tuning::TilingTable;
use crate::tuning::qualify::Times;
fn shape() -> Shape {
    Shape {
        heads: 24,
        kv_heads: 4,
        dim: 256,
        lanes: 256,
        window: 0,
    }
}
fn entry() -> Entry {
    let times = || Times {
        baseline_ms: vec![2.0; MIN_REPS],
        candidate_ms: vec![1.0; MIN_REPS],
    };
    let cases = || {
        HISTORIES
            .into_iter()
            .map(|history| Case {
                case: history,
                cold: times(),
                warm: times(),
            })
            .collect()
    };
    Entry {
        key: Key {
            hardware: "test-device".into(),
            driver_api: 13040,
            driver_build: Some("test-driver".into()),
            fingerprint: "0".repeat(SHA256_HEX_LEN),
            schema: SCHEMA.into(),
            shape: shape(),
        },
        selected: Choice::VectorSharedKv,
        verified_fixtures: reference::FIXTURES,
        verified_elements: reference::FIXTURES * 256 * 24 * 256,
        history_verified_elements: HISTORIES.len() * 256 * 24 * 256,
        max_f64_error: 1e-7,
        forward: cases(),
        reverse: cases(),
    }
}
#[test]
fn cache_requires_full_bits_and_both_history_orders() {
    let good = entry();
    assert!(valid(&good));
    for index in 0..12 {
        let mut bad = good.clone();
        match index {
            0 => bad.verified_elements -= 1,
            1 => bad.history_verified_elements -= 1,
            2 => bad.reverse.clear(),
            3 => bad.forward[1].case = 0,
            4 => bad.reverse[2].cold.candidate_ms = vec![2.0; MIN_REPS],
            5 => bad.forward[0].warm.candidate_ms = vec![2.2; MIN_REPS],
            6 => bad.max_f64_error = super::super::MAX_ERROR,
            7 => bad.key.schema = "stale-contract".into(),
            8 => bad.key.fingerprint = "missing-artifact".into(),
            9 => bad.selected = Choice::Tiled { query_tile: 16 },
            10 => bad.key.shape.lanes = usize::MAX,
            11 => bad.forward[1].cold.candidate_ms = vec![f32::NAN; MIN_REPS],
            _ => unreachable!(),
        }
        assert!(!valid(&bad), "variant {index}");
    }
}
#[test]
fn legacy_cache_and_rejected_shared_candidate_remain_exact() -> Result<()> {
    let old: TilingTable = serde_json::from_str("{}").map_err(device_error)?;
    assert!(old.records.is_empty());
    assert!(!supported(Shape {
        lanes: 64,
        ..shape()
    }));
    let mut rejected = entry();
    rejected.selected = Choice::SharedKv;
    rejected.reverse[1].cold.candidate_ms = vec![3.0; MIN_REPS];
    assert!(valid(&rejected));
    let stored = rejected.clone();
    let table: TilingTable = serde_json::from_value(
        serde_json::json!({ "resident_vector_shared_attention": [rejected] }),
    )
    .map_err(device_error)?;
    let restored: TilingTable =
        serde_json::from_str(&serde_json::to_string(&table).map_err(device_error)?)
            .map_err(device_error)?;
    assert!(valid(
        &search::find::<Key, Entry>(&restored, FAMILY, &stored.key, |_| true).unwrap()
    ));
    Ok(())
}
#[test]
#[ignore = "requires CUDA/safe-run; complete F64/bitwise and short/long cold/warm AB/BA gates"]
fn hardware_vector_search_and_cache_reuse() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let session = ResidentTuningSession::default();
    let selected = select(&device, shape(), true, &session, Choice::SharedKv);
    let entry = memo()
        .lock()
        .map_err(|_| Error::invariant("vector shared KV test memo poisoned"))?
        .values()
        .find(|e| e.key.shape == shape())
        .cloned()
        .ok_or_else(|| Error::invalid("vector shared KV hardware proof did not complete"))?;
    assert!(valid(&entry));
    assert_eq!(selected, Choice::VectorSharedKv);
    assert_eq!(
        select(&device, shape(), false, &session, Choice::SharedKv),
        selected
    );
    println!(
        "VECTOR_KV_TUNING {}",
        serde_json::to_string(&entry).map_err(device_error)?
    );
    Ok(())
}
