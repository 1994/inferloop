use super::*;
use crate::device::device_error;
use crate::resident::attention_oxide::shared_identity;
use crate::resident::attention_oxide::vector_identity;
use crate::tuning::MIN_REPS;
use crate::tuning::TilingTable;
use crate::tuning::qualify::Times;
use sha2::{Digest, Sha256};
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
        selected: Choice::VectorSharedKvCombined,
        verified_fixtures: reference::FIXTURES,
        verified_elements: reference::FIXTURES * 256 * 24 * 256,
        history_verified_elements: HISTORIES.len() * 256 * 24 * 256,
        max_f64_error: 1e-7,
        forward: cases(),
        reverse: cases(),
    }
}
#[test]
fn combined_cache_requires_full_bits_and_both_history_orders() {
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
fn empty_evidence_keeps_the_qualified_vector_baseline() -> Result<()> {
    let empty: TilingTable = serde_json::from_str("{}").map_err(device_error)?;
    assert!(empty.records.is_empty());
    assert!(empty.records.is_empty());
    assert_eq!(
        decide(Choice::VectorSharedKv, None)?,
        Choice::VectorSharedKv
    );
    assert_eq!(decide(Choice::SharedKv, None)?, Choice::SharedKv);
    Ok(())
}
#[test]
fn rejected_combined_candidate_and_legacy_cache_remain_exact() -> Result<()> {
    let mut rejected = entry();
    rejected.selected = Choice::VectorSharedKv;
    rejected.reverse[1].cold.candidate_ms = vec![3.0; MIN_REPS];
    assert!(valid(&rejected));
    let stored = rejected.clone();
    let table: TilingTable = serde_json::from_value(
        serde_json::json!({ "resident_vector_combined_attention": [rejected] }),
    )
    .map_err(device_error)?;
    let restored: TilingTable =
        serde_json::from_str(&serde_json::to_string(&table).map_err(device_error)?)
            .map_err(device_error)?;
    assert!(valid(
        &search::find::<Key, Entry>(&restored, FAMILY, &stored.key, |_| true).unwrap()
    ));
    assert_eq!(
        decide(
            Choice::VectorSharedKv,
            Some(&search::find::<Key, Entry>(&restored, FAMILY, &stored.key, |_| true).unwrap())
        )?,
        Choice::VectorSharedKv
    );
    assert!(!supported(Shape {
        lanes: 64,
        ..shape()
    }));
    Ok(())
}
#[test]
fn combined_fingerprint_is_isolated_from_the_vector_layer() {
    let base = "0".repeat(SHA256_HEX_LEN);
    let without_combined = format!(
        "{:x}",
        Sha256::digest(format!(
            "{}:{}:{}",
            base,
            shared_identity(),
            vector_identity(),
        ))
    );
    assert_ne!(fingerprint(&base), without_combined);
    assert_ne!(fingerprint(&base), base);
    assert_ne!(fingerprint("a"), fingerprint("b"));
    assert!(SCHEMA.contains("combined"));
}
