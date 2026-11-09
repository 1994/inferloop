use super::*;
use crate::tuning::TilingTable;

fn entry(hardware: &str, tile: usize) -> Entry {
    let times = paired_times(&[2.0; MIN_REPS], &[1.0; MIN_REPS]).unwrap();
    let trial = |order| Trial {
        choice: tile,
        order,
        cold: times.clone(),
        warm: times.clone(),
    };
    let rejected = protocol()
        .candidates
        .into_iter()
        .filter(|candidate| *candidate != tile)
        .map(|candidate| (candidate, "rejected".to_owned()))
        .collect();
    Entry {
        key: Key {
            hardware: hardware.into(),
            driver_version: 13020,
            driver_build: Some("driver-build".into()),
            kernel_fingerprint: "kernel-A".into(),
            rows: 1,
            vocabulary: 248_320,
        },
        evidence: Evidence {
            selected: tile,
            exact_candidates_verified: true,
            trials: vec![trial(MeasurementOrder::BaselineFirst)],
            rejected: Some(rejected),
            confirmation: trial(MeasurementOrder::CandidateFirst),
        },
    }
}
fn table(entries: &[Entry]) -> TilingTable {
    let mut table = TilingTable::default();
    let records = table.records.entry(FAMILY.into()).or_default();
    records.extend(
        entries
            .iter()
            .map(|entry| serde_json::to_value(entry).unwrap()),
    );
    table
}
fn cached(table: &TilingTable, key: &Key) -> Option<usize> {
    search::lookup(table, FAMILY, key, Entry::valid).map(|entry| entry.evidence.selected)
}
#[test]
fn persisted_greedy_decisions_require_exact_device_driver_kernel_and_geometry() {
    let first = entry("GPU-A", TILE_SEARCH[1]);
    let second = entry("GPU-B", TILE_SEARCH[2]);
    let entries = table(&[first.clone(), second.clone()]);
    assert_eq!(cached(&entries, &first.key), Some(TILE_SEARCH[1]));
    assert_eq!(cached(&entries, &second.key), Some(TILE_SEARCH[2]));
    for changed in 0..5 {
        let mut key = first.key.clone();
        match changed {
            0 => key.driver_version += 1,
            1 => key.driver_build = Some("other-build".into()),
            2 => key.kernel_fingerprint = "other-kernel".into(),
            3 => key.rows += 1,
            _ => key.vocabulary += 1,
        }
        assert_eq!(cached(&entries, &key), None);
    }
}
#[test]
fn persisted_greedy_decisions_reject_missing_confirmation_or_bad_cold_warm_samples() {
    let good = entry("GPU-A", TILE_SEARCH[1]);
    assert!(good.valid());
    let encoded = serde_json::to_string(&good).unwrap();
    let restored: Entry = serde_json::from_str(&encoded).unwrap();
    assert!(restored.valid());
    for bad in 0..6 {
        let mut entry = good.clone();
        let evidence = &mut entry.evidence;
        match bad {
            0 => evidence.confirmation.order = MeasurementOrder::BaselineFirst,
            1 => evidence.confirmation.cold.candidate_ms[0] = f32::NAN,
            2 => {
                evidence.confirmation.warm =
                    paired_times(&[2.0; MIN_REPS], &[3.0; MIN_REPS]).unwrap();
            }
            3 => evidence.trials.clear(),
            4 => {
                evidence.confirmation.cold = paired_times(&[2.0; 2], &[1.0; 2]).unwrap();
            }
            _ => {
                evidence.confirmation.cold.candidate_ms.pop();
            }
        }
        assert_eq!(cached(&table(&[entry]), &good.key), None);
    }
}
#[test]
fn former_greedy_records_keep_their_choice() {
    let good = entry("GPU-A", TILE_SEARCH[1]);
    let times = serde_json::json!({ "baseline_ms": vec![2.0_f32; MIN_REPS],
        "candidate_ms": vec![1.0_f32; MIN_REPS] });
    let trial = |tile: usize, candidate_first: bool| {
        serde_json::json!({ "tile": tile, "candidate_first": candidate_first,
            "cold": times, "warm": times })
    };
    let former = serde_json::json!({
        "key": good.key,
        "selected": TILE_SEARCH[1],
        "search": [trial(BASELINE_TILE, false), trial(TILE_SEARCH[1], false)],
        "confirmation": trial(TILE_SEARCH[1], true),
    });
    let migrated: Entry = serde_json::from_value(legacy(former)).unwrap();
    assert_eq!(migrated.evidence.selected, TILE_SEARCH[1]);
    assert_eq!(migrated.evidence.rejected, None);
    assert!(migrated.valid());
    let unreadable = serde_json::json!({ "key": good.key, "selected": "lost" });
    assert_eq!(legacy(unreadable.clone()), unreadable);
}

#[test]
#[ignore = "requires CUDA hardware and a writable tuning cache; run under safe-run"]
fn greedy_real_vocabulary_search_records_verified_cold_and_warm_confirmation() -> Result<()> {
    CudaDevice::enable_kernel_cache()?;
    let device = CudaDevice::new(0)?;
    let session = ResidentTuningSession::default();
    for rows in [1, 4, 12] {
        let key = key(&device, rows, 248_320)?;
        let entry = tune(&device, key, &session)?
            .ok_or_else(|| Error::invalid("greedy search did not complete"))?;
        assert!(entry.valid());
        let path = super::super::table_path()
            .ok_or_else(|| Error::invalid("greedy test requires tuning table path"))?;
        assert!(search::store(Some(&path), FAMILY, &entry));
        println!("{}", serde_json::to_string(&entry).map_err(device_error)?);
    }
    Ok(())
}
