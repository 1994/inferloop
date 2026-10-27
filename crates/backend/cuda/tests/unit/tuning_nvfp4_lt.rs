//! Cache completeness, identity isolation and old-table compatibility without CUDA.
use super::{Entry, Key, SCHEMA, Trial, cached, save};
use crate::device::vendor_fp8::{AlgorithmConfig, LibraryIdentity};
use crate::tuning::resident_gemm::{GemmConfig, MeasurementOrder, PairedTimes, ResidentGemmKey};
use crate::tuning::{MIN_REPS, TilingTable};
use std::env::temp_dir;
use std::process::id;
use std::{
    error::Error,
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
type TestResult<T = ()> = Result<T, Box<dyn Error>>;
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> TestResult<Self> {
        let p = temp_dir().join(format!(
            "infer-fp4-lt-cache-{}-{}",
            id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&p)?;
        Ok(Self(p))
    }
    fn file(&self) -> PathBuf {
        self.0.join("tuning.json")
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn entry() -> Entry {
    let key = Key {
        native: ResidentGemmKey {
            hardware: "unit-device".into(),
            driver_version: 13000,
            driver_build: Some("unit-driver".into()),
            kernel_schema: "unit-native".into(),
            kernel_fingerprint: "unit-native-kernel".into(),
            activation_mode: "nvfp4-w4a4-block16".into(),
            m: 1024,
            n: 5120,
            k: 17408,
        },
        baseline: GemmConfig::baseline(1024),
        library: LibraryIdentity {
            lt_version: 130_800,
            cudart_version: 13080,
        },
        algorithm: AlgorithmConfig {
            id: 1,
            tile: 0,
            split_k: 0,
            reduction: 0,
            swizzle: 0,
            custom: 0,
            stages: 0,
            inner: 0,
            cluster: 0,
        },
        pack_fingerprint: "unit-pack-kernel".into(),
        schema: SCHEMA.into(),
    };
    let t = PairedTimes {
        baseline_ms: vec![2.0; MIN_REPS],
        candidate_ms: vec![1.0; MIN_REPS],
        baseline_median_ms: 2.0,
        candidate_median_ms: 1.0,
        speedup: 2.0,
    };
    Entry {
        key,
        trials: vec![
            Trial {
                order: MeasurementOrder::BaselineFirst,
                cold: t.clone(),
                warm: t.clone(),
            },
            Trial {
                order: MeasurementOrder::CandidateFirst,
                cold: t.clone(),
                warm: t,
            },
        ],
    }
}
#[test]
fn qualification_roundtrip_preserves_other_sections_and_identity() -> TestResult {
    let temp = Temp::new()?;
    let path = temp.file();
    let mut table = TilingTable::default();
    table.devices.insert("keep".into(), Vec::new());
    fs::write(&path, serde_json::to_vec(&table)?)?;
    let e = entry();
    assert!(save(Some(&path), e.clone()));
    assert!(cached(Some(&path), &e.key)?);
    assert!(TilingTable::load(&path)?.devices.contains_key("keep"));
    for field in [
        "driver",
        "native",
        "pack",
        "library",
        "algorithm",
        "shape",
        "schema",
    ] {
        let mut other = e.key.clone();
        match field {
            "driver" => other.native.driver_build = Some("different".into()),
            "native" => other.native.kernel_fingerprint.push('!'),
            "pack" => other.pack_fingerprint.push('!'),
            "library" => other.library.lt_version += 1,
            "algorithm" => other.algorithm.id += 1,
            "shape" => other.native.m += 128,
            _ => other.schema.push('!'),
        }
        assert!(!cached(Some(&path), &other)?, "identity {field}");
    }
    Ok(())
}
#[test]
fn partial_forged_or_duplicate_qualifications_are_not_reused() -> TestResult {
    let temp = Temp::new()?;
    let path = temp.file();
    let e = entry();
    for damage in [
        "order",
        "missing",
        "samples",
        "median",
        "regression",
        "driver",
    ] {
        let mut bad = e.clone();
        match damage {
            "order" => bad.trials[1].order = MeasurementOrder::BaselineFirst,
            "missing" => {
                bad.trials.pop();
            }
            "samples" => {
                bad.trials[0].warm.baseline_ms.pop();
            }
            "median" => bad.trials[1].cold.baseline_median_ms = 3.0,
            "regression" => {
                bad.trials[1].warm = PairedTimes {
                    baseline_ms: vec![1.0; MIN_REPS],
                    candidate_ms: vec![2.0; MIN_REPS],
                    baseline_median_ms: 1.0,
                    candidate_median_ms: 2.0,
                    speedup: 0.5,
                }
            }
            _ => bad.key.native.driver_build = None,
        }
        assert!(!bad.valid(), "{damage}");
        assert!(!save(Some(&path), bad.clone()));
        let mut table = TilingTable::default();
        table.resident_fp4_lt.push(bad);
        fs::write(&path, serde_json::to_vec(&table)?)?;
        if damage != "driver" {
            assert!(cached(Some(&path), &e.key).is_err(), "{damage}");
        }
    }
    let table = TilingTable {
        resident_fp4_lt: vec![e.clone(), e.clone()],
        ..TilingTable::default()
    };
    fs::write(&path, serde_json::to_vec(&table)?)?;
    assert!(cached(Some(&path), &e.key).is_err());
    Ok(())
}
#[test]
fn legacy_tables_have_no_implicit_fp4_qualification() -> TestResult {
    let table: TilingTable = serde_json::from_str(r#"{"devices":{}}"#)?;
    assert!(table.resident_fp4_lt.is_empty());
    assert!(!cached(None, &entry().key)?);
    Ok(())
}
