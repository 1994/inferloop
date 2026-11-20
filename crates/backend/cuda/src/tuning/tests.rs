use super::*;
use std::env::temp_dir;
use std::fs::create_dir;
use std::fs::read_dir;
use std::fs::read_to_string;
use std::fs::remove_dir_all;
use std::fs::write;
use std::io::Error as IoError;
use std::process::id;

const DEVICE_A: &str = "test-a|sm_90|81920MiB";
const DEVICE_B: &str = "test-b|sm_120|32768MiB";
const TABLE: &str = r#"{
    "device": "test-a|sm_90|81920MiB",
    "entries": [
        {"dtype": "nvfp4", "rows": 17408, "columns": 5120, "tile_rows": 16, "tile_columns": 512},
        {"dtype": "bf16", "rows": 48, "columns": 5120, "tile_rows": 1, "tile_columns": 512}
    ]
}"#;

fn decode(text: &str) -> Result<TilingTable> {
    serde_json::from_str(text).map_err(|error| Error::invalid(format!("test table: {error}")))
}

fn trial(rows: usize, columns: usize, selected: LinearTiling) -> ProjectionTrial {
    ProjectionTrial {
        key: format!("bf16:{rows}x{columns}"),
        rows,
        columns,
        gpu: "test GPU".to_owned(),
        target: "test".to_owned(),
        baseline: selected,
        selected,
        trials: Vec::new(),
        confirmation: Trial {
            tiling: selected,
            baseline_ms: Vec::new(),
            candidate_ms: Vec::new(),
            baseline_median_ms: 1.0,
            candidate_median_ms: 1.0,
            speedup: 1.0,
        },
        clear_l2: true,
    }
}

fn filesystem(error: &IoError) -> Error {
    Error::invalid(format!("test filesystem: {error}"))
}

struct TemporaryDirectory(PathBuf);

impl TemporaryDirectory {
    fn new() -> Result<Self> {
        static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);
        let path = temp_dir().join(format!(
            "infer-cuda-tuning-test-{}-{}",
            id(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed)
        ));
        create_dir(&path).map_err(|error| filesystem(&error))?;
        Ok(Self(path))
    }
}

impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        let cleanup = remove_dir_all(&self.0);
        drop(cleanup);
    }
}

#[test]
fn legacy_table_migrates_and_round_trips_with_device_scope() -> Result<()> {
    let table = decode(TABLE)?;
    assert_eq!(table.devices.len(), 1);
    assert_eq!(table.devices[DEVICE_A].len(), 2);
    assert_eq!(
        table.tiling_for(DEVICE_A, "nvfp4", 17408, 5120)?,
        Some(LinearTiling::new(16, 512)?)
    );
    assert_eq!(
        table.tiling_for(DEVICE_A, "bf16", 48, 5120)?,
        Some(LinearTiling::new(1, 512)?)
    );
    assert_eq!(table.tiling_for(DEVICE_A, "nvfp4", 5120, 5120)?, None);
    assert_eq!(table.tiling_for(DEVICE_A, "fp8-channel", 48, 5120)?, None);
    assert_eq!(table.tiling_for(DEVICE_B, "nvfp4", 17408, 5120)?, None);
    let encoded = serde_json::to_string(&table)
        .map_err(|error| Error::invalid(format!("test table: {error}")))?;
    assert!(!encoded.contains("\"entries\":"));
    assert!(!encoded.contains("\"device\":"));
    assert_eq!(decode(&encoded)?.devices, table.devices);
    // The earlier `gpu` alias identifies the same legacy hardware and also migrates.
    assert_eq!(
        decode(&TABLE.replace("\"device\"", "\"gpu\""))?.devices,
        table.devices
    );
    Ok(())
}

#[test]
fn recording_one_device_preserves_other_devices_and_geometries() -> Result<()> {
    let mut table = decode(TABLE)?;
    let a_tile = LinearTiling::new(4, 128)?;
    let b_tile = LinearTiling::new(8, 256)?;
    table.record(DEVICE_B, &trial(48, 5120, b_tile));
    table.record(DEVICE_A, &trial(48, 5120, a_tile));
    assert_eq!(table.devices.len(), 2);
    assert_eq!(table.devices[DEVICE_A].len(), 2);
    assert_eq!(table.devices[DEVICE_B].len(), 1);
    assert_eq!(table.tiling_for(DEVICE_A, "bf16", 48, 5120)?, Some(a_tile));
    assert_eq!(table.tiling_for(DEVICE_B, "bf16", 48, 5120)?, Some(b_tile));
    assert_eq!(
        table.tiling_for(DEVICE_A, "nvfp4", 17408, 5120)?,
        Some(LinearTiling::new(16, 512)?)
    );
    assert_eq!(table.tiling_for(DEVICE_B, "nvfp4", 17408, 5120)?, None);
    Ok(())
}

#[test]
fn migration_keeps_explicit_per_device_entries_over_legacy_duplicates() -> Result<()> {
    let mut value: serde_json::Value = serde_json::from_str(TABLE)
        .map_err(|error| Error::invalid(format!("test table: {error}")))?;
    value["devices"] = serde_json::json!({ (DEVICE_A): [
        {"dtype": "bf16", "rows": 48, "columns": 5120, "tile_rows": 4, "tile_columns": 128}
    ] });
    let table: TilingTable = serde_json::from_value(value)
        .map_err(|error| Error::invalid(format!("test table: {error}")))?;
    assert_eq!(table.devices[DEVICE_A].len(), 2);
    assert_eq!(
        table.tiling_for(DEVICE_A, "bf16", 48, 5120)?,
        Some(LinearTiling::new(4, 128)?)
    );
    assert_eq!(
        table.tiling_for(DEVICE_A, "nvfp4", 17408, 5120)?,
        Some(LinearTiling::new(16, 512)?)
    );
    Ok(())
}

#[test]
fn unidentified_entries_never_match_or_migrate_to_a_device() -> Result<()> {
    let mut table = decode(&TABLE.replace(DEVICE_A, ""))?;
    assert!(table.devices.is_empty());
    assert_eq!(table.tiling_for(DEVICE_A, "bf16", 48, 5120)?, None);
    assert_eq!(table.tiling_for("", "bf16", 48, 5120)?, None);
    table.record("", &trial(48, 5120, LinearTiling::new(4, 128)?));
    assert!(table.devices.is_empty());
    let table = decode(
        r#"{"devices":{"": [
        {"dtype":"bf16","rows":48,"columns":5120,"tile_rows":4,"tile_columns":128}
    ]}}"#,
    )?;
    assert!(table.devices.is_empty());
    Ok(())
}

#[test]
fn measured_cache_keys_include_device_identity() -> Result<()> {
    let a_tile = LinearTiling::new(4, 128)?;
    let b_tile = LinearTiling::new(8, 256)?;
    let memo: MeasuredTilings = Mutex::new(BTreeMap::new());
    let mut entries = memo
        .lock()
        .map_err(|_| Error::invariant("test tuning cache poisoned"))?;
    entries.insert(TilingKey::new(DEVICE_A, "bf16:48x5120", 48, 5120), a_tile);
    entries.insert(TilingKey::new(DEVICE_B, "bf16:48x5120", 48, 5120), b_tile);
    assert_eq!(entries.len(), 2);
    assert_eq!(
        entries.get(&TilingKey::new(DEVICE_A, "bf16:48x5120", 48, 5120)),
        Some(&a_tile)
    );
    assert_eq!(
        entries.get(&TilingKey::new(DEVICE_B, "bf16:48x5120", 48, 5120)),
        Some(&b_tile)
    );
    Ok(())
}

#[test]
fn persist_atomically_migrates_and_retains_multiple_devices() -> Result<()> {
    let directory = TemporaryDirectory::new()?;
    let path = directory.0.join("tuning.json");
    write(&path, TABLE).map_err(|error| filesystem(&error))?;
    let a_tile = LinearTiling::new(4, 128)?;
    let b_tile = LinearTiling::new(8, 256)?;
    assert!(persist(Some(&path), DEVICE_B, &trial(48, 5120, b_tile)));
    assert!(persist(Some(&path), DEVICE_A, &trial(48, 5120, a_tile)));
    let table = TilingTable::load(&path)?;
    assert_eq!(table.devices.len(), 2);
    assert_eq!(table.tiling_for(DEVICE_A, "bf16", 48, 5120)?, Some(a_tile));
    assert_eq!(table.tiling_for(DEVICE_B, "bf16", 48, 5120)?, Some(b_tile));
    assert_eq!(table.devices[DEVICE_A].len(), 2);
    assert_eq!(
        read_dir(&directory.0)
            .map_err(|error| filesystem(&error))?
            .count(),
        1
    );
    let fresh = directory.0.join("nested/tuning.json");
    assert!(persist(Some(&fresh), DEVICE_B, &trial(48, 5120, b_tile)));
    assert_eq!(
        TilingTable::load(&fresh)?.tiling_for(DEVICE_B, "bf16", 48, 5120)?,
        Some(b_tile)
    );
    Ok(())
}

#[test]
fn persist_reports_failures_and_preserves_existing_data() -> Result<()> {
    let directory = TemporaryDirectory::new()?;
    let trial = trial(48, 5120, LinearTiling::new(4, 128)?);
    assert!(!persist(None, DEVICE_A, &trial));
    let path = directory.0.join("tuning.json");
    assert!(!persist(Some(&path), "", &trial));
    assert!(!path.exists());
    write(&path, "malformed table").map_err(|error| filesystem(&error))?;
    assert!(!persist(Some(&path), DEVICE_A, &trial));
    assert_eq!(
        read_to_string(&path).map_err(|error| filesystem(&error))?,
        "malformed table"
    );
    // An existing file cannot serve as a parent directory, independent of test user permissions.
    assert!(!persist(Some(&path.join("child.json")), DEVICE_A, &trial));
    // A rename onto a directory fails after creating a temporary file; that file must be removed.
    let target = directory.0.join("directory.json");
    create_dir(&target).map_err(|error| filesystem(&error))?;
    assert!(!persist(Some(&target), DEVICE_A, &trial));
    assert!(target.is_dir());
    assert_eq!(
        read_dir(&directory.0)
            .map_err(|error| filesystem(&error))?
            .count(),
        2
    );
    Ok(())
}

#[test]
fn shipped_baseline_answers_only_the_device_it_was_measured_on() -> Result<()> {
    let table = shipped_table()?;
    assert!(
        !table.devices.is_empty(),
        "shipped entries must identify their hardware"
    );
    for (device, entries) in &table.devices {
        assert_ne!(device, "");
        assert_ne!(entries.len(), 0);
        for entry in entries {
            let expected = LinearTiling::new(entry.tile_rows, entry.tile_columns)?;
            assert_eq!(
                table.tiling_for(device, &entry.dtype, entry.rows, entry.columns)?,
                Some(expected)
            );
            assert_eq!(
                table.tiling_for(
                    "unknown|sm_000|0MiB",
                    &entry.dtype,
                    entry.rows,
                    entry.columns
                )?,
                None
            );
        }
    }
    Ok(())
}
