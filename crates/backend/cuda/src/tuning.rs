//! Paired graph measurements on loaded model weights, isolated from model adapters.
pub mod attention;
pub mod fp8_gemm;
pub mod gated_norm;
pub mod greedy;
pub(crate) mod memory;
pub(crate) mod prefill;
pub mod recurrent;
pub mod resident_gemm;

use crate::device::DeviceProfile;
use crate::strategy::default_tiling;
use crate::target::CudaTarget;
use crate::{
    benchmark::Trial,
    device::{CudaDevice, device_error},
    mlp::ProjectionWeight,
    strategy::LinearTiling,
};
use cutile::error::tensor_error;
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    prelude::*,
};
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::env::var_os;
use std::fs::OpenOptions;
use std::fs::create_dir_all;
use std::fs::read_to_string;
use std::fs::remove_file;
use std::fs::rename;
use std::io::Error as IoError;
use std::io::Result as IoResult;
use std::process::id;
use std::{
    collections::BTreeMap,
    io::Write,
    path::{Path, PathBuf},
    sync::{
        Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

/// Winners must beat the baseline by this factor on an independent trial.
const MIN_CONFIRM_SPEEDUP: f32 = 1.02;
/// Search space of graph GEMV tile dimensions.
const TILE_ROW_SEARCH: [usize; 4] = [1, 4, 8, 16];
/// Search space of graph GEMV tile dimensions.
const TILE_COLUMN_SEARCH: [usize; 3] = [128, 256, 512];
/// Relative plus absolute slack between candidate and baseline outputs.
const MATCH_TOLERANCE: f32 = 5e-4;
/// Deterministic sample value bounds: modulo, center and scale.
const SAMPLE_MODULUS: usize = 31;
/// Deterministic sample value bounds: modulo, center and scale.
const SAMPLE_CENTER: i16 = 15;
/// Deterministic sample value bounds: modulo, center and scale.
const SAMPLE_SCALE: f32 = 16.0;
/// Measurement wall clocks and repetition bounds.
const WARMUP_MS: u64 = 50;
/// Measurement wall clocks and repetition bounds.
const REP_MS: u64 = 100;
/// Measurement wall clocks and repetition bounds.
const MIN_REPS: usize = 10;
/// Measurement wall clocks and repetition bounds.
const MAX_REPS: usize = 30;

/// Median of paired timing samples, or NaN when there are none.
///
/// Every tuning search reports a median through this one implementation so a speedup cannot depend
/// on which module computed it. The halve-then-add form keeps the midpoint of two large samples
/// from overflowing the way `(lo + hi) / 2` would.
fn median(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return f32::NAN;
    }
    let mut ordered = samples.to_vec();
    ordered.sort_by(f32::total_cmp);
    let lo = (ordered.len() - 1) / 2;
    let hi = ordered.len() / 2;
    if lo == hi {
        ordered[lo]
    } else {
        ordered[lo] / 2.0 + ordered[hi] / 2.0
    }
}

#[derive(Serialize)]
pub struct ProjectionTrial {
    pub key: String,
    pub rows: usize,
    pub columns: usize,
    pub gpu: String,
    pub target: CudaTarget,
    pub baseline: LinearTiling,
    pub selected: LinearTiling,
    pub trials: Vec<Trial>,
    pub confirmation: Trial,
    pub clear_l2: bool,
}

/// One projection measured during a load, with the confirmed winner.
#[derive(Debug, Clone, Serialize)]
pub struct MeasuredTiling {
    /// `dtype:rowsxcolumns` key of the measured projection.
    pub key: String,
    /// Confirmed winning tile.
    pub tiling: LinearTiling,
    /// Confirmed speedup over the conservative baseline; without a win the baseline is kept.
    pub speedup: f32,
}

/// Automatic tiling decisions from one model load, for reporting rather than control.
#[derive(Debug, Clone, Default, Serialize)]
pub struct TuningReport {
    /// Geometries measured and confirmed on this load.
    pub measured: Vec<MeasuredTiling>,
    /// Geometries whose measurement failed, with the reason; the conservative tile was kept.
    pub fallback: Vec<String>,
    /// Whether at least one measurement reached the machine-local cache for the next load.
    pub cached: bool,
}

/// Stable dtype and geometry key, without model-specific weight names.
/// # Errors
/// Rejects invalid projection rank/dimensions.
pub fn projection_key(weight: &ProjectionWeight) -> Result<(String, usize, usize)> {
    let (encoding, shape, packing) = match weight {
        ProjectionWeight::Dense(w) => ("bf16", w.shape(), 1),
        ProjectionWeight::Fp8(w, _) => ("fp8-channel", w.shape(), 1),
        ProjectionWeight::Fp8Block(w, _) => ("fp8-block", w.shape(), 1),
        ProjectionWeight::Fp4(w, _, _) => ("nvfp4", w.shape(), 2),
    };
    if shape.len() != 2 {
        return Err(Error::invalid("projection rank"));
    }
    let rows = usize::try_from(shape[0]).map_err(device_error)?;
    let columns = usize::try_from(shape[1]).map_err(device_error)? * packing;
    Ok((format!("{encoding}:{rows}x{columns}"), rows, columns))
}

/// Dtype part of a `dtype:rowsxcolumns` key, matching what [`TuningEntry::dtype`] records.
fn key_dtype(key: &str) -> &str {
    key.split(':').next().unwrap_or_default()
}

/// Tune once per dtype/shape, then confirm the winner in a fresh paired trial.
/// # Errors
/// Rejects numerical divergence, invalid storage or failed CUDA measurements.
pub fn tune(device: &CudaDevice, weight: &ProjectionWeight) -> Result<ProjectionTrial> {
    let (key, rows, columns) = projection_key(weight)?;
    let baseline = default_tiling()?;
    let (base, expected) = capture(device, weight, rows, columns, baseline)?;
    base.launch()
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let expected = device.read(&expected)?;
    let options = BenchOptions {
        warmup: Duration::from_millis(WARMUP_MS),
        rep: Duration::from_millis(REP_MS),
        min_reps: MIN_REPS,
        max_reps: MAX_REPS,
        clear_l2: true,
    };
    let mut trials = Vec::new();
    let mut selected = baseline;
    let mut best = 1.0f32;
    for tile_rows in TILE_ROW_SEARCH {
        for tile_columns in TILE_COLUMN_SEARCH {
            let tiling = LinearTiling::new(tile_rows, tile_columns)?;
            let (candidate, output) = capture(device, weight, rows, columns, tiling)?;
            candidate
                .launch()
                .sync_on(&device.stream)
                .map_err(device_error)?;
            verify(&device.read(&output)?, &expected)?;
            let trial = measure(device, &options, &base, &candidate, tiling)?;
            if trial.speedup > best {
                best = trial.speedup;
                selected = tiling;
            }
            trials.push(trial);
        }
    }
    let (candidate, _) = capture(device, weight, rows, columns, selected)?;
    let confirmation = measure(device, &options, &base, &candidate, selected)?;
    // Require a measurable win on the independent trial; noise keeps the baseline.
    if confirmation.speedup < MIN_CONFIRM_SPEEDUP {
        selected = baseline;
    }
    Ok(ProjectionTrial {
        key,
        rows,
        columns,
        gpu: device.stream.device().name().map_err(device_error)?,
        target: device.target().clone(),
        baseline,
        selected,
        trials,
        confirmation,
        clear_l2: true,
    })
}

fn capture(
    device: &CudaDevice,
    weight: &ProjectionWeight,
    rows: usize,
    columns: usize,
    tile: LinearTiling,
) -> Result<(CudaGraph<()>, Arc<Tensor<f32>>)> {
    weight.validate(rows, columns)?;
    let values = (0..columns)
        .map(|i| {
            f32::from(i16::try_from(i % SAMPLE_MODULUS).unwrap_or_default() - SAMPLE_CENTER)
                / SAMPLE_SCALE
        })
        .collect();
    let input = device.upload(values, &[columns])?;
    let mut output = api::zeros::<f32>(&[rows])
        .sync_on(&device.stream)
        .map_err(device_error)?;
    let graph = CudaGraph::scope(&device.stream, |scope| {
        weight.record(scope, &mut output, &input, columns, tile)
    })
    .map_err(device_error)?;
    Ok((graph, Arc::new(output)))
}

fn measure(
    device: &CudaDevice,
    options: &BenchOptions,
    base: &CudaGraph<()>,
    candidate: &CudaGraph<()>,
    tiling: LinearTiling,
) -> Result<Trial> {
    let run = |graph: &CudaGraph<()>| {
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(|e| tensor_error(&e.to_string()))
    };
    let (a, b) = do_bench_paired(&device.stream, options, |_| run(base), |_| run(candidate))
        .map_err(device_error)?;
    Ok(Trial {
        tiling,
        baseline_ms: a.times_ms().to_vec(),
        candidate_ms: b.times_ms().to_vec(),
        baseline_median_ms: a.median_ms(),
        candidate_median_ms: b.median_ms(),
        speedup: a.median_ms() / b.median_ms(),
    })
}

fn verify(actual: &[f32], expected: &[f32]) -> Result<()> {
    if actual.iter().zip(expected).any(|(a, b)| {
        !a.is_finite()
            || !b.is_finite()
            || (a - b).abs() > b.abs().mul_add(MATCH_TOLERANCE, MATCH_TOLERANCE)
    }) {
        return Err(Error::invariant(
            "projection tuning changed output beyond F32 tolerance",
        ));
    }
    Ok(())
}

/// Environment override for the machine-local tuning table path.
const TUNING_TABLE_ENV: &str = "INFER_CUDA_TUNING_TABLE";

/// One measured tile decision, keyed by projection dtype and geometry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuningEntry {
    /// Projection dtype key: `bf16`, `fp8-channel` or `nvfp4`.
    pub dtype: String,
    /// Output rows of the projection.
    pub rows: usize,
    /// Input columns of the projection.
    pub columns: usize,
    /// Winning tile rows.
    pub tile_rows: usize,
    /// Winning tile columns.
    pub tile_columns: usize,
}

/// Machine-local GEMV tiling decisions, scoped to each measured device identity.
///
/// Old single-device tables are migrated on read. Entries without a device identity are discarded:
/// there is no evidence that their measurements apply to the running hardware.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(from = "StoredTilingTable")]
pub struct TilingTable {
    /// Measured entries per `DeviceProfile::identity`; one cache can retain several GPUs.
    pub devices: BTreeMap<String, Vec<TuningEntry>>,
    /// Multirow W4A4 decisions are independent of scalar GEMV entries.
    #[serde(default)]
    pub resident_gemm: Vec<resident_gemm::ResidentGemmEntry>,
    #[serde(default)]
    pub resident_fp8_gemm: Vec<fp8_gemm::Entry>,
    /// Device/driver/kernel/geometry scoped exact greedy reduction decisions.
    #[serde(default)]
    pub greedy: Vec<greedy::Entry>,
    /// Native gated norm compiler occupancy with full-output bitwise evidence.
    #[serde(default)]
    pub resident_gated_norm: Vec<gated_norm::NormEntry>,
    /// F32-state recurrent column decisions with independent F64 and paired evidence.
    #[serde(default)]
    pub resident_recurrent: Vec<recurrent::Entry>,
    /// Prefill query sharing with independent F64 and complete graph measurements.
    #[serde(default)]
    pub resident_attention: Vec<attention::Entry>,
    /// Exact Rust SIMT alternatives with hardware and artifact scoped evidence.
    #[serde(default)]
    pub resident_exact_attention: Vec<attention::exact::Entry>,
    /// Exact F32 shared KV with short-history regression and long-history paired evidence.
    #[serde(default)]
    pub resident_shared_attention: Vec<attention::shared::Entry>,
    /// Exact vectorized shared-key refinement of the qualified shared kernel.
    #[serde(default)]
    pub resident_vector_shared_attention: Vec<attention::vector::Entry>,
    /// Bit-exact combined vector key/consumer refinement of the qualified vector kernel.
    #[serde(default)]
    pub resident_vector_combined_attention: Vec<attention::vector_combined::Entry>,
    /// FA4 BF16 profile, artifact and architecture scoped cluster search evidence.
    #[serde(default)]
    pub resident_mma_attention: Vec<attention::mma::Entry>,
    /// F32-cache FA4 profile with full short-output and long-row F64 checks.
    #[serde(default)]
    pub resident_mma_f32_attention: Vec<attention::mma_f32::Entry>,
    #[serde(default)]
    pub resident_exact_recurrent: Vec<recurrent::exact::Entry>,
}

/// Read both the current per-device table and the former top-level `device`/`entries` format.
#[derive(Deserialize)]
struct StoredTilingTable {
    #[serde(default, alias = "gpu")]
    device: String,
    #[serde(default)]
    entries: Vec<TuningEntry>,
    #[serde(default)]
    devices: BTreeMap<String, Vec<TuningEntry>>,
    #[serde(default)]
    resident_gemm: Vec<resident_gemm::ResidentGemmEntry>,
    #[serde(default)]
    resident_fp8_gemm: Vec<fp8_gemm::Entry>,
    #[serde(default)]
    greedy: Vec<greedy::Entry>,
    #[serde(default)]
    resident_gated_norm: Vec<gated_norm::NormEntry>,
    #[serde(default)]
    resident_recurrent: Vec<recurrent::Entry>,
    #[serde(default)]
    resident_attention: Vec<attention::Entry>,
    #[serde(default)]
    resident_exact_attention: Vec<attention::exact::Entry>,
    #[serde(default)]
    resident_shared_attention: Vec<attention::shared::Entry>,
    #[serde(default)]
    resident_vector_shared_attention: Vec<attention::vector::Entry>,
    #[serde(default)]
    resident_vector_combined_attention: Vec<attention::vector_combined::Entry>,
    #[serde(default)]
    resident_mma_attention: Vec<attention::mma::Entry>,
    #[serde(default)]
    resident_mma_f32_attention: Vec<attention::mma_f32::Entry>,
    #[serde(default)]
    resident_exact_recurrent: Vec<recurrent::exact::Entry>,
}

impl From<StoredTilingTable> for TilingTable {
    fn from(stored: StoredTilingTable) -> Self {
        let mut devices = stored.devices;
        devices.remove("");
        if !stored.device.is_empty() {
            let entries = devices.entry(stored.device).or_default();
            for legacy in stored.entries {
                // Explicit per-device entries win over a legacy duplicate during migration.
                if !entries.iter().any(|entry| {
                    entry.dtype == legacy.dtype
                        && entry.rows == legacy.rows
                        && entry.columns == legacy.columns
                }) {
                    entries.push(legacy);
                }
            }
        }
        Self {
            devices,
            resident_gemm: stored.resident_gemm,
            resident_fp8_gemm: stored.resident_fp8_gemm,
            greedy: stored.greedy,
            resident_gated_norm: stored.resident_gated_norm,
            resident_recurrent: stored.resident_recurrent,
            resident_attention: stored.resident_attention,
            resident_exact_attention: stored.resident_exact_attention,
            resident_shared_attention: stored.resident_shared_attention,
            resident_vector_shared_attention: stored.resident_vector_shared_attention,
            resident_vector_combined_attention: stored.resident_vector_combined_attention,
            resident_mma_attention: stored.resident_mma_attention,
            resident_mma_f32_attention: stored.resident_mma_f32_attention,
            resident_exact_recurrent: stored.resident_exact_recurrent,
        }
    }
}

impl TilingTable {
    /// # Errors
    /// Rejects unreadable or malformed table files instead of silently ignoring them.
    pub fn load(path: &Path) -> Result<Self> {
        let text = read_to_string(path)
            .map_err(|error| Error::invalid(format!("CUDA tuning table: {error}")))?;
        serde_json::from_str(&text)
            .map_err(|error| Error::invalid(format!("CUDA tuning table: {error}")))
    }
    /// Record one trial for `device`, replacing any previous entry for the same geometry.
    pub fn record(&mut self, device: &str, trial: &ProjectionTrial) {
        if device.is_empty() {
            return;
        }
        let dtype = key_dtype(&trial.key).to_owned();
        let entries = self.devices.entry(device.to_owned()).or_default();
        entries.retain(|entry| {
            entry.dtype != dtype || entry.rows != trial.rows || entry.columns != trial.columns
        });
        entries.push(TuningEntry {
            dtype,
            rows: trial.rows,
            columns: trial.columns,
            tile_rows: trial.selected.rows(),
            tile_columns: trial.selected.columns(),
        });
    }
    /// Measured tile for one device, dtype and geometry; `None` when nothing measured it.
    ///
    /// Entries measured on another device never apply: tiles are a property of the machine. The
    /// dtype is the bare encoding (`nvfp4`), not the composite key, because that is what
    /// [`TuningEntry::dtype`] records.
    /// # Errors
    /// Rejects entries whose tile dimensions are not a legal tiling.
    pub fn tiling_for(
        &self,
        device: &str,
        dtype: &str,
        rows: usize,
        columns: usize,
    ) -> Result<Option<LinearTiling>> {
        if device.is_empty() {
            return Ok(None);
        }
        if let Some(entries) = self.devices.get(device) {
            for entry in entries {
                if entry.dtype == dtype && entry.rows == rows && entry.columns == columns {
                    return LinearTiling::new(entry.tile_rows, entry.tile_columns).map(Some);
                }
            }
        }
        Ok(None)
    }
}

/// Derive the machine-local tuning table path.
///
/// The path comes from the user cache directory and entries are keyed by device identity, so
/// one file serves every GPU and model on this host.
#[must_use]
pub fn table_path() -> Option<PathBuf> {
    if let Some(explicit) = var_os(TUNING_TABLE_ENV).filter(|value| !value.is_empty()) {
        return Some(PathBuf::from(explicit));
    }
    let root = var_os("XDG_CACHE_HOME")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .or_else(|| var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))?;
    Some(root.join("infer-cuda").join("tuning.json"))
}

/// Load the configured tuning table; an absent file simply means "nothing measured yet".
/// # Errors
/// Rejects unreadable or malformed tables instead of silently ignoring them.
pub(crate) fn resolve_table(path: Option<&Path>) -> Result<Option<TilingTable>> {
    let Some(path) = path.map(Path::to_path_buf).or_else(table_path) else {
        return Ok(None);
    };
    if !path.is_file() {
        return Ok(None);
    }
    TilingTable::load(&path).map(Some)
}

/// Baseline measured on known devices and shipped with the backend.
///
/// This is data, not a policy branch: an entry applies only when the running device's identity
/// matches the one it was measured on, and any other device or geometry is measured on first use
/// through [`TilingSource::autotune`]. A new card therefore needs no code change, only its first
/// load; a card with a shipped entry starts at its measured best tile immediately.
const SHIPPED_BASELINE: &str = include_str!("../baselines/tuning.json");

/// Parse the shipped baseline once per process.
/// # Errors
/// Rejects a malformed embedded baseline, which would be a packaging mistake.
pub(crate) fn shipped_table() -> Result<&'static TilingTable> {
    static SHIPPED: OnceLock<TilingTable> = OnceLock::new();
    if let Some(table) = SHIPPED.get() {
        return Ok(table);
    }
    let table: TilingTable = serde_json::from_str(SHIPPED_BASELINE)
        .map_err(|error| Error::invalid(format!("shipped CUDA tuning baseline: {error}")))?;
    Ok(SHIPPED.get_or_init(|| table))
}

/// Where tile decisions come from for one model load.
pub(crate) struct TilingSource<'a> {
    /// Machine-local measurements for this device, when the cache exists.
    pub local: Option<&'a TilingTable>,
    /// Shipped baseline for known devices; read-only and never written back.
    pub shipped: Option<&'a TilingTable>,
    /// Path new measurements are appended to.
    pub path: Option<PathBuf>,
    /// Whether tables leave a geometry uncovered, so it may be measured now.
    pub autotune: bool,
}

/// Device identity plus geometry of one measured projection tiling.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct TilingKey {
    device: String,
    projection: String,
    rows: usize,
    columns: usize,
}

impl TilingKey {
    fn new(device: &str, projection: &str, rows: usize, columns: usize) -> Self {
        Self {
            device: device.to_owned(),
            projection: projection.to_owned(),
            rows,
            columns,
        }
    }
}
/// In-process memo of autotuned tilings.
type MeasuredTilings = Mutex<BTreeMap<TilingKey, LinearTiling>>;

/// In-process memo so one autotuned geometry is never measured twice.
fn measured() -> &'static MeasuredTilings {
    static MEASURED: OnceLock<MeasuredTilings> = OnceLock::new();
    MEASURED.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Merge one trial into the derived table file so the measurement is reused next start.
///
/// Best effort by design: a missing, unreadable or read-only cache only costs re-measurement on the
/// next start. Returns whether the trial was written, so the caller can report it truthfully.
fn persist(path: Option<&Path>, device: &str, trial: &ProjectionTrial) -> bool {
    if device.is_empty() {
        return false;
    }
    update_table(path, |table| table.record(device, trial))
}

/// Shared read/merge/write lock for scalar and resident measurements in the same table.
fn update_table(path: Option<&Path>, update: impl FnOnce(&mut TilingTable)) -> bool {
    static WRITER: Mutex<()> = Mutex::new(());
    let Some(path) = path else {
        return false;
    };
    let Ok(_writer) = WRITER.lock() else {
        return false;
    };
    let mut table = if path.is_file() {
        let Ok(table) = TilingTable::load(path) else {
            return false;
        };
        table
    } else {
        TilingTable::default()
    };
    update(&mut table);
    let Ok(text) = serde_json::to_string_pretty(&table) else {
        return false;
    };
    write_atomic(path, &text).is_ok()
}

/// Replace a cache only after its full contents reach a file in the same directory.
fn write_atomic(path: &Path, text: &str) -> IoResult<()> {
    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    create_dir_all(parent)?;
    let filename = path
        .file_name()
        .ok_or_else(|| IoError::other("CUDA tuning cache has no filename"))?;
    let temporary = parent.join(format!(
        ".{}.{}.{}.tmp",
        filename.to_string_lossy(),
        id(),
        NEXT_FILE.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let written = file
        .write_all(text.as_bytes())
        .and_then(|()| file.sync_all());
    drop(file);
    let result = written.and_then(|()| rename(&temporary, path));
    if result.is_err() {
        // Only remove files created by this call, including failed writes or renames.
        let cleanup = remove_file(&temporary);
        drop(cleanup);
    }
    result
}

/// Tile for one projection: local measurements, the shipped baseline, then measuring it here.
///
/// A measurement is an optional optimization, so a failed one keeps the conservative tile and is
/// recorded in `report`; only malformed tables or invalid dimensions fail the load.
/// # Errors
/// Rejects malformed tables and invalid tile dimensions.
pub(crate) fn select_tiling(
    device: &CudaDevice,
    weight: &ProjectionWeight,
    profile: &DeviceProfile,
    source: &TilingSource<'_>,
    report: &mut TuningReport,
) -> Result<LinearTiling> {
    let (key, rows, columns) = projection_key(weight)?;
    let identity = profile.identity();
    let memo_key = TilingKey::new(&identity, &key, rows, columns);
    // Local measurements win over the shipped baseline; both are scoped to this device identity.
    for table in [source.local, source.shipped].into_iter().flatten() {
        if let Some(tiling) = table.tiling_for(&identity, key_dtype(&key), rows, columns)? {
            return Ok(tiling);
        }
    }
    let baseline = default_tiling()?;
    if !source.autotune {
        return Ok(baseline);
    }
    let cached = measured()
        .lock()
        .map_err(|_| Error::invariant("CUDA tuning cache poisoned"))?
        .get(&memo_key)
        .copied();
    if let Some(tiling) = cached {
        return Ok(tiling);
    }
    let trial = match tune(device, weight) {
        Ok(trial) => trial,
        Err(error) => {
            // The conservative tile is already numerically verified, so an optional measurement
            // that fails must not fail the load; the caller reports why it was skipped.
            report.fallback.push(format!("{key}: {error}"));
            return Ok(baseline);
        }
    };
    measured()
        .lock()
        .map_err(|_| Error::invariant("CUDA tuning cache poisoned"))?
        .insert(memo_key, trial.selected);
    report.cached |= persist(source.path.as_deref(), &identity, &trial);
    report.measured.push(MeasuredTiling {
        key,
        tiling: trial.selected,
        speedup: trial.confirmation.speedup,
    });
    Ok(trial.selected)
}

#[cfg(test)]
#[path = "../tests/unit/tuning.rs"]
mod tests;
