//! Device-scoped query sharing search; complete attention graphs retain the FP8 KV contract.
mod capture;
pub(crate) mod decode;
pub mod exact;
pub mod mma;
pub mod mma_f32;
mod reference;
pub mod shared;
pub mod vector;
pub mod vector_combined;
use super::{
    MAX_REPS, MIN_CONFIRM_SPEEDUP, MIN_REPS, REP_MS, WARMUP_MS, median,
    resident_gemm::ResidentTuningSession,
};
use crate::constants::PREFILL_LANES;
use crate::device::{CudaDevice, device_error};
use crate::resident::attention_mma::catalog::Variant;
use crate::tuning::preparation::driver_build;
use capture::Captured;
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    prelude::*,
};
use infer_core::{Error, Result};
use reference::Fixture;
use serde::{Deserialize, Serialize};
#[cfg(test)]
use std::env::var;
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
    time::Duration,
};

pub(crate) const SCHEMA: &str = "prefill-fp8-kv:precise-bf16-products6:f32-accum:original-append:gqa:query-causal-bound:qt16-32:f64-full1e-4:tail-history-window-permuted:ab-ba-v2";
const QUERY_TILES: [usize; 2] = [16, 32];
const MAX_ERROR: f64 = 1e-4;
const VALIDATED_DIM: usize = 256;
const CUDA_OPT_LEVEL: u8 = 3;
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Shape {
    pub heads: usize,
    pub kv_heads: usize,
    pub dim: usize,
    pub lanes: usize,
    pub window: usize,
}
impl Shape {
    fn supported(self) -> bool {
        self.heads > 0
            && self.kv_heads > 0
            && self.heads.is_multiple_of(self.kv_heads)
            && self.dim == VALIDATED_DIM
            && self.lanes >= PREFILL_LANES
            && self.lanes.is_multiple_of(QUERY_TILES[1])
            && i32::try_from(self.window).is_ok()
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Choice {
    Simt,
    RustSimt,
    SharedKv,
    VectorSharedKv,
    VectorSharedKvCombined,
    Mma {
        variant: Variant,
    },
    /// Validated SM120 plain-FP16 target prefill; never selected by the draft path.
    MmaF16,
    Tiled {
        query_tile: usize,
    },
}
impl Choice {
    pub(crate) const fn query_tile(self) -> Option<usize> {
        match self {
            Self::Simt
            | Self::RustSimt
            | Self::SharedKv
            | Self::VectorSharedKv
            | Self::VectorSharedKvCombined
            | Self::Mma { .. }
            | Self::MmaF16 => None,
            Self::Tiled { query_tile } => Some(query_tile),
        }
    }
    pub(crate) const fn uses_oxide(self) -> bool {
        matches!(
            self,
            Self::RustSimt | Self::SharedKv | Self::VectorSharedKv | Self::VectorSharedKvCombined
        )
    }
    fn valid(self) -> bool {
        matches!(self, Self::Simt) || self.query_tile().is_some_and(|q| QUERY_TILES.contains(&q))
    }
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Key {
    pub hardware: String,
    pub driver_api: u32,
    pub driver_build: Option<String>,
    pub fingerprint: String,
    pub schema: String,
    pub shape: Shape,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Times {
    pub baseline_ms: Vec<f32>,
    pub candidate_ms: Vec<f32>,
}
impl Times {
    fn valid(&self) -> bool {
        (MIN_REPS..=MAX_REPS).contains(&self.baseline_ms.len())
            && self.baseline_ms.len() == self.candidate_ms.len()
            && self
                .baseline_ms
                .iter()
                .chain(&self.candidate_ms)
                .all(|x| x.is_finite() && *x > 0.0)
    }
    fn speedup(&self) -> f32 {
        median(&self.baseline_ms) / median(&self.candidate_ms)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    pub choice: Choice,
    pub candidate_first: bool,
    pub max_f64_error: f64,
    pub cold: Times,
    pub warm: Times,
}
impl Trial {
    fn valid(&self) -> bool {
        self.choice.valid()
            && self.cold.valid()
            && self.warm.valid()
            && self.max_f64_error.is_finite()
            && (0.0..MAX_ERROR).contains(&self.max_f64_error)
    }
    fn confirmed(&self) -> bool {
        self.cold.speedup() >= MIN_CONFIRM_SPEEDUP && self.warm.speedup() >= MIN_CONFIRM_SPEEDUP
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub key: Key,
    pub selected: Choice,
    pub search: Vec<Trial>,
    pub confirmation: Trial,
    pub fixtures: usize,
}
impl Entry {
    fn valid(&self) -> bool {
        self.fixtures == reference::FIXTURES
            && self.key.shape.supported()
            && self.key.schema == SCHEMA
            && !self.key.hardware.is_empty()
            && !self.key.fingerprint.is_empty()
            && self.key.driver_build.as_ref().is_none_or(|s| !s.is_empty())
            && self.search.len() == QUERY_TILES.len()
            && self.search.iter().zip(QUERY_TILES).all(|(trial, q)| {
                trial.valid()
                    && !trial.candidate_first
                    && trial.choice == Choice::Tiled { query_tile: q }
            })
            && self.confirmation.valid()
            && self.confirmation.candidate_first
            && self.confirmation.choice == self.selected
            && (self.selected == Choice::Simt
                || self.confirmation.confirmed()
                    && self
                        .search
                        .iter()
                        .any(|t| t.choice == self.selected && t.confirmed()))
    }
}
type Memo = Mutex<BTreeMap<Key, Entry>>;
fn memo() -> &'static Memo {
    static MEMO: OnceLock<Memo> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(BTreeMap::new()))
}
/// The full task-quality gate rejected tiled arithmetic. Internal tests may compare it;
/// production has no user-selectable path into an unqualified candidate.
#[cfg(not(test))]
pub(crate) const fn enabled() -> bool {
    false
}
#[cfg(test)]
pub(crate) fn enabled() -> bool {
    var("INFER_CUDA_TILED_ATTENTION").is_ok_and(|v| v == "1" || v == "true")
}
pub(crate) fn select(
    device: &CudaDevice,
    shape: Shape,
    autotune: bool,
    session: &ResidentTuningSession,
) -> Choice {
    if !shape.supported() {
        return Choice::Simt;
    }
    match resolve(device, shape, autotune, session) {
        Ok(choice) => choice,
        Err(error) => {
            tracing::warn!(target:"infer::tuning",?shape,%error,"prefill attention search failed: SIMT fallback");
            Choice::Simt
        }
    }
}
fn key(device: &CudaDevice, shape: Shape) -> Result<Key> {
    let driver_build = driver_build();
    Ok(Key {
        hardware: serde_json::to_string(device.profile()?).map_err(device_error)?,
        driver_api: CudaDevice::driver_version()?,
        driver_build,
        fingerprint: capture::fingerprint(device, shape)?,
        schema: SCHEMA.into(),
        shape,
    })
}
fn resolve(
    device: &CudaDevice,
    shape: Shape,
    autotune: bool,
    session: &ResidentTuningSession,
) -> Result<Choice> {
    let key = key(device, shape)?;
    let cached = memo()
        .lock()
        .map_err(|_| Error::invariant("attention memo poisoned"))?
        .get(&key)
        .cloned();
    if let Some(entry) = cached {
        return Ok(entry.selected);
    }
    let path = super::table_path();
    let table = if key.driver_build.is_some() {
        super::resolve_table(path.as_deref())?
    } else {
        None
    };
    if let Some(entry) = table
        .as_ref()
        .and_then(|t| t.resident_attention.iter().find(|e| e.key == key))
    {
        if !entry.valid() {
            return Err(Error::invalid(
                "invalid attention numerical/paired evidence",
            ));
        }
        memo()
            .lock()
            .map_err(|_| Error::invariant("attention memo poisoned"))?
            .insert(key, entry.clone());
        tracing::info!(target:"infer::tuning",?shape,?entry.selected,source="table-cache","prefill attention selected");
        return Ok(entry.selected);
    }
    if !autotune || !session.permits_attention_search() {
        return Ok(Choice::Simt);
    }
    let Some(entry) = tune(device, key.clone(), session)? else {
        return Ok(Choice::Simt);
    };
    let cached = key.driver_build.is_some()
        && super::update_table(path.as_deref(), |table| {
            table.resident_attention.retain(|e| e.key != key);
            table.resident_attention.push(entry.clone());
        });
    tracing::info!(target:"infer::tuning",?shape,?entry.selected,cached,
        cold_speedup=entry.confirmation.cold.speedup(),warm_speedup=entry.confirmation.warm.speedup(),
        max_f64_error=entry.confirmation.max_f64_error,"prefill attention hardware choice");
    let choice = entry.selected;
    memo()
        .lock()
        .map_err(|_| Error::invariant("attention memo poisoned"))?
        .insert(key, entry);
    Ok(choice)
}
fn measure(
    device: &CudaDevice,
    base: &Captured,
    candidate: &Captured,
    cold: bool,
    reverse: bool,
) -> Result<Times> {
    let options = BenchOptions {
        warmup: Duration::from_millis(WARMUP_MS),
        rep: Duration::from_millis(REP_MS),
        min_reps: MIN_REPS,
        max_reps: MAX_REPS,
        clear_l2: cold,
    };
    let (left, right) = if reverse {
        (candidate, base)
    } else {
        (base, candidate)
    };
    let run = |c: &Captured| {
        c.graph
            .launch()
            .sync_on(&device.stream)
            .map_err(|e| tensor_error(&e.to_string()))
    };
    let (a, b) = do_bench_paired(&device.stream, &options, |_| run(left), |_| run(right))
        .map_err(device_error)?;
    let times = if reverse {
        Times {
            baseline_ms: b.times_ms().to_vec(),
            candidate_ms: a.times_ms().to_vec(),
        }
    } else {
        Times {
            baseline_ms: a.times_ms().to_vec(),
            candidate_ms: b.times_ms().to_vec(),
        }
    };
    if !times.valid() {
        return Err(Error::invalid("attention paired timing samples"));
    }
    Ok(times)
}
fn trial(
    device: &CudaDevice,
    base: &Captured,
    fixtures: &[Fixture],
    choice: Choice,
    reverse: bool,
) -> Result<Trial> {
    let candidate = Captured::new(device, &fixtures[0], choice)?;
    let mut max_error = candidate.verify(device, &fixtures[0])?;
    for fixture in &fixtures[1..] {
        let check = Captured::new(device, fixture, choice)?;
        max_error = max_error.max(check.verify(device, fixture)?);
    }
    let cold = measure(device, base, &candidate, true, reverse)?;
    let warm = measure(device, base, &candidate, false, reverse)?;
    max_error = max_error.max(candidate.verify(device, &fixtures[0])?);
    Ok(Trial {
        choice,
        candidate_first: reverse,
        max_f64_error: max_error,
        cold,
        warm,
    })
}
fn tune(device: &CudaDevice, key: Key, session: &ResidentTuningSession) -> Result<Option<Entry>> {
    let options = CompileOptions::default();
    if cfg!(debug_assertions)
        || options.device_debug
        || options.opt_level.is_some_and(|l| l != CUDA_OPT_LEVEL)
    {
        return Err(Error::invalid(
            "attention tuning requires release and CUDA O3",
        ));
    }
    let fixtures = (0..reference::FIXTURES)
        .map(|index| Fixture::new(key.shape, index))
        .collect::<Result<Vec<_>>>()?;
    let base = Captured::new(device, &fixtures[0], Choice::Simt)?;
    for fixture in &fixtures {
        Captured::new(device, fixture, Choice::Simt)?.verify(device, fixture)?;
    }
    let mut search = Vec::new();
    let mut selected = Choice::Simt;
    let mut best = 1.0;
    for query_tile in QUERY_TILES {
        if !session.permits_attention_search() {
            return Ok(None);
        }
        let choice = Choice::Tiled { query_tile };
        let measured = trial(device, &base, &fixtures, choice, false)?;
        let speedup = measured.cold.speedup().min(measured.warm.speedup());
        if measured.confirmed() && speedup > best {
            selected = choice;
            best = speedup;
        }
        search.push(measured);
    }
    if !session.permits_attention_search() {
        return Ok(None);
    }
    let mut confirmation = trial(device, &base, &fixtures, selected, true)?;
    if selected != Choice::Simt && !confirmation.confirmed() {
        if !session.permits_attention_search() {
            return Ok(None);
        }
        selected = Choice::Simt;
        confirmation = trial(device, &base, &fixtures, selected, true)?;
    }
    let entry = Entry {
        key,
        selected,
        search,
        confirmation,
        fixtures: reference::FIXTURES,
    };
    if !entry.valid() {
        return Err(Error::invariant("incomplete attention tuning evidence"));
    }
    Ok(Some(entry))
}
#[cfg(test)]
#[path = "../../tests/unit/tuning_attention.rs"]
mod tests;
