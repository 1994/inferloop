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
use super::qualify::Times;
use super::resident_gemm::{MeasurementOrder, PairedTimes, ResidentTuningSession, paired_times};
use super::search::{self, Evidence, FormerTrial, Protocol, Record};
use super::{MAX_REPS, MIN_REPS, REP_MS, WARMUP_MS};
use crate::constants::PREFILL_LANES;
use crate::device::{CudaDevice, device_error};
use crate::resident::attention_mma::catalog::Variant;
use crate::tuning::preparation::driver_build;
use capture::Captured;
use infer_core::{Error, Result};
use infer_cuda_kernels::jit::JitOptions;
use infer_cuda_kernels::runtime::Timing;
use reference::Fixture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cell::Cell;
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
/// The family's records in the shared table section.
pub const FAMILY: &str = "attention";
/// A query tile measured against SIMT prefill.
pub type Trial = search::Trial<Choice>;
pub type Entry = Record<Key, Choice>;
/// Every query tile against SIMT prefill.
fn protocol() -> Protocol<Choice> {
    Protocol {
        baseline: Choice::Simt,
        candidates: QUERY_TILES
            .map(|query_tile| Choice::Tiled { query_tile })
            .to_vec(),
    }
}
impl Entry {
    fn valid(&self) -> bool {
        self.key.shape.supported()
            && self.key.schema == SCHEMA
            && !self.key.hardware.is_empty()
            && !self.key.fingerprint.is_empty()
            && self.key.driver_build.as_ref().is_none_or(|s| !s.is_empty())
            && self.evidence.sampled(MIN_REPS)
            && self.evidence.within(MAX_ERROR)
            && self.evidence.valid(&protocol(), Choice::valid)
    }
}

#[derive(Deserialize)]
struct LegacyTrial {
    choice: Choice,
    max_f64_error: f64,
    #[serde(flatten)]
    trial: FormerTrial,
}
#[derive(Deserialize)]
struct LegacyEntry {
    key: Key,
    selected: Choice,
    search: Vec<LegacyTrial>,
    confirmation: LegacyTrial,
    fixtures: usize,
}

/// A record from the family's former table section in the shared form; one that does not
/// convert stays as stored. A failed trial ended the former search, so a stored one is
/// complete; it verified every candidate when it covered every fixture, and its deviation is
/// the largest of its trials'.
pub(super) fn legacy(stored: Value) -> Value {
    convert(&stored).unwrap_or(stored)
}

fn convert(stored: &Value) -> Option<Value> {
    let legacy = LegacyEntry::deserialize(stored).ok()?;
    let max_error = legacy
        .search
        .iter()
        .chain([&legacy.confirmation])
        .map(|trial| trial.max_f64_error)
        .reduce(f64::max);
    let trial = |legacy: LegacyTrial| legacy.trial.into_trial(legacy.choice);
    let entry = Entry {
        key: legacy.key,
        evidence: Evidence {
            selected: legacy.selected,
            exact_candidates_verified: legacy.fixtures == reference::FIXTURES,
            trials: legacy
                .search
                .into_iter()
                .map(trial)
                .collect::<Option<_>>()?,
            rejected: Some(Vec::new()),
            confirmation: trial(legacy.confirmation)?,
            max_error,
        },
    };
    serde_json::to_value(entry).ok()
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
        .map(|entry| entry.evidence.selected);
    if let Some(selected) = cached {
        return Ok(selected);
    }
    let path = super::table_path();
    let table = if key.driver_build.is_some() {
        super::resolve_table(path.as_deref())?
    } else {
        None
    };
    if let Some(entry) = table
        .as_ref()
        .and_then(|table| search::lookup(table, FAMILY, &key, |_| true))
    {
        if !entry.valid() {
            return Err(Error::invalid(
                "invalid attention numerical/paired evidence",
            ));
        }
        let selected = entry.evidence.selected;
        memo()
            .lock()
            .map_err(|_| Error::invariant("attention memo poisoned"))?
            .insert(key, entry);
        tracing::info!(target:"infer::tuning",?shape,?selected,source="table-cache","prefill attention selected");
        return Ok(selected);
    }
    if !autotune || !session.permits_attention_search() {
        return Ok(Choice::Simt);
    }
    let Some(entry) = tune(device, key.clone(), session)? else {
        return Ok(Choice::Simt);
    };
    let cached = key.driver_build.is_some() && search::store(path.as_deref(), FAMILY, &entry);
    let evidence = &entry.evidence;
    tracing::info!(target:"infer::tuning",?shape,selected=?evidence.selected,cached,
        cold_speedup=evidence.confirmation.cold.speedup,warm_speedup=evidence.confirmation.warm.speedup,
        max_f64_error=evidence.max_error,"prefill attention hardware choice");
    let choice = evidence.selected;
    memo()
        .lock()
        .map_err(|_| Error::invariant("attention memo poisoned"))?
        .insert(key, entry);
    Ok(choice)
}
/// Unpaired samples for the qualification families, which keep their own record shapes.
fn measure(
    device: &CudaDevice,
    base: &Captured,
    candidate: &Captured,
    cold: bool,
    reverse: bool,
) -> Result<Times> {
    let order = if reverse {
        MeasurementOrder::CandidateFirst
    } else {
        MeasurementOrder::BaselineFirst
    };
    let times = paired(device, base, candidate, cold, order)?;
    let times = Times {
        baseline_ms: times.baseline_ms,
        candidate_ms: times.candidate_ms,
    };
    if !times.valid() {
        return Err(Error::invalid("attention paired timing samples"));
    }
    Ok(times)
}
fn paired(
    device: &CudaDevice,
    base: &Captured,
    candidate: &Captured,
    cold: bool,
    order: MeasurementOrder,
) -> Result<PairedTimes> {
    let options = Timing {
        warmup: Duration::from_millis(WARMUP_MS),
        rep: Duration::from_millis(REP_MS),
        min_reps: MIN_REPS,
        max_reps: MAX_REPS,
        clear_l2: cold,
    };
    let reverse = order == MeasurementOrder::CandidateFirst;
    let (left, right) = if reverse {
        (candidate, base)
    } else {
        (base, candidate)
    };
    let run = |c: &Captured| device.tile().sync(c.graph.launch());
    let (a, b) = device
        .tile()
        .time_paired(&options, || run(left), || run(right))
        .map_err(device_error)?;
    let (baseline, candidate) = if reverse { (b, a) } else { (a, b) };
    paired_times(baseline.times_ms(), candidate.times_ms())
        .filter(|times| times.baseline_ms.len() >= MIN_REPS)
        .ok_or_else(|| Error::invalid("attention paired timing samples"))
}
/// Measure `choice` in `order`, failing when its deviation from the F64 reference on any
/// fixture, before or after timing, reaches the family's tolerance; `error` keeps the largest
/// accepted deviation.
fn trial(
    device: &CudaDevice,
    (base, fixtures): (&Captured, &[Fixture]),
    choice: Choice,
    order: MeasurementOrder,
    error: &Cell<f64>,
) -> Result<Trial> {
    let candidate = Captured::new(device, &fixtures[0], choice)?;
    let mut max_error = candidate.verify(device, &fixtures[0])?;
    for fixture in &fixtures[1..] {
        let check = Captured::new(device, fixture, choice)?;
        max_error = max_error.max(check.verify(device, fixture)?);
    }
    let cold = paired(device, base, &candidate, true, order)?;
    let warm = paired(device, base, &candidate, false, order)?;
    max_error = max_error.max(candidate.verify(device, &fixtures[0])?);
    if !(max_error.is_finite() && (0.0..MAX_ERROR).contains(&max_error)) {
        return Err(Error::invalid(format!(
            "attention F64 deviation {max_error} beyond {MAX_ERROR}"
        )));
    }
    error.set(error.get().max(max_error));
    Ok(Trial {
        choice,
        order,
        cold,
        warm,
    })
}
fn tune(device: &CudaDevice, key: Key, session: &ResidentTuningSession) -> Result<Option<Entry>> {
    let options = JitOptions::default();
    if cfg!(debug_assertions)
        || options.device_debug()
        || options.opt_level().is_some_and(|l| l != CUDA_OPT_LEVEL)
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
    let error = Cell::new(0.0);
    let shape = key.shape;
    let Some(mut evidence) = search::run(
        &protocol(),
        || session.permits_attention_search(),
        |choice, order| trial(device, (&base, &fixtures), choice, order, &error),
        |choice, error| {
            tracing::warn!(target:"infer::tuning",?shape,?choice,%error,"prefill attention candidate rejected");
            Ok(())
        },
    )?
    else {
        return Ok(None);
    };
    evidence.max_error = Some(error.get());
    let entry = Entry { key, evidence };
    if !entry.valid() {
        return Err(Error::invariant("incomplete attention tuning evidence"));
    }
    Ok(Some(entry))
}
#[cfg(test)]
#[path = "../../tests/unit/tuning_attention.rs"]
mod tests;
