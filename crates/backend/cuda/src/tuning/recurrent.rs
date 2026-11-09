//! F32-state WY column partition search; no device name controls a kernel or tile.
mod capture;
pub mod exact;
mod reference;
use super::resident_gemm::{MeasurementOrder, PairedTimes, ResidentTuningSession, paired_times};
use super::search::{self, Evidence, FormerTrial, Protocol, Record};
use super::{MAX_REPS, MIN_REPS, REP_MS, WARMUP_MS, median};
use crate::device::{CudaDevice, device_error};
use crate::tuning::preparation::driver_build;
use capture::{Captured, fingerprint};
use infer_core::{Error, Result};
use infer_cuda_kernels::jit::JitOptions;
use infer_cuda_kernels::runtime::Timing;
use reference::Fixture;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::cell::Cell;
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
    time::Duration,
};

// Quarter columns pass synthetic F64 but fail the predeclared full-model quality gate.
// Keep them diagnostic-only until their full-model quality is independently qualified.
const SCHEMA: &str = "wy32:recursive-unit-triangle:bf16-components3-products6:f32-state:f64-abs2e-5:regular+weak-correlated-v1:columns-full-half-v2";
const SHAPE_RANK: usize = 4;
const SUPPORTED_DIMS: [usize; 2] = [32, 128];
const CHUNK_ROWS: usize = 32;
const MAX_LANES: usize = 256;
const MAX_F64_ERROR: f64 = 2e-5;
const CUDA_OPT_LEVEL: u8 = 3;
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Key {
    pub hardware: String,
    pub driver_version: u32,
    pub driver_build: Option<String>,
    pub kernel_fingerprint: String,
    pub schema: String,
    pub shape: [usize; SHAPE_RANK],
}
/// Unpaired samples of the exact-recurrence qualification, which keeps its own record shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Times {
    pub baseline_ms: Vec<f32>,
    pub candidate_ms: Vec<f32>,
}
impl Times {
    fn valid(&self) -> bool {
        self.baseline_ms.len() >= MIN_REPS
            && self.baseline_ms.len() <= MAX_REPS
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
/// The family's records in the shared table section.
pub const FAMILY: &str = "recurrent";
/// A value block measured against the full-width columns.
pub type Trial = search::Trial<usize>;
pub type Entry = Record<Key, usize>;
/// Half-width columns against full-width ones; quarter columns stay diagnostic-only.
fn protocol(dim: usize) -> Protocol<usize> {
    Protocol {
        baseline: dim,
        candidates: vec![dim / 2],
    }
}
impl Entry {
    fn valid(&self) -> bool {
        let dim = self.key.shape[2];
        supported(self.key.shape)
            && self.key.schema == SCHEMA
            && !self.key.kernel_fingerprint.is_empty()
            && self.key.driver_build.as_ref().is_none_or(|x| !x.is_empty())
            && !self.key.hardware.is_empty()
            && self.evidence.sampled(MIN_REPS)
            && self.evidence.within(MAX_F64_ERROR)
            && self
                .evidence
                .valid(&protocol(dim), |block| [dim, dim / 2].contains(&block))
    }
}

#[derive(Deserialize)]
struct LegacyTrial {
    value_block: usize,
    max_f64_error: f64,
    #[serde(flatten)]
    trial: FormerTrial,
}
#[derive(Deserialize)]
struct LegacyEntry {
    key: Key,
    selected: usize,
    search: Vec<LegacyTrial>,
    confirmation: LegacyTrial,
}

/// A record from the family's former table section in the shared form; one that does not
/// convert stays as stored. A failed trial ended the former search, so a stored one is
/// complete; its deviation is the largest of its trials'.
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
    let trial = |legacy: LegacyTrial| legacy.trial.into_trial(legacy.value_block);
    let entry = Entry {
        key: legacy.key,
        evidence: Evidence {
            selected: legacy.selected,
            exact_candidates_verified: true,
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
fn supported([kh, vh, dim, lanes]: [usize; SHAPE_RANK]) -> bool {
    kh > 0
        && vh > 0
        && vh.is_multiple_of(kh)
        && SUPPORTED_DIMS.contains(&dim)
        && lanes > 0
        && lanes <= MAX_LANES
        && lanes.is_multiple_of(CHUNK_ROWS)
}
type Memo = Mutex<BTreeMap<Key, Entry>>;
fn memo() -> &'static Memo {
    static M: OnceLock<Memo> = OnceLock::new();
    M.get_or_init(|| Mutex::new(BTreeMap::new()))
}
pub(crate) fn select(
    device: &CudaDevice,
    shape: [usize; SHAPE_RANK],
    autotune: bool,
    session: &ResidentTuningSession,
) -> Option<usize> {
    if !supported(shape) {
        return None;
    }
    match resolve(device, shape, autotune, session) {
        Ok(block) => Some(block),
        Err(error) => {
            tracing::warn!(target:"infer::tuning",?shape,%error,"WY tuning failed: per-token fallback");
            None
        }
    }
}
fn resolve(
    device: &CudaDevice,
    shape: [usize; SHAPE_RANK],
    autotune: bool,
    session: &ResidentTuningSession,
) -> Result<usize> {
    let driver_build = driver_build();
    let key = Key {
        hardware: serde_json::to_string(device.profile()?).map_err(device_error)?,
        driver_version: CudaDevice::driver_version()?,
        driver_build,
        kernel_fingerprint: fingerprint(device, shape)?,
        schema: SCHEMA.into(),
        shape,
    };
    let cached = memo()
        .lock()
        .map_err(|_| Error::invariant("WY tuning cache poisoned"))?
        .get(&key)
        .filter(|x| x.valid())
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
            return Err(Error::invalid("invalid WY paired/numerical cache evidence"));
        }
        let selected = entry.evidence.selected;
        memo()
            .lock()
            .map_err(|_| Error::invariant("WY tuning cache poisoned"))?
            .insert(key, entry);
        tracing::info!(target:"infer::tuning",?shape,value_block=selected,source="table-cache","WY selected");
        return Ok(selected);
    }
    if !autotune || !session.permits_recurrent_search() {
        return Ok(shape[2]);
    }
    let Some(entry) = tune(device, key.clone(), session)? else {
        return Ok(shape[2]);
    };
    let cached = key.driver_build.is_some() && search::store(path.as_deref(), FAMILY, &entry);
    let evidence = &entry.evidence;
    tracing::info!(target:"infer::tuning",?shape,value_block=evidence.selected,cached,
        cold_speedup=evidence.confirmation.cold.speedup,warm_speedup=evidence.confirmation.warm.speedup,
        max_f64_error=evidence.max_error,"WY hardware partition selected");
    let selected = evidence.selected;
    memo()
        .lock()
        .map_err(|_| Error::invariant("WY tuning cache poisoned"))?
        .insert(key, entry);
    Ok(selected)
}
fn measure(
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
    let candidate_first = order == MeasurementOrder::CandidateFirst;
    let (left, right) = if candidate_first {
        (candidate, base)
    } else {
        (base, candidate)
    };
    let run = |x: &Captured| device.tile().sync(x.graph.launch());
    let (a, b) = device
        .tile()
        .time_paired(&options, || run(left), || run(right))
        .map_err(device_error)?;
    let (baseline, candidate) = if candidate_first { (b, a) } else { (a, b) };
    paired_times(baseline.times_ms(), candidate.times_ms())
        .filter(|times| times.baseline_ms.len() >= MIN_REPS)
        .ok_or_else(|| Error::invalid("WY paired timing sample evidence"))
}
/// Measure `block` in `order`, failing when its deviation from the F64 reference on either
/// fixture, before or after timing, reaches the family's tolerance; `error` keeps the largest
/// accepted deviation.
fn trial(
    device: &CudaDevice,
    (base, fixture, correlated): (&Captured, &Fixture, &Fixture),
    block: usize,
    order: MeasurementOrder,
    error: &Cell<f64>,
) -> Result<Trial> {
    let candidate = Captured::new(device, fixture, block)?;
    let mut max_f64_error = candidate.verify(device, fixture)?;
    let adversarial = Captured::new(device, correlated, block)?;
    max_f64_error = max_f64_error.max(adversarial.verify(device, correlated)?);
    drop(adversarial);
    let cold = measure(device, base, &candidate, true, order)?;
    let warm = measure(device, base, &candidate, false, order)?;
    max_f64_error = max_f64_error.max(candidate.verify(device, fixture)?);
    if !(max_f64_error.is_finite() && (0.0..MAX_F64_ERROR).contains(&max_f64_error)) {
        return Err(Error::invalid(format!(
            "WY F64 deviation {max_f64_error} beyond {MAX_F64_ERROR}"
        )));
    }
    error.set(error.get().max(max_f64_error));
    Ok(Trial {
        choice: block,
        order,
        cold,
        warm,
    })
}
fn tune(device: &CudaDevice, key: Key, session: &ResidentTuningSession) -> Result<Option<Entry>> {
    let compile = JitOptions::default();
    if cfg!(debug_assertions)
        || compile.device_debug()
        || compile.opt_level().is_some_and(|x| x != CUDA_OPT_LEVEL)
    {
        return Err(Error::invalid("WY tuning requires release and CUDA O3"));
    }
    let fixture = Fixture::new(key.shape, false)?;
    let correlated = Fixture::new(key.shape, true)?;
    let baseline = key.shape[2];
    let base = Captured::new(device, &fixture, baseline)?;
    base.verify(device, &fixture)?;
    let error = Cell::new(0.0);
    let shape = key.shape;
    let Some(mut evidence) = search::run(
        &protocol(baseline),
        || session.permits_recurrent_search(),
        |block, order| trial(device, (&base, &fixture, &correlated), block, order, &error),
        |block, error| {
            tracing::warn!(target:"infer::tuning",?shape,block,%error,"WY candidate rejected");
            Ok(())
        },
    )?
    else {
        return Ok(None);
    };
    evidence.max_error = Some(error.get());
    let entry = Entry { key, evidence };
    if !entry.valid() {
        return Err(Error::invariant("incomplete WY tuning entry"));
    }
    Ok(Some(entry))
}

#[cfg(test)]
#[path = "../../tests/unit/tuning_recurrent.rs"]
mod tests;
