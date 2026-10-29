//! F32-state WY column partition search; no device name controls a kernel or tile.
mod capture;
pub mod exact;
mod reference;
use super::{
    MAX_REPS, MIN_CONFIRM_SPEEDUP, MIN_REPS, REP_MS, WARMUP_MS, median,
    resident_gemm::ResidentTuningSession,
};
use crate::device::platform::driver_build;
use crate::device::{CudaDevice, device_error};
use capture::{Captured, fingerprint};
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    prelude::*,
};
use infer_core::{Error, Result};
use reference::Fixture;
use serde::{Deserialize, Serialize};
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    pub value_block: usize,
    pub candidate_first: bool,
    pub max_f64_error: f64,
    pub cold: Times,
    pub warm: Times,
}
impl Trial {
    fn valid(&self, dim: usize) -> bool {
        [dim, dim / 2].contains(&self.value_block)
            && self.cold.valid()
            && self.warm.valid()
            && self.max_f64_error.is_finite()
            && (0.0..MAX_F64_ERROR).contains(&self.max_f64_error)
    }
    fn confirmed(&self) -> bool {
        self.cold.speedup() >= MIN_CONFIRM_SPEEDUP && self.warm.speedup() >= MIN_CONFIRM_SPEEDUP
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub key: Key,
    pub selected: usize,
    pub search: Vec<Trial>,
    pub confirmation: Trial,
}
impl Entry {
    fn valid(&self) -> bool {
        supported(self.key.shape)
            && self.key.schema == SCHEMA
            && !self.key.kernel_fingerprint.is_empty()
            && self.key.driver_build.as_ref().is_none_or(|x| !x.is_empty())
            && !self.key.hardware.is_empty()
            && self.search.len() == 1
            && self
                .search
                .first()
                .is_some_and(|trial| trial.value_block == self.key.shape[2] / 2)
            && self.confirmation.valid(self.key.shape[2])
            && self.confirmation.candidate_first
            && self.confirmation.value_block == self.selected
            && self
                .search
                .iter()
                .all(|x| !x.candidate_first && x.valid(self.key.shape[2]))
            && (self.selected == self.key.shape[2]
                || self.confirmation.confirmed()
                    && self
                        .search
                        .iter()
                        .any(|x| x.value_block == self.selected && x.confirmed()))
    }
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
        .and_then(|t| t.resident_recurrent.iter().find(|x| x.key == key))
    {
        if !entry.valid() {
            return Err(Error::invalid("invalid WY paired/numerical cache evidence"));
        }
        memo()
            .lock()
            .map_err(|_| Error::invariant("WY tuning cache poisoned"))?
            .insert(key, entry.clone());
        tracing::info!(target:"infer::tuning",?shape,value_block=entry.selected,source="table-cache","WY selected");
        return Ok(entry.selected);
    }
    if !autotune || !session.permits_recurrent_search() {
        return Ok(shape[2]);
    }
    let Some(entry) = tune(device, key.clone(), session)? else {
        return Ok(shape[2]);
    };
    let cached = key.driver_build.is_some()
        && super::update_table(path.as_deref(), |t| {
            t.resident_recurrent.retain(|x| x.key != key);
            t.resident_recurrent.push(entry.clone());
        });
    tracing::info!(target:"infer::tuning",?shape,value_block=entry.selected,cached,
        cold_speedup=entry.confirmation.cold.speedup(),warm_speedup=entry.confirmation.warm.speedup(),
        max_f64_error=entry.confirmation.max_f64_error,"WY hardware partition selected");
    memo()
        .lock()
        .map_err(|_| Error::invariant("WY tuning cache poisoned"))?
        .insert(key, entry.clone());
    Ok(entry.selected)
}
fn measure(
    device: &CudaDevice,
    base: &Captured,
    candidate: &Captured,
    cold: bool,
    candidate_first: bool,
) -> Result<Times> {
    let options = BenchOptions {
        warmup: Duration::from_millis(WARMUP_MS),
        rep: Duration::from_millis(REP_MS),
        min_reps: MIN_REPS,
        max_reps: MAX_REPS,
        clear_l2: cold,
    };
    let (left, right) = if candidate_first {
        (candidate, base)
    } else {
        (base, candidate)
    };
    let run = |x: &Captured| {
        x.graph
            .launch()
            .sync_on(&device.stream)
            .map_err(|e| tensor_error(&e.to_string()))
    };
    let (a, b) = do_bench_paired(&device.stream, &options, |_| run(left), |_| run(right))
        .map_err(device_error)?;
    let times = if candidate_first {
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
        return Err(Error::invalid("WY paired timing sample evidence"));
    }
    Ok(times)
}
fn trial(
    device: &CudaDevice,
    base: &Captured,
    fixture: &Fixture,
    correlated: &Fixture,
    block: usize,
    candidate_first: bool,
) -> Result<Trial> {
    let mut candidate = Captured::new(device, fixture, block)?;
    let mut max_f64_error = candidate.verify(device, fixture)?;
    let mut adversarial = Captured::new(device, correlated, block)?;
    max_f64_error = max_f64_error.max(adversarial.verify(device, correlated)?);
    drop(adversarial);
    let cold = measure(device, base, &candidate, true, candidate_first)?;
    let warm = measure(device, base, &candidate, false, candidate_first)?;
    max_f64_error = max_f64_error.max(candidate.verify(device, fixture)?);
    Ok(Trial {
        value_block: block,
        candidate_first,
        max_f64_error,
        cold,
        warm,
    })
}
fn tune(device: &CudaDevice, key: Key, session: &ResidentTuningSession) -> Result<Option<Entry>> {
    let compile = CompileOptions::default();
    if cfg!(debug_assertions)
        || compile.device_debug
        || compile.opt_level.is_some_and(|x| x != CUDA_OPT_LEVEL)
    {
        return Err(Error::invalid("WY tuning requires release and CUDA O3"));
    }
    let fixture = Fixture::new(key.shape, false)?;
    let correlated = Fixture::new(key.shape, true)?;
    let baseline = key.shape[2];
    let mut base = Captured::new(device, &fixture, baseline)?;
    base.verify(device, &fixture)?;
    let mut selected = baseline;
    let mut search = Vec::new();
    let block = baseline / 2;
    if !session.permits_recurrent_search() {
        return Ok(None);
    }
    let measured = trial(device, &base, &fixture, &correlated, block, false)?;
    if measured.confirmed() {
        selected = block;
    }
    search.push(measured);
    if !session.permits_recurrent_search() {
        return Ok(None);
    }
    let mut confirmation = trial(device, &base, &fixture, &correlated, selected, true)?;
    if selected != baseline && !confirmation.confirmed() {
        if !session.permits_recurrent_search() {
            return Ok(None);
        }
        selected = baseline;
        confirmation = trial(device, &base, &fixture, &correlated, baseline, true)?;
    }
    let entry = Entry {
        key,
        selected,
        search,
        confirmation,
    };
    if !entry.valid() {
        return Err(Error::invariant("incomplete WY tuning entry"));
    }
    Ok(Some(entry))
}

#[cfg(test)]
#[path = "../../tests/unit/tuning_recurrent.rs"]
mod tests;
