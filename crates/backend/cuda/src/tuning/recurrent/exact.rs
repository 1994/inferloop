//! Exact Rust SIMT prepared recurrence selected by measured hardware/shape evidence.
mod capture;
mod identity;
use super::{CUDA_OPT_LEVEL, Key, SHAPE_RANK, Times, reference::Fixture};
use crate::tuning::MAX_REPS;
use crate::tuning::MIN_CONFIRM_SPEEDUP;
use crate::tuning::MIN_REPS;
use crate::tuning::REP_MS;
use crate::tuning::WARMUP_MS;
use crate::tuning::preparation::driver_build;
use crate::tuning::resolve_table;
use crate::tuning::shipped_table;
use crate::tuning::table_path;
use crate::tuning::update_table;
use crate::{
    device::{CudaDevice, device_error},
    resident::recurrent_prefill::oxide,
    tuning::resident_gemm::ResidentTuningSession,
};
use capture::Captured;
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use cutile::{
    bench::{BenchOptions, do_bench_paired},
    prelude::*,
};
use infer_core::{Error, Result};
use infer_cuda_kernels::tile::runtime::Graph;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
    time::Duration,
};
const SCHEMA: &str = "rust-simt-sm120:gdn-f32-16-48-128:canonical-pred-readout:no-fma:original-prepare-transpose:bits-f64-abs2e-5:fixtures9:counts51-tail-full:cold-warm-ab-ba-v1";
const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 48;
const DIM: usize = 128;
const GEOMETRY_RANK: usize = 3;
const SUPPORTED_LANES: [usize; 5] = [64, 256, 512, 1024, 2048];
const SHA256_HEX_LEN: usize = 64;
const SHORT_COUNT: i32 = 51;
const OFFSET_COUNT: i32 = 52;
const OFFSET_BASE: i32 = 17;
const NEGATIVE_OFFSET: i32 = -18;
const LATE_BASE: i32 = 128;
const FIXTURES: usize = 9;
const TIMED: [usize; 3] = [2, 3, 7];
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case {
    pub fixture: usize,
    pub cold: Times,
    pub warm: Times,
}
impl Case {
    fn valid(&self) -> bool {
        self.cold.valid() && self.warm.valid()
    }
    fn confirmed(&self) -> bool {
        self.cold.speedup() >= MIN_CONFIRM_SPEEDUP && self.warm.speedup() >= MIN_CONFIRM_SPEEDUP
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub key: Key,
    pub rust_simt: bool,
    pub verified_fixtures: usize,
    pub verified_elements: usize,
    pub max_f64_error: f64,
    pub forward: Vec<Case>,
    pub reverse: Vec<Case>,
}
impl Entry {
    fn valid(&self) -> bool {
        supported(self.key.shape)
            && self.key.schema == SCHEMA
            && !self.key.hardware.is_empty()
            && self.key.kernel_fingerprint.len() == SHA256_HEX_LEN
            && self
                .key
                .kernel_fingerprint
                .bytes()
                .all(|c| c.is_ascii_hexdigit())
            && self.key.driver_build.as_ref().is_none_or(|s| !s.is_empty())
            && self.verified_fixtures == FIXTURES
            && self.verified_elements
                == FIXTURES * (VALUE_HEADS * DIM * DIM + self.key.shape[3] * VALUE_HEADS * DIM)
            && self.max_f64_error.is_finite()
            && (0.0..super::MAX_F64_ERROR).contains(&self.max_f64_error)
            && [&self.forward, &self.reverse].iter().all(|cases| {
                cases.len() == TIMED.len()
                    && cases
                        .iter()
                        .zip(TIMED)
                        .all(|(case, index)| case.fixture == index && case.valid())
            })
            && (!self.rust_simt
                || self
                    .forward
                    .iter()
                    .chain(&self.reverse)
                    .all(Case::confirmed))
    }
}
fn supported(shape: [usize; SHAPE_RANK]) -> bool {
    shape[..GEOMETRY_RANK] == [KEY_HEADS, VALUE_HEADS, DIM] && SUPPORTED_LANES.contains(&shape[3])
}
type Memo = Mutex<BTreeMap<Key, Entry>>;
fn memo() -> &'static Memo {
    static MEMO: OnceLock<Memo> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(BTreeMap::new()))
}
pub(crate) fn select(
    device: &CudaDevice,
    shape: [usize; SHAPE_RANK],
    autotune: bool,
    session: &ResidentTuningSession,
) -> bool {
    if !supported(shape) {
        return false;
    }
    match resolve(device, shape, autotune, session) {
        Ok(value) => value,
        Err(error) => {
            tracing::warn!(target:"infer::tuning",?shape,%error,"exact GDN tuning failed: Rust SIMT default");
            true
        }
    }
}
fn resolve(
    device: &CudaDevice,
    shape: [usize; SHAPE_RANK],
    autotune: bool,
    session: &ResidentTuningSession,
) -> Result<bool> {
    if !device.target().has_simt_artifacts() {
        return Ok(false);
    }
    let driver_build = driver_build();
    let key = Key {
        hardware: serde_json::to_string(device.profile()?).map_err(device_error)?,
        driver_version: CudaDevice::driver_version()?,
        driver_build,
        kernel_fingerprint: identity::fingerprint(device, shape)?,
        schema: SCHEMA.into(),
        shape,
    };
    if let Some(entry) = memo()
        .lock()
        .map_err(|_| Error::invariant("exact GDN memo poisoned"))?
        .get(&key)
    {
        return Ok(entry.rust_simt);
    }
    let path = table_path();
    let local = if key.driver_build.is_some() {
        resolve_table(path.as_deref())?
    } else {
        None
    };
    for table in [local.as_ref(), Some(shipped_table()?)]
        .into_iter()
        .flatten()
    {
        if let Some(entry) = table
            .resident_exact_recurrent
            .iter()
            .find(|entry| key.driver_build.is_some() && entry.key == key)
        {
            if !entry.valid() {
                return Err(Error::invalid("incomplete exact GDN cache evidence"));
            }
            memo()
                .lock()
                .map_err(|_| Error::invariant("exact GDN memo poisoned"))?
                .insert(key, entry.clone());
            tracing::info!(target:"infer::tuning",?shape,rust_simt=entry.rust_simt,source="table-cache","exact GDN selected");
            return Ok(entry.rust_simt);
        }
    }
    // The Rust SIMT kernel won every measured width; an unmeasured width takes it rather than
    // the cuTile recurrence, whose per-chunk tile grows with the prompt width.
    if !autotune || !session.permits_recurrent_search() {
        return Ok(true);
    }
    let Some(entry) = tune(device, key.clone(), session)? else {
        return Ok(true);
    };
    let cached = key.driver_build.is_some()
        && update_table(path.as_deref(), |table| {
            table
                .resident_exact_recurrent
                .retain(|entry| entry.key != key);
            table.resident_exact_recurrent.push(entry.clone());
        });
    tracing::info!(target:"infer::tuning",?shape,rust_simt=entry.rust_simt,cached,verified_elements=entry.verified_elements,max_f64_error=entry.max_f64_error,"exact GDN hardware choice");
    let selected = entry.rust_simt;
    memo()
        .lock()
        .map_err(|_| Error::invariant("exact GDN memo poisoned"))?
        .insert(key, entry);
    Ok(selected)
}
fn fixtures(shape: [usize; SHAPE_RANK]) -> Result<Vec<Fixture>> {
    let width = i32::try_from(shape[3]).map_err(device_error)?;
    let cases = [
        (false, [0, 0, 0, 0]),
        (false, [0, 1, 0, 0]),
        (false, [0, SHORT_COUNT, 0, 0]),
        (false, [0, width - 1, 0, 0]),
        (false, [OFFSET_BASE, OFFSET_COUNT, NEGATIVE_OFFSET, 0]),
        (false, [0, width, -1, 0]),
        (false, [LATE_BASE, width, 0, 0]),
        (false, [0, width, 0, 0]),
        (true, [0, width, 0, 0]),
    ];
    cases
        .into_iter()
        .map(|(correlated, meta)| Fixture::new_case(shape, correlated, meta))
        .collect()
}
fn measure(
    device: &CudaDevice,
    native: &Graph,
    rust: &Graph,
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
        (rust, native)
    } else {
        (native, rust)
    };
    let run = |graph: &Graph| {
        graph
            .launch()
            .sync_on(&device.stream)
            .map_err(|e| tensor_error(&e.to_string()))
    };
    let (a, b) = do_bench_paired(&device.stream, &options, |_| run(left), |_| run(right))
        .map_err(device_error)?;
    let (a, b) = if reverse { (b, a) } else { (a, b) };
    let times = Times {
        baseline_ms: a.times_ms().to_vec(),
        candidate_ms: b.times_ms().to_vec(),
    };
    if !times.valid() {
        return Err(Error::invalid("exact GDN paired timing evidence"));
    }
    Ok(times)
}
fn cases(
    device: &CudaDevice,
    fixtures: &[Fixture],
    rust: &oxide::Support,
    reverse: bool,
    session: &ResidentTuningSession,
) -> Result<Option<Vec<Case>>> {
    let mut cases = Vec::new();
    for index in TIMED {
        if !session.permits_recurrent_search() {
            return Ok(None);
        }
        let mut data = Captured::new(device, &fixtures[index])?;
        let native = data.graph(device, None)?;
        let candidate = data.graph(device, Some(rust))?;
        data.verify(device, &native, &candidate, &fixtures[index])?;
        cases.push(Case {
            fixture: index,
            cold: measure(device, &native, &candidate, true, reverse)?,
            warm: measure(device, &native, &candidate, false, reverse)?,
        });
        data.verify(device, &native, &candidate, &fixtures[index])?;
    }
    Ok(Some(cases))
}
fn tune(device: &CudaDevice, key: Key, session: &ResidentTuningSession) -> Result<Option<Entry>> {
    let compile = CompileOptions::default();
    if cfg!(debug_assertions)
        || compile.device_debug
        || compile.opt_level.is_some_and(|x| x != CUDA_OPT_LEVEL)
    {
        return Err(Error::invalid(
            "exact GDN tuning requires release and CUDA O3",
        ));
    }
    let fixtures = fixtures(key.shape)?;
    let rust = oxide::Support::load(device)?;
    let mut max_f64_error = 0.0_f64;
    for fixture in &fixtures {
        if !session.permits_recurrent_search() {
            return Ok(None);
        }
        let mut data = Captured::new(device, fixture)?;
        let native = data.graph(device, None)?;
        let candidate = data.graph(device, Some(&rust))?;
        max_f64_error = max_f64_error.max(data.verify(device, &native, &candidate, fixture)?);
    }
    let Some(forward) = cases(device, &fixtures, &rust, false, session)? else {
        return Ok(None);
    };
    let Some(reverse) = cases(device, &fixtures, &rust, true, session)? else {
        return Ok(None);
    };
    let entry = Entry {
        verified_fixtures: FIXTURES,
        verified_elements: FIXTURES * (VALUE_HEADS * DIM * DIM + key.shape[3] * VALUE_HEADS * DIM),
        rust_simt: forward.iter().chain(&reverse).all(Case::confirmed),
        key,
        max_f64_error,
        forward,
        reverse,
    };
    if !entry.valid() {
        return Err(Error::invariant("incomplete exact GDN tuning entry"));
    }
    Ok(Some(entry))
}
#[cfg(test)]
#[path = "../../../tests/unit/tuning_recurrent_exact.rs"]
mod tests;
