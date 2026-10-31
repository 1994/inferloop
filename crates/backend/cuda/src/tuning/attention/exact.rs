//! Exact Rust SIMT alternatives use the same device-scoped pre-capture tuning contract.
use super::{
    Choice, Key, Shape, Times,
    capture::Captured,
    reference::{self, Fixture},
};
use crate::device::{CudaDevice, device_error};
use crate::tuning::MIN_CONFIRM_SPEEDUP;
use crate::tuning::preparation::driver_build;
use crate::tuning::resident_gemm::ResidentTuningSession;
use crate::tuning::resolve_table;
use crate::tuning::shipped_table;
use crate::tuning::table_path;
use crate::tuning::update_table;
use cutile::cutile_compiler::hints::CompileOptions;
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
};

const SCHEMA: &str = "oxide-sm120:fp8-e4m3-kv:f32-q-out:canonical-d256:original-append:full-bits-and-f64:fixtures4:history-cold-warm-ab-ba-v1";
const SUPPORTED_GEOMETRY: (usize, usize, usize) = (24, 4, 256);
const SUPPORTED_LANES: [usize; 4] = [64, 256, 1024, 2048];
const SHA256_HEX_LEN: usize = 64;
const TIMED_FIXTURES: [usize; 2] = [0, 1];

fn supported(s: Shape) -> bool {
    (s.heads, s.kv_heads, s.dim) == SUPPORTED_GEOMETRY
        && SUPPORTED_LANES.contains(&s.lanes)
        && i32::try_from(s.window).is_ok()
}
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
    pub selected: Choice,
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
            && self.key.fingerprint.len() == SHA256_HEX_LEN
            && self.key.fingerprint.bytes().all(|c| c.is_ascii_hexdigit())
            && self.key.driver_build.as_ref().is_none_or(|s| !s.is_empty())
            && self.verified_fixtures == reference::FIXTURES
            && self.verified_elements
                == reference::FIXTURES
                    * self.key.shape.lanes
                    * self.key.shape.heads
                    * self.key.shape.dim
            && self.max_f64_error.is_finite()
            && (0.0..super::MAX_ERROR).contains(&self.max_f64_error)
            && [&self.forward, &self.reverse].iter().all(|cases| {
                cases.len() == TIMED_FIXTURES.len()
                    && cases
                        .iter()
                        .zip(TIMED_FIXTURES)
                        .all(|(case, index)| case.fixture == index && case.valid())
            })
            && (self.selected == Choice::Simt
                || self.selected == Choice::RustSimt
                    && self
                        .forward
                        .iter()
                        .chain(&self.reverse)
                        .all(Case::confirmed))
    }
}
type Memo = Mutex<BTreeMap<Key, Entry>>;
fn memo() -> &'static Memo {
    static MEMO: OnceLock<Memo> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(BTreeMap::new()))
}
pub(super) fn key(device: &CudaDevice, shape: Shape) -> Result<Key> {
    let driver_build = driver_build();
    Ok(Key {
        hardware: serde_json::to_string(device.profile()?).map_err(device_error)?,
        driver_api: CudaDevice::driver_version()?,
        driver_build,
        fingerprint: super::capture::native_fingerprint(device, shape)?,
        schema: SCHEMA.into(),
        shape,
    })
}
pub(crate) fn select(
    device: &CudaDevice,
    shape: Shape,
    autotune: bool,
    session: &ResidentTuningSession,
) -> Choice {
    if !supported(shape) {
        return Choice::Simt;
    }
    match resolve(device, shape, autotune, session) {
        Ok(choice) => choice,
        Err(error) => {
            tracing::warn!(target:"infer::tuning",?shape,%error,"exact attention tuning failed: cuTile fallback");
            Choice::Simt
        }
    }
}
fn resolve(
    device: &CudaDevice,
    shape: Shape,
    autotune: bool,
    session: &ResidentTuningSession,
) -> Result<Choice> {
    if device.profile()?.architecture != "sm_120" {
        return Ok(Choice::Simt);
    }
    let key = key(device, shape)?;
    if let Some(entry) = memo()
        .lock()
        .map_err(|_| Error::invariant("exact attention memo poisoned"))?
        .get(&key)
    {
        return Ok(entry.selected);
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
            .resident_exact_attention
            .iter()
            .find(|entry| key.driver_build.is_some() && entry.key == key)
        {
            if !entry.valid() {
                return Err(Error::invalid("incomplete exact attention cache evidence"));
            }
            memo()
                .lock()
                .map_err(|_| Error::invariant("exact attention memo poisoned"))?
                .insert(key, entry.clone());
            tracing::info!(target:"infer::tuning",?shape,?entry.selected,source="table-cache","exact attention selected");
            return Ok(entry.selected);
        }
    }
    // Wide quadratic F64 proofs are prepared offline, never added to model startup.
    if shape.lanes > SUPPORTED_LANES[1] || !autotune || !session.permits_attention_search() {
        return Ok(Choice::Simt);
    }
    let Some(entry) = tune(device, key.clone(), session)? else {
        return Ok(Choice::Simt);
    };
    let cached = key.driver_build.is_some()
        && update_table(path.as_deref(), |table| {
            table
                .resident_exact_attention
                .retain(|entry| entry.key != key);
            table.resident_exact_attention.push(entry.clone());
        });
    tracing::info!(target:"infer::tuning",?shape,?entry.selected,cached,
        verified_elements=entry.verified_elements,max_f64_error=entry.max_f64_error,"exact attention hardware choice");
    let choice = entry.selected;
    memo()
        .lock()
        .map_err(|_| Error::invariant("exact attention memo poisoned"))?
        .insert(key, entry);
    Ok(choice)
}
fn cases(
    device: &CudaDevice,
    key: &Key,
    fixtures: &[Fixture],
    reverse: bool,
    session: &ResidentTuningSession,
) -> Result<Option<Vec<Case>>> {
    let mut cases = Vec::new();
    for index in TIMED_FIXTURES {
        if !session.permits_attention_search() {
            return Ok(None);
        }
        let native = Captured::new(device, &fixtures[index], Choice::Simt)?;
        let rust = Captured::new(device, &fixtures[index], Choice::RustSimt)?;
        rust.exact_match(device, &native)?;
        cases.push(Case {
            fixture: index,
            cold: super::measure(device, &native, &rust, true, reverse)?,
            warm: super::measure(device, &native, &rust, false, reverse)?,
        });
    }
    if cases.len() != TIMED_FIXTURES.len() {
        return Err(Error::invariant(format!(
            "incomplete exact attention cases {:?}",
            key.shape
        )));
    }
    Ok(Some(cases))
}
fn tune(device: &CudaDevice, key: Key, session: &ResidentTuningSession) -> Result<Option<Entry>> {
    let options = CompileOptions::default();
    if cfg!(debug_assertions)
        || options.device_debug
        || options
            .opt_level
            .is_some_and(|level| level != super::CUDA_OPT_LEVEL)
    {
        return Err(Error::invalid(
            "exact attention tuning requires release and CUDA O3",
        ));
    }
    let mut fixtures = Vec::new();
    let mut verified_elements = 0;
    let mut max_error = 0.0_f64;
    for index in 0..reference::FIXTURES {
        if !session.permits_attention_search() {
            return Ok(None);
        }
        let fixture = Fixture::new(key.shape, index)?;
        let native = Captured::new(device, &fixture, Choice::Simt)?;
        let rust = Captured::new(device, &fixture, Choice::RustSimt)?;
        max_error = max_error
            .max(native.verify(device, &fixture)?)
            .max(rust.verify(device, &fixture)?);
        verified_elements += rust.exact_match(device, &native)?;
        fixtures.push(fixture);
    }
    let Some(forward) = cases(device, &key, &fixtures, false, session)? else {
        return Ok(None);
    };
    let Some(reverse) = cases(device, &key, &fixtures, true, session)? else {
        return Ok(None);
    };
    let selected = if forward.iter().chain(&reverse).all(Case::confirmed) {
        Choice::RustSimt
    } else {
        Choice::Simt
    };
    let entry = Entry {
        key,
        selected,
        verified_fixtures: fixtures.len(),
        verified_elements,
        max_f64_error: max_error,
        forward,
        reverse,
    };
    if !entry.valid() {
        return Err(Error::invariant("incomplete exact attention proof"));
    }
    Ok(Some(entry))
}
#[cfg(test)]
#[path = "../../../tests/unit/tuning_attention_exact.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../tests/unit/tuning_attention_fp8_convert.rs"]
mod fp8_convert_tests;

#[cfg(test)]
pub(crate) fn validate_and_cache(
    device: &CudaDevice,
    shape: Shape,
    session: &ResidentTuningSession,
) -> Result<Choice> {
    let key = key(device, shape)?;
    let entry = tune(device, key.clone(), session)?
        .ok_or_else(|| Error::invalid("offline exact attention proof budget expired"))?;
    if !entry.valid()
        || !update_table(table_path().as_deref(), |table| {
            table.resident_exact_attention.retain(|e| e.key != key);
            table.resident_exact_attention.push(entry.clone());
        })
    {
        return Err(Error::invalid(
            "offline exact attention proof could not be cached",
        ));
    }
    println!(
        "WIDE_EXACT_ATTENTION_TUNING {}",
        serde_json::to_string(&entry).map_err(device_error)?
    );
    Ok(entry.selected)
}
