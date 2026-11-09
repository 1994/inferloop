//! Lossless shared KV is selected automatically after exact output and history-aware cost gates.
use super::{
    Choice, Key, Shape,
    capture::Captured,
    reference::{self, Fixture},
};
use crate::resident::attention_oxide::shared_identity;
use crate::tuning::MIN_CONFIRM_SPEEDUP;
use crate::tuning::qualify::{Case, Qualification};
use crate::tuning::resolve_table;
use crate::tuning::search;
use crate::tuning::shipped_table;
use crate::tuning::table_path;
use crate::{device::CudaDevice, tuning::resident_gemm::ResidentTuningSession};
use infer_core::{Error, Result};
use infer_cuda_kernels::jit::JitOptions;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
};

const SCHEMA: &str = "shared-kv32:gqa6:q2:canonical-f32:original-append:short-no-regress5:history8k32k:cold-warm-ab-ba-v1";
const GEOMETRY: (usize, usize, usize) = (24, 4, 256);
const LANES: [usize; 3] = [256, 1024, 2048];
const CAPACITY: usize = 32768;
const HISTORY_CASES: usize = 3;
const MID_HISTORY: usize = 8192;
const fn histories(shape: Shape) -> [usize; HISTORY_CASES] {
    [0, MID_HISTORY, CAPACITY - shape.lanes]
}
const MAX_SHORT_REGRESSION: f32 = 1.05;
const SHA256_HEX_LEN: usize = 64;

fn supported(shape: Shape) -> bool {
    (shape.heads, shape.kv_heads, shape.dim) == GEOMETRY
        && LANES.contains(&shape.lanes)
        && i32::try_from(shape.window).is_ok()
}
/// The case confirms the candidate, or regresses by at most the short-history allowance.
fn accepted(case: &Case) -> bool {
    let minimum = if case.case == 0 {
        1.0 / MAX_SHORT_REGRESSION
    } else {
        MIN_CONFIRM_SPEEDUP
    };
    case.cold.speedup() >= minimum && case.warm.speedup() >= minimum
}
/// The family's records in the shared table section.
pub const FAMILY: &str = "shared_attention";
pub type Entry = Qualification<Key, Choice>;
fn valid(entry: &Entry) -> bool {
    if !supported(entry.key.shape) {
        return false;
    }
    let elements = entry.key.shape.lanes * entry.key.shape.heads * entry.key.shape.dim;
    entry.key.schema == SCHEMA
        && !entry.key.hardware.is_empty()
        && entry.key.fingerprint.len() == SHA256_HEX_LEN
        && entry.key.fingerprint.bytes().all(|x| x.is_ascii_hexdigit())
        && entry
            .key
            .driver_build
            .as_ref()
            .is_none_or(|x| !x.is_empty())
        && entry.verified_fixtures == reference::FIXTURES
        && entry.verified_elements == reference::FIXTURES * elements
        && entry.history_verified_elements == HISTORY_CASES * elements
        && entry.within(super::MAX_ERROR)
        && entry.timed(&histories(entry.key.shape))
        && (entry.selected == Choice::RustSimt
            || entry.selected == Choice::SharedKv
                && entry.forward.iter().chain(&entry.reverse).all(accepted))
}
type Memo = Mutex<BTreeMap<Key, Entry>>;
fn memo() -> &'static Memo {
    static MEMO: OnceLock<Memo> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(BTreeMap::new()))
}
fn key(device: &CudaDevice, shape: Shape) -> Result<Key> {
    let mut key = super::exact::key(device, shape)?;
    key.fingerprint = format!(
        "{:x}",
        Sha256::digest(format!("{}:{}", key.fingerprint, shared_identity()))
    );
    key.schema = SCHEMA.into();
    Ok(key)
}
pub(crate) fn select(
    device: &CudaDevice,
    shape: Shape,
    autotune: bool,
    session: &ResidentTuningSession,
    baseline: Choice,
) -> Choice {
    if baseline != Choice::RustSimt || !supported(shape) {
        return baseline;
    }
    match resolve(device, shape, autotune, session) {
        Ok(choice) => choice,
        Err(error) => {
            tracing::warn!(target:"infer::tuning",?shape,%error,"shared KV tuning failed: exact fallback");
            baseline
        }
    }
}
fn resolve(
    device: &CudaDevice,
    shape: Shape,
    autotune: bool,
    session: &ResidentTuningSession,
) -> Result<Choice> {
    if !device.target().has_simt_artifacts() {
        return Ok(Choice::RustSimt);
    }
    let key = key(device, shape)?;
    if let Some(entry) = memo()
        .lock()
        .map_err(|_| Error::invariant("shared KV memo poisoned"))?
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
        if let Some(entry) = key
            .driver_build
            .is_some()
            .then(|| search::find::<Key, Entry>(table, FAMILY, &key, |_| true))
            .flatten()
        {
            if !valid(&entry) {
                return Err(Error::invalid("incomplete shared KV cache evidence"));
            }
            memo()
                .lock()
                .map_err(|_| Error::invariant("shared KV memo poisoned"))?
                .insert(key, entry.clone());
            tracing::info!(target:"infer::tuning",?shape,?entry.selected,source="table-cache","shared KV selected");
            return Ok(entry.selected);
        }
    }
    // Wide quadratic F64 proofs are prepared offline, never added to model startup.
    if shape.lanes > LANES[0] || !autotune || !session.permits_attention_search() {
        return Ok(Choice::RustSimt);
    }
    let Some(entry) = tune(device, key.clone(), session)? else {
        return Ok(Choice::RustSimt);
    };
    let cached = key.driver_build.is_some() && search::put(path.as_deref(), FAMILY, &key, &entry);
    tracing::info!(target:"infer::tuning",?shape,?entry.selected,cached,
        verified_elements=entry.verified_elements,history_verified_elements=entry.history_verified_elements,
        max_f64_error=entry.max_f64_error,"shared KV hardware choice");
    let choice = entry.selected;
    memo()
        .lock()
        .map_err(|_| Error::invariant("shared KV memo poisoned"))?
        .insert(key, entry);
    Ok(choice)
}
fn cases(
    device: &CudaDevice,
    shape: Shape,
    fixtures: &[Fixture],
    reverse: bool,
    session: &ResidentTuningSession,
) -> Result<Option<Vec<Case>>> {
    let mut result = Vec::new();
    for (fixture, history) in fixtures.iter().zip(histories(shape)) {
        if !session.permits_attention_search() {
            return Ok(None);
        }
        let base = Captured::new(device, fixture, Choice::RustSimt)?;
        let shared = Captured::new(device, fixture, Choice::SharedKv)?;
        shared.exact_match(device, &base)?;
        result.push(Case {
            case: history,
            cold: super::measure(device, &base, &shared, true, reverse)?,
            warm: super::measure(device, &base, &shared, false, reverse)?,
        });
    }
    Ok(Some(result))
}
fn tune(device: &CudaDevice, key: Key, session: &ResidentTuningSession) -> Result<Option<Entry>> {
    let options = JitOptions::default();
    if cfg!(debug_assertions)
        || options.device_debug()
        || options
            .opt_level()
            .is_some_and(|level| level != super::CUDA_OPT_LEVEL)
    {
        return Err(Error::invalid(
            "shared KV tuning requires release and CUDA O3",
        ));
    }
    let mut verified_elements = 0;
    let mut max_f64_error = 0.0_f64;
    for index in 0..reference::FIXTURES {
        if !session.permits_attention_search() {
            return Ok(None);
        }
        let f = Fixture::new(key.shape, index)?;
        let baseline = Captured::new(device, &f, Choice::RustSimt)?;
        let shared = Captured::new(device, &f, Choice::SharedKv)?;
        max_f64_error = max_f64_error
            .max(baseline.verify(device, &f)?)
            .max(shared.verify(device, &f)?);
        verified_elements += shared.exact_match(device, &baseline)?;
    }
    let mut fixtures = Vec::new();
    for history in histories(key.shape) {
        fixtures.push(if history == 0 {
            Fixture::new(key.shape, 0)?
        } else {
            Fixture::history(key.shape, history, CAPACITY)?
        });
    }
    let Some(forward) = cases(device, key.shape, &fixtures, false, session)? else {
        return Ok(None);
    };
    let Some(reverse) = cases(device, key.shape, &fixtures, true, session)? else {
        return Ok(None);
    };
    let selected = if forward.iter().chain(&reverse).all(accepted) {
        Choice::SharedKv
    } else {
        Choice::RustSimt
    };
    let history_verified_elements =
        fixtures.len() * key.shape.heads * key.shape.dim * key.shape.lanes;
    let entry = Entry {
        key,
        selected,
        verified_fixtures: reference::FIXTURES,
        verified_elements,
        history_verified_elements,
        max_f64_error,
        forward,
        reverse,
    };
    if !valid(&entry) {
        return Err(Error::invariant("incomplete shared KV proof"));
    }
    Ok(Some(entry))
}
#[cfg(test)]
#[path = "../../../tests/unit/tuning_attention_shared.rs"]
mod tests;
