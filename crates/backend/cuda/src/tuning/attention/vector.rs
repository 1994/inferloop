//! Aligned vector-key vector shared KV is selected automatically after exact output and history-aware cost gates.
use super::{
    Choice, Key, Shape,
    capture::Captured,
    reference::{self, Fixture},
};
use crate::resident::attention_oxide::shared_identity;
use crate::resident::attention_oxide::vector_identity;
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

const SCHEMA: &str = "vector-key128:shared-kv32:gqa6:q2:canonical-f32:original-append:short-no-regress5:history8k32k:cold-warm-ab-ba-v1";
const GEOMETRY: (usize, usize, usize, usize) = (24, 4, 256, 256);
const CAPACITY: usize = 32768;
const HISTORIES: [usize; 3] = [0, 8192, 32512];
const MAX_SHORT_REGRESSION: f32 = 1.05;
const SHA256_HEX_LEN: usize = 64;

fn supported(shape: Shape) -> bool {
    (shape.heads, shape.kv_heads, shape.dim, shape.lanes) == GEOMETRY
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
pub const FAMILY: &str = "vector_shared_attention";
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
        && entry.history_verified_elements == HISTORIES.len() * elements
        && entry.within(super::MAX_ERROR)
        && entry.timed(&HISTORIES)
        && (entry.selected == Choice::SharedKv
            || entry.selected == Choice::VectorSharedKv
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
        Sha256::digest(format!(
            "{}:{}:{}",
            key.fingerprint,
            shared_identity(),
            vector_identity(),
        ))
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
    if baseline != Choice::SharedKv || !supported(shape) {
        return baseline;
    }
    match resolve(device, shape, autotune, session) {
        Ok(choice) => choice,
        Err(error) => {
            tracing::warn!(target:"infer::tuning",?shape,%error,"vector shared KV tuning failed: exact fallback");
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
        return Ok(Choice::SharedKv);
    }
    let key = key(device, shape)?;
    if let Some(entry) = memo()
        .lock()
        .map_err(|_| Error::invariant("vector shared KV memo poisoned"))?
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
                return Err(Error::invalid("incomplete vector shared KV cache evidence"));
            }
            memo()
                .lock()
                .map_err(|_| Error::invariant("vector shared KV memo poisoned"))?
                .insert(key, entry.clone());
            tracing::info!(target:"infer::tuning",?shape,?entry.selected,source="table-cache","vector shared KV selected");
            return Ok(entry.selected);
        }
    }
    if !autotune || !session.permits_attention_search() {
        return Ok(Choice::SharedKv);
    }
    let Some(entry) = tune(device, key.clone(), session)? else {
        return Ok(Choice::SharedKv);
    };
    let cached = key.driver_build.is_some() && search::put(path.as_deref(), FAMILY, &key, &entry);
    tracing::info!(target:"infer::tuning",?shape,?entry.selected,cached,
        verified_elements=entry.verified_elements,history_verified_elements=entry.history_verified_elements,
        max_f64_error=entry.max_f64_error,"vector shared KV hardware choice");
    let choice = entry.selected;
    memo()
        .lock()
        .map_err(|_| Error::invariant("vector shared KV memo poisoned"))?
        .insert(key, entry);
    Ok(choice)
}
fn cases(
    device: &CudaDevice,
    fixtures: &[Fixture],
    reverse: bool,
    session: &ResidentTuningSession,
) -> Result<Option<Vec<Case>>> {
    let mut result = Vec::new();
    for (fixture, history) in fixtures.iter().zip(HISTORIES) {
        if !session.permits_attention_search() {
            return Ok(None);
        }
        let base = Captured::new(device, fixture, Choice::SharedKv)?;
        let shared = Captured::new(device, fixture, Choice::VectorSharedKv)?;
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
            "vector shared KV tuning requires release and CUDA O3",
        ));
    }
    let mut verified_elements = 0;
    let mut max_f64_error = 0.0_f64;
    for index in 0..reference::FIXTURES {
        if !session.permits_attention_search() {
            return Ok(None);
        }
        let f = Fixture::new(key.shape, index)?;
        let baseline = Captured::new(device, &f, Choice::SharedKv)?;
        let shared = Captured::new(device, &f, Choice::VectorSharedKv)?;
        max_f64_error = max_f64_error
            .max(baseline.verify(device, &f)?)
            .max(shared.verify(device, &f)?);
        verified_elements += shared.exact_match(device, &baseline)?;
    }
    let mut fixtures = Vec::new();
    for history in HISTORIES {
        fixtures.push(if history == 0 {
            Fixture::new(key.shape, 0)?
        } else {
            Fixture::history(key.shape, history, CAPACITY)?
        });
    }
    let Some(forward) = cases(device, &fixtures, false, session)? else {
        return Ok(None);
    };
    let Some(reverse) = cases(device, &fixtures, true, session)? else {
        return Ok(None);
    };
    let selected = if forward.iter().chain(&reverse).all(accepted) {
        Choice::VectorSharedKv
    } else {
        Choice::SharedKv
    };
    let entry = Entry {
        key,
        selected,
        verified_fixtures: reference::FIXTURES,
        verified_elements,
        history_verified_elements: fixtures.len() * GEOMETRY.0 * GEOMETRY.2 * GEOMETRY.3,
        max_f64_error,
        forward,
        reverse,
    };
    if !valid(&entry) {
        return Err(Error::invariant("incomplete vector shared KV proof"));
    }
    Ok(Some(entry))
}
#[cfg(test)]
#[path = "../../../tests/unit/tuning_attention_vector.rs"]
mod tests;
