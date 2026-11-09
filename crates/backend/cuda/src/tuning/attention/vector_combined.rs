//! Combined vector key/consumer refinement, selected only from verified tuning entries.
//!
//! The hardware search that produces those entries is deliberately deferred: an entry reaches
//! this layer only after a GPU gate has proven the candidate bit-exact and measured it against
//! the qualified vector kernel. With no such entry the selection path returns the incoming
//! baseline unchanged, so adding this layer cannot alter existing behaviour.
use super::{Choice, Key, Shape, reference};
use crate::device::CudaDevice;
use crate::resident::attention_oxide::Support;
use crate::resident::attention_oxide::shared_identity;
use crate::resident::attention_oxide::vector_identity;
use crate::tuning::MIN_CONFIRM_SPEEDUP;
use crate::tuning::qualify::{Case, Qualification};
use crate::tuning::resolve_table;
use crate::tuning::search;
use crate::tuning::shipped_table;
use crate::tuning::table_path;
use infer_core::{Error, Result};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
};

const SCHEMA: &str = "vector-combined-key128:shared-kv32:gqa6:q2:canonical-f32:original-append:short-no-regress5:history8k32k:cold-warm-ab-ba-v1";
const GEOMETRY: (usize, usize, usize, usize) = (24, 4, 256, 256);
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
pub const FAMILY: &str = "vector_combined_attention";
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
        && (entry.selected == Choice::VectorSharedKv
            || entry.selected == Choice::VectorSharedKvCombined
                && entry.forward.iter().chain(&entry.reverse).all(accepted))
}
type Memo = Mutex<BTreeMap<Key, Entry>>;
fn memo() -> &'static Memo {
    static MEMO: OnceLock<Memo> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(BTreeMap::new()))
}
/// Combined artifact identity, kept out of the vector layer's fingerprint; the method is the
/// key's schema.
fn fingerprint(base: &str) -> String {
    format!(
        "{:x}",
        Sha256::digest(format!(
            "{}:{}:{}:{}",
            base,
            shared_identity(),
            vector_identity(),
            Support::combined_identity(),
        ))
    )
}
fn key(device: &CudaDevice, shape: Shape) -> Result<Key> {
    let mut key = super::exact::key(device, shape)?;
    key.fingerprint = fingerprint(&key.fingerprint);
    key.schema = SCHEMA.into();
    Ok(key)
}
/// The selection is decided purely by validated evidence; an absent entry keeps `baseline`.
fn decide(baseline: Choice, entry: Option<&Entry>) -> Result<Choice> {
    let Some(entry) = entry else {
        return Ok(baseline);
    };
    if !valid(entry) {
        return Err(Error::invalid("incomplete vector combined cache evidence"));
    }
    Ok(entry.selected)
}
pub(crate) fn select(device: &CudaDevice, shape: Shape, baseline: Choice) -> Choice {
    if baseline != Choice::VectorSharedKv || !supported(shape) {
        return baseline;
    }
    match resolve(device, shape, baseline) {
        Ok(choice) => choice,
        Err(error) => {
            tracing::warn!(target:"infer::tuning",?shape,%error,"vector combined tuning failed: qualified vector fallback");
            baseline
        }
    }
}
fn resolve(device: &CudaDevice, shape: Shape, baseline: Choice) -> Result<Choice> {
    if !device.target().has_simt_artifacts() {
        return Ok(baseline);
    }
    let key = key(device, shape)?;
    if let Some(entry) = memo()
        .lock()
        .map_err(|_| Error::invariant("vector combined memo poisoned"))?
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
            let choice = decide(baseline, Some(&entry))?;
            memo()
                .lock()
                .map_err(|_| Error::invariant("vector combined memo poisoned"))?
                .insert(key, entry);
            tracing::info!(target:"infer::tuning",?shape,?choice,source="table-cache","vector combined selected");
            return Ok(choice);
        }
    }
    Ok(baseline)
}
#[cfg(test)]
#[path = "../../../tests/unit/tuning_attention_vector_combined.rs"]
mod tests;
