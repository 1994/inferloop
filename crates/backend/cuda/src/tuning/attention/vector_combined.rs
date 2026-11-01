//! Combined vector key/consumer refinement, selected only from verified tuning entries.
//!
//! The hardware search that produces those entries is deliberately deferred: an entry reaches
//! this layer only after a GPU gate has proven the candidate bit-exact and measured it against
//! the qualified vector kernel. With no such entry the selection path returns the incoming
//! baseline unchanged, so adding this layer cannot alter existing behaviour.
use super::{Choice, Key, Shape, Times, reference};
use crate::device::CudaDevice;
use crate::resident::attention_oxide::Support;
use crate::resident::attention_oxide::shared_identity;
use crate::resident::attention_oxide::vector_identity;
use crate::tuning::MIN_CONFIRM_SPEEDUP;
use crate::tuning::resolve_table;
use crate::tuning::shipped_table;
use crate::tuning::table_path;
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case {
    pub history: usize,
    pub cold: Times,
    pub warm: Times,
}
impl Case {
    fn accepted(&self) -> bool {
        let minimum = if self.history == 0 {
            1.0 / MAX_SHORT_REGRESSION
        } else {
            MIN_CONFIRM_SPEEDUP
        };
        self.cold.speedup() >= minimum && self.warm.speedup() >= minimum
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub key: Key,
    pub selected: Choice,
    pub verified_fixtures: usize,
    pub verified_elements: usize,
    pub history_verified_elements: usize,
    pub max_f64_error: f64,
    pub forward: Vec<Case>,
    pub reverse: Vec<Case>,
}
impl Entry {
    fn valid(&self) -> bool {
        if !supported(self.key.shape) {
            return false;
        }
        let elements = self.key.shape.lanes * self.key.shape.heads * self.key.shape.dim;
        self.key.schema == SCHEMA
            && !self.key.hardware.is_empty()
            && self.key.fingerprint.len() == SHA256_HEX_LEN
            && self.key.fingerprint.bytes().all(|x| x.is_ascii_hexdigit())
            && self.key.driver_build.as_ref().is_none_or(|x| !x.is_empty())
            && self.verified_fixtures == reference::FIXTURES
            && self.verified_elements == reference::FIXTURES * elements
            && self.history_verified_elements == HISTORIES.len() * elements
            && self.max_f64_error.is_finite()
            && (0.0..super::MAX_ERROR).contains(&self.max_f64_error)
            && [&self.forward, &self.reverse].iter().all(|cases| {
                cases.len() == HISTORIES.len()
                    && cases.iter().zip(HISTORIES).all(|(case, history)| {
                        case.history == history && case.cold.valid() && case.warm.valid()
                    })
            })
            && (self.selected == Choice::VectorSharedKv
                || self.selected == Choice::VectorSharedKvCombined
                    && self.forward.iter().chain(&self.reverse).all(Case::accepted))
    }
}
type Memo = Mutex<BTreeMap<Key, Entry>>;
fn memo() -> &'static Memo {
    static MEMO: OnceLock<Memo> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(BTreeMap::new()))
}
/// Combined artifact plus tuning-source identity, kept out of the vector layer's fingerprint.
fn fingerprint(base: &str) -> String {
    format!(
        "{:x}",
        Sha256::digest(format!(
            "{}:{}:{}:{}:{:x}",
            base,
            shared_identity(),
            vector_identity(),
            Support::combined_identity(),
            Sha256::digest(include_str!("vector_combined.rs")),
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
    if !entry.valid() {
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
        if let Some(entry) = table
            .resident_vector_combined_attention
            .iter()
            .find(|entry| key.driver_build.is_some() && entry.key == key)
        {
            let choice = decide(baseline, Some(entry))?;
            memo()
                .lock()
                .map_err(|_| Error::invariant("vector combined memo poisoned"))?
                .insert(key, entry.clone());
            tracing::info!(target:"infer::tuning",?shape,?choice,source="table-cache","vector combined selected");
            return Ok(choice);
        }
    }
    Ok(baseline)
}
#[cfg(test)]
#[path = "../../../tests/unit/tuning_attention_vector_combined.rs"]
mod tests;
