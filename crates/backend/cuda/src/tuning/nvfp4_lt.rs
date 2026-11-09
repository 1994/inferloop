//! Persist only complete FP4 Lt performance confirmations; live-weight proofs stay model-local.
use super::resident_gemm::{
    GemmConfig, MeasurementOrder, PairedTimes, ResidentGemmKey, times_valid,
};
use super::search;
use super::{MIN_CONFIRM_SPEEDUP, resolve_table};
use crate::device::vendor_fp8::{AlgorithmConfig, LibraryIdentity};
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Bump when SDK types, layout, accumulation, alpha arithmetic or confirmation protocol changes.
pub(crate) const SCHEMA: &str =
    "nvfp4-lt-v1:prequantized-fused-down:f32:vec16-ue4m3:row128-k4:alpha1/(wg*xs):ab-ba-cold-warm";
/// One queried algorithm, current native reference and complete kernel/toolchain identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Key {
    pub native: ResidentGemmKey,
    pub baseline: GemmConfig,
    pub library: LibraryIdentity,
    pub algorithm: AlgorithmConfig,
    pub pack_fingerprint: String,
    pub schema: String,
}
/// One measured arm order, with separately recorded cold and warm source samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    pub order: MeasurementOrder,
    pub cold: PairedTimes,
    pub warm: PairedTimes,
}
/// A search and independent reversed confirmation; no opaque algorithm bytes or weight pointers.
/// The family's records in the shared table section.
pub const FAMILY: &str = "fp4_lt";
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub key: Key,
    pub trials: Vec<Trial>,
}
impl Entry {
    pub(crate) fn valid(&self) -> bool {
        self.key.schema == SCHEMA
            && !self.key.pack_fingerprint.is_empty()
            && self
                .key
                .native
                .driver_build
                .as_ref()
                .is_some_and(|s| !s.is_empty())
            && self.trials.len() == 2
            && self.trials[0].order == MeasurementOrder::BaselineFirst
            && self.trials[1].order == MeasurementOrder::CandidateFirst
            && self.trials.iter().all(|trial| {
                times_valid(&trial.cold)
                    && times_valid(&trial.warm)
                    && trial.cold.speedup >= MIN_CONFIRM_SPEEDUP
                    && trial.warm.speedup >= MIN_CONFIRM_SPEEDUP
            })
    }
}
pub(crate) fn cached(path: Option<&Path>, key: &Key) -> Result<bool> {
    if key.native.driver_build.is_none() {
        return Ok(false);
    }
    let Some(table) = resolve_table(path)? else {
        return Ok(false);
    };
    let matching: Vec<Entry> = search::matching(&table, FAMILY, key).collect();
    if matching.len() > 1 || matching.iter().any(|entry| !entry.valid()) {
        return Err(Error::invalid("invalid or duplicate FP4 Lt qualification"));
    }
    Ok(!matching.is_empty())
}
pub(crate) fn save(path: Option<&Path>, entry: &Entry) -> bool {
    if !entry.valid() {
        return false;
    }
    search::put(path, FAMILY, &entry.key, entry)
}

#[cfg(test)]
#[path = "../../tests/unit/tuning_nvfp4_lt.rs"]
mod tests;
