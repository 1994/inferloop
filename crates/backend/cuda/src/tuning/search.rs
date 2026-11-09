//! One search protocol, record shape and table section for every tuning family.
//!
//! What runs unmeasured is the baseline. Each candidate is measured against it baseline-first,
//! the winner is confirmed candidate-first, and the record keeps the complete evidence.
//!
//! A family supplies its key, its choices and how one trial is measured. How the search runs,
//! what makes its evidence complete and where records live are the same for every family, so a
//! new candidate or family needs no new key structure and no new table field.
use super::resident_gemm::{MeasurementOrder, PairedTimes, times_valid};
use super::{MIN_CONFIRM_SPEEDUP, TilingTable, update_table};
use infer_core::{Error, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;

/// One candidate measured against the baseline, cold and warm, in a recorded arm order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial<C> {
    #[serde(alias = "config")]
    pub choice: C,
    /// Entries from before arm orders were recorded load as unspecified, which no protocol
    /// accepts.
    #[serde(default)]
    pub order: MeasurementOrder,
    pub cold: PairedTimes,
    pub warm: PairedTimes,
}

impl<C> Trial<C> {
    /// Both measurements are complete and internally consistent.
    pub(crate) fn valid(&self) -> bool {
        times_valid(&self.cold) && times_valid(&self.warm)
    }

    /// The candidate beat the baseline by the confirmation margin both cold and warm.
    pub(crate) fn confirmed(&self) -> bool {
        self.valid()
            && self.cold.speedup >= MIN_CONFIRM_SPEEDUP
            && self.warm.speedup >= MIN_CONFIRM_SPEEDUP
    }

    /// The speedup a candidate is ranked by: its worse cache state.
    const fn score(&self) -> f32 {
        self.cold.speedup.min(self.warm.speedup)
    }
}

/// What one search measured and chose.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(deserialize = "C: Deserialize<'de>"))]
pub struct Evidence<C> {
    pub selected: C,
    /// Every measured candidate reproduced its complete reference output.
    #[serde(alias = "bitwise_verified")]
    pub exact_candidates_verified: bool,
    pub trials: Vec<Trial<C>>,
    /// Candidates that failed to build or to match their reference, with the reason. `None` only
    /// in records written before a family kept them, which therefore cannot show the search
    /// covered every candidate; such a record is held to the rules it was written under, and the
    /// next measurement of its key writes the complete form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rejected: Option<Vec<(C, String)>>,
    pub confirmation: Trial<C>,
}

/// The choices one search covers: what runs unmeasured, and what is measured against it.
pub(crate) struct Protocol<C> {
    pub(crate) baseline: C,
    pub(crate) candidates: Vec<C>,
}

impl<C: Copy + PartialEq> Evidence<C> {
    /// Whether this is a complete run of `protocol` whose every choice passes `allowed`: each
    /// candidate measured or rejected exactly once, measurements baseline-first, a candidate-first
    /// confirmation of the selection, and a selection other than the baseline confirmed twice.
    /// A record without its rejections is checked for everything but coverage.
    pub(crate) fn valid(&self, protocol: &Protocol<C>, allowed: impl Fn(C) -> bool) -> bool {
        let covered = |rejected: &[(C, String)]| {
            let count = |choice: C| {
                self.trials
                    .iter()
                    .filter(|trial| trial.choice == choice)
                    .count()
                    + rejected
                        .iter()
                        .filter(|(rejected, _)| *rejected == choice)
                        .count()
            };
            protocol.candidates.iter().all(|&choice| count(choice) == 1)
                && self.trials.len() + rejected.len() == protocol.candidates.len()
                && rejected
                    .iter()
                    .all(|(choice, reason)| allowed(*choice) && !reason.is_empty())
        };
        self.exact_candidates_verified
            && self.rejected.as_deref().is_none_or(covered)
            && allowed(self.selected)
            && self.confirmation.choice == self.selected
            && self.confirmation.order == MeasurementOrder::CandidateFirst
            && self.confirmation.valid()
            && self.trials.iter().all(|trial| {
                trial.order == MeasurementOrder::BaselineFirst
                    && allowed(trial.choice)
                    && trial.valid()
            })
            && (self.selected == protocol.baseline
                || self.confirmation.confirmed()
                    && self
                        .trials
                        .iter()
                        .any(|trial| trial.choice == self.selected && trial.confirmed()))
    }
}

/// Run `protocol`. `trial` measures one choice in one arm order, failing when the choice cannot
/// be built or does not match its reference; `reject` handles such a failure before the search
/// moves on, for example by rechecking the baseline. `None` when `permits` stops the search.
///
/// The best candidate confirmed in both cache states wins; it is then measured again
/// candidate-first, and one that does not confirm there falls back to the baseline.
/// # Errors
/// Propagates a failed confirmation trial and whatever `reject` returns.
pub(crate) fn run<C: Copy + PartialEq>(
    protocol: &Protocol<C>,
    permits: impl Fn() -> bool,
    mut trial: impl FnMut(C, MeasurementOrder) -> Result<Trial<C>>,
    mut reject: impl FnMut(C, &Error) -> Result<()>,
) -> Result<Option<Evidence<C>>> {
    let mut selected = protocol.baseline;
    let mut best = 1.0_f32;
    let mut trials = Vec::new();
    let mut rejected = Vec::new();
    for &choice in &protocol.candidates {
        if !permits() {
            return Ok(None);
        }
        match trial(choice, MeasurementOrder::BaselineFirst) {
            Ok(measured) => {
                let score = measured.score();
                if measured.confirmed() && score > best {
                    best = score;
                    selected = choice;
                }
                trials.push(measured);
            }
            Err(error) => {
                reject(choice, &error)?;
                rejected.push((choice, error.to_string()));
            }
        }
    }
    if !permits() {
        return Ok(None);
    }
    let mut confirmation = trial(selected, MeasurementOrder::CandidateFirst)?;
    if selected != protocol.baseline && !confirmation.confirmed() {
        if !permits() {
            return Ok(None);
        }
        selected = protocol.baseline;
        confirmation = trial(selected, MeasurementOrder::CandidateFirst)?;
    }
    Ok(Some(Evidence {
        selected,
        exact_candidates_verified: true,
        trials,
        rejected: Some(rejected),
        confirmation,
    }))
}

/// A family's decision for one key, with the evidence behind it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record<K, C> {
    pub key: K,
    #[serde(flatten)]
    pub evidence: Evidence<C>,
}

/// Every record of `family` this build can decode; others stay in the table untouched.
#[must_use]
pub(crate) fn records<K: DeserializeOwned, C: DeserializeOwned>(
    table: &TilingTable,
    family: &str,
) -> Vec<Record<K, C>> {
    table
        .records
        .get(family)
        .into_iter()
        .flatten()
        .filter_map(|stored| Record::deserialize(stored).ok())
        .collect()
}

/// The record of `family` for `key` that satisfies `valid`.
#[must_use]
pub(crate) fn lookup<K, C>(
    table: &TilingTable,
    family: &str,
    key: &K,
    valid: impl Fn(&Record<K, C>) -> bool,
) -> Option<Record<K, C>>
where
    K: DeserializeOwned + PartialEq,
    C: DeserializeOwned,
{
    records(table, family)
        .into_iter()
        .find(|record| record.key == *key && valid(record))
}

/// Store `record` in `family` at `path`, replacing any record with its key. False when nothing
/// was written.
pub(crate) fn store<K, C>(path: Option<&Path>, family: &str, record: &Record<K, C>) -> bool
where
    K: Serialize + DeserializeOwned + PartialEq,
    C: Serialize,
{
    let Ok(value) = serde_json::to_value(record) else {
        return false;
    };
    update_table(path, |table| {
        let stored = table.records.entry(family.to_owned()).or_default();
        stored.retain(|old| key_of::<K>(old).is_none_or(|key| key != record.key));
        stored.push(value);
    })
}

/// The key of a stored record, when this build can decode it.
fn key_of<K: DeserializeOwned>(stored: &Value) -> Option<K> {
    stored.get("key").and_then(|key| K::deserialize(key).ok())
}

#[cfg(test)]
#[path = "../../tests/unit/tuning_search.rs"]
mod tests;
