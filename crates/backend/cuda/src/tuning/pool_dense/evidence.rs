//! Complete paired evidence; new choices preserve the private reduction tree.
use super::capture::tile;
use super::{Choice, Key, baseline, candidates_for, prompt_searchable, searchable};
use crate::tuning::{
    MIN_CONFIRM_SPEEDUP,
    resident_gemm::{MeasurementOrder, PairedTimes, times_valid},
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    pub choice: Choice,
    pub order: MeasurementOrder,
    pub cold: PairedTimes,
    pub warm: PairedTimes,
}
impl Trial {
    pub(super) fn confirmed(&self) -> bool {
        times_valid(&self.cold)
            && times_valid(&self.warm)
            && self.cold.speedup >= MIN_CONFIRM_SPEEDUP
            && self.warm.speedup >= MIN_CONFIRM_SPEEDUP
    }
    fn valid(&self) -> bool {
        times_valid(&self.cold) && times_valid(&self.warm)
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub key: Key,
    pub selected: Choice,
    /// Every measured new choice matched the complete private output; the legacy baseline
    /// retains its own already-admitted arithmetic when the row count selects tensor cores.
    pub exact_candidates_verified: bool,
    pub trials: Vec<Trial>,
    pub rejected: Vec<(Choice, String)>,
    pub confirmation: Trial,
}
impl Entry {
    pub(crate) fn valid(&self) -> bool {
        let shape = [self.key.native.m, self.key.native.n, self.key.native.k];
        let baseline = if self.key.baseline.prompt_family() {
            Choice::PromptBase
        } else {
            baseline(shape)
        };
        let prompt = baseline.prompt_family();
        let allowed = |choice: Choice| {
            choice.prompt_family() == prompt
                && (!choice.native_family() || baseline.native_family())
        };
        let candidates = candidates_for(baseline, shape);
        self.exact_candidates_verified
            && if prompt {
                prompt_searchable(shape)
            } else {
                searchable(shape)
            }
            && tile(&self.key).is_ok()
            && candidates.iter().all(|&choice| {
                self.trials
                    .iter()
                    .filter(|trial| trial.choice == choice)
                    .count()
                    + self
                        .rejected
                        .iter()
                        .filter(|(rejected, _)| *rejected == choice)
                        .count()
                    == 1
            })
            && self.trials.len() + self.rejected.len() == candidates.len()
            && self.key.baseline == baseline
            && !self.key.native.kernel_fingerprint.is_empty()
            && self
                .key
                .native
                .driver_build
                .as_ref()
                .is_none_or(|s| !s.is_empty())
            && allowed(self.selected)
            && self.confirmation.choice == self.selected
            && self.confirmation.order == MeasurementOrder::CandidateFirst
            && self.confirmation.valid()
            && self.trials.iter().all(|trial| {
                trial.order == MeasurementOrder::BaselineFirst
                    && allowed(trial.choice)
                    && trial.valid()
            })
            && self
                .rejected
                .iter()
                .all(|(choice, reason)| allowed(*choice) && !reason.is_empty())
            && (self.selected == baseline
                || self.confirmation.confirmed()
                    && self
                        .trials
                        .iter()
                        .any(|trial| trial.choice == self.selected && trial.confirmed()))
    }
}
