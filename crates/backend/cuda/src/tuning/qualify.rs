//! Two-arm qualifications: a candidate replaces the baseline only when it matches the
//! independent reference and beats the baseline on every timed case, cold and warm, in both
//! arm orders.
//!
//! Unlike a search, a qualification has one candidate but several timed cases (fixtures or
//! history lengths), so its record keeps one paired measurement per case and order. Records
//! live in the shared table section under their family's name, like search records.
use super::{MAX_REPS, MIN_CONFIRM_SPEEDUP, MIN_REPS, median};
use serde::{Deserialize, Serialize};

/// Unpaired samples of one case in one cache state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Times {
    pub baseline_ms: Vec<f32>,
    pub candidate_ms: Vec<f32>,
}

impl Times {
    /// A complete paired window: as many baseline as candidate samples, all finite and positive.
    pub(crate) fn valid(&self) -> bool {
        (MIN_REPS..=MAX_REPS).contains(&self.baseline_ms.len())
            && self.baseline_ms.len() == self.candidate_ms.len()
            && self
                .baseline_ms
                .iter()
                .chain(&self.candidate_ms)
                .all(|x| x.is_finite() && *x > 0.0)
    }

    pub(crate) fn speedup(&self) -> f32 {
        median(&self.baseline_ms) / median(&self.candidate_ms)
    }
}

/// One timed case in one arm order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Case {
    /// The fixture index or history length the case times.
    #[serde(alias = "fixture", alias = "history")]
    pub case: usize,
    pub cold: Times,
    pub warm: Times,
}

impl Case {
    pub(crate) fn valid(&self) -> bool {
        self.cold.valid() && self.warm.valid()
    }

    /// The candidate beat the baseline by the confirmation margin both cold and warm.
    pub(crate) fn confirmed(&self) -> bool {
        self.cold.speedup() >= MIN_CONFIRM_SPEEDUP && self.warm.speedup() >= MIN_CONFIRM_SPEEDUP
    }
}

/// A family's qualification for one key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qualification<K, C> {
    pub key: K,
    #[serde(alias = "rust_simt")]
    pub selected: C,
    pub verified_fixtures: usize,
    pub verified_elements: usize,
    /// Elements checked over long histories, for families that time history lengths.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub history_verified_elements: usize,
    pub max_f64_error: f64,
    /// Cases timed baseline-first.
    pub forward: Vec<Case>,
    /// The same cases timed candidate-first.
    pub reverse: Vec<Case>,
}

impl<K, C> Qualification<K, C> {
    /// Both orders time exactly `cases`, in that order, with complete samples.
    pub(crate) fn timed(&self, cases: &[usize]) -> bool {
        [&self.forward, &self.reverse].iter().all(|timed| {
            timed.len() == cases.len()
                && timed
                    .iter()
                    .zip(cases)
                    .all(|(case, expected)| case.case == *expected && case.valid())
        })
    }

    /// Every timed case confirmed the candidate in both orders.
    pub(crate) fn confirmed(&self) -> bool {
        self.forward
            .iter()
            .chain(&self.reverse)
            .all(Case::confirmed)
    }

    /// The reference deviation is finite and below `tolerance`.
    pub(crate) fn within(&self, tolerance: f64) -> bool {
        self.max_f64_error.is_finite() && (0.0..tolerance).contains(&self.max_f64_error)
    }
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "serde's skip_serializing_if passes the field by reference"
)]
const fn is_zero(count: &usize) -> bool {
    *count == 0
}
