use super::{BASELINE, CANDIDATES, SUPPORTED_DIMENSION};
use crate::tuning::MIN_REPS;
use crate::tuning::search::{self, Evidence, FormerTrial, Protocol, Record};
use serde::{Deserialize, Serialize};
use serde_json::Value;
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Shape {
    pub rows: usize,
    pub dimension: usize,
    pub epsilon_bits: u32,
}
impl Shape {
    pub fn new(rows: usize, dimension: usize, epsilon: f32) -> Option<Self> {
        (rows > 0
            && rows.checked_mul(dimension).is_some()
            && i32::try_from(rows).is_ok()
            && dimension == SUPPORTED_DIMENSION
            && epsilon.is_finite()
            && epsilon > 0.0)
            .then_some(Self {
                rows,
                dimension,
                epsilon_bits: epsilon.to_bits(),
            })
    }
    fn valid(self) -> bool {
        Self::new(self.rows, self.dimension, f32::from_bits(self.epsilon_bits)) == Some(self)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Key {
    pub hardware: String,
    pub driver_version: u32,
    pub driver_build: Option<String>,
    pub kernel_fingerprint: String,
    pub shape: Shape,
}
/// The family's records in the shared table section.
pub const FAMILY: &str = "gated_norm";
/// An occupancy measured against the compiler policy.
pub type Trial = search::Trial<i32>;
pub type Entry = Record<Key, i32>;
/// Every occupancy hint against the compiler policy.
pub fn protocol() -> Protocol<i32> {
    Protocol {
        baseline: BASELINE,
        candidates: CANDIDATES.to_vec(),
    }
}
impl Entry {
    #[must_use]
    pub fn valid(&self) -> bool {
        self.key.shape.valid()
            && !self.key.kernel_fingerprint.is_empty()
            && !self.key.hardware.is_empty()
            && self.evidence.sampled(MIN_REPS)
            && self.evidence.valid(&protocol(), |occupancy| {
                occupancy == BASELINE || CANDIDATES.contains(&occupancy)
            })
    }
}

#[derive(Deserialize)]
struct LegacyTrial {
    occupancy: i32,
    verified_elements: usize,
    #[serde(flatten)]
    trial: FormerTrial,
}
#[derive(Deserialize)]
struct LegacyEntry {
    key: Key,
    selected: i32,
    search: Vec<LegacyTrial>,
    confirmation: LegacyTrial,
}

/// A record from the family's former table section in the shared form. One that does not
/// decode or has incomplete samples stays as stored, so a rewrite keeps it. Every trial of the
/// former search compared the full output, which the verified element counts record; every
/// candidate was measured, since a failed one ended the search.
pub(in crate::tuning) fn legacy(stored: Value) -> Value {
    convert(&stored).unwrap_or(stored)
}

fn convert(stored: &Value) -> Option<Value> {
    let legacy = LegacyEntry::deserialize(stored).ok()?;
    let elements = legacy
        .key
        .shape
        .rows
        .checked_mul(legacy.key.shape.dimension)?;
    let exact_candidates_verified = legacy
        .search
        .iter()
        .chain([&legacy.confirmation])
        .all(|trial| trial.verified_elements == elements);
    let trial = |legacy: LegacyTrial| legacy.trial.into_trial(legacy.occupancy);
    let entry = Entry {
        key: legacy.key,
        evidence: Evidence {
            selected: legacy.selected,
            exact_candidates_verified,
            trials: legacy
                .search
                .into_iter()
                .map(trial)
                .collect::<Option<_>>()?,
            rejected: Some(Vec::new()),
            confirmation: trial(legacy.confirmation)?,
            max_error: None,
        },
    };
    serde_json::to_value(entry).ok()
}
