//! Per-shape search allowances derived from the declared paired measurement work.
use super::{GEMM_RANK, MAX_REPS, MIN_REPS, REP_MS, WARMUP_MS};
use infer_core::{Error, Result};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Each timing compares the baseline arm with the candidate arm.
const MEASUREMENT_ARMS: u32 = 2;
/// Candidates must be measured with both cold and warm weights.
const CACHE_STATES: u32 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Family {
    Nvfp4,
    Fp8Channel,
    PoolDense,
    FusedSwiGLU,
}

type Key = (Family, [usize; GEMM_RANK]);
type Shapes = BTreeMap<Key, Allowance>;

struct Allowance {
    budget: Duration,
    started: Option<Instant>,
    complete: bool,
}

/// Registration does not start a timer. A shape receives its own allowance when its search
/// starts, so JIT or measurements of an earlier shape cannot silently consume its opportunity.
#[derive(Default)]
pub(super) struct Schedule {
    shapes: Mutex<Shapes>,
    override_budget: Option<Duration>,
}

fn measurement_budget(trials: usize) -> Result<Duration> {
    let trials =
        u32::try_from(trials).map_err(|_| Error::invalid("GEMM search trial count overflow"))?;
    let repetitions = u32::try_from(MAX_REPS.div_ceil(MIN_REPS))
        .map_err(|_| Error::invalid("GEMM repetition allowance overflow"))?;
    // The max/min repetition ratio reserves the same relative slack as the measurement
    // protocol. Increasing the candidate space expands the allowance instead of stealing time
    // from later shapes; confirmations must be included in `trials` by the caller.
    Duration::from_millis(WARMUP_MS + REP_MS)
        .checked_mul(MEASUREMENT_ARMS)
        .and_then(|budget| budget.checked_mul(CACHE_STATES))
        .and_then(|budget| budget.checked_mul(repetitions))
        .and_then(|budget| budget.checked_mul(trials))
        .ok_or_else(|| Error::invalid("GEMM measurement allowance overflow"))
}

impl Schedule {
    #[cfg(test)]
    pub(super) fn with_limit(budget: Duration) -> Self {
        Self {
            override_budget: Some(budget),
            ..Self::default()
        }
    }

    fn budget(&self, trials: usize) -> Result<Duration> {
        self.override_budget
            .map_or_else(|| measurement_budget(trials), Ok)
    }

    pub(super) fn plan(
        &self,
        family: Family,
        shapes: &[[usize; GEMM_RANK]],
        trials: usize,
    ) -> Result<()> {
        let budget = self.budget(trials)?;
        let mut state = self
            .shapes
            .lock()
            .map_err(|_| Error::invariant("GEMM search schedule poisoned"))?;
        let previous = state.len();
        let mut expanded = false;
        for &shape in shapes {
            let entry = state.entry((family, shape)).or_insert(Allowance {
                budget,
                started: None,
                complete: false,
            });
            if budget > entry.budget {
                entry.budget = budget;
                expanded = true;
            }
        }
        let planned_shapes = state.len();
        let total_budget = state.values().fold(Duration::ZERO, |total, entry| {
            total.saturating_add(entry.budget)
        });
        drop(state);
        if previous != planned_shapes || expanded {
            tracing::info!(target: "infer::tuning", ?family, planned_shapes,
                total_budget_ms=total_budget.as_millis(),
                shape_budget_ms=budget.as_millis(), "GEMM search work planned");
        }
        Ok(())
    }

    pub(super) fn permits(
        &self,
        family: Family,
        shape: [usize; GEMM_RANK],
        trials: usize,
    ) -> Result<bool> {
        self.permits_at(family, shape, trials, Instant::now())
    }

    fn permits_at(
        &self,
        family: Family,
        shape: [usize; GEMM_RANK],
        trials: usize,
        now: Instant,
    ) -> Result<bool> {
        self.plan(family, &[shape], trials)?;
        let mut state = self
            .shapes
            .lock()
            .map_err(|_| Error::invariant("GEMM search schedule poisoned"))?;
        let entry = state
            .get_mut(&(family, shape))
            .ok_or_else(|| Error::invariant("unplanned GEMM search"))?;
        if entry.complete {
            drop(state);
            return Ok(false);
        }
        let started = *entry.started.get_or_insert(now);
        let elapsed = now.saturating_duration_since(started);
        let budget = entry.budget;
        let permitted = elapsed < budget;
        let missing = if permitted {
            Vec::new()
        } else {
            unfinished(&state)
        };
        drop(state);
        if !permitted {
            tracing::info!(target: "infer::tuning", ?family, ?shape, ?missing,
                elapsed_ms=elapsed.as_millis(), budget_ms=budget.as_millis(),
                "GEMM shape search budget exhausted: incomplete evidence discarded");
        }
        Ok(permitted)
    }

    pub(super) fn complete(&self, family: Family, shape: [usize; GEMM_RANK]) -> Result<()> {
        let mut state = self
            .shapes
            .lock()
            .map_err(|_| Error::invariant("GEMM search schedule poisoned"))?;
        if let Some(entry) = state.get_mut(&(family, shape)) {
            entry.complete = true;
        }
        drop(state);
        Ok(())
    }

    pub(super) fn is_exhausted(&self, family: Family, shape: [usize; GEMM_RANK]) -> Result<bool> {
        let state = self
            .shapes
            .lock()
            .map_err(|_| Error::invariant("GEMM search schedule poisoned"))?;
        let exhausted = state.get(&(family, shape)).is_some_and(|entry| {
            entry
                .started
                .is_some_and(|started| started.elapsed() >= entry.budget)
        });
        drop(state);
        Ok(exhausted)
    }
}

fn unfinished(state: &Shapes) -> Vec<Key> {
    state
        .iter()
        .filter_map(|(&key, entry)| (!entry.complete).then_some(key))
        .collect()
}

#[cfg(test)]
#[path = "../../../tests/unit/resident_search_budget.rs"]
mod tests;
