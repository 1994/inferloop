//! Measurement allowance for the unique projection searches a model actually retains.
use super::{REP_MS, WARMUP_MS};
use infer_core::{Error, Result};
use std::{
    collections::BTreeMap,
    sync::Mutex,
    time::{Duration, Instant},
};

/// Every search measures cold and warm cache states; each paired window includes both arms.
const CACHE_STATES: u32 = 2;

/// Separate numeric/search contracts can share a matrix geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProjectionKind {
    Fp4,
    Fp8,
    Fp4Lt,
    Swiglu,
    PoolDense {
        output_tile: usize,
        reduction_tile: usize,
    },
}
/// Matrix geometry is activation rows, output columns and reduction columns.
pub(super) const GEMM_RANK: usize = 3;
/// Shared shape contract for each projection numeric path.
pub type ProjectionShape = [usize; GEMM_RANK];
type Work = (ProjectionKind, ProjectionShape);

#[derive(Debug)]
pub(super) struct Snapshot {
    pub(super) planned: usize,
    pub(super) allowance: Duration,
    pub(super) elapsed: Duration,
    pub(super) pending: Vec<Work>,
}

/// A model session registers its live shapes before searching. Repeated weights and states
/// do not renew the allowance. Late program widths add only their new measurement work.
#[derive(Default)]
pub(super) struct SearchBudget {
    work: BTreeMap<Work, (Duration, bool)>,
    estimate: Duration,
    elapsed: Duration,
    active: Option<Instant>,
    reported: bool,
    #[cfg(test)]
    limit: Option<Duration>,
}
impl SearchBudget {
    #[cfg(test)]
    pub(super) fn with_limit(limit: Duration) -> Self {
        Self {
            limit: Some(limit),
            ..Self::default()
        }
    }

    /// Trials count AB search and independently recaptured BA confirmation windows.
    /// This estimates timed work, not an interruptible deadline for CUDA JIT or validation.
    pub(super) fn plan(
        &mut self,
        kind: ProjectionKind,
        shape: ProjectionShape,
        trials: usize,
    ) -> Result<bool> {
        if shape.contains(&0) || trials == 0 {
            return Err(Error::invalid(
                "projection search workload must be nonempty",
            ));
        }
        let trials = u32::try_from(trials).map_err(|e| Error::invalid(e.to_string()))?;
        let allowance = Duration::from_millis(WARMUP_MS + REP_MS)
            .checked_mul(CACHE_STATES)
            .and_then(|window| window.checked_mul(trials))
            .ok_or_else(|| Error::invalid("projection search estimate overflow"))?;
        let key = (kind, shape);
        let old = self
            .work
            .get(&key)
            .map_or(Duration::ZERO, |&(cost, _)| cost);
        if old >= allowance {
            return Ok(false);
        }
        let increase = allowance
            .checked_sub(old)
            .ok_or_else(|| Error::invariant("projection search estimate decreased"))?;
        self.estimate = self
            .estimate
            .checked_add(increase)
            .ok_or_else(|| Error::invalid("model projection search estimate overflow"))?;
        self.work.insert(key, (allowance, false));
        self.reported = false;
        Ok(true)
    }

    pub(super) fn complete(&mut self, kind: ProjectionKind, shape: ProjectionShape) {
        if let Some((_, complete)) = self.work.get_mut(&(kind, shape)) {
            *complete = true;
        }
    }

    pub(super) fn elapsed(&self) -> Duration {
        self.elapsed
            .saturating_add(self.active.map_or(Duration::ZERO, |start| start.elapsed()))
    }

    pub(super) const fn allowance(&self) -> Duration {
        #[cfg(test)]
        if let Some(limit) = self.limit {
            return limit;
        }
        self.estimate
    }

    pub(super) fn permits(&self, elapsed: Duration) -> bool {
        !self.work.is_empty() && elapsed < self.allowance()
    }

    pub(super) fn report_expiration(&mut self, elapsed: Duration) -> bool {
        if self.reported || self.permits(elapsed) || self.work.is_empty() {
            return false;
        }
        self.reported = true;
        true
    }

    pub(super) fn snapshot(&self) -> Snapshot {
        Snapshot {
            planned: self.work.len(),
            allowance: self.allowance(),
            elapsed: self.elapsed(),
            pending: self
                .work
                .iter()
                .filter_map(|(&key, &(_, complete))| (!complete).then_some(key))
                .collect(),
        }
    }
}

/// Counts only active new-search work. Cache proofs and other preparation cannot spend it.
/// The guard owns host accounting only and never retains the mutex across device operations.
pub struct SearchScope<'a> {
    budget: &'a Mutex<SearchBudget>,
}
impl<'a> SearchScope<'a> {
    pub(super) fn begin(budget: &'a Mutex<SearchBudget>) -> Result<Self> {
        let mut state = budget
            .lock()
            .map_err(|_| Error::invariant("projection search budget poisoned"))?;
        if state.active.is_some() {
            return Err(Error::invariant("overlapping model projection searches"));
        }
        state.active = Some(Instant::now());
        drop(state);
        Ok(Self { budget })
    }
}
impl Drop for SearchScope<'_> {
    fn drop(&mut self) {
        let Ok(mut state) = self.budget.lock() else {
            tracing::error!(target:"infer::tuning", "projection search accounting poisoned; search remains disabled");
            return;
        };
        if let Some(start) = state.active.take() {
            state.elapsed = state.elapsed.saturating_add(start.elapsed());
        }
        drop(state);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/projection_search_budget.rs"]
mod tests;
