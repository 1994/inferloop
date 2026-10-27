//! Measure the actual pooled row geometry before capture; new paths retain private F32 bits.
mod capture;
mod evidence;
use crate::constants::{CB_DECODE_SLOTS, SMALL_GEMM_TILE};
use crate::device::CudaDevice;
use crate::resident::{linear_batch::batched, pool_gemv, small_gemm_kernels};
use crate::strategy::LinearTiling;
use crate::tuning::projection_budget::{ProjectionKind, ProjectionShape};
use crate::tuning::resident_gemm::{
    MeasurementOrder, ResidentGemmKey, ResidentTuningSession, measure_graphs,
};
use crate::tuning::{resolve_table, table_path, update_table};
use capture::{Captured, key, verify};
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::{half::bf16, prelude::*};
pub use evidence::{Entry, Trial};
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};
#[cfg(test)]
use std::cell::Cell;
use std::collections::BTreeMap;
use std::result::Result as StdResult;
use std::sync::{Arc, Mutex, OnceLock, Weak};

/// ABI of the existing row-tiled private-reduction kernel.
pub(crate) const SHARED_ROWS: usize = 4;
/// Existing qualified small-output family, independent of the row crossover.
const MAX_SMALL_OUTPUTS: usize = 128;
/// Existing qualified reduction-width family.
const MIN_SMALL_COLUMNS: usize = 4096;
/// One reverse confirmation and a baseline confirmation after rejecting a winner.
const CONFIRMATIONS: usize = 2;
/// The benchmark protocol requires the release compiler policy.
const JIT_OPT_LEVEL: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Choice {
    Gemv,
    Shared,
    Native,
}
impl Choice {
    pub(crate) fn record(
        self,
        scope: &Scope,
        weight: &Arc<Tensor<bf16>>,
        input: &TensorView<'_, f32>,
        output: &mut Tensor<f32>,
        columns: usize,
        tile: LinearTiling,
    ) -> StdResult<(), DeviceError> {
        let generics = vec![
            bf16::DTYPE.as_str().into(),
            tile.rows().to_string(),
            tile.columns().to_string(),
            columns.to_string(),
        ];
        match self {
            Self::Gemv => {
                scope.record(
                    pool_gemv::kernels::dense(output.partition([1, tile.rows()]), input, weight)
                        .generics(generics),
                )?;
            }
            Self::Shared => {
                scope.record(
                    batched::dense(output.partition([SHARED_ROWS, tile.rows()]), input, weight)
                        .generics(generics),
                )?;
            }
            Self::Native => {
                scope.record(
                    small_gemm_kernels::dense(output.partition(SMALL_GEMM_TILE), input, weight)
                        .generics(vec![bf16::DTYPE.as_str().into(), columns.to_string()]),
                )?;
            }
        }
        Ok(())
    }
}
/// Only the unmeasured/disabled path retains the previously admitted selection.
/// Automatic choices come from the measured geometry, not this compatibility fallback.
pub(crate) const fn legacy(rows: usize) -> Choice {
    if rows <= CB_DECODE_SLOTS {
        Choice::Gemv
    } else {
        Choice::Native
    }
}
pub(crate) const fn supported(shape: ProjectionShape) -> bool {
    let [m, n, k] = shape;
    m > 0 && n > 0 && n <= MAX_SMALL_OUTPUTS && k >= MIN_SMALL_COLUMNS
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Key {
    pub native: ResidentGemmKey,
    pub tile: [usize; 2],
    pub baseline: Choice,
}
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Binding {
    weight: usize,
    shape: ProjectionShape,
    tile: [usize; 2],
}
impl Binding {
    pub(crate) fn new(
        weight: &Arc<Tensor<bf16>>,
        shape: ProjectionShape,
        tile: LinearTiling,
    ) -> Self {
        Self {
            weight: Arc::as_ptr(weight) as usize,
            shape,
            tile: [tile.rows(), tile.columns()],
        }
    }
}
pub(crate) const fn kind(tile: LinearTiling) -> ProjectionKind {
    ProjectionKind::PoolDense {
        output_tile: tile.rows(),
        reduction_tile: tile.columns(),
    }
}
fn candidates(shape: ProjectionShape) -> impl Iterator<Item = Choice> {
    [Choice::Gemv, Choice::Shared]
        .into_iter()
        .filter(move |choice| *choice != legacy(shape[0]))
}
pub(crate) fn search_trials(shape: ProjectionShape) -> usize {
    candidates(shape).count() + CONFIRMATIONS
}
type Memo = Mutex<BTreeMap<Key, Choice>>;
fn memo() -> &'static Memo {
    static MEMO: OnceLock<Memo> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(BTreeMap::new()))
}
struct Proof {
    weight: Weak<Tensor<bf16>>,
    accepted: bool,
}
type Proofs = Mutex<BTreeMap<(Key, usize), Proof>>;
fn proofs() -> &'static Proofs {
    static PROOFS: OnceLock<Proofs> = OnceLock::new();
    PROOFS.get_or_init(|| Mutex::new(BTreeMap::new()))
}
fn verified(
    device: &CudaDevice,
    weight: &Arc<Tensor<bf16>>,
    key: &Key,
    choice: Choice,
) -> Result<Choice> {
    // The row GEMV is the canonical private arithmetic; Native is only the admitted legacy path.
    if choice != Choice::Shared {
        return Ok(choice);
    }
    let binding = (key.clone(), Arc::as_ptr(weight) as usize);
    let accepted = proofs()
        .lock()
        .map_err(|_| Error::invariant("pooled dense proof cache poisoned"))?
        .get(&binding)
        .filter(|p| p.weight.strong_count() > 0)
        .map(|p| p.accepted);
    if let Some(accepted) = accepted {
        return Ok(if accepted { choice } else { key.baseline });
    }
    let shape = [key.native.m, key.native.n, key.native.k];
    let tile = capture::tile(key)?;
    let reference = Captured::new(device, weight, shape, tile, Choice::Gemv)?;
    let candidate = Captured::new(device, weight, shape, tile, choice)?;
    reference.run(device)?;
    candidate.run(device)?;
    let accepted = verify(&candidate.output(device)?, &reference.output(device)?).is_ok();
    if !accepted {
        tracing::warn!(target:"infer::tuning",?shape,?choice,
            "pooled dense cached choice rejected by complete real-weight proof");
    }
    let mut proofs = proofs()
        .lock()
        .map_err(|_| Error::invariant("pooled dense proof cache poisoned"))?;
    proofs.retain(|_, proof| proof.weight.strong_count() > 0);
    proofs.insert(
        binding,
        Proof {
            weight: Arc::downgrade(weight),
            accepted,
        },
    );
    drop(proofs);
    Ok(if accepted { choice } else { key.baseline })
}
pub(crate) fn select(
    device: &CudaDevice,
    weight: &Arc<Tensor<bf16>>,
    shape: ProjectionShape,
    tile: LinearTiling,
    autotune: bool,
    session: &ResidentTuningSession,
) -> Choice {
    match resolve(device, weight, shape, tile, autotune, session) {
        Ok(choice) => choice,
        Err(error) => {
            tracing::warn!(target:"infer::tuning",?shape,%error,"pooled dense legacy fallback");
            legacy(shape[0])
        }
    }
}
fn resolve(
    device: &CudaDevice,
    weight: &Arc<Tensor<bf16>>,
    shape: ProjectionShape,
    tile: LinearTiling,
    autotune: bool,
    session: &ResidentTuningSession,
) -> Result<Choice> {
    if !supported(shape) {
        return Err(Error::invalid("unsupported pooled dense geometry"));
    }
    let key = key(device, shape, tile)?;
    let cached = memo()
        .lock()
        .map_err(|_| Error::invariant("pooled dense cache poisoned"))?
        .get(&key)
        .copied();
    if let Some(choice) = cached {
        let choice = verified(device, weight, &key, choice)?;
        session.complete_projection(kind(tile), shape)?;
        return Ok(choice);
    }
    let path = table_path();
    let table = if key.native.driver_build.is_some() {
        resolve_table(path.as_deref())?
    } else {
        None
    };
    let cached = table
        .as_ref()
        .and_then(|table| {
            table
                .resident_pool_dense
                .iter()
                .find(|entry| entry.key == key && entry.valid())
        })
        .cloned();
    let (choice, source) = if let Some(entry) = cached {
        (entry.selected, "table-cache")
    } else if !autotune {
        return Ok(key.baseline);
    } else {
        let Some(entry) = tune(device, weight, key.clone(), session)? else {
            return Ok(key.baseline);
        };
        let choice = entry.selected;
        let cached = key.native.driver_build.is_some()
            && update_table(path.as_deref(), |table| {
                table.resident_pool_dense.retain(|old| old.key != entry.key);
                table.resident_pool_dense.push(entry.clone());
            });
        tracing::info!(target:"infer::tuning",?shape,?choice,cached,
            cold_speedup=entry.confirmation.cold.speedup,
            warm_speedup=entry.confirmation.warm.speedup,"pooled dense measured crossover");
        (choice, "measured")
    };
    memo()
        .lock()
        .map_err(|_| Error::invariant("pooled dense cache poisoned"))?
        .insert(key.clone(), choice);
    let choice = verified(device, weight, &key, choice)?;
    session.complete_projection(kind(tile), shape)?;
    tracing::info!(target:"infer::tuning",?shape,?choice,source,"pooled dense choice");
    Ok(choice)
}
struct Search<'a> {
    device: &'a CudaDevice,
    weight: &'a Arc<Tensor<bf16>>,
    key: &'a Key,
    tile: LinearTiling,
    baseline: &'a Captured,
    legacy_expected: &'a [f32],
    private_expected: &'a [f32],
}
impl Search<'_> {
    fn trial(&self, choice: Choice, order: MeasurementOrder) -> Result<Trial> {
        let shape = [self.key.native.m, self.key.native.n, self.key.native.k];
        let candidate = Captured::new(self.device, self.weight, shape, self.tile, choice)?;
        candidate.run(self.device)?;
        let expected = if choice == Choice::Native {
            self.legacy_expected
        } else {
            self.private_expected
        };
        verify(&candidate.output(self.device)?, expected)?;
        let trial = Trial {
            choice,
            order,
            cold: measure_graphs(
                self.device,
                &self.baseline.graph,
                &candidate.graph,
                true,
                order,
            )?,
            warm: measure_graphs(
                self.device,
                &self.baseline.graph,
                &candidate.graph,
                false,
                order,
            )?,
        };
        verify(&candidate.output(self.device)?, expected)?;
        self.check_baseline()?;
        Ok(trial)
    }
    fn check_baseline(&self) -> Result<()> {
        self.baseline.run(self.device)?;
        verify(&self.baseline.output(self.device)?, self.legacy_expected)
    }
}
fn tune(
    device: &CudaDevice,
    weight: &Arc<Tensor<bf16>>,
    key: Key,
    session: &ResidentTuningSession,
) -> Result<Option<Entry>> {
    let compile = CompileOptions::default();
    if cfg!(debug_assertions)
        || compile.device_debug
        || compile.opt_level.is_some_and(|opt| opt != JIT_OPT_LEVEL)
    {
        return Err(Error::invalid(
            "pooled dense tuning requires release and CUDA JIT O3",
        ));
    }
    let shape = [key.native.m, key.native.n, key.native.k];
    let tile = capture::tile(&key)?;
    session.plan_projection(kind(tile), shape, search_trials(shape))?;
    let _search = session.projection_search()?;
    if !session.permits_search() {
        return Ok(None);
    }
    let baseline = Captured::new(device, weight, shape, tile, key.baseline)?;
    let reference = Captured::new(device, weight, shape, tile, Choice::Gemv)?;
    baseline.run(device)?;
    reference.run(device)?;
    let legacy_expected = baseline.output(device)?;
    let private_expected = reference.output(device)?;
    let search = Search {
        device,
        weight,
        key: &key,
        tile,
        baseline: &baseline,
        legacy_expected: &legacy_expected,
        private_expected: &private_expected,
    };
    let mut selected = key.baseline;
    let mut best = 1.0_f32;
    let mut trials = Vec::new();
    let mut rejected = Vec::new();
    for choice in candidates(shape) {
        if !session.permits_search() {
            return Ok(None);
        }
        match search.trial(choice, MeasurementOrder::BaselineFirst) {
            Ok(trial) => {
                let score = trial.cold.speedup.min(trial.warm.speedup);
                if trial.confirmed() && score > best {
                    best = score;
                    selected = choice;
                }
                trials.push(trial);
            }
            Err(error) => {
                tracing::warn!(target:"infer::tuning",?shape,?choice,%error,
                    "pooled dense candidate rejected; baseline rechecked");
                rejected.push((choice, error.to_string()));
                search.check_baseline()?;
            }
        }
    }
    if !session.permits_search() {
        return Ok(None);
    }
    let mut confirmation = search.trial(selected, MeasurementOrder::CandidateFirst)?;
    if selected != key.baseline && !confirmation.confirmed() {
        if !session.permits_search() {
            return Ok(None);
        }
        selected = key.baseline;
        confirmation = search.trial(selected, MeasurementOrder::CandidateFirst)?;
    }
    let entry = Entry {
        selected,
        key,
        exact_candidates_verified: true,
        trials,
        rejected,
        confirmation,
    };
    if !entry.valid() {
        return Err(Error::invariant("invalid pooled dense confirmation"));
    }
    Ok(Some(entry))
}

#[cfg(test)]
#[path = "../../tests/unit/pool_dense_tuning.rs"]
mod tests;

#[cfg(test)]
std::thread_local! {
    static PRIVATE_REFERENCE: Cell<bool> = const { Cell::new(false) };
}
#[cfg(test)]
pub(crate) fn recorded_choice(choice: Choice) -> Choice {
    if PRIVATE_REFERENCE.with(Cell::get) {
        Choice::Gemv
    } else {
        choice
    }
}
#[cfg(test)]
pub(crate) fn with_private_reference<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            PRIVATE_REFERENCE.with(|enabled| enabled.set(self.0));
        }
    }
    let _restore = Restore(PRIVATE_REFERENCE.with(|enabled| enabled.replace(true)));
    run()
}
