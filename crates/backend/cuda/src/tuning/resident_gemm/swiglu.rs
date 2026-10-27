//! Actual-weight, complete-code `SwiGLU` tile search with reverse confirmation.
mod capture;
mod evidence;
use super::{
    GemmConfig, GemmTrial, JIT_OPT_LEVEL, MeasurementOrder, PairedTimes, ResidentGemmKey,
    ResidentTuningSession, Verification, WeightKey, confirmed, ordered_times,
};
use crate::device::{CudaDevice, device_error};
use crate::mlp::ProjectionWeight;
use crate::tuning::projection_budget::{ProjectionKind, ProjectionShape};
use crate::tuning::{
    MAX_REPS, MIN_REPS, REP_MS, WARMUP_MS, resolve_table, table_path, update_table,
};
pub use capture::record;
use capture::{Captured, key, verify};
use cutile::bench::{BenchOptions, do_bench_paired};
use cutile::cutile_compiler::hints::CompileOptions;
use cutile::error::tensor_error;
use cutile::prelude::*;
pub use evidence::Entry;
use evidence::candidates;
pub use evidence::{baseline, compatible};
use infer_core::{Error, Result};
#[cfg(test)]
use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

/// Exact immutable operands and scalar values bound to one prompt geometry.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Binding {
    weights: [WeightKey; 2],
    down_scale: u32,
    shape: ProjectionShape,
}
impl Binding {
    #[cfg(test)]
    pub const fn shape(&self) -> ProjectionShape {
        self.shape
    }
    pub fn new(
        weights: (&ProjectionWeight, &ProjectionWeight),
        scales: (f32, f32),
        shape: ProjectionShape,
    ) -> Result<Self> {
        Ok(Self {
            weights: [
                WeightKey::new(weights.0, scales.0)?,
                WeightKey::new(weights.1, scales.0)?,
            ],
            down_scale: scales.1.to_bits(),
            shape,
        })
    }
}
/// One independent confirmation plus a baseline confirmation if a candidate is rejected.
const CONFIRMATIONS: usize = 2;
pub fn search_trials(shape: ProjectionShape) -> usize {
    candidates(shape).count().saturating_sub(1) + CONFIRMATIONS
}
type Memo = Mutex<BTreeMap<ResidentGemmKey, GemmConfig>>;
fn memo() -> &'static Memo {
    static MEMO: OnceLock<Memo> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(BTreeMap::new()))
}
type Proofs = Mutex<BTreeMap<(ResidentGemmKey, Binding, GemmConfig), [Verification; 2]>>;
fn proofs() -> &'static Proofs {
    static PROOFS: OnceLock<Proofs> = OnceLock::new();
    PROOFS.get_or_init(|| Mutex::new(BTreeMap::new()))
}
fn alive(proof: &[Verification; 2]) -> bool {
    proof
        .iter()
        .all(|p| p.codes.strong_count() > 0 && p.scales.strong_count() > 0)
}
fn weak(weight: &ProjectionWeight, accepted: bool) -> Result<Verification> {
    let ProjectionWeight::Fp4(codes, scales, _) = weight else {
        return Err(Error::invalid("SwiGLU verification weight"));
    };
    Ok(Verification {
        codes: Arc::downgrade(codes),
        scales: Arc::downgrade(scales),
        accepted,
    })
}
fn verified(
    device: &CudaDevice,
    weights: (&ProjectionWeight, &ProjectionWeight),
    scales: (f32, f32),
    key: &ResidentGemmKey,
    config: GemmConfig,
) -> Result<GemmConfig> {
    if config == baseline() {
        return Ok(config);
    }
    let shape = [key.m, key.n, key.k];
    let binding = (key.clone(), Binding::new(weights, scales, shape)?, config);
    let accepted = proofs()
        .lock()
        .map_err(|_| Error::invariant("SwiGLU proof cache poisoned"))?
        .get(&binding)
        .filter(|proof| alive(proof))
        .map(|proof| proof[0].accepted);
    if let Some(accepted) = accepted {
        return Ok(if accepted { config } else { baseline() });
    }
    let base = Captured::new(device, weights, scales, shape, baseline())?;
    let candidate = Captured::new(device, weights, scales, shape, config)?;
    base.run(device)?;
    let expected = base.bytes(device)?;
    candidate.run(device)?;
    let accepted = verify(&candidate.bytes(device)?, &expected).is_ok();
    if !accepted {
        tracing::warn!(target: "infer::tuning", ?shape, ?config,
            "SwiGLU cached tile rejected by complete real-weight code/scale proof");
    }
    let mut proofs = proofs()
        .lock()
        .map_err(|_| Error::invariant("SwiGLU proof cache poisoned"))?;
    proofs.retain(|_, proof| alive(proof));
    proofs.insert(
        binding,
        [weak(weights.0, accepted)?, weak(weights.1, accepted)?],
    );
    drop(proofs);
    Ok(if accepted { config } else { baseline() })
}
pub fn select(
    device: &CudaDevice,
    weights: (&ProjectionWeight, &ProjectionWeight),
    scales: (f32, f32),
    shape: ProjectionShape,
    autotune: bool,
    session: &ResidentTuningSession,
) -> GemmConfig {
    match resolve(device, weights, scales, shape, autotune, session) {
        Ok(config) => config,
        Err(error) => {
            tracing::warn!(target: "infer::tuning", ?shape, %error,
                "fused SwiGLU conservative tile fallback");
            baseline()
        }
    }
}
fn resolve(
    device: &CudaDevice,
    weights: (&ProjectionWeight, &ProjectionWeight),
    scales: (f32, f32),
    shape: ProjectionShape,
    autotune: bool,
    session: &ResidentTuningSession,
) -> Result<GemmConfig> {
    if !compatible(baseline(), shape) {
        return Err(Error::invalid("SwiGLU fusion geometry"));
    }
    let key = key(device, shape)?;
    let cached = memo()
        .lock()
        .map_err(|_| Error::invariant("SwiGLU cache poisoned"))?
        .get(&key)
        .copied();
    if let Some(config) = cached {
        let config = verified(device, weights, scales, &key, config)?;
        session.complete_projection(ProjectionKind::Swiglu, shape)?;
        return Ok(config);
    }
    let path = table_path();
    let table = if key.driver_build.is_some() {
        resolve_table(path.as_deref())?
    } else {
        None
    };
    let entry = table
        .as_ref()
        .and_then(|table| {
            table
                .resident_swiglu
                .iter()
                .find(|entry| entry.key == key && entry.valid())
        })
        .cloned();
    let (config, source) = if let Some(entry) = entry {
        (entry.selected, "table-cache")
    } else if !autotune {
        return Ok(baseline());
    } else {
        let Some(entry) = tune(device, weights, scales, key.clone(), session)? else {
            return Ok(baseline());
        };
        let config = entry.selected;
        let cached = key.driver_build.is_some()
            && update_table(path.as_deref(), |table| {
                table.resident_swiglu.retain(|old| old.key != entry.key);
                table.resident_swiglu.push(entry.clone());
            });
        tracing::info!(target: "infer::tuning", ?shape, ?config, cached,
            cold_speedup=entry.confirmation.cold.speedup,
            warm_speedup=entry.confirmation.warm.speedup, "fused SwiGLU measured tile");
        (config, "measured")
    };
    memo()
        .lock()
        .map_err(|_| Error::invariant("SwiGLU cache poisoned"))?
        .insert(key.clone(), config);
    let config = verified(device, weights, scales, &key, config)?;
    session.complete_projection(ProjectionKind::Swiglu, shape)?;
    tracing::info!(target: "infer::tuning", ?shape, ?config, source, "fused SwiGLU tile selected");
    Ok(config)
}
fn measure(
    device: &CudaDevice,
    baseline: &Captured,
    candidate: &Captured,
    cold: bool,
    order: MeasurementOrder,
) -> Result<PairedTimes> {
    let options = BenchOptions {
        warmup: Duration::from_millis(WARMUP_MS),
        rep: Duration::from_millis(REP_MS),
        min_reps: MIN_REPS,
        max_reps: MAX_REPS,
        clear_l2: cold,
    };
    let (first, second) = match order {
        MeasurementOrder::BaselineFirst => (baseline, candidate),
        MeasurementOrder::CandidateFirst => (candidate, baseline),
        MeasurementOrder::Unspecified => {
            return Err(Error::invalid("SwiGLU measurement arm order"));
        }
    };
    let run = |graph: &Captured| {
        graph
            .graph
            .launch()
            .sync_on(&device.stream)
            .map_err(|error| tensor_error(&error.to_string()))
    };
    let (a, b) = do_bench_paired(&device.stream, &options, |_| run(first), |_| run(second))
        .map_err(device_error)?;
    ordered_times(a.times_ms(), b.times_ms(), order)
}
struct Search<'a> {
    device: &'a CudaDevice,
    weights: (&'a ProjectionWeight, &'a ProjectionWeight),
    scales: (f32, f32),
    shape: ProjectionShape,
    base: &'a Captured,
    expected: &'a (Vec<u8>, Vec<u8>),
}
impl Search<'_> {
    fn trial(&self, config: GemmConfig, order: MeasurementOrder) -> Result<GemmTrial> {
        let candidate = Captured::new(self.device, self.weights, self.scales, self.shape, config)?;
        candidate.run(self.device)?;
        verify(&candidate.bytes(self.device)?, self.expected)?;
        let measured = GemmTrial {
            config,
            order,
            cold: measure(self.device, self.base, &candidate, true, order)?,
            warm: measure(self.device, self.base, &candidate, false, order)?,
        };
        verify(&candidate.bytes(self.device)?, self.expected)?;
        self.check_baseline()?;
        Ok(measured)
    }
    fn check_baseline(&self) -> Result<()> {
        self.base.run(self.device)?;
        verify(&self.base.bytes(self.device)?, self.expected)
    }
}
fn tune(
    device: &CudaDevice,
    weights: (&ProjectionWeight, &ProjectionWeight),
    scales: (f32, f32),
    key: ResidentGemmKey,
    session: &ResidentTuningSession,
) -> Result<Option<Entry>> {
    let compile = CompileOptions::default();
    if cfg!(debug_assertions)
        || compile.device_debug
        || compile.opt_level.is_some_and(|opt| opt != JIT_OPT_LEVEL)
    {
        return Err(Error::invalid(
            "SwiGLU tuning requires release and CUDA JIT O3",
        ));
    }
    let shape = [key.m, key.n, key.k];
    session.plan_projection(ProjectionKind::Swiglu, shape, search_trials(shape))?;
    let _search = session.projection_search()?;
    if session.exhausted(&key) {
        return Ok(None);
    }
    let base = Captured::new(device, weights, scales, shape, baseline())?;
    base.run(device)?;
    let expected = base.bytes(device)?;
    verify(&expected, &expected)?;
    let search = Search {
        device,
        weights,
        scales,
        shape,
        base: &base,
        expected: &expected,
    };
    let mut selected = baseline();
    let mut best = 1.0_f32;
    let mut trials = Vec::new();
    let mut rejected = Vec::new();
    for config in candidates(shape).filter(|config| *config != baseline()) {
        if session.exhausted(&key) {
            return Ok(None);
        }
        match search.trial(config, MeasurementOrder::BaselineFirst) {
            Ok(trial) => {
                let score = trial.cold.speedup.min(trial.warm.speedup);
                if confirmed(&trial) && score > best {
                    best = score;
                    selected = config;
                }
                trials.push(trial);
            }
            Err(error) => {
                tracing::warn!(target: "infer::tuning", ?shape, ?config, %error,
                    "fused SwiGLU candidate rejected; baseline rechecked");
                rejected.push((config, error.to_string()));
                search.check_baseline()?;
            }
        }
    }
    if session.exhausted(&key) {
        return Ok(None);
    }
    let mut confirmation = search.trial(selected, MeasurementOrder::CandidateFirst)?;
    if selected != baseline() && !confirmed(&confirmation) {
        if session.exhausted(&key) {
            return Ok(None);
        }
        selected = baseline();
        confirmation = search.trial(selected, MeasurementOrder::CandidateFirst)?;
    }
    let entry = Entry {
        key,
        selected,
        bitwise_verified: true,
        trials,
        rejected,
        confirmation,
    };
    if !entry.valid() {
        return Err(Error::invariant("incomplete SwiGLU tuning evidence"));
    }
    Ok(Some(entry))
}

#[cfg(test)]
std::thread_local! {
    static BASELINE_ONLY: Cell<bool> = const { Cell::new(false) };
    static CAPTURES: Cell<(usize,usize)> = const { Cell::new((0,0)) };
}
#[cfg(test)]
pub fn capture_counts() -> (usize, usize) {
    CAPTURES.with(Cell::get)
}
#[cfg(test)]
pub fn recorded_config(config: GemmConfig) -> GemmConfig {
    let baseline_only = BASELINE_ONLY.with(Cell::get);
    CAPTURES.with(|count| {
        let (tuned, reference) = count.get();
        count.set(if baseline_only {
            (tuned, reference + 1)
        } else {
            (tuned + 1, reference)
        });
    });
    if baseline_only { baseline() } else { config }
}
#[cfg(test)]
pub fn with_baseline<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            BASELINE_ONLY.with(|enabled| enabled.set(self.0));
        }
    }
    let _restore = Restore(BASELINE_ONLY.with(|enabled| enabled.replace(true)));
    run()
}

#[cfg(test)]
#[path = "../../../tests/unit/resident_swiglu_tuning.rs"]
mod tests;
