//! Automatic architecture and cluster selection for the distinct FA4 BF16 profile.
//! Operator F64 sanity is not a substitute for the unchanged model task-quality gate.
use super::{
    Choice, Key, Shape,
    capture::Captured,
    reference::{self, Fixture},
};
use crate::device::device_error;
use crate::resident::ProgramWeights;
use crate::resident::attention_oxide::shared_identity;
use crate::resident::attention_oxide::vector_identity;
use crate::tuning::MIN_CONFIRM_SPEEDUP;
use crate::tuning::resolve_table;
use crate::tuning::search;
use crate::tuning::table_path;
use crate::{
    device::CudaDevice,
    resident::attention_mma::{
        self,
        catalog::{self, Variant},
    },
    tuning::resident_gemm::ResidentTuningSession,
};
use infer_core::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::{
    collections::BTreeMap,
    sync::{Mutex, OnceLock},
};
pub(super) const MAX_ERROR: f64 = 0.01;
pub(crate) const SCHEMA: &str = "fa4-bf16:full-f64-operator-sanity1e-2:original-append:history0-8k-32k:cluster-search:cold-warm-ab-ba-v1";
const SHA256_HEX_LEN: usize = 64;
const HISTORY_CASES: usize = 3;
const HISTORY_DIVISOR: usize = 4;
const CAPACITY: usize = 32768;
const MAX_SHORT_REGRESSION: f32 = 1.05;
/// A timed history length; stored records name it `history`, which the shared case reads.
pub use crate::tuning::qualify::Case;
/// Each timed history beat the baseline by the confirmation margin cold and warm. The empty
/// history only has to stay within the short-prompt regression bound.
pub(super) fn accepted(case: &Case) -> bool {
    let minimum = if case.case == 0 {
        1.0 / MAX_SHORT_REGRESSION
    } else {
        MIN_CONFIRM_SPEEDUP
    };
    case.valid() && case.cold.speedup() >= minimum && case.warm.speedup() >= minimum
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    pub variant: Variant,
    pub fixtures: usize,
    pub elements: usize,
    pub max_f64_error: f64,
    pub forward: Vec<Case>,
    pub reverse: Vec<Case>,
}
impl Trial {
    pub(super) fn score(&self) -> f32 {
        self.forward
            .iter()
            .chain(&self.reverse)
            .filter(|c| c.case > 0)
            .map(|c| c.cold.speedup().min(c.warm.speedup()))
            .fold(f32::INFINITY, f32::min)
    }
    pub(super) fn accepted(&self) -> bool {
        self.forward.iter().chain(&self.reverse).all(accepted)
    }
    pub(super) fn valid(&self, shape: Shape) -> bool {
        let capacity = capacity(shape);
        let histories = histories(shape, capacity);
        self.fixtures == reference::FIXTURES
            && self.elements == reference::FIXTURES * shape.lanes * shape.heads * shape.dim
            && self.max_f64_error.is_finite()
            && (0.0..MAX_ERROR).contains(&self.max_f64_error)
            && [&self.forward, &self.reverse].iter().all(|cases| {
                cases.len() == histories.len()
                    && cases
                        .iter()
                        .zip(histories)
                        .all(|(c, h)| c.case == h && c.valid())
            })
    }
}
/// The family's records in the shared table section.
pub const FAMILY: &str = "mma_attention";
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    pub key: Key,
    pub architecture: String,
    pub profile: String,
    pub baseline: Choice,
    pub selected: Choice,
    pub trials: Vec<Trial>,
}
impl Entry {
    pub(crate) fn valid(&self) -> bool {
        let shape = self.key.shape;
        let variants = Variant::candidates(&self.architecture);
        self.key.schema == SCHEMA
            && self.profile == catalog::PROFILE
            && serde_json::from_str::<serde_json::Value>(&self.key.hardware)
                .ok()
                .and_then(|v| {
                    v.get("architecture")
                        .and_then(|a| a.as_str())
                        .map(str::to_owned)
                })
                .is_some_and(|a| a == self.architecture)
            && self.key.driver_build.as_ref().is_none_or(|v| !v.is_empty())
            && !self.key.hardware.is_empty()
            && self.key.fingerprint.len() == SHA256_HEX_LEN
            && self.key.fingerprint.bytes().all(|b| b.is_ascii_hexdigit())
            && catalog::supported(
                &self.architecture,
                shape.heads,
                shape.kv_heads,
                shape.dim,
                shape.lanes,
                shape.window,
            )
            && !matches!(self.baseline, Choice::Mma { .. } | Choice::Tiled { .. })
            && self.trials.len() == variants.len()
            && self
                .trials
                .iter()
                .zip(variants)
                .all(|(trial, variant)| trial.variant == *variant && trial.valid(shape))
            && (self.selected == self.baseline
                || self.trials.iter().any(|trial| {
                    self.selected
                        == Choice::Mma {
                            variant: trial.variant,
                        }
                        && trial.accepted()
                }))
    }
}
pub(super) fn capacity(shape: Shape) -> usize {
    CAPACITY.max(shape.lanes.saturating_mul(2))
}
pub(super) const fn histories(shape: Shape, capacity: usize) -> [usize; HISTORY_CASES] {
    [0, capacity / HISTORY_DIVISOR, capacity - shape.lanes]
}
type Memo = Mutex<BTreeMap<Key, Entry>>;
fn memo() -> &'static Memo {
    static MEMO: OnceLock<Memo> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(BTreeMap::new()))
}
/// Cold searches run at model loading, before request resource acknowledgement deadlines.
/// Request capture may only reuse completed, artifact-scoped choices.
pub(crate) fn prepare(
    device: &CudaDevice,
    graph: &infer_ir::DataflowGraph,
    weights: &ProgramWeights,
) -> Result<()> {
    // A target FP16 program never consults the BF16 target search. The draft program and the
    // explicit BF16 profile keep the existing BF16 behavior.
    let target_f16 = weights.attention_profile.is_fp16() && weights.fusion.is_none();
    tracing::info!(
        target: "infer::tuning",
        attention_profile = weights.attention_profile.name(),
        draft = weights.fusion.is_some(),
        schema = if target_f16 { attention_mma::f16::SCHEMA } else { SCHEMA },
        "attention numerical profile"
    );
    // Shape order prepares narrow graphs first, matching resident capture order.
    for (shape, fp8, mma_eligible, f16_eligible) in prepared_shapes(device, graph, weights)? {
        if !fp8 && weights.fusion.is_some() {
            super::mma_f32::select(device, shape, weights.autotune, &weights.resident_tuning);
        }
        if !fp8 {
            continue;
        }
        let baseline =
            super::exact::select(device, shape, weights.autotune, &weights.resident_tuning);
        let baseline = super::shared::select(
            device,
            shape,
            weights.autotune,
            &weights.resident_tuning,
            baseline,
        );
        let baseline = super::vector::select(
            device,
            shape,
            weights.autotune,
            &weights.resident_tuning,
            baseline,
        );
        let baseline = super::vector_combined::select(device, shape, baseline);
        // Exact target tuning must finish before proposal search starts the shared
        // wall-clock allowance. Only the shared eligibility decision may narrow precision.
        if f16_eligible {
            if let Err(error) = attention_mma::f16::select(device, shape) {
                tracing::warn!(target:"infer::tuning",?shape,%error,"FP16 target attention unavailable: canonical fallback");
            }
        } else if mma_eligible {
            select(
                device,
                shape,
                weights.autotune,
                &weights.resident_tuning,
                baseline,
            );
        }
    }
    Ok(())
}

/// Enumerate the loaded program's attention geometries before selecting kernels.
fn prepared_shapes(
    device: &CudaDevice,
    graph: &infer_ir::DataflowGraph,
    weights: &ProgramWeights,
) -> Result<BTreeSet<(Shape, bool, bool, bool)>> {
    let architecture = &device.profile()?.architecture;
    let target_f16 = weights.attention_profile.is_fp16() && weights.fusion.is_none();
    let mut shapes = BTreeSet::new();
    for lanes in weights.prompt_widths() {
        let mma_eligible = attention_mma::eligible(architecture, graph, weights, lanes);
        for node in &graph.nodes {
            let infer_ir::TensorOp::Attention {
                query_heads,
                kv_heads,
                head_dim,
                window,
            } = node.op
            else {
                continue;
            };
            if !node.states.is_empty()
                && catalog::supported(
                    architecture,
                    query_heads,
                    kv_heads,
                    head_dim,
                    lanes,
                    window.unwrap_or(0),
                )
            {
                let fp8 = node
                    .states
                    .first()
                    .is_some_and(|id| weights.kv_scales.contains_key(id));
                let f16_eligible = target_f16
                    && fp8
                    && attention_mma::f16::supported(
                        architecture,
                        query_heads,
                        kv_heads,
                        head_dim,
                        lanes,
                        window.unwrap_or(0),
                    );
                shapes.insert((
                    Shape {
                        heads: query_heads,
                        kv_heads,
                        dim: head_dim,
                        lanes,
                        window: window.unwrap_or(0),
                    },
                    fp8,
                    mma_eligible && !target_f16,
                    f16_eligible,
                ));
            }
        }
    }
    Ok(shapes)
}

pub(crate) fn select(
    device: &CudaDevice,
    shape: Shape,
    autotune: bool,
    session: &ResidentTuningSession,
    baseline: Choice,
) -> Choice {
    match resolve(device, shape, autotune, session, baseline) {
        Ok(choice) => choice,
        Err(error) => {
            tracing::warn!(target:"infer::tuning",?shape,%error,"MMA attention unavailable: canonical fallback");
            baseline
        }
    }
}
fn resolve(
    device: &CudaDevice,
    shape: Shape,
    autotune: bool,
    session: &ResidentTuningSession,
    baseline: Choice,
) -> Result<Choice> {
    let architecture = &device.profile()?.architecture;
    if !catalog::supported(
        architecture,
        shape.heads,
        shape.kv_heads,
        shape.dim,
        shape.lanes,
        shape.window,
    ) {
        return Ok(baseline);
    }
    let mut key = super::exact::key(device, shape)?;
    key.schema = SCHEMA.into();
    key.fingerprint = format!(
        "{:x}",
        Sha256::digest(format!(
            "{}:{}:{}:{}:{}",
            key.fingerprint,
            shared_identity(),
            vector_identity(),
            attention_mma::artifact_identity(),
            serde_json::to_string(&baseline).map_err(device_error)?
        ))
    );
    if let Some(entry) = memo()
        .lock()
        .map_err(|_| Error::invariant("MMA memo poisoned"))?
        .get(&key)
    {
        return Ok(entry.selected);
    }
    let path = table_path();
    let table = if key.driver_build.is_some() {
        resolve_table(path.as_deref())?
    } else {
        None
    };
    if let Some(entry) = table
        .and_then(|table| search::find(&table, FAMILY, &key, |e: &Entry| e.baseline == baseline))
    {
        if !entry.valid() {
            return Err(Error::invalid("incomplete MMA profile/cluster evidence"));
        }
        let choice = entry.selected;
        memo()
            .lock()
            .map_err(|_| Error::invariant("MMA memo poisoned"))?
            .insert(key, entry);
        tracing::info!(target:"infer::tuning",?shape,?choice,profile=catalog::PROFILE,source="table-cache","MMA attention selected");
        return Ok(choice);
    }
    // Tensor-core attention is the default arithmetic for these programs; a shape without a
    // measured entry takes the architecture's first variant instead of the scalar baseline,
    // which is several times slower at every prompt width.
    let untuned = || {
        let choice = Variant::candidates(architecture)
            .first()
            .map_or(baseline, |&variant| Choice::Mma { variant });
        tracing::info!(target:"infer::tuning",?shape,?choice,profile=catalog::PROFILE,source="default","MMA attention selected");
        choice
    };
    if !autotune || !session.permits_attention_search() || cfg!(debug_assertions) {
        return Ok(untuned());
    }
    let Some(entry) = tune(device, key.clone(), session, baseline)? else {
        return Ok(untuned());
    };
    let cached = key.driver_build.is_some() && search::put(path.as_deref(), FAMILY, &key, &entry);
    let choice = entry.selected;
    tracing::info!(target:"infer::tuning",?shape,?choice,cached,profile=catalog::PROFILE,"MMA attention hardware choice");
    memo()
        .lock()
        .map_err(|_| Error::invariant("MMA memo poisoned"))?
        .insert(key, entry);
    Ok(choice)
}
fn cases(
    device: &CudaDevice,
    shape: Shape,
    baseline: Choice,
    variant: Variant,
    reverse: bool,
    session: &ResidentTuningSession,
) -> Result<Option<Vec<Case>>> {
    let mut result = Vec::new();
    let capacity = capacity(shape);
    for history in histories(shape, capacity) {
        if !session.permits_attention_search() {
            return Ok(None);
        }
        let fixture = Fixture::history(shape, history, capacity)?;
        let base = Captured::new(device, &fixture, baseline)?;
        let candidate = Captured::new(device, &fixture, Choice::Mma { variant })?;
        candidate.verify_mma_history(device, &fixture)?;
        result.push(Case {
            case: history,
            cold: super::measure(device, &base, &candidate, true, reverse)?,
            warm: super::measure(device, &base, &candidate, false, reverse)?,
        });
        candidate.verify_mma_history(device, &fixture)?;
    }
    Ok(Some(result))
}
fn tune(
    device: &CudaDevice,
    key: Key,
    session: &ResidentTuningSession,
    baseline: Choice,
) -> Result<Option<Entry>> {
    let architecture = device.profile()?.architecture.clone();
    let mut trials = Vec::new();
    let mut selected = baseline;
    let mut best = 1.0;
    for &variant in Variant::candidates(&architecture) {
        let mut max_error = 0.0_f64;
        let mut elements = 0;
        for index in 0..reference::FIXTURES {
            if !session.permits_attention_search() {
                return Ok(None);
            }
            let fixture = Fixture::mma(key.shape, index)?;
            let candidate = Captured::new(device, &fixture, Choice::Mma { variant })?;
            max_error = max_error.max(candidate.verify_mma(device, &fixture)?);
            elements += fixture.expected.len();
        }
        let Some(forward) = cases(device, key.shape, baseline, variant, false, session)? else {
            return Ok(None);
        };
        let Some(reverse) = cases(device, key.shape, baseline, variant, true, session)? else {
            return Ok(None);
        };
        let trial = Trial {
            variant,
            fixtures: reference::FIXTURES,
            elements,
            max_f64_error: max_error,
            forward,
            reverse,
        };
        if trial.accepted() && trial.score() > best {
            best = trial.score();
            selected = Choice::Mma { variant };
        }
        trials.push(trial);
    }
    let entry = Entry {
        key,
        architecture,
        profile: catalog::PROFILE.into(),
        baseline,
        selected,
        trials,
    };
    if !entry.valid() {
        return Err(Error::invariant("incomplete MMA tuning evidence"));
    }
    Ok(Some(entry))
}

#[cfg(test)]
#[path = "../../../tests/unit/tuning_attention_mma.rs"]
mod tests;
