//! Native gated norm compiler occupancy, selected before capture from measured evidence.
mod capture;
mod evidence;
use self::{
    capture::Buffers,
    evidence::{Entry, Key, Shape},
};
use super::resident_gemm::ResidentTuningSession;
use crate::constants::PREFILL_LANES;
use crate::{device::CudaDevice, resident::ProgramWeights};
use cutile::tile_kernel::CompileOptions;
pub use evidence::Entry as NormEntry;
use infer_core::{Error, Result};
use infer_ir::{DataflowGraph, TensorOp};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Mutex, OnceLock},
};

/// Only this per-head reduction has complete native-equivalence and real-model evidence.
const SUPPORTED_DIMENSION: usize = 128;
/// Zero denotes the original compiler policy, rather than an occupancy hint.
const BASELINE: i32 = 0;
const CANDIDATES: [i32; 3] = [2, 4, 6];
/// Hint for a prompt shape without a measured entry: occupancy 4 won every measured prompt shape
/// (2.0-2.8x over the compiler policy) with identical bits.
const UNMEASURED: i32 = 4;
const SCHEMA: &str = "native-gated-norm-occupancy:same-address:finite-all-bits:cold-warm-AB-BA-v1";
type ProcessKey = (usize, Shape);
#[derive(Clone, Copy)]
struct Prepared {
    occupancy: i32,
    measured: bool,
}
type Memo = Mutex<BTreeMap<ProcessKey, Prepared>>;
fn memo() -> &'static Memo {
    static MEMO: OnceLock<Memo> = OnceLock::new();
    MEMO.get_or_init(|| Mutex::new(BTreeMap::new()))
}
fn process_key(device: &CudaDevice, shape: Shape) -> ProcessKey {
    (device.stream.device().ordinal(), shape)
}
pub(crate) fn compile_options(occupancy: i32) -> CompileOptions {
    if occupancy == BASELINE {
        CompileOptions::default()
    } else {
        CompileOptions::default().occupancy(occupancy)
    }
}
/// Capture only consults already-prepared process state; it never measures or reads files.
pub(crate) fn options(
    device_id: usize,
    dimension: usize,
    rows: usize,
    epsilon: f32,
) -> CompileOptions {
    let Some(shape) = Shape::new(rows, dimension, epsilon) else {
        return CompileOptions::default();
    };
    let occupancy = memo()
        .lock()
        .ok()
        .and_then(|m| m.get(&(device_id, shape)).map(|p| p.occupancy))
        .unwrap_or(BASELINE);
    compile_options(occupancy)
}
/// Prompt widths come from the actual program geometry; ordinary decode keeps its old policy.
pub(crate) fn prepare(
    device: &CudaDevice,
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    widths: &[usize],
) {
    let mut shapes = BTreeSet::new();
    for node in &graph.nodes {
        let TensorOp::GatedNorm { head_dim, epsilon } = node.op else {
            continue;
        };
        if head_dim != SUPPORTED_DIMENSION {
            continue;
        }
        let Some(spec) = node
            .outputs
            .first()
            .and_then(|id| graph.tensors.iter().find(|t| t.id == *id))
        else {
            continue;
        };
        let Some(elements) = spec.shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) else {
            continue;
        };
        if !elements.is_multiple_of(head_dim) {
            continue;
        }
        for &width in widths.iter().filter(|w| **w >= PREFILL_LANES) {
            if let Some(rows) = width.checked_mul(elements / head_dim)
                && let Some(shape) = Shape::new(rows, head_dim, epsilon)
            {
                shapes.insert(shape);
            }
        }
    }
    for shape in shapes {
        if let Err(error) = resolve(device, shape, weights.autotune, &weights.resident_tuning) {
            tracing::warn!(target:"infer::tuning",?shape,%error,"gated norm original compiler fallback");
            if let Ok(mut memo) = memo().lock() {
                memo.insert(
                    process_key(device, shape),
                    Prepared {
                        occupancy: BASELINE,
                        measured: false,
                    },
                );
            }
        }
    }
}
fn resolve(
    device: &CudaDevice,
    shape: Shape,
    autotune: bool,
    session: &ResidentTuningSession,
) -> Result<()> {
    let local = process_key(device, shape);
    if let Some(prepared) = memo()
        .lock()
        .map_err(|_| Error::invariant("norm memo poisoned"))?
        .get(&local)
        && (prepared.measured || !autotune)
    {
        return Ok(());
    }
    let buffers = Buffers::metadata(device, shape)?;
    let key = buffers.key(device, shape)?;
    let path = super::table_path();
    let table = if key.driver_build.is_some() {
        super::resolve_table(path.as_deref())?
    } else {
        None
    };
    let cached = table
        .as_ref()
        .and_then(|t| {
            t.resident_gated_norm
                .iter()
                .find(|e| e.key == key && e.valid())
        })
        .cloned();
    let entry = if cached.is_some() {
        cached
    } else if autotune && session.permits_norm_search() {
        tune(device, key.clone(), session)?
    } else {
        None
    };
    let selected = entry.as_ref().map_or(UNMEASURED, |e| e.selected);
    if let Some(entry) = &entry {
        let from_table = table.as_ref().is_some_and(|t| {
            t.resident_gated_norm
                .iter()
                .any(|e| e.key == key && e.valid())
        });
        if !from_table && key.driver_build.is_some() {
            super::update_table(path.as_deref(), |table| {
                table.resident_gated_norm.retain(|old| old.key != key);
                table.resident_gated_norm.push(entry.clone());
            });
        }
        tracing::info!(target:"infer::tuning",?shape,occupancy=selected,from_table,
            cold_speedup=entry.confirmation.cold.speedup(),warm_speedup=entry.confirmation.warm.speedup(),
            "gated norm compiler selected");
    }
    memo()
        .lock()
        .map_err(|_| Error::invariant("norm memo poisoned"))?
        .insert(
            local,
            Prepared {
                occupancy: selected,
                measured: entry.is_some(),
            },
        );
    Ok(())
}
fn tune(device: &CudaDevice, key: Key, session: &ResidentTuningSession) -> Result<Option<Entry>> {
    let compile = CompileOptions::default();
    if cfg!(debug_assertions)
        || compile.device_debug
        || compile
            .opt_level
            .is_some_and(|n| n != capture::JIT_OPT_LEVEL)
    {
        return Err(Error::invalid(
            "gated norm timing requires release CUDA JIT O3",
        ));
    }
    let buffers = Buffers::new(device, key.shape)?;
    let baseline = buffers.capture(device, BASELINE)?;
    let expected = buffers.read(device, &baseline)?;
    capture::bits(&expected, &expected)?;
    let mut search = Vec::new();
    let mut selected = BASELINE;
    let mut best = 1.0f32;
    for occupancy in CANDIDATES {
        if !session.permits_norm_search() {
            return Ok(None);
        }
        let candidate = buffers.capture(device, occupancy)?;
        capture::bits(&buffers.read(device, &candidate)?, &expected)?;
        let trial = capture::measure(
            device,
            &baseline,
            &candidate,
            occupancy,
            false,
            expected.len(),
        )?;
        capture::bits(&buffers.read(device, &candidate)?, &expected)?;
        if trial.confirmed() && trial.score() > best {
            selected = occupancy;
            best = trial.score();
        }
        search.push(trial);
    }
    if !session.permits_norm_search() {
        return Ok(None);
    }
    let candidate = buffers.capture(device, selected)?;
    capture::bits(&buffers.read(device, &candidate)?, &expected)?;
    let mut confirmation = capture::measure(
        device,
        &baseline,
        &candidate,
        selected,
        true,
        expected.len(),
    )?;
    capture::bits(&buffers.read(device, &candidate)?, &expected)?;
    if selected != BASELINE && !confirmation.confirmed() {
        selected = BASELINE;
        confirmation =
            capture::measure(device, &baseline, &baseline, BASELINE, true, expected.len())?;
    }
    let entry = Entry {
        key,
        selected,
        search,
        confirmation,
    };
    if !entry.valid() {
        return Err(Error::invariant("incomplete gated norm evidence"));
    }
    Ok(Some(entry))
}

#[cfg(test)]
#[path = "../../tests/unit/tuning_gated_norm.rs"]
mod tests;
