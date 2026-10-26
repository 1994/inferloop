//! Retained, shape-derived scratch for pooled GDN verification pairs.
use super::{ProgramWeights, batch::gdn_ba_pairs, capture::error, pool_gemv, small_gemm::kernels};
use crate::constants::{F32_BYTES, SMALL_GEMM_TILE, SMALL_GEMM_TILE_K};
use crate::device::{CudaDevice, DeviceProfile, device_error};
use crate::mlp::ProjectionWeight;
use cutile::half::bf16;
use cutile::prelude::*;
use infer_core::{Error, Result};
use infer_ir::DataflowGraph;
#[cfg(test)]
use std::cell::Cell;
use std::collections::BTreeMap;
use std::result::Result as StdResult;

/// Bounds the retained pair workspace over all distinct shapes in one pool.
const SCRATCH_LIMIT: usize = 16 * 1024 * 1024;
/// Keeps the bounded K-window search independent of model names.
const SPLIT_LIMIT: usize = 64;
/// Beta and alpha have independent partial accumulators.
const PAIR_OUTPUTS: usize = 2;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Shape {
    rows: usize,
    outputs: usize,
    columns: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Plan {
    pub splits: usize,
    pub k_tiles: usize,
    pub shape: [usize; 2],
    pub scratch_bytes: usize,
}

pub(super) fn plan_for(rows: usize, outputs: usize, columns: usize, sms: usize) -> Option<Plan> {
    if rows == 0 || outputs == 0 || columns == 0 || sms == 0 {
        return None;
    }
    let row_tiles = rows.div_ceil(SMALL_GEMM_TILE[0]);
    let ctas = row_tiles.checked_mul(outputs.div_ceil(SMALL_GEMM_TILE[1]))?;
    let steps = columns.div_ceil(SMALL_GEMM_TILE_K);
    let maximum = sms.div_ceil(ctas).min(steps).min(SPLIT_LIMIT);
    for splits in (1..=maximum).rev() {
        if splits <= 1 || !steps.is_multiple_of(splits) {
            continue;
        }
        let shape = [
            row_tiles
                .checked_mul(SMALL_GEMM_TILE[0])?
                .checked_mul(splits)?,
            outputs
                .div_ceil(SMALL_GEMM_TILE[1])
                .checked_mul(SMALL_GEMM_TILE[1])?,
        ];
        let bytes = shape[0]
            .checked_mul(shape[1])?
            .checked_mul(F32_BYTES)?
            .checked_mul(PAIR_OUTPUTS)?;
        if bytes <= SCRATCH_LIMIT {
            return Some(Plan {
                splits,
                k_tiles: steps / splits,
                shape,
                scratch_bytes: bytes,
            });
        }
    }
    None
}

fn plans(
    graph: &DataflowGraph,
    weights: &ProgramWeights,
    rows: usize,
    profile: &DeviceProfile,
) -> Result<BTreeMap<Shape, Plan>> {
    let mut plans = BTreeMap::new();
    #[cfg(test)]
    if !ENABLED.with(Cell::get) {
        return Ok(plans);
    }
    if rows == 0 || !weights.expert_banks.is_empty() {
        return Ok(plans);
    }
    let sms = usize::try_from(profile.multiprocessors)
        .map_err(|_| Error::invalid("GDN pair SM count"))?;
    let mut retained = 0usize;
    for &index in gdn_ba_pairs(graph, weights).keys() {
        let node = &graph.nodes[index];
        let Some(ProjectionWeight::Dense(weight)) = weights.projections.get(&node.inputs[1]) else {
            continue;
        };
        let [outputs, columns] = weight.shape() else {
            continue;
        };
        let outputs = usize::try_from(*outputs).map_err(|_| Error::invalid("GDN pair outputs"))?;
        let columns = usize::try_from(*columns).map_err(|_| Error::invalid("GDN pair columns"))?;
        let shape = Shape {
            rows,
            outputs,
            columns,
        };
        if pool_gemv::supports(rows, outputs, columns) || plans.contains_key(&shape) {
            continue;
        }
        let Some(plan) = plan_for(rows, outputs, columns, sms) else {
            continue;
        };
        let total = retained
            .checked_add(plan.scratch_bytes)
            .ok_or_else(|| Error::invalid("GDN pair retained scratch overflow"))?;
        if total <= SCRATCH_LIMIT {
            retained = total;
            plans.insert(shape, plan);
        }
    }
    Ok(plans)
}

struct Buffers {
    plan: Plan,
    partials: [Tensor<f32>; PAIR_OUTPUTS],
}

/// One graph's scratch owners; a reduction consumes each pair before its reuse.
pub struct Workspace {
    buffers: BTreeMap<Shape, Buffers>,
}

impl Workspace {
    /// Retained tensor bytes for the selected distinct shapes.
    pub fn storage_bytes(
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        rows: usize,
        profile: &DeviceProfile,
    ) -> Result<u64> {
        let bytes = plans(graph, weights, rows, profile)?
            .values()
            .map(|plan| plan.scratch_bytes)
            .sum::<usize>();
        u64::try_from(bytes).map_err(|_| Error::invalid("GDN pair scratch bytes"))
    }

    pub(super) fn new(
        device: &CudaDevice,
        graph: &DataflowGraph,
        weights: &ProgramWeights,
        rows: usize,
    ) -> Result<Self> {
        let mut buffers = BTreeMap::new();
        for (shape, plan) in plans(graph, weights, rows, device.profile()?)? {
            let allocate = || {
                api::zeros::<f32>(&plan.shape)
                    .sync_on(&device.stream)
                    .map_err(device_error)
            };
            buffers.insert(
                shape,
                Buffers {
                    plan,
                    partials: [allocate()?, allocate()?],
                },
            );
        }
        Ok(Self { buffers })
    }

    pub(super) fn record(
        &mut self,
        scope: &Scope,
        input: &TensorView<'_, f32>,
        (weight_b, weight_a): (&Tensor<bf16>, &Tensor<bf16>),
        (out_b, out_a): (&mut Tensor<f32>, &mut Tensor<f32>),
    ) -> StdResult<bool, DeviceError> {
        let shape = Shape {
            rows: usize::try_from(out_b.shape()[0]).map_err(error)?,
            outputs: usize::try_from(out_b.shape()[1]).map_err(error)?,
            columns: usize::try_from(input.shape()[1]).map_err(error)?,
        };
        let Some(work) = self.buffers.get_mut(&shape) else {
            return Ok(false);
        };
        let [partial_b, partial_a] = &mut work.partials;
        scope.record(
            kernels::dense_pair_split(
                (&mut *partial_b).partition(SMALL_GEMM_TILE),
                (&mut *partial_a).partition(SMALL_GEMM_TILE),
                input,
                weight_b,
                weight_a,
                i32::try_from(work.plan.k_tiles).map_err(error)?,
                i32::try_from(shape.rows.div_ceil(SMALL_GEMM_TILE[0])).map_err(error)?,
            )
            .generics(vec![shape.columns.to_string()]),
        )?;
        scope.record(kernels::reduce_pair_split(
            out_b.partition(SMALL_GEMM_TILE),
            out_a.partition(SMALL_GEMM_TILE),
            &*partial_b,
            &*partial_a,
            i32::try_from(work.plan.splits).map_err(error)?,
            i32::try_from(shape.rows.div_ceil(SMALL_GEMM_TILE[0])).map_err(error)?,
        ))?;
        #[cfg(test)]
        RECORDS.with(|records| records.set(records.get().saturating_add(1)));
        Ok(true)
    }
}

#[cfg(test)]
thread_local! {
    static ENABLED: Cell<bool> = const { Cell::new(true) };
    static RECORDS: Cell<usize> = const { Cell::new(0) };
}

#[cfg(test)]
pub(super) fn captures() -> usize {
    RECORDS.with(Cell::get)
}

#[cfg(test)]
pub(super) fn with_disabled<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            ENABLED.with(|enabled| enabled.set(self.0));
        }
    }
    let _restore = Restore(ENABLED.with(|enabled| enabled.replace(false)));
    run()
}
